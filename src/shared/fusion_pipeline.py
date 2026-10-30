"""Train Model 3 (concat) or Model 4 (gated attention) on the shared cohort.

Joins frozen labels/split/residual text with usable waveforms, fits a
train-only tokenizer and ECG normalizer, optionally warms encoders from
Model 1/2 checkpoints, then applies the same BCE / validation-threshold /
test-metric protocol as the unimodal models.

--smoke writes synthetic artifacts and disables checkpoint loading so vocab
sizes cannot clash with a real Model 2 run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from model_1_ecg_only.dataset import fit_normalizer
from model_2_text_only.tokenizer import Vocabulary
from shared.cohort import join_shared_cohort, split_frames
from shared.constants import LABEL_ORDER
from shared.ecg_explainability import occlude_ecg, save_occlusion_heatmap
from shared.predictions import write_test_predictions
from shared.error_analysis import collect_error_examples, error_summary
from shared.evaluate import evaluate_multilabel
from shared.fusion import ConcatFusionModel, GatedAttentionFusionModel, load_encoder_subsets
from shared.io_utils import ensure_dir, load_yaml, package_versions, repo_root, resolve_path, save_json, set_seed
from shared.multimodal_dataset import MultimodalDataset
from shared.smoke import build_smoke_cohort
from shared.thresholds import apply_thresholds
from shared.training import (
    compute_pos_weight,
    predict_proba,
    select_and_save_thresholds,
    train_model,
    write_metric_tables,
)
from shared.waveforms import attach_usable_waveforms


def run_fusion_pipeline(
    config_path: Path,
    fusion_type: str,
    artifacts_dir: Path | None = None,
    output_dir: Path | None = None,
    waveform_root: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train concat or gated fusion and write metrics, predictions, and occlusions."""
    if fusion_type not in {"concat", "gated"}:
        raise ValueError(fusion_type)
    config_path = config_path.resolve()
    cfg = load_yaml(config_path)
    root = repo_root()
    set_seed(int(cfg["random_seed"]))
    if smoke:
        dest = Path(artifacts_dir).parent if artifacts_dir is not None else root / "artifacts" / f"fusion_{fusion_type}_smoke_bundle"
        built = build_smoke_cohort(dest, n_studies=48, seed=int(cfg["random_seed"]))
        artifacts_dir = built["artifacts_dir"]
        waveform_root = built["waveform_root"]
        if output_dir is None:
            output_dir = root / "outputs" / f"model_fusion_{fusion_type}_smoke"
        cfg["training"]["max_epochs"] = min(int(cfg["training"]["max_epochs"]), 2)
        cfg["explainability"]["n_examples"] = 4
        cfg["explainability"]["max_windows"] = 4
        cfg.setdefault("init", {})
        cfg["init"]["ecg_checkpoint"] = None
        cfg["init"]["text_checkpoint"] = None
    artifacts_dir = ensure_dir(Path(artifacts_dir) if artifacts_dir else resolve_path(root, cfg["artifacts_dir"]))
    output_dir = ensure_dir(Path(output_dir) if output_dir else resolve_path(root, cfg["output_dir"]))
    waveform_root = Path(waveform_root) if waveform_root else resolve_path(root, cfg["waveform_root"])

    cohort, join_notes = join_shared_cohort(artifacts_dir, require_text=True)
    usable, _index, wf_stats = attach_usable_waveforms(
        cohort, waveform_root, index_path=artifacts_dir / "waveform_index.csv"
    )
    save_json({"join": join_notes, "waveforms": wf_stats, "fusion": fusion_type}, output_dir / "inclusion_exclusion_report.json")
    train_df, val_df, test_df = split_frames(usable)

    tok_cfg = cfg["tokenizer"]
    vocab = Vocabulary(
        pad_token=tok_cfg["pad_token"],
        unk_token=tok_cfg["unk_token"],
        mask_token=str(tok_cfg["mask_token"]).lower(),
        min_freq=int(tok_cfg["min_token_freq"]),
    )
    vocab.fit(
        train_df["clinical_text"].tolist(),
        max_seq_len=tok_cfg.get("max_seq_len"),
        percentile=int(tok_cfg.get("max_seq_len_percentile", 95)),
        cap=int(tok_cfg.get("max_seq_len_cap", 64)),
    )
    vocab.save(output_dir / "vocabulary.json")

    cache: dict[int, Any] = {}
    normalizer = fit_normalizer(train_df, cache=cache)
    save_json(normalizer.to_dict(), output_dir / "ecg_normalizer.json")

    train_ds = MultimodalDataset(train_df, vocab, normalizer, cache=cache)
    val_ds = MultimodalDataset(val_df, vocab, normalizer, cache=cache)
    test_ds = MultimodalDataset(test_df, vocab, normalizer, cache=cache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common_kwargs = dict(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        text_cfg=cfg["text_model"],
        ecg_cfg=cfg["ecg_model"],
        dropout=float(cfg["training"].get("dropout", 0.3)),
    )
    if fusion_type == "concat":
        model = ConcatFusionModel(**common_kwargs).to(device)
    else:
        model = GatedAttentionFusionModel(**common_kwargs, fused_dim=int(cfg.get("fusion", {}).get("fused_dim", 128))).to(device)

    init_notes = load_encoder_subsets(
        model,
        resolve_path(root, cfg["init"].get("ecg_checkpoint", "")) if cfg.get("init", {}).get("ecg_checkpoint") else None,
        resolve_path(root, cfg["init"].get("text_checkpoint", "")) if cfg.get("init", {}).get("text_checkpoint") else None,
        device,
    )

    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    nw = int(train_cfg["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw)

    pos_weight = compute_pos_weight(train_ds.labels) if train_cfg.get("use_pos_weight", True) else None
    text_lr = float(train_cfg.get("text_learning_rate", train_cfg["learning_rate"]))
    ecg_lr = float(train_cfg["learning_rate"])
    param_groups = [
        {"params": model.ecg_encoder.parameters(), "lr": ecg_lr},
        {"params": model.text_encoder.parameters(), "lr": text_lr},
    ]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("ecg_encoder.") and not n.startswith("text_encoder.")]
    param_groups.append({"params": head_params, "lr": ecg_lr})

    train_result = train_model(
        model,
        train_loader,
        val_loader,
        device,
        output_dir,
        max_epochs=int(train_cfg["max_epochs"]),
        learning_rate=ecg_lr,
        weight_decay=float(train_cfg["weight_decay"]),
        patience=int(train_cfg["early_stopping_patience"]),
        pos_weight=pos_weight,
        param_groups=param_groups,
    )
    save_json(train_result, output_dir / "training_history.json")

    val_out = predict_proba(model, val_loader, device)
    thresholds = select_and_save_thresholds(
        val_out["true"],
        val_out["prob"],
        output_dir,
        strategy=str(cfg["thresholds"]["strategy"]),
        fallback=float(cfg["thresholds"]["fallback"]),
        min_val_positives=int(cfg["thresholds"]["min_val_positives"]),
    )
    test_out = predict_proba(model, test_loader, device, extra_keys=("study_id",))
    test_pred = apply_thresholds(test_out["prob"], thresholds)
    metrics = evaluate_multilabel(
        test_out["true"],
        test_out["prob"],
        test_pred,
        thresholds,
        n_studies=len(test_df),
        n_subjects=int(test_df["subject_id"].nunique()),
    )
    write_metric_tables(metrics, output_dir)
    pred_frame = write_test_predictions(
        output_dir / "test_predictions.csv",
        test_out["study_id"],
        test_out["true"],
        test_out["prob"],
        test_pred,
    )
    if fusion_type == "gated":
        gates = _collect_gates(model, test_loader, device)
        pred_frame["gate_ecg"] = gates[:, 0]
        pred_frame["gate_text"] = gates[:, 1]
        pred_frame.to_csv(output_dir / "test_predictions.csv", index=False)
        save_json(
            {
                "mean_gate_ecg": float(gates[:, 0].mean()),
                "mean_gate_text": float(gates[:, 1].mean()),
                "note": "Softmax modality gates. Not causal explanations.",
            },
            output_dir / "attention_gates.json",
        )

    examples = collect_error_examples(
        test_df.reset_index(drop=True),
        test_out["true"],
        test_out["prob"],
        test_pred,
        thresholds,
        n_examples=int(cfg["explainability"]["n_examples"]),
        extra_fields=("clinical_text",),
    )
    explained = []
    for ex in examples:
        idx = ex["example_id"]
        signal = test_ds.normalizer.transform(test_ds._signal(idx))
        extra = {
            "input_ids": torch.tensor(test_ds.input_ids[idx][None, ...], dtype=torch.long),
            "lengths": torch.tensor([int(test_ds.lengths[idx])], dtype=torch.long),
        }
        occ_ecg = occlude_ecg(
            model,
            signal,
            device,
            window=int(cfg["explainability"]["window"]),
            max_windows=int(cfg["explainability"]["max_windows"]),
            extra_batch=extra,
        )
        occ_text = _occlude_text_with_ecg(
            model,
            vocab,
            signal,
            str(ex.get("clinical_text", "")),
            device,
            int(cfg["explainability"].get("max_tokens_to_occlude", 40)),
        )
        item = {**ex, "ecg_occlusion": occ_ecg, "text_occlusion": occ_text}
        if len(explained) < 3:
            top_label = max(occ_ecg["base_probabilities"], key=occ_ecg["base_probabilities"].get)
            save_occlusion_heatmap(
                occ_ecg,
                output_dir / "explanations" / f"example_{ex['example_id']}_{top_label}.png",
                label=top_label,
            )
        if fusion_type == "gated":
            item["gates"] = {
                "ecg": float(pred_frame.iloc[idx]["gate_ecg"]) if idx < len(pred_frame) else None,
                "text": float(pred_frame.iloc[idx]["gate_text"]) if idx < len(pred_frame) else None,
            }
        explained.append(item)
    save_json(
        {
            "summary": error_summary(test_out["true"], test_pred),
            "examples": explained,
            "note": "example_id is a split-local index. Patient identifiers are omitted. Gate weights are not causal explanations.",
        },
        output_dir / "error_analysis.json",
    )
    save_json(
        {
            "fusion": fusion_type,
            "init": init_notes,
            "finetune_policy": "finetune_both_encoders_lower_text_lr",
            "text_learning_rate": text_lr,
            "ecg_learning_rate": ecg_lr,
            "device": str(device),
            "label_order": LABEL_ORDER,
            "seed": cfg["random_seed"],
            "formula": (
                "concat([z_ecg, z_text])"
                if fusion_type == "concat"
                else "z = a_ecg * P(z_ecg) + a_text * P(z_text), a = softmax(W [P(z_ecg); P(z_text)])"
            ),
        },
        output_dir / "model_config.json",
    )
    save_json({"package_versions": package_versions(), "seed": cfg["random_seed"], "smoke": smoke}, output_dir / "reproducibility.json")
    summary = {
        "stopped_before_training": False,
        "fusion": fusion_type,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "waveform_stats": wf_stats,
        "test_metrics": metrics,
        "device": str(device),
        "output_dir": str(output_dir),
    }
    save_json(summary, output_dir / "run_summary.json")
    return summary


@torch.no_grad()
def _collect_gates(model: GatedAttentionFusionModel, loader: DataLoader, device: torch.device) -> Any:
    import numpy as np

    model.eval()
    rows = []
    for batch in loader:
        _fused, gates = model.fuse(
            batch["ecg"].to(device),
            batch["input_ids"].to(device),
            batch["lengths"].to(device),
        )
        rows.append(gates.cpu().numpy())
    return np.concatenate(rows, axis=0)


def _occlude_text_with_ecg(model, vocab, signal, text, device, max_tokens: int) -> dict[str, Any]:
    from model_2_text_only.tokenizer import tokenize
    import numpy as np

    model.eval()
    tokens = tokenize(text)[: vocab.max_seq_len]
    ecg = torch.tensor(signal[None, ...], dtype=torch.float32, device=device)

    def prob_for(txt: str) -> np.ndarray:
        ids, length, _ = vocab.encode(txt)
        logits = model(
            ecg,
            torch.tensor([ids], dtype=torch.long, device=device),
            torch.tensor([length], dtype=torch.long, device=device),
        )
        return torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()

    base = prob_for(text)
    attrs = []
    for i, tok in enumerate(tokens[:max_tokens]):
        occluded = tokens.copy()
        occluded[i] = vocab.unk_token
        delta = base - prob_for(" ".join(occluded))
        attrs.append({"position": i, "token": tok, "delta_prob": {n: float(delta[j]) for j, n in enumerate(LABEL_ORDER)}})
    return {
        "text": text,
        "tokens": tokens,
        "base_probabilities": {n: float(base[i]) for i, n in enumerate(LABEL_ORDER)},
        "attributions": attrs,
        "method": "token_occlusion_with_ecg_held_fixed",
    }

