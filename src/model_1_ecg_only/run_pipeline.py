"""Train Model 1 on the frozen split, score test, write occlusion examples.

Requires waveform_root matching the cohort (WFDB or .npy). --smoke builds
synthetic files under the caller’s artifact directory. Outputs include
test_metrics, test_predictions.csv, and explanations/*.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from shared.cohort import join_shared_cohort, split_frames
from shared.constants import LABEL_ORDER
from shared.error_analysis import collect_error_examples, error_summary
from shared.ecg_explainability import occlude_ecg, save_occlusion_heatmap
from shared.predictions import write_test_predictions
from shared.io_utils import default_config, ensure_dir, load_yaml, package_versions, repo_root, resolve_path, save_json, set_seed
from shared.smoke import build_smoke_cohort
from shared.training import (
    compute_pos_weight,
    predict_proba,
    select_and_save_thresholds,
    train_model,
    write_metric_tables,
)
from shared.thresholds import apply_thresholds
from shared.evaluate import evaluate_multilabel
from shared.waveforms import attach_usable_waveforms

from .dataset import EcgDataset, fit_normalizer
from .model import EcgOnlyModel


def run_pipeline(
    config_path: Path,
    artifacts_dir: Path | None = None,
    output_dir: Path | None = None,
    waveform_root: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    cfg = load_yaml(config_path)
    root = repo_root()
    set_seed(int(cfg["random_seed"]))
    if smoke:
        dest = Path(artifacts_dir).parent if artifacts_dir is not None else root / "artifacts" / "model_1_smoke_bundle"
        built = build_smoke_cohort(dest, n_studies=48, seed=int(cfg["random_seed"]))
        artifacts_dir = built["artifacts_dir"]
        waveform_root = built["waveform_root"]
        if output_dir is None:
            output_dir = root / "outputs" / "model_1_ecg_only_smoke"
        cfg["training"]["max_epochs"] = min(int(cfg["training"]["max_epochs"]), 2)
        cfg["explainability"]["n_examples"] = 4
        cfg["explainability"]["max_windows"] = 4
    artifacts_dir = ensure_dir(Path(artifacts_dir) if artifacts_dir else resolve_path(root, cfg["artifacts_dir"]))
    output_dir = ensure_dir(Path(output_dir) if output_dir else resolve_path(root, cfg["output_dir"]))
    waveform_root = Path(waveform_root) if waveform_root else resolve_path(root, cfg["waveform_root"])

    cohort, join_notes = join_shared_cohort(artifacts_dir, require_text=False)
    usable, index, wf_stats = attach_usable_waveforms(
        cohort, waveform_root, index_path=artifacts_dir / "waveform_index.csv"
    )
    save_json({"join": join_notes, "waveforms": wf_stats}, output_dir / "inclusion_exclusion_report.json")
    if usable.empty:
        raise RuntimeError(
            "No usable waveforms matched the frozen cohort. Place MIMIC-IV-ECG WFDB or .npy files under waveform_root."
        )
    train_df, val_df, test_df = split_frames(usable)

    cache: dict[int, Any] = {}
    normalizer = fit_normalizer(train_df, cache=cache)
    save_json(normalizer.to_dict(), output_dir / "ecg_normalizer.json")

    train_ds = EcgDataset(train_df, normalizer, cache=cache)
    val_ds = EcgDataset(val_df, normalizer, cache=cache)
    test_ds = EcgDataset(test_df, normalizer, cache=cache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    model = EcgOnlyModel(
        channels=list(model_cfg["channels"]),
        kernel_size=int(model_cfg.get("kernel_size", 7)),
        embedding_dim=int(model_cfg["embedding_dim"]),
        dropout=float(model_cfg["dropout"]),
        num_labels=int(model_cfg["num_labels"]),
    ).to(device)

    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    nw = int(train_cfg["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw)

    pos_weight = compute_pos_weight(train_ds.labels) if train_cfg.get("use_pos_weight", True) else None
    train_result = train_model(
        model,
        train_loader,
        val_loader,
        device,
        output_dir,
        max_epochs=int(train_cfg["max_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        patience=int(train_cfg["early_stopping_patience"]),
        pos_weight=pos_weight,
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
    write_test_predictions(
        output_dir / "test_predictions.csv",
        test_out["study_id"],
        test_out["true"],
        test_out["prob"],
        test_pred,
    )

    examples = collect_error_examples(
        test_df.reset_index(drop=True),
        test_out["true"],
        test_out["prob"],
        test_pred,
        thresholds,
        n_examples=int(cfg["explainability"]["n_examples"]),
        extra_fields=(),
    )
    explained = []
    for ex in examples:
        row = test_df.reset_index(drop=True).iloc[ex["example_id"]]
        signal = test_ds.normalizer.transform(test_ds._signal(ex["example_id"]))
        occ = occlude_ecg(
            model,
            signal,
            device,
            window=int(cfg["explainability"]["window"]),
            max_windows=int(cfg["explainability"]["max_windows"]),
        )
        explained.append({**ex, "ecg_occlusion": occ, "path_basename": Path(str(row["path"])).name})
        if len(explained) <= 3:
            top_label = max(occ["base_probabilities"], key=occ["base_probabilities"].get)
            save_occlusion_heatmap(
                occ,
                output_dir / "explanations" / f"example_{ex['example_id']}_{top_label}.png",
                label=top_label,
            )
    save_json(
        {
            "summary": error_summary(test_out["true"], test_pred),
            "examples": explained,
            "note": "example_id is a split-local index. Patient identifiers are omitted from this public report.",
        },
        output_dir / "error_analysis.json",
    )
    save_json(
        {
            "architecture": model_cfg,
            "device": str(device),
            "label_order": LABEL_ORDER,
            "seed": cfg["random_seed"],
            "experiment_type": "ecg_waveform_only",
            "finetune_policy": "trained_from_scratch",
        },
        output_dir / "model_config.json",
    )
    save_json(
        {"package_versions": package_versions(), "seed": cfg["random_seed"], "smoke": smoke},
        output_dir / "reproducibility.json",
    )
    summary = {
        "stopped_before_training": False,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model 1 ECG-only CNN baseline")
    parser.add_argument("--config", default=str(default_config("model_1_ecg_only")))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run_pipeline(Path(args.config), smoke=args.smoke)
    metrics = summary.get("test_metrics") or {}
    print("macro_f1", metrics.get("macro_f1"), "micro_f1", metrics.get("micro_f1"))


if __name__ == "__main__":
    main()
