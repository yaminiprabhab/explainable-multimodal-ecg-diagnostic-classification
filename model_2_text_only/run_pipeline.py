"""End-to-end Model 2 pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data_loader import load_measurements, report_columns_of
from .data_splitting import sample_working_cohort, split_patients
from .data_validation import drop_missing_identifiers, resolve_duplicate_studies, validate_measurements
from .dataset import ClinicalTextDataset
from .error_analysis import collect_error_examples, error_summary
from .evaluate import evaluate_multilabel
from .explainability import occlude_tokens
from .label_generation import LabelMapper, generate_labels, label_prevalence
from .leakage_audit import audit_leakage
from .model import TextSequenceEncoder
from .text_preprocessing import build_clinical_text
from .thresholds import apply_thresholds, select_thresholds
from .tokenizer import Vocabulary
from .train import compute_pos_weight, predict_proba, train_model
from .utils import LABEL_ORDER, ensure_dir, load_yaml, package_versions, resolve_path, save_json, save_yaml, set_seed


def run_pipeline(
    config_path: Path,
    smoke_csv: Path | None = None,
    artifacts_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    cfg = load_yaml(config_path)
    root = config_path.parent.parent
    set_seed(int(cfg["random_seed"]))
    artifacts_dir = ensure_dir(
        Path(artifacts_dir) if artifacts_dir else resolve_path(root, cfg["artifacts_dir"])
    )
    output_dir = ensure_dir(
        Path(output_dir) if output_dir else resolve_path(root, cfg["output_dir"])
    )
    mapping_path = resolve_path(root, cfg["label_mapping_path"])
    mapper = LabelMapper.from_path(mapping_path)

    exclusion_log: dict[str, Any] = {"steps": []}

    input_csv = Path(smoke_csv) if smoke_csv else resolve_path(root, cfg["input_csv"])
    frame = load_measurements(input_csv, report_prefix=cfg["data"]["report_prefix"])
    validation = validate_measurements(frame, report_prefix=cfg["data"]["report_prefix"])
    exclusion_log["input_validation"] = validation
    _log_step(exclusion_log, "loaded", len(frame))

    frame, id_log = drop_missing_identifiers(frame)
    exclusion_log["identifier_filter"] = id_log
    _log_step(exclusion_log, "after_missing_id_drop", len(frame))

    frame, dup_log = resolve_duplicate_studies(frame)
    exclusion_log["duplicate_studies"] = dup_log
    _log_step(exclusion_log, "after_study_dedup", len(frame))

    print(f"Generating labels for {len(frame)} studies...")
    labels = generate_labels(frame, mapper, report_prefix=cfg["data"]["report_prefix"])
    prevalence = label_prevalence(labels)
    labels_full_path = artifacts_dir / "labels_full.csv"
    labels.to_csv(labels_full_path, index=False)
    prevalence.to_csv(output_dir / "label_prevalence_full.csv", index=False)
    save_yaml(mapper.mapping, artifacts_dir / "label_mapping.yaml")

    eligible = labels.copy()
    eligible["exclude_reason"] = ""
    no_text = eligible["n_statements"] == 0
    eligible.loc[no_text, "exclude_reason"] = "no_report_text"
    if cfg["data"].get("require_at_least_one_label", True):
        unmapped = eligible["unmapped"] & eligible["exclude_reason"].eq("")
        eligible.loc[unmapped, "exclude_reason"] = "unmapped_no_target_class"
    if cfg["data"].get("exclude_poor_quality_only_records", True):
        poor_only = eligible["poor_quality"] & eligible["unmapped"] & eligible["exclude_reason"].eq("")
        eligible.loc[poor_only, "exclude_reason"] = "poor_quality_unmapped"
    labeled = eligible[eligible["exclude_reason"].eq("")].copy()
    _log_step(exclusion_log, "after_label_eligibility", len(labeled), extra={"dropped": int(len(eligible) - len(labeled))})

    max_studies = int(cfg["data"]["working_cohort_max_studies"])
    if smoke_csv is not None:
        max_studies = max(max_studies, len(labeled))
    cohort = sample_working_cohort(labeled, max_studies=max_studies, seed=int(cfg["random_seed"]))
    _log_step(exclusion_log, "after_working_cohort_sample", len(cohort))

    leakage_cfg = cfg["leakage"]
    cohort = build_clinical_text(
        cohort,
        mapper,
        separator=cfg["data"]["text_separator"],
        mask_token=leakage_cfg["mask_token"],
        also_mask_summary=bool(leakage_cfg.get("also_mask_summary_phrases", True)),
        summary_phrases=list(leakage_cfg.get("summary_phrases") or []),
        leakage_mode=leakage_cfg.get("mode", "mask_trigger_statements"),
    )
    audit = audit_leakage(cohort, mapper)
    save_json(audit, artifacts_dir / "leakage_audit.json")
    save_json(audit, output_dir / "leakage_audit.json")

    if leakage_cfg.get("drop_study_if_no_remaining_text", True):
        no_remain = ~cohort["has_usable_text"]
        cohort.loc[no_remain, "exclude_reason"] = "no_text_after_leakage_mask"
        text_cohort = cohort[cohort["has_usable_text"]].copy()
    else:
        text_cohort = cohort.copy()
    _log_step(exclusion_log, "after_leakage_text_filter", len(text_cohort))

    if audit.get("stop_training") or len(text_cohort) < 20:
        result = {
            "stopped_before_training": True,
            "reason": "insufficient_independent_text_or_tiny_cohort",
            "leakage_audit": audit,
            "exclusion_log": exclusion_log,
            "n_text_cohort": int(len(text_cohort)),
        }
        save_json(result, output_dir / "run_summary.json")
        _write_cohort_and_eligibility(artifacts_dir, output_dir, eligible, text_cohort, exclusion_log, prevalence)
        return result

    split_cfg = cfg["splitting"]
    manifest, split_stats = split_patients(
        text_cohort,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(cfg["random_seed"]),
    )
    manifest_path = artifacts_dir / "patient_split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    save_json(split_stats, output_dir / "split_stats.json")

    merged = text_cohort.merge(manifest, on=["subject_id", "study_id"], how="inner")
    train_df = merged[merged["split"] == "train"].copy()
    val_df = merged[merged["split"] == "val"].copy()
    test_df = merged[merged["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("A split is empty; increase the working cohort or adjust ratios")

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

    train_ds = ClinicalTextDataset(train_df, vocab)
    val_ds = ClinicalTextDataset(val_df, vocab)
    test_ds = ClinicalTextDataset(test_df, vocab)
    save_json(
        {
            "train_truncation_rate": train_ds.truncation_rate,
            "val_truncation_rate": val_ds.truncation_rate,
            "test_truncation_rate": test_ds.truncation_rate,
            "train_length_stats": vocab.train_length_stats,
        },
        output_dir / "tokenizer_stats.json",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    model = TextSequenceEncoder(
        vocab_size=len(vocab),
        embedding_dim=int(model_cfg["embedding_dim"]),
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        bidirectional=bool(model_cfg["bidirectional"]),
        num_labels=int(model_cfg["num_labels"]),
        pad_id=vocab.pad_id,
        encoder=str(model_cfg["encoder"]),
    ).to(device)

    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg["num_workers"]))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg["num_workers"]))

    pos_weight = None
    if train_cfg.get("use_pos_weight", True):
        pos_weight = compute_pos_weight(train_ds.labels)

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
    save_json(
        {
            "architecture": model_cfg,
            "tokenizer": vocab.train_length_stats,
            "device": str(device),
            "label_order": LABEL_ORDER,
            "seed": cfg["random_seed"],
            "leakage_mode": leakage_cfg.get("mode"),
            "experiment_type": audit.get("experiment_type"),
        },
        output_dir / "model_config.json",
    )

    val_prob, val_true = predict_proba(model, val_loader, device)
    thresh_out = select_thresholds(
        val_true,
        val_prob,
        strategy=str(cfg["thresholds"]["strategy"]),
        fallback=float(cfg["thresholds"]["fallback"]),
        min_val_positives=int(cfg["thresholds"]["min_val_positives"]),
    )
    save_json(thresh_out, output_dir / "thresholds.json")
    thresholds = thresh_out["thresholds"]

    test_prob, test_true = predict_proba(model, test_loader, device)
    test_pred = apply_thresholds(test_prob, thresholds)
    metrics = evaluate_multilabel(
        test_true,
        test_prob,
        test_pred,
        thresholds,
        n_studies=len(test_df),
        n_subjects=int(test_df["subject_id"].nunique()),
    )
    save_json(metrics, output_dir / "test_metrics.json")
    _metrics_to_csv(metrics, output_dir / "test_metrics.csv")
    _per_label_csv(metrics, output_dir / "test_per_label_metrics.csv")

    examples = collect_error_examples(
        test_df.reset_index(drop=True),
        test_true,
        test_prob,
        test_pred,
        thresholds,
        n_examples=int(cfg["explainability"]["n_examples"]),
    )
    explained = []
    for ex in examples:
        occ = occlude_tokens(
            model,
            vocab,
            ex["clinical_text"],
            device,
            max_tokens=int(cfg["explainability"]["max_tokens_to_occlude"]),
        )
        explained.append({**ex, "occlusion": occ})
    save_json(
        {
            "summary": error_summary(test_true, test_pred),
            "examples": explained,
            "note": "example_id is a split-local index. Patient identifiers are omitted from this public report.",
        },
        output_dir / "error_analysis.json",
    )

    cohort_out = merged.copy()
    cohort_path = artifacts_dir / "cohort_manifest.csv"
    keep_cols = [
        "subject_id",
        "study_id",
        "split",
        "clinical_text",
        "raw_report_text",
        "trigger_statements",
        "poor_quality",
        "unmapped",
        "ambiguous",
        "uncertain_labels",
        "has_usable_text",
        "n_remaining_statements",
        "n_masked_statements",
        *LABEL_ORDER,
        *[f"evidence_{n}" for n in LABEL_ORDER],
    ]
    cohort_out[keep_cols].to_csv(cohort_path, index=False)
    labels[LABEL_ORDER + ["subject_id", "study_id", "unmapped", "ambiguous", "poor_quality"]].to_csv(
        artifacts_dir / "labels.csv", index=False
    )
    eligible.to_csv(output_dir / "eligibility_full.csv", index=False)
    save_json(exclusion_log, output_dir / "exclusion_log.json")
    save_json(
        {
            "package_versions": package_versions(),
            "seed": cfg["random_seed"],
            "config_path": str(config_path),
            "input_csv": str(input_csv),
        },
        output_dir / "reproducibility.json",
    )

    summary = {
        "stopped_before_training": False,
        "files": {
            "labels_full": str(labels_full_path),
            "labels": str(artifacts_dir / "labels.csv"),
            "cohort_manifest": str(cohort_path),
            "patient_split_manifest": str(manifest_path),
            "label_mapping": str(artifacts_dir / "label_mapping.yaml"),
            "output_dir": str(output_dir),
        },
        "n_input_rows": validation["n_rows"],
        "n_labeled_eligible": int(len(labeled)),
        "n_text_cohort": int(len(text_cohort)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "split_stats": split_stats,
        "prevalence_full": prevalence.to_dict(orient="records"),
        "leakage_audit": audit,
        "test_metrics": metrics,
        "experiment_type": audit.get("experiment_type"),
        "device": str(device),
    }
    save_json(summary, output_dir / "run_summary.json")
    return summary


def _log_step(log: dict[str, Any], name: str, n: int, extra: dict[str, Any] | None = None) -> None:
    rec = {"step": name, "n": int(n)}
    if extra:
        rec.update(extra)
    log["steps"].append(rec)
    print(f"[{name}] n={n}")


def _write_cohort_and_eligibility(
    artifacts_dir: Path,
    output_dir: Path,
    eligible: pd.DataFrame,
    text_cohort: pd.DataFrame,
    exclusion_log: dict[str, Any],
    prevalence: pd.DataFrame,
) -> None:
    eligible.to_csv(output_dir / "eligibility_full.csv", index=False)
    if len(text_cohort):
        text_cohort.to_csv(artifacts_dir / "cohort_manifest.csv", index=False)
    save_json(exclusion_log, output_dir / "exclusion_log.json")
    prevalence.to_csv(output_dir / "label_prevalence_full.csv", index=False)


def _metrics_to_csv(metrics: dict[str, Any], path: Path) -> None:
    rows = [
        {"metric": k, "value": v}
        for k, v in metrics.items()
        if k not in {"per_label", "label_order", "thresholds"}
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _per_label_csv(metrics: dict[str, Any], path: Path) -> None:
    rows = []
    for label, vals in metrics["per_label"].items():
        row = {"label": label, **vals}
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_smoke_csv(path: Path) -> Path:
    rows = []
    phrases = [
        ["Sinus rhythm", "Normal ECG"],
        ["Sinus rhythm", "Left ventricular hypertrophy", "Abnormal ECG"],
        ["Sinus tachycardia", "Nonspecific T abnormalities, diffuse leads", "Abnormal ECG"],
        ["Sinus rhythm", "Inferior infarct - age undetermined", "Abnormal ECG"],
        ["Sinus rhythm", "Right bundle branch block", "Abnormal ECG"],
        ["Atrial fibrillation", "Possible left ventricular hypertrophy"],
        ["Sinus bradycardia", "Lateral ST-T changes are nonspecific"],
        ["Ventricular pacing", "Pacemaker rhythm - no further analysis"],
        ["Sinus rhythm", "Possible anterior infarct - age undetermined"],
        ["Sinus rhythm", "Normal ECG except for rate"],
        ["Sinus rhythm", "No evidence of inferior infarct", "Normal ECG"],
        ["Sinus rhythm", "Extensive T wave changes may be due to myocardial ischemia"],
    ]
    sid = 1
    subj = 1
    for i in range(36):
        stmt = phrases[i % len(phrases)]
        row = {
            "subject_id": subj,
            "study_id": sid,
            "cart_id": 1,
            "ecg_time": "2000-01-01",
            **{f"report_{k}": stmt[k] if k < len(stmt) else None for k in range(18)},
            "bandwidth": "0.5-40",
            "filtering": "60",
            "rr_interval": 800,
            "p_onset": 1,
            "p_end": 2,
            "qrs_onset": 3,
            "qrs_end": 4,
            "t_end": 5,
            "p_axis": 1,
            "qrs_axis": 1,
            "t_axis": 1,
        }
        rows.append(row)
        sid += 1
        if i % 2 == 1:
            subj += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model 2 clinical-text-only baseline")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to config.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="Run on a synthetic fixture instead of the real CSV")
    args = parser.parse_args()
    smoke_csv = None
    artifacts_dir = None
    output_dir = None
    if args.smoke:
        smoke_csv = Path(__file__).resolve().parent / "tests" / "fixtures" / "smoke_measurements.csv"
        write_smoke_csv(smoke_csv)
        artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts" / "smoke"
        output_dir = Path(__file__).resolve().parent.parent / "outputs" / "model_2_text_only_smoke"
    summary = run_pipeline(
        Path(args.config),
        smoke_csv=smoke_csv,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
    )
    print("Run complete. stopped_before_training=", summary.get("stopped_before_training"))
    if not summary.get("stopped_before_training"):
        metrics = summary.get("test_metrics") or {}
        print("macro_f1", metrics.get("macro_f1"), "micro_f1", metrics.get("micro_f1"))


if __name__ == "__main__":
    main()
