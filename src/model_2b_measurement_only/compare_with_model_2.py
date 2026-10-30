"""Compare Model 2B with the frozen Model 2 text-only test results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_2_text_only.dataset import ClinicalTextDataset
from model_2_text_only.model import TextSequenceEncoder
from model_2_text_only.thresholds import apply_thresholds
from model_2_text_only.tokenizer import Vocabulary
from model_2_text_only.train import predict_proba
from model_2_text_only.utils import LABEL_ORDER, load_yaml, save_json

from .evaluate import evaluate_multilabel


def compare_reports(
    model_2_metrics: dict[str, Any] | None,
    model_2b_results: dict[str, dict[str, Any]],
    common_model_2_metrics: dict[str, Any] | None,
    cohort_note: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "label_order": list(LABEL_ORDER),
        "weak_labels": True,
        "clinically_verified": False,
        "cohort": cohort_note,
        "model_2_original_test": model_2_metrics,
        "model_2_common_test": common_model_2_metrics,
        "model_2b": {},
        "deltas_vs_model_2_original": {},
        "interpretation_constraints": [
            "Both models are scored against the same report-derived weak labels.",
            "Model 2 uses leakage-controlled residual report text; Model 2B uses numerical measurements only.",
            "Do not attribute differences to modality if the test populations differ.",
            "These results do not evaluate raw-waveform CNNs or multimodal fusion.",
        ],
    }
    original = model_2_metrics or {}
    for name, payload in model_2b_results.items():
        metrics = payload.get("test_metrics") or {}
        comparison["model_2b"][name] = metrics
        comparison["deltas_vs_model_2_original"][name] = _delta(metrics, original)
    if common_model_2_metrics:
        comparison["deltas_vs_model_2_common"] = {
            name: _delta((payload.get("test_metrics") or {}), common_model_2_metrics)
            for name, payload in model_2b_results.items()
        }
    return comparison


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = ["macro_f1", "micro_f1", "macro_precision", "macro_recall", "macro_auroc", "macro_auprc"]
    out: dict[str, Any] = {}
    for key in keys:
        a, b = left.get(key), right.get(key)
        out[key] = None if a is None or b is None else float(a) - float(b)
    per = {}
    for label in LABEL_ORDER:
        l_lab = (left.get("per_label") or {}).get(label) or {}
        r_lab = (right.get("per_label") or {}).get(label) or {}
        rec = {}
        for metric in ("f1", "precision", "recall", "auroc", "auprc"):
            a, b = l_lab.get(metric), r_lab.get(metric)
            rec[metric] = None if a is None or b is None else float(a) - float(b)
        per[label] = rec
    out["per_label"] = per
    return out


def evaluate_model_2_on_studies(
    study_ids: list[Any],
    model_2_output_dir: Path,
    config_path: Path,
    cohort_manifest_path: Path,
) -> dict[str, Any] | None:
    checkpoint = model_2_output_dir / "best_model.pt"
    vocab_path = model_2_output_dir / "vocabulary.json"
    thresh_path = model_2_output_dir / "thresholds.json"
    if not (checkpoint.exists() and vocab_path.exists() and thresh_path.exists() and cohort_manifest_path.exists()):
        return None
    cohort = pd.read_csv(cohort_manifest_path)
    subset = cohort[cohort["study_id"].isin(set(study_ids))].copy()
    if "split" in subset.columns:
        subset = subset[subset["split"].eq("test")].copy()
    if subset.empty or "clinical_text" not in subset.columns:
        return None
    vocab = Vocabulary.load(vocab_path)
    cfg = load_yaml(config_path)
    model_cfg = cfg["model"]
    device = torch.device("cpu")
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
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    dataset = ClinicalTextDataset(subset, vocab)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    proba, true = predict_proba(model, loader, device)
    import json

    thresh_payload = json.loads(thresh_path.read_text(encoding="utf-8"))
    thresholds = thresh_payload["thresholds"]
    pred = apply_thresholds(proba, thresholds)
    return evaluate_multilabel(
        true,
        proba,
        pred,
        thresholds,
        n_studies=len(subset),
        n_subjects=int(subset["subject_id"].nunique()),
    )


def write_markdown_report(comparison: dict[str, Any], path: Path) -> None:
    lines = [
        "# Model 2 vs Model 2B comparison",
        "",
        "Labels are weak, report-derived targets. This is not independent clinical validation.",
        "",
        "## Cohort",
        "",
        "```json",
        _pretty(comparison.get("cohort")),
        "```",
        "",
        "## Overall metrics",
        "",
        "| Model | Macro F1 | Micro F1 | Macro AUROC | Macro AUPRC | N studies | N patients |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    m2 = comparison.get("model_2_original_test") or {}
    lines.append(_row("Model 2 original test", m2))
    common = comparison.get("model_2_common_test")
    if common:
        lines.append(_row("Model 2 common test", common))
    for name, metrics in (comparison.get("model_2b") or {}).items():
        lines.append(_row(f"Model 2B {name}", metrics or {}))
    lines.extend(["", "## Per-label F1", "", "| Model | NORM | HYP | STTC | MI | CD |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    lines.append(_label_f1_row("Model 2 original test", m2))
    if common:
        lines.append(_label_f1_row("Model 2 common test", common))
    for name, metrics in (comparison.get("model_2b") or {}).items():
        lines.append(_label_f1_row(f"Model 2B {name}", metrics or {}))
    lines.extend(["", "## Notes", ""])
    for note in comparison.get("interpretation_constraints") or []:
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pretty(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)


def _row(name: str, metrics: dict[str, Any]) -> str:
    def fmt(key: str) -> str:
        val = metrics.get(key)
        return "NA" if val is None else f"{val:.4f}"

    return (
        f"| {name} | {fmt('macro_f1')} | {fmt('micro_f1')} | {fmt('macro_auroc')} | "
        f"{fmt('macro_auprc')} | {metrics.get('n_studies', '')} | {metrics.get('n_subjects', '')} |"
    )


def _label_f1_row(name: str, metrics: dict[str, Any]) -> str:
    per = metrics.get("per_label") or {}
    cells = []
    for label in LABEL_ORDER:
        val = (per.get(label) or {}).get("f1")
        cells.append("NA" if val is None else f"{val:.4f}")
    return f"| {name} | {' | '.join(cells)} |"
