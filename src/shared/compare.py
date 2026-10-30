"""Intersect test study_ids and re-score each model with its frozen thresholds.

Interprets Hamming wins between pairs (e.g. concat vs gated). Missing
test_predictions.csv files are listed rather than inferred. Results are
agreement with weak labels, not diagnostic accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import LABEL_ORDER
from .evaluate import evaluate_multilabel
from .io_utils import ensure_dir, save_json
from .thresholds import apply_thresholds


MODEL_DIRS = {
    "model_2_text": "outputs/model_2_text_only",
    "model_2b_measurements": "outputs/model_2b_measurement_only",
    "model_1_ecg": "outputs/model_1_ecg_only",
    "model_3_concat": "outputs/model_3_concat_fusion",
    "model_4_gated": "outputs/model_4_attention_fusion",
}

METRIC_KEYS = ["macro_f1", "micro_f1", "macro_precision", "macro_recall", "macro_auroc", "macro_auprc"]


def load_prediction_table(output_dir: Path) -> pd.DataFrame | None:
    for name in ("test_predictions.csv", "random_forest_test_predictions.csv"):
        path = output_dir / name
        if path.exists():
            return pd.read_csv(path)
    return None


def load_thresholds(output_dir: Path) -> list[float] | None:
    import json

    path = None
    for name in ("thresholds.json", "random_forest_thresholds.json"):
        candidate = output_dir / name
        if candidate.exists():
            path = candidate
            break
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("thresholds"), dict):
        return [float(data["thresholds"][name]) for name in LABEL_ORDER]
    return [float(t) for t in data["thresholds"]]


def common_study_ids(tables: dict[str, pd.DataFrame]) -> set[int]:
    ids = None
    for frame in tables.values():
        cur = set(frame["study_id"].astype(int).tolist())
        ids = cur if ids is None else ids & cur
    return ids or set()


def metrics_on_common(
    frame: pd.DataFrame,
    thresholds: list[float],
    n_subjects: int | None = None,
) -> dict[str, Any]:
    y_true = frame[[f"true_{n}" for n in LABEL_ORDER]].to_numpy(dtype=int)
    y_prob = frame[[f"prob_{n}" for n in LABEL_ORDER]].to_numpy(dtype=float)
    y_pred = apply_thresholds(y_prob, thresholds)
    return evaluate_multilabel(
        y_true,
        y_prob,
        y_pred,
        thresholds,
        n_studies=len(frame),
        n_subjects=int(n_subjects or len(frame)),
    )


def who_wins(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> dict[str, Any]:
    """Count Hamming-better studies and cases where the right model fixes/breaks the left."""
    merged = left.merge(right, on="study_id", suffixes=("_L", "_R"))
    left_better = 0
    right_better = 0
    ties = 0
    fusion_fixes = 0
    fusion_breaks = 0
    for _, row in merged.iterrows():
        true = {n: int(row[f"true_{n}_L"]) for n in LABEL_ORDER}
        h_l = sum(int(row[f"pred_{n}_L"]) == true[n] for n in LABEL_ORDER)
        h_r = sum(int(row[f"pred_{n}_R"]) == true[n] for n in LABEL_ORDER)
        if h_l > h_r:
            left_better += 1
        elif h_r > h_l:
            right_better += 1
        else:
            ties += 1
        l_err = h_l < 5
        r_err = h_r < 5
        if l_err and not r_err:
            fusion_fixes += 1
        if (not l_err) and r_err:
            fusion_breaks += 1
    return {
        "left": left_name,
        "right": right_name,
        "n": int(len(merged)),
        "left_better_hamming": left_better,
        "right_better_hamming": right_better,
        "ties": ties,
        "right_corrects_left_error": fusion_fixes,
        "right_introduces_error": fusion_breaks,
    }


def compare_output_dirs(named_dirs: dict[str, Path], dest: Path) -> dict[str, Any]:
    """Write comparison.json/md/csv for models that have prediction tables."""
    ensure_dir(dest)
    tables = {}
    thresholds = {}
    missing = []
    for name, path in named_dirs.items():
        table = load_prediction_table(path)
        thresh = load_thresholds(path)
        if table is None or thresh is None:
            missing.append(name)
            continue
        tables[name] = table
        thresholds[name] = thresh
    report: dict[str, Any] = {
        "missing_outputs": missing,
        "n_models_compared": len(tables),
        "label_order": LABEL_ORDER,
        "interpretation": (
            "Metrics measure agreement with weak report-derived labels, not clinical diagnostic accuracy. "
            "Models 2/3/4 use residual text after masking; residual leakage can remain. "
            "Model 2B measurements may correlate with the same cart software that wrote the reports."
        ),
    }
    if len(tables) < 2:
        save_json(report, dest / "comparison.json")
        (dest / "comparison.md").write_text(
            "# Model comparison\n\nNeed at least two `test_predictions.csv` files.\n",
            encoding="utf-8",
        )
        return report
    ids = common_study_ids(tables)
    report["n_common_test_studies"] = int(len(ids))
    summary_rows = []
    common_tables = {}
    for name, frame in tables.items():
        subset = frame[frame["study_id"].astype(int).isin(ids)].copy()
        common_tables[name] = subset
        metrics = metrics_on_common(subset, thresholds[name])
        report.setdefault("metrics", {})[name] = metrics
        row = {"model": name, "n_studies": metrics["n_studies"]}
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        for lab in LABEL_ORDER:
            row[f"f1_{lab}"] = metrics["per_label"][lab]["f1"]
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(dest / "comparison_table.csv", index=False)
    pairwise = []
    names = list(common_tables)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairwise.append(who_wins(common_tables[a], common_tables[b], a, b))
    report["pairwise_hamming"] = pairwise
    save_json(report, dest / "comparison.json")
    (dest / "comparison.md").write_text(_markdown(report, summary_rows, pairwise), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any], rows: list[dict[str, Any]], pairwise: list[dict[str, Any]]) -> str:
    lines = [
        "# Common-cohort model comparison",
        "",
        report["interpretation"],
        "",
        f"Common test studies: **{report.get('n_common_test_studies', 0)}**.",
        "",
        "| Model | Macro F1 | Micro F1 | Macro AUROC | Macro AUPRC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {_fmt(row.get('macro_f1'))} | {_fmt(row.get('micro_f1'))} | "
            f"{_fmt(row.get('macro_auroc'))} | {_fmt(row.get('macro_auprc'))} |"
        )
    lines.extend(["", "## Per-label F1", "", "| Model | " + " | ".join(LABEL_ORDER) + " |", "| --- | " + " | ".join(["---:" ] * 5) + " |"])
    for row in rows:
        vals = " | ".join(_fmt(row.get(f"f1_{n}")) for n in LABEL_ORDER)
        lines.append(f"| {row['model']} | {vals} |")
    lines.extend(["", "## Pairwise Hamming wins", ""])
    for p in pairwise:
        lines.append(
            f"- `{p['left']}` vs `{p['right']}` (n={p['n']}): "
            f"left better {p['left_better_hamming']}, right better {p['right_better_hamming']}, ties {p['ties']}; "
            f"right corrects left {p['right_corrects_left_error']}, right introduces error {p['right_introduces_error']}."
        )
    if report.get("missing_outputs"):
        lines.extend(["", "Missing outputs: " + ", ".join(report["missing_outputs"])])
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def default_named_dirs(root: Path) -> dict[str, Path]:
    return {name: root / path for name, path in MODEL_DIRS.items()}
