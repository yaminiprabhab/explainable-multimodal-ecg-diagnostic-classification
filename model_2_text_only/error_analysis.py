"""Test-set error analysis using leakage-controlled text only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import LABEL_ORDER


def collect_error_examples(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    thresholds: list[float],
    n_examples: int = 12,
    label_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    order = label_order or LABEL_ORDER
    records = []
    for i in range(len(frame)):
        true_vec = y_true[i]
        pred_vec = y_pred[i]
        fp = [order[j] for j in range(len(order)) if pred_vec[j] == 1 and true_vec[j] == 0]
        fn = [order[j] for j in range(len(order)) if pred_vec[j] == 0 and true_vec[j] == 1]
        kind = "correct" if not fp and not fn else "error"
        records.append(
            {
                "example_id": i,
                "kind": kind,
                "clinical_text": str(frame.iloc[i]["clinical_text"]),
                "true_labels": {order[j]: int(true_vec[j]) for j in range(len(order))},
                "predicted_labels": {order[j]: int(pred_vec[j]) for j in range(len(order))},
                "probabilities": {order[j]: float(y_prob[i, j]) for j in range(len(order))},
                "thresholds": {order[j]: float(thresholds[j]) for j in range(len(order))},
                "false_positives": fp,
                "false_negatives": fn,
            }
        )
    errors = [r for r in records if r["kind"] == "error"]
    correct = [r for r in records if r["kind"] == "correct"]
    selected = errors[: n_examples // 2 + n_examples % 2] + correct[: n_examples // 2]
    if len(selected) < n_examples:
        selected.extend(records[len(selected) : n_examples])
    return selected[:n_examples]


def error_summary(y_true: np.ndarray, y_pred: np.ndarray, label_order: list[str] | None = None) -> dict[str, Any]:
    order = label_order or LABEL_ORDER
    out: dict[str, Any] = {}
    for i, name in enumerate(order):
        fp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        out[name] = {"false_positives": fp, "false_negatives": fn}
    return out
