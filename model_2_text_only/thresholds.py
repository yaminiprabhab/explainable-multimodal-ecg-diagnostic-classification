"""Validation-only decision thresholds."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from .utils import LABEL_ORDER


def select_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "per_label_f1",
    fallback: float = 0.5,
    min_val_positives: int = 5,
    label_order: list[str] | None = None,
) -> dict[str, Any]:
    order = label_order or LABEL_ORDER
    if y_true.shape != y_prob.shape or y_true.shape[1] != len(order):
        raise ValueError("y_true/y_prob shapes must match the five-label order")
    thresholds = []
    notes = []
    grid = np.linspace(0.05, 0.95, 19)
    for i, name in enumerate(order):
        pos = int(y_true[:, i].sum())
        if strategy != "per_label_f1" or pos < min_val_positives:
            thresholds.append(float(fallback))
            notes.append(
                {
                    "label": name,
                    "reason": "fallback_insufficient_validation_positives" if pos < min_val_positives else "fallback_strategy",
                    "val_positives": pos,
                    "threshold": float(fallback),
                }
            )
            continue
        best_t = fallback
        best_f1 = -1.0
        for t in grid:
            pred = (y_prob[:, i] >= t).astype(int)
            score = f1_score(y_true[:, i], pred, zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_t = float(t)
        thresholds.append(best_t)
        notes.append(
            {
                "label": name,
                "reason": "per_label_f1",
                "val_positives": pos,
                "threshold": best_t,
                "val_f1_at_threshold": best_f1,
            }
        )
    return {"thresholds": thresholds, "label_order": order, "notes": notes}


def apply_thresholds(y_prob: np.ndarray, thresholds: list[float]) -> np.ndarray:
    thresh = np.asarray(thresholds, dtype=float).reshape(1, -1)
    return (y_prob >= thresh).astype(int)
