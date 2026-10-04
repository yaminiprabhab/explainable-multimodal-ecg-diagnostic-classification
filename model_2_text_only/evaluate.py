"""Multi-label evaluation with undefined-safe AUROC/AUPRC."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import LABEL_ORDER


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if y_true.sum() == 0:
        return None
    return float(average_precision_score(y_true, y_prob))


def evaluate_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    thresholds: list[float],
    n_studies: int,
    n_subjects: int,
    label_order: list[str] | None = None,
) -> dict[str, Any]:
    order = label_order or LABEL_ORDER
    report: dict[str, Any] = {
        "n_studies": int(n_studies),
        "n_subjects": int(n_subjects),
        "label_order": order,
        "thresholds": {name: float(t) for name, t in zip(order, thresholds)},
        "macro_f1": _maybe_f1(y_true, y_pred, average="macro"),
        "micro_f1": _maybe_f1(y_true, y_pred, average="micro"),
        "macro_precision": _maybe_prf(y_true, y_pred, precision_score, "macro"),
        "macro_recall": _maybe_prf(y_true, y_pred, recall_score, "macro"),
        "micro_precision": _maybe_prf(y_true, y_pred, precision_score, "micro"),
        "micro_recall": _maybe_prf(y_true, y_pred, recall_score, "micro"),
        "per_label": {},
    }
    aurocs = []
    auprcs = []
    for i, name in enumerate(order):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        pr = y_prob[:, i]
        tn, fp, fn, tp = _confusion_counts(yt, yp)
        auroc = _safe_auroc(yt, pr)
        auprc = _safe_auprc(yt, pr)
        if auroc is not None:
            aurocs.append(auroc)
        if auprc is not None:
            auprcs.append(auprc)
        report["per_label"][name] = {
            "support": int(yt.sum()),
            "prevalence": float(yt.mean()) if len(yt) else None,
            "threshold": float(thresholds[i]),
            "precision": _binary_prf(yt, yp, precision_score),
            "recall": _binary_prf(yt, yp, recall_score),
            "f1": _binary_prf(yt, yp, f1_score),
            "auroc": auroc,
            "auprc": auprc,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "undefined_auroc_reason": None if auroc is not None else "single_class_in_split",
            "undefined_auprc_reason": None if auprc is not None else "no_positive_examples",
        }
    report["macro_auroc"] = float(np.mean(aurocs)) if aurocs else None
    report["macro_auprc"] = float(np.mean(auprcs)) if auprcs else None
    if not aurocs:
        report["macro_auroc_undefined_reason"] = "no_label_with_both_classes"
    return report


def _maybe_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> float | None:
    if y_true.size == 0:
        return None
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def _maybe_prf(y_true: np.ndarray, y_pred: np.ndarray, fn, average: str) -> float | None:
    if y_true.size == 0:
        return None
    return float(fn(y_true, y_pred, average=average, zero_division=0))


def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray, fn) -> float:
    return float(fn(y_true, y_pred, zero_division=0))


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)
