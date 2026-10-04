"""Multi-label evaluation, matching Model 2 metrics plus micro AUROC."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from model_2_text_only.evaluate import evaluate_multilabel as model2_evaluate
from model_2_text_only.utils import LABEL_ORDER


def evaluate_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    thresholds: list[float],
    n_studies: int,
    n_subjects: int,
    label_order: list[str] | None = None,
) -> dict[str, Any]:
    report = model2_evaluate(
        y_true,
        y_prob,
        y_pred,
        thresholds,
        n_studies=n_studies,
        n_subjects=n_subjects,
        label_order=label_order or LABEL_ORDER,
    )
    order = label_order or LABEL_ORDER
    for i, name in enumerate(order):
        rec = report["per_label"][name]
        rec["positive_support"] = int(y_true[:, i].sum())
        rec["negative_support"] = int((y_true[:, i] == 0).sum())
        rec["support"] = rec["positive_support"]
    try:
        report["micro_auroc"] = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))
    except ValueError:
        report["micro_auroc"] = None
        report["micro_auroc_undefined_reason"] = "single_class_across_all_label_entries"
    return report
