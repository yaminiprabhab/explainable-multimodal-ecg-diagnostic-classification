"""Per-label permutation importance on frozen models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from model_2_text_only.utils import LABEL_ORDER

from .classifiers import ConstantLabelFallback, MultiLabelBinaryEnsemble


def permutation_importance_by_label(
    model: MultiLabelBinaryEnsemble,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int,
    seed: int,
    dataset_name: str,
) -> pd.DataFrame:
    rows = []
    for i, label in enumerate(LABEL_ORDER):
        est = model.estimators[i]
        if isinstance(est, ConstantLabelFallback) or len(np.unique(y[:, i])) < 2:
            rows.append(
                {
                    "feature": None,
                    "importance_mean": None,
                    "importance_std": None,
                    "label": label,
                    "dataset": dataset_name,
                    "method": "permutation_roc_auc",
                    "status": "skipped_single_class_or_fallback",
                }
            )
            continue
        result = permutation_importance(
            est,
            X,
            y[:, i],
            n_repeats=int(n_repeats),
            random_state=seed + i,
            scoring=_roc_auc_scorer,
        )
        for name, mean, std in zip(feature_names, result.importances_mean, result.importances_std):
            rows.append(
                {
                    "feature": name,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                    "label": label,
                    "dataset": dataset_name,
                    "method": "permutation_roc_auc",
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def impurity_importance(model: MultiLabelBinaryEnsemble, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for i, label in enumerate(LABEL_ORDER):
        est = model.estimators[i]
        values = getattr(est, "feature_importances_", None)
        if values is None:
            continue
        for name, score in zip(feature_names, values):
            rows.append(
                {
                    "feature": name,
                    "importance": float(score),
                    "label": label,
                    "method": "impurity_or_gain",
                    "limitation": "Can be biased toward high-cardinality or correlated features.",
                }
            )
    return pd.DataFrame(rows)


def plot_top_features(importance: pd.DataFrame, path: Path, top_k: int = 12) -> None:
    import matplotlib.pyplot as plt

    usable = importance.dropna(subset=["importance_mean"])
    if usable.empty:
        return
    labels = [lab for lab in LABEL_ORDER if lab in set(usable["label"])]
    n = max(len(labels), 1)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.6 * n), squeeze=False)
    for ax, label in zip(axes.ravel(), labels):
        part = usable[usable["label"] == label].nlargest(top_k, "importance_mean")
        ax.barh(part["feature"][::-1], part["importance_mean"][::-1])
        ax.set_title(f"{label} permutation importance (validation)")
        ax.set_xlabel("Mean AUROC drop")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _roc_auc_scorer(estimator, X, y) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        classes = list(getattr(estimator, "classes_", [0, 1]))
        if 1 in [int(c) for c in classes] and proba.ndim == 2 and proba.shape[1] > 1:
            idx = [int(c) for c in classes].index(1)
            scores = proba[:, idx]
        elif proba.ndim == 2:
            scores = proba[:, -1]
        else:
            scores = proba
    else:
        scores = estimator.predict(X)
    return float(roc_auc_score(y, scores))
