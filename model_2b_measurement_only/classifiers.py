"""Independent binary classifiers for the five-label task."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model_2_text_only.evaluate import _safe_auroc
from model_2_text_only.utils import LABEL_ORDER


def _positive_proba(estimator, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        classes = getattr(estimator, "classes_", np.array([0, 1]))
        if len(classes) == 1:
            fill = 1.0 if int(classes[0]) == 1 else 0.0
            return np.full(X.shape[0], fill, dtype=float)
        class_list = [int(c) for c in classes]
        if 1 not in class_list:
            return np.zeros(X.shape[0], dtype=float)
        return np.asarray(proba[:, class_list.index(1)], dtype=float)
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Estimator {type(estimator)} cannot produce probabilities")


def train_pos_weight(y: np.ndarray) -> float:
    pos = float(y.sum())
    neg = float(len(y) - pos)
    if pos == 0 or neg == 0:
        return 1.0
    return neg / pos


@dataclass
class ConstantLabelFallback:
    label_value: int
    classes_: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.classes_ = np.array([self.label_value])

    def predict_proba(self, X) -> np.ndarray:
        n = X.shape[0]
        p_pos = 1.0 if self.label_value == 1 else 0.0
        return np.column_stack([np.full(n, 1.0 - p_pos), np.full(n, p_pos)])

    def predict(self, X) -> np.ndarray:
        return np.full(X.shape[0], self.label_value, dtype=int)


class MultiLabelBinaryEnsemble:
    def __init__(self, estimators: list[Any], notes: list[dict[str, Any]], params: dict[str, Any], name: str):
        self.estimators = estimators
        self.notes = notes
        self.params = params
        self.name = name
        self.label_order = list(LABEL_ORDER)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cols = [_positive_proba(est, X) for est in self.estimators]
        proba = np.column_stack(cols)
        if proba.shape != (X.shape[0], 5):
            raise RuntimeError(f"{self.name} produced shape {proba.shape}, expected {(X.shape[0], 5)}")
        return proba


def fit_binary_models(
    X: np.ndarray,
    y: np.ndarray,
    factory: Callable[[int, np.ndarray], Any],
    params: dict[str, Any],
    name: str,
) -> MultiLabelBinaryEnsemble:
    if y.ndim != 2 or y.shape[1] != 5:
        raise ValueError("Targets must have shape (n_studies, 5) in LABEL_ORDER")
    estimators = []
    notes = []
    for i, label in enumerate(LABEL_ORDER):
        y_i = y[:, i]
        n_pos = int(y_i.sum())
        n_neg = int(len(y_i) - n_pos)
        if n_pos == 0 or n_neg == 0:
            value = 1 if n_pos else 0
            estimators.append(ConstantLabelFallback(value))
            notes.append(
                {
                    "label": label,
                    "fallback": True,
                    "reason": "single_class_in_training",
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "constant": value,
                    "limited_evaluability": True,
                }
            )
            continue
        est = factory(i, y_i)
        est.fit(X, y_i)
        estimators.append(est)
        notes.append(
            {
                "label": label,
                "fallback": False,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "pos_weight": train_pos_weight(y_i),
            }
        )
    return MultiLabelBinaryEnsemble(estimators, notes, params, name)


def rf_factory(params: dict[str, Any], seed: int) -> Callable[[int, np.ndarray], Any]:
    def _make(idx: int, y_i: np.ndarray):
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight=params.get("class_weight", "balanced"),
            random_state=seed + idx,
            n_jobs=int(params.get("n_jobs", -1)),
        )

    return _make


def logistic_factory(params: dict[str, Any], seed: int) -> Callable[[int, np.ndarray], Any]:
    def _make(idx: int, y_i: np.ndarray):
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=int(params.get("max_iter", 2000)),
            solver="lbfgs",
            random_state=seed + idx,
        )

    return _make


def xgb_factory(params: dict[str, Any], seed: int) -> Callable[[int, np.ndarray], Any]:
    from xgboost import XGBClassifier

    def _make(idx: int, y_i: np.ndarray):
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            scale_pos_weight=train_pos_weight(y_i),
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=seed + idx,
            verbosity=0,
        )

    return _make


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def select_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    factory_builder: Callable[[dict[str, Any]], Callable[[int, np.ndarray], Any]],
    grid: dict[str, list[Any]],
    name: str,
    metric: str = "macro_auroc",
) -> tuple[MultiLabelBinaryEnsemble, dict[str, Any]]:
    trials = []
    best_model = None
    best_score = -np.inf
    best_params = None
    for params in expand_grid(grid):
        model = fit_binary_models(X_train, y_train, factory_builder(params), params, name)
        proba = model.predict_proba(X_val)
        score, defined = _selection_score(y_val, proba, metric)
        trials.append({"params": params, "score": score, "n_defined_labels": defined})
        comparable = -np.inf if score is None else score
        if comparable > best_score:
            best_score = comparable
            best_model = model
            best_params = params
    if best_model is None:
        raise RuntimeError(f"No {name} configuration could be fit")
    return best_model, {
        "selected_params": best_params,
        "selected_score": None if best_score == -np.inf else float(best_score),
        "selection_metric": metric,
        "trials": trials,
        "label_notes": best_model.notes,
    }


def _selection_score(y_true: np.ndarray, y_prob: np.ndarray, metric: str) -> tuple[float | None, int]:
    if metric == "macro_auroc":
        scores = []
        for i in range(y_true.shape[1]):
            value = _safe_auroc(y_true[:, i], y_prob[:, i])
            if value is not None:
                scores.append(value)
        if not scores:
            return None, 0
        return float(np.mean(scores)), len(scores)
    if metric == "micro_auroc":
        try:
            return float(roc_auc_score(y_true.ravel(), y_prob.ravel())), 5
        except ValueError:
            return None, 0
    raise ValueError(f"Unknown selection metric {metric}")


def clone_ensemble_unfitted(model: MultiLabelBinaryEnsemble) -> list[Any]:
    return [clone(est) if not isinstance(est, ConstantLabelFallback) else ConstantLabelFallback(est.label_value) for est in model.estimators]
