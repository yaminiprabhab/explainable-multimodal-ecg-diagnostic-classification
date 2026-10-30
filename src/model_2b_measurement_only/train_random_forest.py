"""Random Forest one-vs-label trainers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .classifiers import rf_factory, select_on_validation


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    grid: dict[str, list[Any]],
    seed: int,
    selection_metric: str,
    extra_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    base = dict(extra_params or {})
    def builder(params: dict[str, Any]):
        merged = {**base, **params}
        return rf_factory(merged, seed)

    return select_on_validation(
        X_train,
        y_train,
        X_val,
        y_val,
        builder,
        grid,
        name="random_forest",
        metric=selection_metric,
    )
