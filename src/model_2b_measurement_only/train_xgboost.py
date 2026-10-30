"""XGBoost one-vs-label trainers."""

from __future__ import annotations

from typing import Any

import numpy as np


def xgboost_available() -> tuple[bool, str | None]:
    try:
        import xgboost  # noqa: F401

        return True, getattr(xgboost, "__version__", "unknown")
    except ImportError:
        return False, None


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    grid: dict[str, list[Any]],
    seed: int,
    selection_metric: str,
    extra_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    available, version = xgboost_available()
    if not available:
        raise ImportError("xgboost is not installed")
    from .classifiers import select_on_validation, xgb_factory

    base = dict(extra_params or {})

    def builder(params: dict[str, Any]):
        merged = {**base, **params}
        return xgb_factory(merged, seed)

    model, info = select_on_validation(
        X_train,
        y_train,
        X_val,
        y_val,
        builder,
        grid,
        name="xgboost",
        metric=selection_metric,
    )
    info["xgboost_version"] = version
    return model, info
