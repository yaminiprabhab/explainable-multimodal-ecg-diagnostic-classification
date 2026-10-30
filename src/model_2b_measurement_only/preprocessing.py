"""Training-only median imputation and missingness indicators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .feature_engineering import AXIS_COLUMNS
from .utils import assert_feature_allowlist, forbidden_columns


@dataclass
class FittedPreprocessor:
    feature_columns: list[str]
    model_columns: list[str]
    imputer: SimpleImputer
    missing_indicator_features: list[str]
    circular_axis_columns: list[str] = field(default_factory=list)
    scaler: StandardScaler | None = None
    imputer_statistics: dict[str, float] = field(default_factory=dict)

    def transform(self, frame: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
        assert_feature_allowlist(self.feature_columns, self.feature_columns)
        values = frame[self.feature_columns].to_numpy(dtype=float)
        missing = pd.DataFrame(index=frame.index)
        for col in self.missing_indicator_features:
            missing[f"{col}_missing"] = frame[col].isna().astype(int)
        imputed = self.imputer.transform(values)
        out = pd.DataFrame(imputed, columns=self.feature_columns, index=frame.index)
        out = _append_circular_axes(out, self.circular_axis_columns)
        if len(self.missing_indicator_features):
            out = pd.concat([out, missing], axis=1)
        leaked = forbidden_columns(list(out.columns))
        if leaked:
            raise ValueError(f"Forbidden columns after transform: {leaked}")
        if scale:
            if self.scaler is None:
                raise RuntimeError("Scaler was not fitted")
            scaled = self.scaler.transform(out[self.model_columns].to_numpy(dtype=float))
            out = pd.DataFrame(scaled, columns=self.model_columns, index=out.index)
        else:
            out = out[self.model_columns]
        return out


def _append_circular_axes(frame: pd.DataFrame, axis_columns: list[str]) -> pd.DataFrame:
    if not axis_columns:
        return frame
    out = frame.copy()
    for col in axis_columns:
        radians = np.deg2rad(out[col].astype(float))
        out[f"{col}_sin"] = np.sin(radians)
        out[f"{col}_cos"] = np.cos(radians)
    return out


def circular_feature_names(axis_columns: list[str]) -> list[str]:
    names: list[str] = []
    for col in axis_columns:
        names.extend([f"{col}_sin", f"{col}_cos"])
    return names


def fit_preprocessor(
    train: pd.DataFrame,
    feature_columns: list[str],
    missing_indicator_min_rate: float = 0.05,
    scale: bool = False,
    circular_axes: bool = True,
) -> tuple[FittedPreprocessor, dict[str, Any]]:
    assert_feature_allowlist(feature_columns, feature_columns)
    missing_rates = {col: float(train[col].isna().mean()) for col in feature_columns}
    indicator_cols = [
        col for col, rate in missing_rates.items() if rate >= float(missing_indicator_min_rate)
    ]
    entirely_missing = [col for col, rate in missing_rates.items() if rate >= 1.0]
    if entirely_missing:
        raise ValueError(
            "These features are entirely missing in the training set and cannot be imputed "
            f"from training statistics: {entirely_missing}"
        )
    imputer = SimpleImputer(strategy="median")
    train_values = train[feature_columns].to_numpy(dtype=float)
    imputed = imputer.fit_transform(train_values)
    stats = {
        col: (float(imputer.statistics_[i]) if np.isfinite(imputer.statistics_[i]) else None)
        for i, col in enumerate(feature_columns)
    }
    circular_cols = [c for c in AXIS_COLUMNS if c in feature_columns] if circular_axes else []
    design = pd.DataFrame(imputed, columns=feature_columns, index=train.index)
    design = _append_circular_axes(design, circular_cols)
    for col in indicator_cols:
        design[f"{col}_missing"] = train[col].isna().astype(int)
    model_columns = (
        list(feature_columns)
        + circular_feature_names(circular_cols)
        + [f"{c}_missing" for c in indicator_cols]
    )
    scaler = None
    if scale:
        scaler = StandardScaler()
        scaler.fit(design[model_columns].to_numpy(dtype=float))
    preprocessor = FittedPreprocessor(
        feature_columns=list(feature_columns),
        model_columns=model_columns,
        imputer=imputer,
        missing_indicator_features=indicator_cols,
        circular_axis_columns=circular_cols,
        scaler=scaler,
        imputer_statistics=stats,
    )
    report = {
        "missing_rates_train": missing_rates,
        "missing_indicator_features": indicator_cols,
        "imputer_strategy": "median",
        "imputer_statistics_train_only": stats,
        "circular_axis_columns": circular_cols,
        "circular_features": circular_feature_names(circular_cols),
        "model_columns": model_columns,
        "scaler_fitted": scale,
        "n_train_rows": int(len(train)),
    }
    return preprocessor, report
