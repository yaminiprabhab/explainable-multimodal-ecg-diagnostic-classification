"""Documented conversion, sentinel handling, and derived ECG intervals."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .data_loader import PRIMARY_MEASUREMENT_COLUMNS
from .utils import assert_feature_allowlist, forbidden_columns

FIDUCIAL_COLUMNS = ["p_onset", "p_end", "qrs_onset", "qrs_end", "t_end"]
AXIS_COLUMNS = ["p_axis", "qrs_axis", "t_axis"]
TIMING_COLUMNS = ["rr_interval", *FIDUCIAL_COLUMNS]


def default_validity_config() -> dict[str, Any]:
    return {
        "timing_sentinels": [29999, 32767, 65534, 65535],
        "axis_sentinels": [29999, 32767, -32767, -32768, 65534, 65535],
        "fiducial_min_ms": 0,
        "fiducial_max_ms": 10000,
        "rr_min_ms": 0,
        "rr_max_ms": 10000,
        "axis_min_deg": -180,
        "axis_max_deg": 180,
        "require_positive_derived_intervals": True,
    }


def coerce_numeric(series: pd.Series) -> tuple[pd.Series, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    failed = int((series.notna() & numeric.isna()).sum())
    return numeric, failed


def _sentinel_set(values: list[Any]) -> set[float]:
    return {float(v) for v in values}


def apply_timing_validity(series: pd.Series, name: str, cfg: dict[str, Any]) -> tuple[pd.Series, dict[str, int]]:
    sentinels = _sentinel_set(cfg["timing_sentinels"])
    numeric, failed = coerce_numeric(series)
    counts = {
        "failed_conversions": failed,
        "original_missing": int(series.isna().sum()),
        "sentinel": 0,
        "out_of_range": 0,
        "nonpositive_rr": 0,
        "final_missing": 0,
        "replaced_with_missing": 0,
    }
    invalid = numeric.isna()
    sentinel_mask = numeric.isin(sentinels)
    counts["sentinel"] = int(sentinel_mask.sum())
    invalid = invalid | sentinel_mask
    if name == "rr_interval":
        low = float(cfg["rr_min_ms"])
        high = float(cfg["rr_max_ms"])
        # RR of 0 ms is not a cardiac cycle length.
        nonpositive = numeric <= 0
        counts["nonpositive_rr"] = int((~sentinel_mask & nonpositive).sum())
        out = (numeric < low) | (numeric > high) | nonpositive
    else:
        low = float(cfg["fiducial_min_ms"])
        high = float(cfg["fiducial_max_ms"])
        out = (numeric < low) | (numeric > high)
    counts["out_of_range"] = int((~sentinel_mask & out).sum())
    invalid = invalid | out
    cleaned = numeric.mask(invalid)
    counts["final_missing"] = int(cleaned.isna().sum())
    counts["replaced_with_missing"] = int((numeric.notna() & cleaned.isna()).sum())
    return cleaned, counts


def apply_axis_validity(series: pd.Series, cfg: dict[str, Any]) -> tuple[pd.Series, dict[str, int]]:
    sentinels = _sentinel_set(cfg["axis_sentinels"])
    numeric, failed = coerce_numeric(series)
    counts = {
        "failed_conversions": failed,
        "original_missing": int(series.isna().sum()),
        "sentinel": 0,
        "out_of_range": 0,
        "final_missing": 0,
        "replaced_with_missing": 0,
    }
    sentinel_mask = numeric.isin(sentinels)
    counts["sentinel"] = int(sentinel_mask.sum())
    low = float(cfg["axis_min_deg"])
    high = float(cfg["axis_max_deg"])
    out = (numeric < low) | (numeric > high)
    counts["out_of_range"] = int((~sentinel_mask & out).sum())
    cleaned = numeric.mask(numeric.isna() | sentinel_mask | out)
    counts["final_missing"] = int(cleaned.isna().sum())
    counts["replaced_with_missing"] = int((numeric.notna() & cleaned.isna()).sum())
    return cleaned, counts


def derived_interval(left: pd.Series, right: pd.Series, require_positive: bool = True) -> pd.Series:
    both = left.notna() & right.notna()
    delta = (left - right).where(both)
    if require_positive:
        delta = delta.where(delta > 0)
    return delta


def engineer_features(
    frame: pd.DataFrame,
    feature_cfg: dict[str, Any],
    validity_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    validity_cfg = {**default_validity_config(), **(validity_cfg or {})}
    primary = list(feature_cfg.get("primary") or PRIMARY_MEASUREMENT_COLUMNS)
    missing_primary = [c for c in primary if c not in frame.columns]
    if missing_primary:
        raise ValueError(f"Missing approved measurement columns: {missing_primary}")

    audit: dict[str, Any] = {
        "validity_config": validity_cfg,
        "units": {
            "rr_interval": "milliseconds, R-R cycle length",
            "p_onset": "milliseconds, P-wave onset time position in the ECG window",
            "p_end": "milliseconds, P-wave end time position",
            "qrs_onset": "milliseconds, QRS onset time position",
            "qrs_end": "milliseconds, QRS end time position",
            "t_end": "milliseconds, T-wave end time position",
            "p_axis": "degrees, frontal-plane P axis",
            "qrs_axis": "degrees, frontal-plane QRS axis",
            "t_axis": "degrees, frontal-plane T axis",
            "pr_interval": "milliseconds, p_end - p_onset when both valid and p_end > p_onset",
            "qrs_duration": "milliseconds, qrs_end - qrs_onset when both valid and qrs_end > qrs_onset",
            "qt_interval": "milliseconds, t_end - qrs_onset when both valid and t_end > qrs_onset",
        },
        "column_audit": {},
        "derived": {},
        "sentinel_policy_uncertainty": {
            "official_codebook_in_repo": False,
            "primary_policy": "replace documented 16-bit and 29999 encodings plus range bounds with missing",
            "sensitivity_note": (
                "If only 29999 were invalid, remaining 32767/65534/65535 counts are "
                "recorded per column in column_audit.sentinel. Those encodings are "
                "still treated as missing in the primary experiment because they are "
                "not plausible timing or axis values at the documented units."
            ),
        },
        "notes": [
            "MIMIC-IV-ECG documents onset/end fields as time positions in msec, not as precomputed durations.",
            "29999 and 32767 appear as unavailable machine encodings in the public PhysioNet sample and as 16-bit sentinels.",
            "Fiducials outside 0-10000 ms fall outside a 10-second recording window.",
            "Electrical axis values outside [-180, 180] degrees are treated as invalid frontal-plane angles.",
            "report_*, identifiers, bandwidth, and filtering are never features.",
        ],
    }

    out = pd.DataFrame(index=frame.index)
    out["subject_id"] = frame["subject_id"]
    out["study_id"] = frame["study_id"]
    n_valid_primary = pd.Series(0, index=frame.index, dtype=int)

    for col in primary:
        if col in AXIS_COLUMNS:
            cleaned, counts = apply_axis_validity(frame[col], validity_cfg)
        else:
            cleaned, counts = apply_timing_validity(frame[col], col, validity_cfg)
        out[col] = cleaned
        n_valid_primary += cleaned.notna().astype(int)
        audit["column_audit"][col] = counts

    out["n_valid_primary_features"] = n_valid_primary
    feature_cols = list(primary)
    derived_requested = list(feature_cfg.get("derived") or [])
    require_pos = bool(validity_cfg.get("require_positive_derived_intervals", True))

    derived_map = {
        "pr_interval": ("p_end", "p_onset"),
        "qrs_duration": ("qrs_end", "qrs_onset"),
        "qt_interval": ("t_end", "qrs_onset"),
    }
    for name in derived_requested:
        if name not in derived_map:
            raise ValueError(f"Unknown derived feature {name}")
        left, right = derived_map[name]
        values = derived_interval(out[left], out[right], require_positive=require_pos)
        out[name] = values
        feature_cols.append(name)
        both_present = out[left].notna() & out[right].notna()
        n_neg = int((both_present & ((out[left] - out[right]) <= 0)).sum()) if require_pos else 0
        audit["derived"][name] = {
            "formula": f"{left} - {right}",
            "n_defined": int(values.notna().sum()),
            "n_missing": int(values.isna().sum()),
            "n_nonpositive_when_both_present": n_neg,
            "retained": True,
        }

    if feature_cfg.get("circular_axes", True):
        audit["circular_axes"] = {
            "method": "sin/cos of imputed axis degrees, computed after training-only median imputation",
            "reason": "Frontal-plane axes are circular; -179 and +179 are adjacent angles.",
            "applied_in": "preprocessing.FittedPreprocessor after imputation",
            "source_columns": AXIS_COLUMNS,
            "note": (
                "Sin/cos are not computed on raw missing axes. Computing them before "
                "imputation would impute sine and cosine independently and could yield "
                "pairs that are not on the unit circle."
            ),
        }

    leaked = forbidden_columns(feature_cols)
    if leaked:
        raise ValueError(f"Engineered feature list contains forbidden names: {leaked}")
    assert_feature_allowlist(feature_cols, feature_cols)
    audit["final_feature_list"] = feature_cols
    return out, feature_cols, audit
