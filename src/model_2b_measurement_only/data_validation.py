"""Input validation, duplicate-study policy, and measurement quality logs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from model_2_text_only.data_loader import report_columns_of
from model_2_text_only.data_validation import drop_missing_identifiers, validate_measurements

from .data_loader import PRIMARY_MEASUREMENT_COLUMNS


KNOWN_SENTINEL_CANDIDATES = [29999, 32767, -32767, -32768, 65534, 65535]


def validate_source_table(frame: pd.DataFrame) -> dict[str, Any]:
    report = validate_measurements(frame)
    present = [c for c in PRIMARY_MEASUREMENT_COLUMNS if c in frame.columns]
    report["primary_measurement_columns_present"] = present
    report["primary_measurement_columns_missing"] = [
        c for c in PRIMARY_MEASUREMENT_COLUMNS if c not in frame.columns
    ]
    missingness = {}
    distributions = {}
    for col in present:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        n_valid = int(numeric.notna().sum())
        missingness[col] = {
            "n_missing_or_non_numeric": int(numeric.isna().sum()),
            "rate_missing_or_non_numeric": float(numeric.isna().mean()),
            "n_failed_numeric_conversion": int((frame[col].notna() & numeric.isna()).sum()),
            "min": None if n_valid == 0 else float(numeric.min()),
            "p50": None if n_valid == 0 else float(numeric.median()),
            "max": None if n_valid == 0 else float(numeric.max()),
        }
        distributions[col] = _distribution_audit(numeric)
    report["primary_measurement_missingness_raw"] = missingness
    report["primary_measurement_distributions"] = distributions
    return report


def _distribution_audit(numeric: pd.Series) -> dict[str, Any]:
    observed = numeric.dropna()
    quantiles = {}
    if len(observed):
        for q in (0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0):
            quantiles[f"p{int(q * 100):02d}"] = float(observed.quantile(q))
    sentinel_counts = {}
    for value in KNOWN_SENTINEL_CANDIDATES:
        sentinel_counts[str(value)] = int((numeric == value).sum())
    top = observed.value_counts().head(8)
    return {
        "n_numeric": int(len(observed)),
        "n_unique_numeric": int(observed.nunique(dropna=True)),
        "quantiles": quantiles,
        "sentinel_candidate_counts": sentinel_counts,
        "most_frequent_values": [
            {"value": float(idx), "count": int(cnt)} for idx, cnt in top.items()
        ],
        "note": (
            "Frequent spikes at 29999/32767/65535 are treated as candidate sentinels "
            "during feature cleaning. Official PhysioNet per-field codebook is not "
            "stored in this repository, so the replacement policy is documented and "
            "unit-tested rather than claimed as vendor-complete."
        ),
    }


def resolve_duplicate_studies(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep one row per study_id.

    Exact duplicate rows are collapsed. Conflicting duplicate study_ids are
    excluded rather than silently keeping the first row.
    """
    log: dict[str, Any] = {"input_rows": int(len(frame)), "policy": "exclude_conflicting_duplicates"}
    if "study_id" not in frame.columns:
        raise ValueError("study_id is required to define the unit of analysis")
    exact_dups = int(frame.duplicated().sum())
    log["exact_duplicate_rows"] = exact_dups
    collapsed = frame.drop_duplicates(keep="first")
    dup_mask = collapsed["study_id"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    log["duplicate_study_rows_after_exact_collapse"] = n_dup_rows
    if n_dup_rows == 0:
        log["conflicting_duplicate_studies"] = 0
        log["excluded_conflicting_rows"] = 0
        log["rows_after_study_dedup"] = int(len(collapsed))
        return collapsed.reset_index(drop=True), log

    compare_cols = [c for c in collapsed.columns if c not in ("subject_id", "study_id")]
    conflict_ids = []
    for study_id, grp in collapsed[dup_mask].groupby("study_id", sort=False):
        subset = grp[compare_cols]
        if not all(subset[col].nunique(dropna=False) <= 1 for col in compare_cols):
            conflict_ids.append(study_id)
    conflict_set = set(conflict_ids)
    excluded = collapsed[collapsed["study_id"].isin(conflict_set)]
    kept = collapsed[~collapsed["study_id"].isin(conflict_set)]
    kept = kept.drop_duplicates(subset=["study_id"], keep="first")
    log["conflicting_duplicate_studies"] = len(conflict_ids)
    log["conflict_study_id_sample"] = [str(x) for x in conflict_ids[:20]]
    log["excluded_conflicting_rows"] = int(len(excluded))
    log["rows_after_study_dedup"] = int(len(kept))
    log["note"] = (
        "Model 2 logs conflicting duplicates but still keeps the first row after "
        "sorting by study_id/ecg_time. Model 2B excludes conflicting studies. "
        "The current MIMIC-IV-ECG table has no duplicate study_id values, so the "
        "practical cohorts are the same."
    )
    return kept.reset_index(drop=True), log


def source_has_not_been_written(path, original_mtime: float, original_size: int) -> bool:
    return path.stat().st_mtime == original_mtime and path.stat().st_size == original_size


def report_column_names(frame: pd.DataFrame) -> list[str]:
    return report_columns_of(frame)
