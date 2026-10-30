"""Input validation and exclusion logging."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .data_loader import REQUIRED_ID_COLUMNS, report_columns_of


def validate_measurements(frame: pd.DataFrame, report_prefix: str = "report_") -> dict[str, Any]:
    report_cols = report_columns_of(frame, prefix=report_prefix)
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_studies = int(frame["study_id"].duplicated().sum()) if "study_id" in frame.columns else None
    n_subject = int(frame["subject_id"].nunique()) if "subject_id" in frame.columns else None
    n_study = int(frame["study_id"].nunique()) if "study_id" in frame.columns else None
    missing_subject = int(frame["subject_id"].isna().sum()) if "subject_id" in frame.columns else None
    missing_study = int(frame["study_id"].isna().sum()) if "study_id" in frame.columns else None
    empty_reports = 0
    if report_cols:
        nonempty_any = pd.Series(False, index=frame.index)
        missing_tokens = {"", "nan", "none", "null", "n/a", "na", ".", "-"}
        for col in report_cols:
            series = frame[col]
            text = series.astype(str).str.strip()
            valid = series.notna() & ~text.str.lower().isin(missing_tokens)
            nonempty_any = nonempty_any | valid
        empty_reports = int((~nonempty_any).sum())
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "columns": list(frame.columns),
        "report_columns": report_cols,
        "duplicate_rows": duplicate_rows,
        "duplicate_study_id_rows": duplicate_studies,
        "unique_subject_id": n_subject,
        "unique_study_id": n_study,
        "missing_subject_id": missing_subject,
        "missing_study_id": missing_study,
        "rows_with_no_report_text": empty_reports,
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
    }


def missing_required_columns(columns: list[str], required: list[str] | None = None) -> list[str]:
    required = required or REQUIRED_ID_COLUMNS
    return [c for c in required if c not in columns]


def resolve_duplicate_studies(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep one row per study_id. Inspect conflicts before dropping."""
    log: dict[str, Any] = {"input_rows": int(len(frame))}
    if "study_id" not in frame.columns:
        raise ValueError("study_id is required to define the unit of analysis")
    dup_mask = frame["study_id"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    log["duplicate_study_rows"] = n_dup_rows
    if n_dup_rows == 0:
        log["conflicting_duplicate_studies"] = 0
        log["rows_after_study_dedup"] = int(len(frame))
        return frame.copy(), log

    sort_cols = [c for c in ("study_id", "ecg_time") if c in frame.columns]
    ordered = frame.sort_values(sort_cols, kind="mergesort")
    grouped = ordered.groupby("study_id", sort=False)
    conflict_ids = []
    compare_cols = [c for c in ordered.columns if c not in ("subject_id", "study_id")]
    for study_id, grp in grouped:
        if len(grp) <= 1:
            continue
        subset = grp[compare_cols]
        if not all(subset[col].nunique(dropna=False) <= 1 for col in compare_cols):
            conflict_ids.append(study_id)
    deduped = ordered.drop_duplicates(subset=["study_id"], keep="first")
    log["conflicting_duplicate_studies"] = len(conflict_ids)
    log["conflict_study_id_sample"] = [str(x) for x in conflict_ids[:20]]
    log["rows_after_study_dedup"] = int(len(deduped))
    return deduped.reset_index(drop=True), log


def drop_missing_identifiers(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(frame)
    cleaned = frame.dropna(subset=["subject_id", "study_id"])
    return cleaned.reset_index(drop=True), {
        "rows_before_id_filter": int(before),
        "rows_missing_ids": int(before - len(cleaned)),
        "rows_after_id_filter": int(len(cleaned)),
    }
