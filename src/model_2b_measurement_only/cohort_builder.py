"""Align Model 2 weak labels and define the measurement-eligible cohort."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from model_2_text_only.label_generation import LabelMapper, generate_labels, label_prevalence
from model_2_text_only.utils import LABEL_ORDER

from .data_loader import PRIMARY_MEASUREMENT_COLUMNS


def load_or_generate_labels(
    measurements: pd.DataFrame,
    labels_path: Path | None,
    mapping_path: Path,
    use_shared: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {
        "label_order": list(LABEL_ORDER),
        "mapping_version": None,
        "source": None,
        "weak_labels": True,
        "clinically_verified": False,
    }
    mapper = LabelMapper.from_path(mapping_path)
    info["mapping_version"] = mapper.mapping.get("version")
    info["uncertain_policy"] = mapper.mapping.get("uncertain_policy")
    fallback = None
    if labels_path is not None:
        fallback = labels_path.with_name("labels_full.csv")
    if use_shared and labels_path is not None and labels_path.exists():
        labels = pd.read_csv(labels_path)
        info["source"] = str(labels_path)
        info["reused_shared_artifact"] = True
    elif use_shared and fallback is not None and fallback.exists():
        labels = pd.read_csv(fallback)
        info["source"] = str(fallback)
        info["reused_shared_artifact"] = True
    else:
        if not measurements.attrs.get("report_columns"):
            raise FileNotFoundError(
                "Shared label artifact is missing and report columns are unavailable; "
                "cannot invent a second label mapping."
            )
        labels = generate_labels(measurements, mapper)
        info["source"] = "generated_with_model_2_LabelMapper"
        info["reused_shared_artifact"] = False
    _validate_label_matrix(labels)
    return labels, info


def _validate_label_matrix(labels: pd.DataFrame) -> None:
    required = ["subject_id", "study_id", *LABEL_ORDER]
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise ValueError(f"Label matrix missing columns: {missing}")
    if labels["study_id"].duplicated().any():
        raise ValueError("Label matrix has duplicate study_id values")
    for name in LABEL_ORDER:
        if labels[name].isna().any():
            raise ValueError(f"Missing target values for {name}")
        values = set(pd.to_numeric(labels[name], errors="raise").unique())
        if values - {0, 1}:
            raise ValueError(f"Non-binary target values for {name}: {values}")


def align_labels(measurements: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    meas = measurements.copy()
    lab = labels.copy()
    meas["study_id"] = pd.to_numeric(meas["study_id"], errors="raise")
    lab["study_id"] = pd.to_numeric(lab["study_id"], errors="raise")
    log: dict[str, Any] = {
        "measurement_studies": int(meas["study_id"].nunique()),
        "label_studies": int(lab["study_id"].nunique()),
    }
    merged = meas.merge(lab, on="study_id", how="left", suffixes=("", "_label"))
    if "subject_id_label" in merged.columns:
        mismatch = merged["subject_id"].notna() & merged["subject_id_label"].notna()
        n_mismatch = int((merged.loc[mismatch, "subject_id"] != merged.loc[mismatch, "subject_id_label"]).sum())
        log["subject_id_mismatches"] = n_mismatch
        if n_mismatch:
            raise ValueError("subject_id disagrees between measurements and labels for some study_id values")
        merged = merged.drop(columns=["subject_id_label"])
    missing_labels = merged[LABEL_ORDER].isna().any(axis=1)
    log["studies_missing_labels"] = int(missing_labels.sum())
    return merged, log


def apply_label_eligibility(
    frame: pd.DataFrame,
    require_at_least_one_label: bool,
    exclude_poor_quality_unmapped: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = frame.copy()
    if "exclude_reason" not in eligible.columns:
        eligible["exclude_reason"] = ""
    eligible["exclude_reason"] = eligible["exclude_reason"].fillna("")
    if require_at_least_one_label:
        if "unmapped" in eligible.columns:
            unmapped = eligible["unmapped"].astype(bool) & eligible["exclude_reason"].eq("")
        else:
            unmapped = eligible[LABEL_ORDER].sum(axis=1).eq(0) & eligible["exclude_reason"].eq("")
        eligible.loc[unmapped, "exclude_reason"] = "unmapped_no_target_class"
    if exclude_poor_quality_unmapped and "poor_quality" in eligible.columns:
        if "unmapped" in eligible.columns:
            poor_only = (
                eligible["poor_quality"].astype(bool)
                & eligible["unmapped"].astype(bool)
                & eligible["exclude_reason"].eq("")
            )
        else:
            poor_only = eligible["poor_quality"].astype(bool) & eligible["exclude_reason"].eq("")
        eligible.loc[poor_only, "exclude_reason"] = "poor_quality_unmapped"
    kept = eligible[eligible["exclude_reason"].eq("")].copy()
    return kept, {
        "n_before": int(len(eligible)),
        "n_after": int(len(kept)),
        "n_excluded_unmapped": int((eligible["exclude_reason"] == "unmapped_no_target_class").sum()),
        "n_excluded_poor_quality_unmapped": int((eligible["exclude_reason"] == "poor_quality_unmapped").sum()),
        "prevalence": label_prevalence(kept).to_dict(orient="records") if len(kept) else [],
    }


def apply_measurement_eligibility(
    frame: pd.DataFrame,
    min_valid_primary_features: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "n_valid_primary_features" not in frame.columns:
        raise ValueError("n_valid_primary_features is required before measurement eligibility")
    eligible = frame.copy()
    if "exclude_reason" not in eligible.columns:
        eligible["exclude_reason"] = ""
    eligible["exclude_reason"] = eligible["exclude_reason"].fillna("")
    none_usable = eligible["n_valid_primary_features"] < int(min_valid_primary_features)
    eligible.loc[none_usable & eligible["exclude_reason"].eq(""), "exclude_reason"] = "no_usable_numerical_measurements"
    kept = eligible[eligible["exclude_reason"].eq("")].copy()
    return kept, {
        "min_valid_primary_features": int(min_valid_primary_features),
        "n_before": int(len(eligible)),
        "n_after": int(len(kept)),
        "n_excluded_no_usable_measurements": int(none_usable.sum()),
        "primary_columns": PRIMARY_MEASUREMENT_COLUMNS,
    }
