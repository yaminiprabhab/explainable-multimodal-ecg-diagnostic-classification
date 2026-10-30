"""Load the shared patient-level split or create a compatible replacement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from model_2_text_only.data_splitting import split_patients
from model_2_text_only.utils import LABEL_ORDER


def load_or_create_split(
    frame: pd.DataFrame,
    split_path: Path | None,
    use_shared: bool,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if use_shared:
        if split_path is not None and split_path.exists():
            manifest = pd.read_csv(split_path)
            info = {
                "source": str(split_path),
                "reused_shared_artifact": True,
                "method": "reuse_model_2_patient_split_manifest",
            }
            _validate_manifest(manifest)
            overlap = _split_subject_overlap(manifest)
            if overlap:
                raise RuntimeError(f"Patient leakage in shared split: {overlap}")
            info["n_manifest_studies"] = int(len(manifest))
            info["n_manifest_subjects"] = int(manifest["subject_id"].nunique())
            info["split_counts"] = manifest["split"].value_counts().to_dict()
            return manifest, info
        return pd.DataFrame(), {
            "reused_shared_artifact": False,
            "source": None,
            "reason": "shared_manifest_missing",
        }

    manifest, stats = split_patients(
        frame,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        label_order=LABEL_ORDER,
    )
    stats["source"] = "created_because_shared_manifest_missing"
    stats["reused_shared_artifact"] = False
    return manifest, stats


def assign_split(frame: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = frame.merge(manifest[["study_id", "subject_id", "split"]], on=["study_id", "subject_id"], how="left")
    missing = merged["split"].isna()
    log = {
        "n_studies_in_frame": int(len(frame)),
        "n_assigned": int((~missing).sum()),
        "n_not_in_manifest": int(missing.sum()),
        "n_subjects_assigned": int(merged.loc[~missing, "subject_id"].nunique()),
    }
    assigned = merged.loc[~missing].copy()
    overlap = _split_subject_overlap(assigned)
    if overlap:
        raise RuntimeError(f"Patient leakage after assigning splits: {overlap}")
    log["split_study_counts"] = assigned["split"].value_counts().to_dict()
    log["split_subject_counts"] = (
        assigned.drop_duplicates("subject_id")["split"].value_counts().to_dict()
    )
    return assigned, log


def _validate_manifest(manifest: pd.DataFrame) -> None:
    required = {"subject_id", "study_id", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest missing columns: {missing}")
    if manifest["study_id"].duplicated().any():
        raise ValueError("Split manifest has duplicate study_id values")
    allowed = {"train", "val", "test"}
    extra = set(manifest["split"].unique()) - allowed
    if extra:
        raise ValueError(f"Unexpected split labels: {extra}")


def _split_subject_overlap(manifest: pd.DataFrame) -> dict[str, int]:
    grouped = manifest.groupby("subject_id")["split"].nunique()
    leaked = grouped[grouped > 1]
    return {"n_subjects_in_multiple_splits": int(len(leaked))} if len(leaked) else {}
