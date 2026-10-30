"""Join the frozen Model 2 split, weak labels, and residual report text.

Requires artifacts written by Model 2 (or smoke fixtures):
patient_split_manifest.csv, labels.csv, and optionally cohort_manifest.csv.
Fusion models set require_text=True so only leakage-controlled residual text
remains. Do not rebuild the split after observing test performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .constants import LABEL_ORDER


def load_frozen_split(artifacts_dir: Path) -> pd.DataFrame:
    path = artifacts_dir / "patient_split_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen split {path}. Run Model 2 first or use --smoke."
        )
    frame = pd.read_csv(path)
    required = {"subject_id", "study_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split manifest missing columns: {sorted(missing)}")
    return frame


def load_labels(artifacts_dir: Path) -> pd.DataFrame:
    for name in ("labels.csv", "labels_full.csv"):
        path = artifacts_dir / name
        if path.exists():
            frame = pd.read_csv(path)
            missing = {"subject_id", "study_id", *LABEL_ORDER} - set(frame.columns)
            if missing:
                raise ValueError(f"{name} missing columns: {sorted(missing)}")
            return frame[["subject_id", "study_id", *LABEL_ORDER]].copy()
    raise FileNotFoundError(f"No labels.csv in {artifacts_dir}. Run Model 2 first or use --smoke.")


def load_residual_text(artifacts_dir: Path) -> pd.DataFrame | None:
    path = artifacts_dir / "cohort_manifest.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    cols = [c for c in ("subject_id", "study_id", "clinical_text", "has_usable_text", "split") if c in frame.columns]
    return frame[cols].copy()


def join_shared_cohort(
    artifacts_dir: Path,
    require_text: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inner-join frozen split and labels; optionally keep residual-text studies only."""
    split = load_frozen_split(artifacts_dir)
    labels = load_labels(artifacts_dir)
    merged = split.merge(labels, on=["subject_id", "study_id"], how="inner")
    notes: dict[str, Any] = {
        "n_split": int(len(split)),
        "n_after_label_join": int(len(merged)),
        "require_text": require_text,
        "label_mapping_frozen": True,
        "mapping_version": "1.0.0",
    }
    text = load_residual_text(artifacts_dir)
    if text is not None:
        on_cols = ["subject_id", "study_id"] if "subject_id" in text.columns else ["study_id"]
        merged = merged.merge(text.drop(columns=["split"], errors="ignore"), on=on_cols, how="left")
        notes["n_with_text_join"] = int(len(merged))
        if require_text:
            if "has_usable_text" in merged.columns:
                merged = merged[merged["has_usable_text"].fillna(False).astype(bool)].copy()
            else:
                merged = merged[merged["clinical_text"].fillna("").astype(str).str.strip().ne("")].copy()
            notes["n_after_text_filter"] = int(len(merged))
    elif require_text:
        raise FileNotFoundError(
            f"cohort_manifest.csv is required for text/fusion models but was not found in {artifacts_dir}"
        )
    if merged.empty:
        raise RuntimeError("Shared cohort is empty after joining frozen artifacts")
    return merged.reset_index(drop=True), notes


def split_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "val"].copy()
    test_df = frame[frame["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("A split is empty after waveform/text filtering")
    return train_df, val_df, test_df
