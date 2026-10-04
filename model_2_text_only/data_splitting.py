"""Patient-level train/validation/test splitting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import LABEL_ORDER


def patient_label_matrix(frame: pd.DataFrame, label_order: list[str] | None = None) -> pd.DataFrame:
    order = label_order or LABEL_ORDER
    grouped = frame.groupby("subject_id", sort=False)[order].max()
    return grouped


def _combo_key(row: np.ndarray) -> str:
    return "".join(str(int(v)) for v in row)


def split_patients(
    frame: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    label_order: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1")
    order = label_order or LABEL_ORDER
    patient_y = patient_label_matrix(frame, order)
    patients = patient_y.index.to_numpy()
    y = patient_y[order].to_numpy()
    keys = np.array([_combo_key(row) for row in y])
    key_counts = pd.Series(keys).value_counts()
    rare = set(key_counts[key_counts < 3].index.tolist())
    strat = np.array(["rare" if k in rare else k for k in keys])
    method = "multilabel_combo_with_rare_fold"
    try:
        rest_idx, test_idx = train_test_split(
            np.arange(len(patients)),
            test_size=test_ratio,
            random_state=seed,
            stratify=strat,
        )
        val_share = val_ratio / (train_ratio + val_ratio)
        rest_strat = strat[rest_idx]
        rest_counts = pd.Series(rest_strat).value_counts()
        rest_strat = np.array(
            ["rare" if rest_counts.get(s, 0) < 2 else s for s in rest_strat]
        )
        train_rel, val_rel = train_test_split(
            rest_idx,
            test_size=val_share,
            random_state=seed,
            stratify=rest_strat,
        )
    except ValueError:
        method = "random_group_split_unstratified"
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(patients))
        n_test = int(round(len(patients) * test_ratio))
        n_val = int(round(len(patients) * val_ratio))
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_rel = idx[n_test + n_val :]
        val_rel = val_idx

    assignment = {}
    for i in train_rel:
        assignment[int(patients[i]) if np.issubdtype(type(patients[i]), np.integer) else patients[i]] = "train"
    for i in val_rel:
        assignment[patients[i]] = "val"
    for i in test_idx:
        assignment[patients[i]] = "test"

    manifest = frame[["subject_id", "study_id"]].copy()
    manifest["split"] = manifest["subject_id"].map(assignment)
    if manifest["split"].isna().any():
        raise RuntimeError("Some studies were not assigned a split")

    overlap = _split_subject_overlap(manifest)
    if overlap:
        raise RuntimeError(f"Patient leakage across splits: {overlap}")

    stats = _split_stats(frame, manifest, order)
    stats["method"] = method
    stats["requested_ratios"] = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    actual = manifest.drop_duplicates("subject_id")["split"].value_counts(normalize=True).to_dict()
    stats["actual_patient_ratios"] = {k: float(v) for k, v in actual.items()}
    return manifest, stats


def _split_subject_overlap(manifest: pd.DataFrame) -> dict[str, int]:
    grouped = manifest.groupby("subject_id")["split"].nunique()
    leaked = grouped[grouped > 1]
    return {"n_subjects_in_multiple_splits": int(len(leaked))} if len(leaked) else {}


def _split_stats(frame: pd.DataFrame, manifest: pd.DataFrame, order: list[str]) -> dict[str, Any]:
    merged = frame.merge(manifest[["study_id", "split"]], on="study_id", how="left")
    out: dict[str, Any] = {"by_split": {}}
    for split, part in merged.groupby("split"):
        rec: dict[str, Any] = {
            "n_studies": int(len(part)),
            "n_subjects": int(part["subject_id"].nunique()),
        }
        for label in order:
            rec[f"{label}_prevalence"] = float(part[label].mean()) if len(part) else None
            rec[f"{label}_positives"] = int(part[label].sum())
        out["by_split"][str(split)] = rec
    return out


def sample_working_cohort(
    frame: pd.DataFrame,
    max_studies: int,
    seed: int,
    label_order: list[str] | None = None,
) -> pd.DataFrame:
    if max_studies is None or max_studies <= 0 or len(frame) <= max_studies:
        return frame.copy()
    order = label_order or LABEL_ORDER
    patient_y = patient_label_matrix(frame, order)
    keys = patient_y[order].astype(int).astype(str).agg("".join, axis=1)
    patients = patient_y.index.to_numpy()
    rng = np.random.RandomState(seed)
    # Stratified shuffle of patients, then greedily add all of a patient's studies.
    grouped_ids = {}
    for key in keys.unique():
        ids = keys.index[keys == key].to_numpy()
        rng.shuffle(ids)
        grouped_ids[key] = list(ids)
    # Round-robin over strata to preserve rare labels.
    chosen: list[Any] = []
    n_studies = 0
    studies_by_patient = frame.groupby("subject_id").size().to_dict()
    keys_cycle = list(grouped_ids.keys())
    rng.shuffle(keys_cycle)
    while n_studies < max_studies and any(grouped_ids[k] for k in keys_cycle):
        progress = False
        for key in keys_cycle:
            if not grouped_ids[key]:
                continue
            pid = grouped_ids[key].pop()
            chosen.append(pid)
            n_studies += int(studies_by_patient.get(pid, 0))
            progress = True
            if n_studies >= max_studies:
                break
        if not progress:
            break
    return frame[frame["subject_id"].isin(set(chosen))].copy()
