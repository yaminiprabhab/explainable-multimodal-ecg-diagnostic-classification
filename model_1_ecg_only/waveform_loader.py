"""Waveform indexing and loading with explicit shape and sampling checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def discover_wfdb_records(root: Path) -> pd.DataFrame:
    """Discover records stored as split/subject/study/study.hea and .dat."""
    rows = []
    for header in sorted(root.rglob("*.hea")):
        relative = header.relative_to(root)
        if len(relative.parts) < 3:
            continue
        record_path = header.with_suffix("")
        if not record_path.with_suffix(".dat").exists():
            raise FileNotFoundError(f"WFDB header has no matching .dat file: {header}")
        split = relative.parts[0]
        subject_text = relative.parts[-3]
        study_text = relative.parts[-2]
        if not subject_text.startswith("p") or not study_text.startswith("s"):
            raise ValueError(
                "Expected waveform layout split/p<subject>/s<study>/<study>.hea; "
                f"found {header}"
            )
        subject_id = int(subject_text[1:])
        study_id = int(study_text[1:])
        row = {
            "subject_id": subject_id,
            "study_id": study_id,
            "split": split,
            "waveform_path": str(record_path.resolve()),
        }
        try:
            metadata = inspect_wfdb_header(record_path)
            row.update(metadata)
            row["header_valid"] = (
                metadata["header_subject_id"] == subject_id
                and metadata["header_study_id"] == study_id
                and metadata["sampling_frequency"] == 500.0
                and metadata["n_samples"] == 5000
                and metadata["n_leads"] == 12
                and metadata["missing_leads"] == []
                and metadata["duplicate_leads"] == []
            )
            reasons = []
            if metadata["header_subject_id"] != subject_id:
                reasons.append("header_subject_id_mismatch")
            if metadata["header_study_id"] != study_id:
                reasons.append("header_study_id_mismatch")
            if metadata["sampling_frequency"] != 500.0:
                reasons.append("sampling_frequency_mismatch")
            if metadata["n_samples"] != 5000:
                reasons.append("duration_or_sample_count_mismatch")
            if metadata["n_leads"] != 12:
                reasons.append("lead_count_mismatch")
            if metadata["missing_leads"]:
                reasons.append("missing_required_leads")
            if metadata["duplicate_leads"]:
                reasons.append("duplicate_leads")
            row["header_exclusion_reason"] = ";".join(reasons)
        except Exception as exc:
            row.update(
                {
                    "header_valid": False,
                    "header_exclusion_reason": f"header_read_error:{type(exc).__name__}:{exc}",
                }
            )
        rows.append(row)
    index = pd.DataFrame(rows)
    if index.empty:
        raise FileNotFoundError(f"No WFDB .hea records found below {root}")
    validate_waveform_index(index)
    return index


def inspect_wfdb_header(path: Path) -> dict[str, Any]:
    import re

    import wfdb

    header = wfdb.rdheader(str(path))
    comments = "\n".join(header.comments)
    subject_match = re.search(r"(?:<subject_id>|subject_id)\s*:\s*(\d+)", comments, flags=re.IGNORECASE)
    study_match = re.search(r"study_id\s*:\s*(\d+)", comments, flags=re.IGNORECASE)
    record_study = Path(str(header.record_name)).name
    header_study_id = int(study_match.group(1)) if study_match else int(record_study)
    header_subject_id = int(subject_match.group(1)) if subject_match else None
    lead_names = [str(name).strip() for name in header.sig_name]
    return {
        "header_subject_id": header_subject_id,
        "header_study_id": header_study_id,
        "sampling_frequency": float(header.fs),
        "n_samples": int(header.sig_len),
        "n_leads": int(header.n_sig),
        "lead_names": lead_names,
        "missing_leads": [name for name in DEFAULT_LEAD_ORDER if name not in lead_names],
        "duplicate_leads": sorted({name for name in lead_names if lead_names.count(name) > 1}),
    }


def load_waveform(
    path: Path,
    n_leads: int,
    n_samples: int,
    sampling_frequency: int,
    lead_order: list[str] | None = None,
    require_exact_length: bool = True,
) -> np.ndarray:
    """Load one waveform as (requested leads, samples), mapping leads by name."""
    lead_order = lead_order or DEFAULT_LEAD_ORDER
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "waveform" not in data:
                raise ValueError(f"NPZ waveform file lacks 'waveform': {path}")
            signal = np.asarray(data["waveform"], dtype=np.float32)
        fs = None
        lead_names = None
    elif suffix == ".npy":
        signal = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        fs = None
        lead_names = None
    else:
        try:
            import wfdb
        except ImportError as exc:
            raise ImportError("Install wfdb to load real WFDB recordings") from exc
        record = wfdb.rdrecord(str(path))
        signal = np.asarray(record.p_signal, dtype=np.float32)
        fs = float(record.fs)
        lead_names = [str(name).strip() for name in record.sig_name]

    if signal.ndim != 2:
        raise ValueError(f"Waveform must be rank 2, got {signal.shape} from {path}")
    if lead_names is not None:
        missing = [name for name in lead_order if name not in lead_names]
        if missing or len(set(lead_names)) != len(lead_names):
            raise ValueError(f"Lead names do not support requested mapping: missing={missing}, names={lead_names}")
        signal = signal[:, [lead_names.index(name) for name in lead_order]].T
    elif signal.shape[0] == n_leads:
        signal = signal[:n_leads]
    elif signal.shape[1] == n_leads:
        signal = signal.T[:n_leads]
    else:
        raise ValueError(f"Expected {n_leads} leads, got shape {signal.shape} from {path}")
    if fs is not None and not np.isclose(fs, sampling_frequency):
        raise ValueError(
            f"Sampling frequency mismatch for {path}: found {fs}, expected {sampling_frequency}. "
            "Resampling must be performed explicitly before this pipeline."
        )
    if not np.isfinite(signal).all():
        signal = np.where(np.isfinite(signal), signal, np.nan)
    if require_exact_length and signal.shape[1] != n_samples:
        raise ValueError(f"Expected exactly {n_samples} samples, got {signal.shape[1]} from {path}")
    if signal.shape[1] >= n_samples:
        signal = signal[:, :n_samples]
    else:
        padded = np.full((n_leads, n_samples), np.nan, dtype=np.float32)
        padded[:, : signal.shape[1]] = signal
        signal = padded
    return signal.astype(np.float32, copy=False)


def validate_waveform_index(index: pd.DataFrame) -> None:
    required = {"subject_id", "study_id", "waveform_path"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Waveform index missing columns: {sorted(missing)}")
    if index["study_id"].duplicated().any():
        raise ValueError("Waveform index contains duplicate study_id values")
    if index["waveform_path"].isna().any():
        raise ValueError("Waveform index contains missing waveform paths")


def load_waveform_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Waveform index not found: {path}. Create it from the downloaded MIMIC-IV-ECG files."
        )
    index = pd.read_csv(path)
    validate_waveform_index(index)
    return index


def compute_normalization(
    paths: list[Path], n_leads: int, n_samples: int, sampling_frequency: int
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for path in paths:
        signal = load_waveform(path, n_leads, n_samples, sampling_frequency)
        values.append(signal)
    stacked = np.concatenate(values, axis=1)
    mean = np.nanmean(stacked, axis=1)
    std = np.nanstd(stacked, axis=1)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Training waveforms contain a lead with no finite samples")
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_waveform(signal: np.ndarray, mean: np.ndarray, std: np.ndarray, clip_value: float) -> np.ndarray:
    normalized = (signal - mean[:, None]) / std[:, None]
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=clip_value, neginf=-clip_value)
    return np.clip(normalized, -clip_value, clip_value).astype(np.float32)
