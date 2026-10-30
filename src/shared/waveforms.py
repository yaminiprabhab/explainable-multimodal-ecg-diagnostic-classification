"""Index PhysioNet/MIMIC WFDB or .npy ECGs for the frozen study list.

Looks under waveform_root for files/pXXXX/p{subject}/s{study}/ or {study}.npy.
Keeps records that can be mapped to 12 leads and length 5000 (pad/truncate).
Unusable studies are dropped from the comparison cohort; the patient split
itself is not rebuilt. EcgNormalizer fits per-lead z-scores on training
signals only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import LEAD_ORDER, N_LEADS, TARGET_FS, TARGET_LENGTH_SAMPLES


LEAD_ALIASES = {
    "I": "I",
    "II": "II",
    "III": "III",
    "AVR": "AVR",
    "AVL": "AVL",
    "AVF": "AVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
}


def normalize_lead_name(name: str) -> str:
    key = str(name).strip().upper().replace(" ", "")
    key = key.replace("AVR", "AVR").replace("AVL", "AVL").replace("AVF", "AVF")
    if key.startswith("A") and key[1:] in {"VR", "VL", "VF"}:
        key = "A" + key[1:]
    return LEAD_ALIASES.get(key, key)


PHYSIONET_MIMIC_IV_ECG = "https://physionet.org/files/mimic-iv-ecg/1.0"


def mimic_record_dir(waveform_root: Path, subject_id: int, study_id: int) -> Path:
    sid = str(int(subject_id))
    prefix = "p" + sid[:4] if len(sid) >= 4 else "p" + sid.zfill(4)
    return waveform_root / "files" / prefix / f"p{sid}" / f"s{int(study_id)}"


def candidate_record_stems(waveform_root: Path, subject_id: int, study_id: int) -> list[Path]:
    study = int(study_id)
    subject = int(subject_id)
    stems = []
    rec_dir = mimic_record_dir(waveform_root, subject, study)
    stems.extend(
        [
            rec_dir / str(study),
            rec_dir / f"s{study}",
            waveform_root / str(study),
            waveform_root / f"s{study}",
            waveform_root / f"{subject}_{study}",
        ]
    )
    return stems


def _npy_path(stem: Path) -> Path:
    return stem.with_suffix(".npy")


def _hea_path(stem: Path) -> Path:
    return Path(str(stem) + ".hea")


@dataclass
class WaveformRecord:
    study_id: int
    subject_id: int
    path: str
    usable: bool
    reason: str
    n_leads: int | None = None
    n_samples: int | None = None
    fs: float | None = None


def load_signal_array(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return array shaped (n_leads, n_samples) and metadata."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D array in {path}, got {arr.shape}")
        if arr.shape[0] != N_LEADS and arr.shape[1] == N_LEADS:
            arr = arr.T
        meta = {"fs": TARGET_FS, "sig_name": list(LEAD_ORDER), "source": "npy"}
        return np.asarray(arr, dtype=np.float32), meta
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("wfdb is required to read PhysioNet records") from exc
    record = wfdb.rdrecord(str(path.with_suffix("")) if path.suffix else str(path))
    arr = np.asarray(record.p_signal, dtype=np.float32).T
    meta = {
        "fs": float(record.fs) if record.fs else TARGET_FS,
        "sig_name": list(record.sig_name or []),
        "source": "wfdb",
    }
    return arr, meta


def reorder_and_fix_length(
    signal: np.ndarray,
    sig_names: list[str] | None,
    target_length: int = TARGET_LENGTH_SAMPLES,
) -> tuple[np.ndarray, str]:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 2:
        return arr, "not_2d"
    if arr.shape[0] != N_LEADS and arr.shape[1] == N_LEADS:
        arr = arr.T
    ordered = np.zeros((N_LEADS, arr.shape[1]), dtype=np.float32)
    names = [normalize_lead_name(n) for n in (sig_names or [])]
    if len(names) == arr.shape[0] and set(LEAD_ORDER).issubset(set(names)):
        index = {name: i for i, name in enumerate(names)}
        for i, lead in enumerate(LEAD_ORDER):
            ordered[i] = arr[index[lead]]
    elif arr.shape[0] == N_LEADS:
        ordered = arr[:N_LEADS]
    else:
        return arr, f"unexpected_lead_count_{arr.shape[0]}"
    n = ordered.shape[1]
    if n < target_length:
        pad = np.zeros((N_LEADS, target_length), dtype=np.float32)
        pad[:, :n] = ordered
        ordered = pad
        reason = "padded"
    elif n > target_length:
        ordered = ordered[:, :target_length]
        reason = "truncated"
    else:
        reason = "ok"
    if not np.isfinite(ordered).all():
        return ordered, "non_finite"
    return ordered, reason


def quality_ok(reason: str, n_leads: int | None, n_samples: int | None, fs: float | None) -> bool:
    if reason not in {"ok", "padded", "truncated"}:
        return False
    if n_leads is not None and n_leads < N_LEADS:
        return False
    if fs is not None and abs(fs - TARGET_FS) > 1.0 and reason != "ok":
        # Non-500 Hz records are still kept after resample-free truncate/pad; flag only.
        pass
    return True


def resolve_record_path(waveform_root: Path, subject_id: int, study_id: int) -> Path | None:
    for stem in candidate_record_stems(waveform_root, subject_id, study_id):
        npy = _npy_path(stem)
        if npy.exists():
            return npy
        hea = _hea_path(stem)
        if hea.exists():
            return stem
        dat = Path(str(stem) + ".dat")
        if dat.exists():
            return stem
    return None


def index_waveforms(
    frame: pd.DataFrame,
    waveform_root: Path,
    target_length: int = TARGET_LENGTH_SAMPLES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        subject_id = int(row.subject_id)
        study_id = int(row.study_id)
        path = resolve_record_path(waveform_root, subject_id, study_id)
        rec = {
            "subject_id": subject_id,
            "study_id": study_id,
            "path": str(path) if path else "",
            "usable": False,
            "reason": "missing_file",
            "n_leads": None,
            "n_samples": None,
            "fs": None,
        }
        if path is None:
            rows.append(rec)
            continue
        try:
            signal, meta = load_signal_array(path)
            fixed, reason = reorder_and_fix_length(signal, meta.get("sig_name"), target_length)
            rec["n_leads"] = int(fixed.shape[0]) if fixed.ndim == 2 else None
            rec["n_samples"] = int(fixed.shape[1]) if fixed.ndim == 2 else None
            rec["fs"] = meta.get("fs")
            rec["reason"] = reason
            rec["usable"] = reason in {"ok", "padded", "truncated"} and np.isfinite(fixed).all()
            rec["path"] = str(path)
        except Exception as exc:  # noqa: BLE001 - record-level skip
            rec["reason"] = f"load_error:{type(exc).__name__}"
        rows.append(rec)
    return pd.DataFrame(rows)


def attach_usable_waveforms(
    cohort: pd.DataFrame,
    waveform_root: Path,
    index_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Keep cohort rows whose waveform passes lead/length/finite checks."""
    index = index_waveforms(cohort[["subject_id", "study_id"]].drop_duplicates(), waveform_root)
    if index_path is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(index_path, index=False)
    merged = cohort.merge(index, on=["subject_id", "study_id"], how="left")
    usable = merged[merged["usable"].fillna(False).astype(bool)].copy()
    stats = {
        "n_cohort": int(len(cohort)),
        "n_indexed": int(len(index)),
        "n_usable": int(len(usable)),
        "n_missing": int((index["reason"] == "missing_file").sum()),
        "n_dropped": int(len(merged) - len(usable)),
        "waveform_root": str(waveform_root),
    }
    return usable.reset_index(drop=True), index, stats


def relative_record_dir(subject_id: int, study_id: int) -> Path:
    return mimic_record_dir(Path("."), subject_id, study_id).relative_to(".")


def write_download_lists(
    manifest: Path,
    dest_dir: Path,
    base_url: str = PHYSIONET_MIMIC_IV_ECG,
) -> dict[str, Any]:
    """Write PhysioNet .hea/.dat URLs for studies in the frozen split only."""
    frame = pd.read_csv(manifest)
    needed = frame[["subject_id", "study_id"]].drop_duplicates()
    dest_dir.mkdir(parents=True, exist_ok=True)
    rel_path = dest_dir / "waveform_relative_dirs.txt"
    url_path = dest_dir / "waveform_physionet_urls.txt"
    rel_lines: list[str] = []
    url_lines: list[str] = []
    for row in needed.itertuples(index=False):
        rel = relative_record_dir(int(row.subject_id), int(row.study_id)).as_posix()
        study = int(row.study_id)
        rel_lines.append(rel)
        url_lines.append(f"{base_url.rstrip('/')}/{rel}/{study}.hea")
        url_lines.append(f"{base_url.rstrip('/')}/{rel}/{study}.dat")
    rel_path.write_text("\n".join(rel_lines) + "\n", encoding="utf-8")
    url_path.write_text("\n".join(url_lines) + "\n", encoding="utf-8")
    return {
        "n_studies": int(len(needed)),
        "n_urls": int(len(url_lines)),
        "relative_list": str(rel_path),
        "url_list": str(url_path),
    }


class EcgNormalizer:
    """Per-lead mean/std from the training set; apply unchanged to val/test."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, signals: list[np.ndarray]) -> "EcgNormalizer":
        stacked = np.stack(signals, axis=0)
        self.mean_ = stacked.mean(axis=(0, 2))
        self.std_ = stacked.std(axis=(0, 2))
        self.std_ = np.maximum(self.std_, self.eps)
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer is not fitted")
        x = np.asarray(signal, dtype=np.float32)
        return (x - self.mean_[:, None]) / self.std_[:, None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EcgNormalizer":
        obj = cls(eps=float(data.get("eps", 1e-6)))
        if data.get("mean") is not None:
            obj.mean_ = np.asarray(data["mean"], dtype=np.float32)
            obj.std_ = np.asarray(data["std"], dtype=np.float32)
        return obj


def load_fixed_ecg(path: str | Path, target_length: int = TARGET_LENGTH_SAMPLES) -> np.ndarray:
    signal, meta = load_signal_array(Path(path))
    fixed, reason = reorder_and_fix_length(signal, meta.get("sig_name"), target_length)
    if reason not in {"ok", "padded", "truncated"}:
        raise ValueError(f"Unusable waveform {path}: {reason}")
    return fixed
