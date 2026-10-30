"""Synthetic labels, residual text, and sine-wave ECGs for CI only.

Never treat smoke metrics as experimental findings. The generator writes a
mini frozen split/labels/cohort plus {study_id}.npy files so Models 1/3/4
can train without MIMIC access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import LABEL_ORDER, LEAD_ORDER, N_LEADS, TARGET_LENGTH_SAMPLES
from .io_utils import ensure_dir, save_json


PHRASES = [
    ("Sinus rhythm", "Normal ECG", [1, 0, 0, 0, 0]),
    ("Sinus rhythm", "Left ventricular hypertrophy", [0, 1, 0, 0, 0]),
    ("Sinus tachycardia", "Nonspecific T abnormalities", [0, 0, 1, 0, 0]),
    ("Sinus rhythm", "Inferior infarct age undetermined", [0, 0, 0, 1, 0]),
    ("Sinus rhythm", "Right bundle branch block", [0, 0, 0, 0, 1]),
    ("Atrial fibrillation", "Possible left ventricular hypertrophy", [0, 1, 0, 0, 0]),
    ("Sinus bradycardia", "Lateral ST-T changes are nonspecific", [0, 0, 1, 0, 0]),
    ("Ventricular pacing", "Pacemaker rhythm", [0, 0, 0, 0, 1]),
    ("Sinus rhythm", "Possible anterior infarct", [0, 0, 0, 1, 0]),
    ("Sinus rhythm", "Normal ECG except for rate", [1, 0, 0, 0, 0]),
    ("Sinus rhythm", "Extensive T wave changes", [0, 0, 1, 0, 0]),
    ("Sinus rhythm", "Left anterior fascicular block", [0, 0, 0, 0, 1]),
]


def synthetic_ecg(label_vec: list[int] | np.ndarray, seed: int, length: int = TARGET_LENGTH_SAMPLES) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0.0, 10.0, length, dtype=np.float32)
    labels = np.asarray(label_vec, dtype=np.float32)
    freq = 1.1 + 0.4 * labels[4] + 0.2 * labels[3]
    signal = np.zeros((N_LEADS, length), dtype=np.float32)
    for i in range(N_LEADS):
        base = np.sin(2 * np.pi * (freq + 0.03 * i) * t)
        qrs = 0.4 * labels[4] * np.sin(2 * np.pi * 8.0 * t)
        st = 0.25 * labels[2] * np.sin(2 * np.pi * 0.4 * t + i)
        mi = 0.3 * labels[3] * np.sin(2 * np.pi * 2.0 * t + 0.5)
        hyp = 0.2 * labels[1] * np.sin(2 * np.pi * 0.8 * t)
        signal[i] = base + qrs + st + mi + hyp + 0.05 * rng.randn(length).astype(np.float32)
    return signal


def write_synthetic_waveforms(
    frame: pd.DataFrame,
    waveform_root: Path,
    length: int = TARGET_LENGTH_SAMPLES,
) -> Path:
    ensure_dir(waveform_root)
    for row in frame.itertuples(index=False):
        labels = [int(getattr(row, name)) for name in LABEL_ORDER]
        arr = synthetic_ecg(labels, seed=int(row.study_id), length=length)
        path = waveform_root / f"{int(row.study_id)}.npy"
        np.save(path, arr)
    (waveform_root / "lead_order.txt").write_text("\n".join(LEAD_ORDER), encoding="utf-8")
    return waveform_root


def build_smoke_cohort(
    dest: Path,
    n_studies: int = 48,
    seed: int = 42,
    length: int = TARGET_LENGTH_SAMPLES,
) -> dict[str, Any]:
    """Write frozen-style artifacts plus synthetic .npy waveforms."""
    artifacts = ensure_dir(dest / "artifacts")
    waveforms = ensure_dir(dest / "waveforms")
    rng = np.random.RandomState(seed)
    rows = []
    sid = 1000
    subj = 1
    for i in range(n_studies):
        phrase = PHRASES[i % len(PHRASES)]
        labels = phrase[2]
        # Residual text: rhythm statement only, matching Model 2 leakage control.
        residual = phrase[0]
        split = "train" if i < int(0.70 * n_studies) else "val" if i < int(0.85 * n_studies) else "test"
        # Keep all studies of a synthetic patient in one split by pairing.
        if i % 2 == 0:
            # assign split by patient id below
            pass
        rows.append(
            {
                "subject_id": subj,
                "study_id": sid,
                "split": None,
                "clinical_text": residual,
                "has_usable_text": True,
                "raw_report_text": f"{phrase[0]} | {phrase[1]}",
                **{name: labels[j] for j, name in enumerate(LABEL_ORDER)},
            }
        )
        sid += 1
        if i % 2 == 1:
            subj += 1
    frame = pd.DataFrame(rows)
    patients = frame["subject_id"].unique()
    rng.shuffle(patients)
    n_train = int(round(0.70 * len(patients)))
    n_val = int(round(0.15 * len(patients)))
    assignment = {}
    for i, pid in enumerate(patients):
        if i < n_train:
            assignment[pid] = "train"
        elif i < n_train + n_val:
            assignment[pid] = "val"
        else:
            assignment[pid] = "test"
    frame["split"] = frame["subject_id"].map(assignment)
    # Ensure none empty
    if frame["split"].nunique() < 3:
        ids = list(frame["subject_id"].unique())
        frame.loc[frame["subject_id"] == ids[0], "split"] = "train"
        frame.loc[frame["subject_id"] == ids[1], "split"] = "val"
        frame.loc[frame["subject_id"] == ids[2], "split"] = "test"

    split = frame[["subject_id", "study_id", "split"]]
    labels = frame[["subject_id", "study_id", *LABEL_ORDER]]
    cohort = frame[
        ["subject_id", "study_id", "split", "clinical_text", "has_usable_text", "raw_report_text", *LABEL_ORDER]
    ]
    split.to_csv(artifacts / "patient_split_manifest.csv", index=False)
    labels.to_csv(artifacts / "labels.csv", index=False)
    cohort.to_csv(artifacts / "cohort_manifest.csv", index=False)
    write_synthetic_waveforms(frame, waveforms, length=length)
    save_json(
        {
            "synthetic": True,
            "n_studies": int(len(frame)),
            "n_patients": int(frame["subject_id"].nunique()),
            "note": "Synthetic smoke fixtures. Not clinical data.",
        },
        artifacts / "smoke_manifest.json",
    )
    return {
        "artifacts_dir": artifacts,
        "waveform_root": waveforms,
        "frame": frame,
        "n_studies": int(len(frame)),
    }
