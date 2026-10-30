"""Load machine_measurements.csv without using report text as features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from model_2_text_only.data_loader import REQUIRED_ID_COLUMNS, detect_report_columns


PRIMARY_MEASUREMENT_COLUMNS = [
    "rr_interval",
    "p_onset",
    "p_end",
    "qrs_onset",
    "qrs_end",
    "t_end",
    "p_axis",
    "qrs_axis",
    "t_axis",
]


def load_measurements(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing_ids = [c for c in REQUIRED_ID_COLUMNS if c not in frame.columns]
    if missing_ids:
        raise ValueError(f"Missing required identifier columns: {missing_ids}")
    frame.attrs["report_columns"] = detect_report_columns(list(frame.columns))
    frame.attrs["source_path"] = str(path.resolve())
    return frame


def missing_required_columns(columns: list[str], required: list[str] | None = None) -> list[str]:
    required = required or REQUIRED_ID_COLUMNS
    return [c for c in required if c not in columns]
