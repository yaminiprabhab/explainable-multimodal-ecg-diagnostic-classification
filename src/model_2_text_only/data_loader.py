"""Load machine_measurements.csv with configurable paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import LABEL_ORDER


REQUIRED_ID_COLUMNS = ["subject_id", "study_id"]


def detect_report_columns(columns: list[str], prefix: str = "report_") -> list[str]:
    report_cols = [c for c in columns if c.startswith(prefix)]

    def sort_key(name: str) -> int:
        suffix = name[len(prefix) :]
        return int(suffix) if suffix.isdigit() else 10**9

    return sorted(report_cols, key=sort_key)


def load_measurements(path: Path, report_prefix: str = "report_") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing_ids = [c for c in REQUIRED_ID_COLUMNS if c not in frame.columns]
    if missing_ids:
        raise ValueError(f"Missing required identifier columns: {missing_ids}")
    report_cols = detect_report_columns(list(frame.columns), prefix=report_prefix)
    if not report_cols:
        raise ValueError(f"No report columns found with prefix {report_prefix!r}")
    frame.attrs["report_columns"] = report_cols
    return frame


def report_columns_of(frame: pd.DataFrame, prefix: str = "report_") -> list[str]:
    stored = frame.attrs.get("report_columns")
    if stored:
        return list(stored)
    return detect_report_columns(list(frame.columns), prefix=prefix)


def empty_label_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({name: [0] * n for name in LABEL_ORDER})
