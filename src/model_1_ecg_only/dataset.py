"""Load cached (12, 5000) arrays and apply a training-fitted z-score.

Does not read report text. Labels come from the frozen weak-label matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from shared.constants import LABEL_ORDER, N_LEADS, TARGET_LENGTH_SAMPLES
from shared.waveforms import EcgNormalizer, load_fixed_ecg


class EcgDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        normalizer: EcgNormalizer,
        label_order: list[str] | None = None,
        cache: dict[int, np.ndarray] | None = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.normalizer = normalizer
        self.label_order = label_order or LABEL_ORDER
        self.labels = self.frame[self.label_order].to_numpy(dtype=np.float32)
        self.cache = cache if cache is not None else {}

    def __len__(self) -> int:
        return len(self.frame)

    def _signal(self, idx: int) -> np.ndarray:
        study_id = int(self.frame.iloc[idx]["study_id"])
        if study_id in self.cache:
            return self.cache[study_id]
        path = self.frame.iloc[idx]["path"]
        arr = load_fixed_ecg(path)
        if arr.shape != (N_LEADS, TARGET_LENGTH_SAMPLES):
            raise ValueError(f"Bad ECG shape {arr.shape} for study {study_id}")
        self.cache[study_id] = arr
        return arr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        signal = self.normalizer.transform(self._signal(idx))
        return {
            "ecg": torch.tensor(signal, dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "study_id": torch.tensor(int(self.frame.iloc[idx]["study_id"]), dtype=torch.long),
        }


def fit_normalizer(frame: pd.DataFrame, cache: dict[int, np.ndarray] | None = None) -> EcgNormalizer:
    cache = cache if cache is not None else {}
    signals = []
    for row in frame.itertuples(index=False):
        study_id = int(row.study_id)
        if study_id not in cache:
            cache[study_id] = load_fixed_ecg(Path(row.path))
        signals.append(cache[study_id])
    return EcgNormalizer().fit(signals)
