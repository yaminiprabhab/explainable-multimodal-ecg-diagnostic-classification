"""Lazy waveform dataset; identifiers are retained only for bookkeeping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .waveform_loader import load_waveform, normalize_waveform


class ECGWaveformDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        n_leads: int,
        n_samples: int,
        sampling_frequency: int,
        mean: np.ndarray,
        std: np.ndarray,
        clip_value: float,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.n_leads = n_leads
        self.n_samples = n_samples
        self.sampling_frequency = sampling_frequency
        self.mean = mean
        self.std = std
        self.clip_value = clip_value
        self.labels = self.frame[["NORM", "HYP", "STTC", "MI", "CD"]].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        signal = load_waveform(
            Path(str(row["waveform_path"])),
            self.n_leads,
            self.n_samples,
            self.sampling_frequency,
        )
        signal = normalize_waveform(signal, self.mean, self.std, self.clip_value)
        return {
            "waveform": torch.from_numpy(signal),
            "labels": torch.from_numpy(self.labels[index].copy()),
        }
