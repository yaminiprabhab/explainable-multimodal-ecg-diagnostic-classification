"""One training example: normalized 12-lead ECG + tokenized residual text.

Text must already be leakage-masked clinical_text. Waveforms are z-scored
with a training-fitted EcgNormalizer. study_id is returned for prediction
tables, not as a network input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from model_2_text_only.tokenizer import Vocabulary
from shared.constants import LABEL_ORDER, N_LEADS, TARGET_LENGTH_SAMPLES
from shared.waveforms import EcgNormalizer, load_fixed_ecg


class MultimodalDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocab: Vocabulary,
        normalizer: EcgNormalizer,
        cache: dict[int, np.ndarray] | None = None,
        label_order: list[str] | None = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.vocab = vocab
        self.normalizer = normalizer
        self.label_order = label_order or LABEL_ORDER
        texts = self.frame["clinical_text"].fillna("").astype(str).tolist()
        encoded = vocab.encode_batch(texts)
        self.input_ids = encoded["input_ids"]
        self.lengths = encoded["lengths"]
        self.labels = self.frame[self.label_order].to_numpy(dtype=np.float32)
        self.cache = cache if cache is not None else {}

    def __len__(self) -> int:
        return len(self.frame)

    def _signal(self, idx: int) -> np.ndarray:
        study_id = int(self.frame.iloc[idx]["study_id"])
        if study_id in self.cache:
            return self.cache[study_id]
        arr = load_fixed_ecg(self.frame.iloc[idx]["path"])
        if arr.shape != (N_LEADS, TARGET_LENGTH_SAMPLES):
            raise ValueError(f"Bad ECG shape {arr.shape} for study {study_id}")
        self.cache[study_id] = arr
        return arr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        signal = self.normalizer.transform(self._signal(idx))
        return {
            "ecg": torch.tensor(signal, dtype=torch.float32),
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "lengths": torch.tensor(self.lengths[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "study_id": torch.tensor(int(self.frame.iloc[idx]["study_id"]), dtype=torch.long),
        }
