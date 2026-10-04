"""PyTorch dataset for leakage-controlled clinical text."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import Vocabulary
from .utils import LABEL_ORDER


class ClinicalTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: Vocabulary, label_order: list[str] | None = None):
        self.frame = frame.reset_index(drop=True)
        self.vocab = vocab
        self.label_order = label_order or LABEL_ORDER
        texts = self.frame["clinical_text"].fillna("").astype(str).tolist()
        encoded = vocab.encode_batch(texts)
        self.input_ids = encoded["input_ids"]
        self.lengths = encoded["lengths"]
        self.truncation_rate = encoded["truncation_rate"]
        self.labels = self.frame[self.label_order].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "lengths": torch.tensor(self.lengths[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
