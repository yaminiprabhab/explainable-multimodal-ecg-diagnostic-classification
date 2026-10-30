"""Train-only vocabulary fitting, padding, and truncation."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"\[MASKED\]|[A-Za-z]+|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]


class Vocabulary:
    def __init__(
        self,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        mask_token: str = "[masked]",
        min_freq: int = 2,
        max_seq_len: int = 64,
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token.lower()
        self.min_freq = min_freq
        self.max_seq_len = max_seq_len
        self.token_to_id = {pad_token: 0, unk_token: 1, self.mask_token: 2}
        self.id_to_token = {0: pad_token, 1: unk_token, 2: self.mask_token}
        self.fitted = False
        self.train_length_stats: dict[str, Any] = {}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def fit(self, texts: list[str], max_seq_len: int | None = None, percentile: int = 95, cap: int = 64) -> None:
        counts: Counter[str] = Counter()
        lengths = []
        for text in texts:
            tokens = tokenize(text)
            lengths.append(len(tokens))
            counts.update(tokens)
        for token, freq in counts.most_common():
            if token in self.token_to_id:
                continue
            if freq < self.min_freq:
                continue
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        if max_seq_len is None:
            if lengths:
                chosen = int(np.percentile(lengths, percentile))
                max_seq_len = max(4, min(cap, max(chosen, 8)))
            else:
                max_seq_len = min(cap, 32)
        self.max_seq_len = int(max_seq_len)
        self.fitted = True
        arr = np.array(lengths) if lengths else np.array([0])
        self.train_length_stats = {
            "n_docs": len(texts),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": int(arr.max()),
            "chosen_max_seq_len": self.max_seq_len,
            "vocab_size": len(self),
            "min_freq": self.min_freq,
        }

    def encode(self, text: str) -> tuple[list[int], int, bool]:
        tokens = tokenize(text)
        truncated = len(tokens) > self.max_seq_len
        tokens = tokens[: self.max_seq_len]
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
        length = len(ids)
        if length == 0:
            ids = [self.unk_id]
            length = 1
        padded = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        return padded, length, truncated

    def encode_batch(self, texts: list[str]) -> dict[str, Any]:
        sequences = []
        lengths = []
        n_trunc = 0
        for text in texts:
            ids, length, truncated = self.encode(text)
            sequences.append(ids)
            lengths.append(length)
            n_trunc += int(truncated)
        array = np.asarray(sequences, dtype=np.int64)
        if array.ndim != 2 or array.shape[1] != self.max_seq_len:
            raise ValueError(f"Unexpected sequence shape {array.shape}")
        if array.min() < 0 or array.max() >= len(self):
            raise ValueError("Token id out of vocabulary range")
        return {
            "input_ids": array,
            "lengths": np.asarray(lengths, dtype=np.int64),
            "n_truncated": n_trunc,
            "truncation_rate": float(n_trunc / len(texts)) if texts else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "mask_token": self.mask_token,
            "min_freq": self.min_freq,
            "max_seq_len": self.max_seq_len,
            "token_to_id": self.token_to_id,
            "fitted": self.fitted,
            "train_length_stats": self.train_length_stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vocabulary":
        vocab = cls(
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
            mask_token=data["mask_token"],
            min_freq=data["min_freq"],
            max_seq_len=data["max_seq_len"],
        )
        vocab.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        vocab.id_to_token = {int(v): k for k, v in vocab.token_to_id.items()}
        vocab.fitted = bool(data.get("fitted", True))
        vocab.train_length_stats = data.get("train_length_stats", {})
        return vocab

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
