"""1D CNN for 12-lead ECG: four conv blocks, GAP, linear head → 5 logits.

Input shape (batch, 12, 5000). encode() returns the 128-d vector used by
fusion models; logits_from_batch is the shared training hook. Five independent
outputs with BCE; no softmax.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from shared.constants import N_LEADS


class EcgCNNEncoder(nn.Module):
    """Temporal conv stack → pooled ECG representation (no classification head)."""

    def __init__(
        self,
        in_channels: int = N_LEADS,
        channels: list[int] | None = None,
        kernel_size: int = 7,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        channels = channels or [32, 64, 128, 128]
        pools = [2, 2, 5, 5]
        blocks = []
        prev = in_channels
        for ch, pool in zip(channels, pools):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(prev, ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(pool),
                )
            )
            prev = ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(prev, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ECG tensor must be (B, C, T), got {tuple(x.shape)}")
        h = self.blocks(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class EcgOnlyModel(nn.Module):
    """Model 1 classifier: CNN encoder plus dropout linear head."""

    def __init__(
        self,
        channels: list[int] | None = None,
        kernel_size: int = 7,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        num_labels: int = 5,
    ):
        super().__init__()
        if num_labels != 5:
            raise ValueError("This project uses exactly five labels in fixed order")
        self.encoder = EcgCNNEncoder(
            channels=channels,
            kernel_size=kernel_size,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_labels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        logits = self.classifier(self.dropout(z))
        if logits.shape[-1] != 5:
            raise RuntimeError(f"Expected 5 logits, got shape {tuple(logits.shape)}")
        return logits

    def logits_from_batch(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        return self.forward(batch["ecg"].to(device))
