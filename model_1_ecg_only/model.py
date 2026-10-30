"""Small reproducible 1D CNN for 12-lead ECG classification."""

from __future__ import annotations

import torch
from torch import nn


class ECG1DCNN(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        channels: list[int] | None = None,
        kernels: list[int] | None = None,
        dropout: float = 0.3,
        num_labels: int = 5,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128]
        kernels = kernels or [15, 9, 7]
        if len(channels) != 3 or len(kernels) != 3:
            raise ValueError("Exactly three CNN blocks are required")
        blocks = []
        in_channels = n_leads
        for out_channels, kernel in zip(channels, kernels):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, num_labels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(f"waveform must have shape (batch, leads, samples), got {tuple(waveform.shape)}")
        return self.classifier(self.dropout(self.pool(self.encoder(waveform)).squeeze(-1)))


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    import numpy as np

    weights = []
    for i in range(labels.shape[1]):
        positive = float(labels[:, i].sum())
        negative = float(len(labels) - positive)
        weights.append(negative / positive if positive and negative else 1.0)
    return torch.tensor(weights, dtype=torch.float32)
