"""Model 4: Gated Modality-Attention Network.

ICT 4442: Deep Learning Course Project
Author: Mugdha Chatterjee
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ECGEncoder(nn.Module):
    """1D-CNN for raw 12-lead ECG waveforms (B, 12, 5000)."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(5),
            # Block 4
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(5),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_blocks(x)
        feat = self.adaptive_pool(feat).squeeze(-1)
        return self.proj(feat)


class TextEncoder(nn.Module):
    """Bi-LSTM for residual clinical text tokens."""

    def __init__(
        self,
        vocab_size: int = 202,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lengths_clamped = torch.clamp(lengths.cpu(), min=1)
        packed = pack_padded_sequence(
            embedded, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        text_emb = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        return self.proj(text_emb)


class Model4GatedFusion(nn.Module):
    """Model 4: Gated Modality-Attention Network.

    Computes dynamic scalar softmax weights [a_ecg, a_text] over projected embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 202,
        embed_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.ecg_encoder = ECGEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, embed_dim=64, hidden_dim=64, out_dim=embed_dim
        )
        self.gate_net = nn.Linear(embed_dim * 2, 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self, ecg: torch.Tensor, input_ids: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_ecg = self.ecg_encoder(ecg)                    # (B, 128)
        p_text = self.text_encoder(input_ids, lengths)   # (B, 128)
        concat_emb = torch.cat([p_ecg, p_text], dim=-1)  # (B, 256)
        gates = torch.softmax(self.gate_net(concat_emb), dim=-1)  # (B, 2) -> [a_ecg, a_text]

        a_ecg = gates[:, 0:1]
        a_text = gates[:, 1:2]
        z_gated = a_ecg * p_ecg + a_text * p_text
        logits = self.classifier(self.dropout(z_gated))
        return logits, gates
