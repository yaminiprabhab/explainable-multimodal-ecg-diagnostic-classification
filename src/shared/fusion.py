"""Fusion heads that reuse the Model 1 CNN encoder and Model 2 LSTM encoder.

ConcatFusionModel: Linear(concat(z_ecg, z_text)).
GatedAttentionFusionModel: project both to d, a = softmax(W [P(z_ecg); P(z_text)]),
z = a_ecg P(z_ecg) + a_text P(z_text). Gates are inspectable, not causal.

Optional checkpoints load with strict=False. Default training finetunes both
encoders; the fusion pipeline uses a lower text learning rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from model_1_ecg_only.model import EcgCNNEncoder
from model_2_text_only.model import TextSequenceEncoder


class ConcatFusionModel(nn.Module):
    """Independent encoders; classification head on concatenated representations."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        text_cfg: dict[str, Any],
        ecg_cfg: dict[str, Any],
        dropout: float = 0.3,
        num_labels: int = 5,
    ):
        super().__init__()
        if num_labels != 5:
            raise ValueError("This project uses exactly five labels in fixed order")
        self.ecg_encoder = EcgCNNEncoder(
            channels=list(ecg_cfg.get("channels") or [32, 64, 128, 128]),
            kernel_size=int(ecg_cfg.get("kernel_size", 7)),
            embedding_dim=int(ecg_cfg.get("embedding_dim", 128)),
            dropout=float(ecg_cfg.get("dropout", dropout)),
        )
        self.text_encoder = TextSequenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=int(text_cfg.get("embedding_dim", 64)),
            hidden_size=int(text_cfg.get("hidden_size", 64)),
            num_layers=int(text_cfg.get("num_layers", 1)),
            dropout=float(text_cfg.get("dropout", dropout)),
            bidirectional=bool(text_cfg.get("bidirectional", True)),
            num_labels=5,
            pad_id=pad_id,
            encoder=str(text_cfg.get("encoder", "lstm")),
        )
        fused = self.ecg_encoder.embedding_dim + self.text_encoder.representation_size()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused, num_labels)

    def encode_modalities(
        self, ecg: torch.Tensor, input_ids: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ecg_encoder.encode(ecg), self.text_encoder.encode(input_ids, lengths)

    def forward(self, ecg: torch.Tensor, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        z_ecg, z_text = self.encode_modalities(ecg, input_ids, lengths)
        logits = self.classifier(self.dropout(torch.cat([z_ecg, z_text], dim=1)))
        if logits.shape[-1] != 5:
            raise RuntimeError(f"Expected 5 logits, got shape {tuple(logits.shape)}")
        return logits

    def logits_from_batch(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        return self.forward(
            batch["ecg"].to(device),
            batch["input_ids"].to(device),
            batch["lengths"].to(device),
        )


class GatedAttentionFusionModel(nn.Module):
    """Softmax gates over projected ECG and text vectors (primary Model 4)."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        text_cfg: dict[str, Any],
        ecg_cfg: dict[str, Any],
        dropout: float = 0.3,
        fused_dim: int = 128,
        num_labels: int = 5,
    ):
        super().__init__()
        if num_labels != 5:
            raise ValueError("This project uses exactly five labels in fixed order")
        self.ecg_encoder = EcgCNNEncoder(
            channels=list(ecg_cfg.get("channels") or [32, 64, 128, 128]),
            kernel_size=int(ecg_cfg.get("kernel_size", 7)),
            embedding_dim=int(ecg_cfg.get("embedding_dim", 128)),
            dropout=float(ecg_cfg.get("dropout", dropout)),
        )
        self.text_encoder = TextSequenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=int(text_cfg.get("embedding_dim", 64)),
            hidden_size=int(text_cfg.get("hidden_size", 64)),
            num_layers=int(text_cfg.get("num_layers", 1)),
            dropout=float(text_cfg.get("dropout", dropout)),
            bidirectional=bool(text_cfg.get("bidirectional", True)),
            num_labels=5,
            pad_id=pad_id,
            encoder=str(text_cfg.get("encoder", "lstm")),
        )
        self.ecg_proj = nn.Linear(self.ecg_encoder.embedding_dim, fused_dim)
        self.text_proj = nn.Linear(self.text_encoder.representation_size(), fused_dim)
        self.gate = nn.Linear(fused_dim * 2, 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused_dim, num_labels)
        self.fused_dim = fused_dim
        self.last_gates: torch.Tensor | None = None

    def fuse(
        self, ecg: torch.Tensor, input_ids: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_ecg = self.ecg_proj(self.ecg_encoder.encode(ecg))
        z_text = self.text_proj(self.text_encoder.encode(input_ids, lengths))
        gates = torch.softmax(self.gate(torch.cat([z_ecg, z_text], dim=1)), dim=1)
        fused = gates[:, 0:1] * z_ecg + gates[:, 1:2] * z_text
        self.last_gates = gates
        return fused, gates

    def forward(self, ecg: torch.Tensor, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        fused, _gates = self.fuse(ecg, input_ids, lengths)
        logits = self.classifier(self.dropout(fused))
        if logits.shape[-1] != 5:
            raise RuntimeError(f"Expected 5 logits, got shape {tuple(logits.shape)}")
        return logits

    def logits_from_batch(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        return self.forward(
            batch["ecg"].to(device),
            batch["input_ids"].to(device),
            batch["lengths"].to(device),
        )


def load_encoder_subsets(model: nn.Module, ecg_ckpt: Path | None, text_ckpt: Path | None, device: torch.device) -> dict[str, Any]:
    """Warm-start fusion encoders from Model 1 / Model 2 checkpoints when files exist."""
    notes: dict[str, Any] = {"ecg_ckpt": None, "text_ckpt": None}
    if ecg_ckpt and Path(ecg_ckpt).exists():
        state = torch.load(ecg_ckpt, map_location=device, weights_only=False)["model_state"]
        encoder_state = {
            k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")
        }
        missing, unexpected = model.ecg_encoder.load_state_dict(encoder_state, strict=False)
        notes["ecg_ckpt"] = str(ecg_ckpt)
        notes["ecg_missing"] = list(missing)
        notes["ecg_unexpected"] = list(unexpected)
    if text_ckpt and Path(text_ckpt).exists():
        state = torch.load(text_ckpt, map_location=device, weights_only=False)["model_state"]
        missing, unexpected = model.text_encoder.load_state_dict(state, strict=False)
        notes["text_ckpt"] = str(text_ckpt)
        notes["text_missing"] = list(missing)
        notes["text_unexpected"] = list(unexpected)
    return notes
