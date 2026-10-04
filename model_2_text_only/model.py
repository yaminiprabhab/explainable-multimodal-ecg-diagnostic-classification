"""Trainable embedding + LSTM/GRU encoder for multi-label text classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextSequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_labels: int = 5,
        pad_id: int = 0,
        encoder: str = "lstm",
    ):
        super().__init__()
        if num_labels != 5:
            raise ValueError("This project uses exactly five labels in fixed order")
        self.pad_id = pad_id
        self.encoder_type = encoder.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        rnn_cls = nn.LSTM if self.encoder_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * directions, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be rank 2, got {tuple(input_ids.shape)}")
        embedded = self.embedding(input_ids)
        lengths_cpu = lengths.detach().cpu().clamp(min=1)
        packed = pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.rnn(packed)
        if isinstance(self.rnn, nn.LSTM):
            h = hidden[0]
        else:
            h = hidden
        if self.rnn.bidirectional:
            representation = torch.cat([h[-2], h[-1]], dim=1)
        else:
            representation = h[-1]
        logits = self.classifier(self.dropout(representation))
        if logits.shape[-1] != 5:
            raise RuntimeError(f"Expected 5 logits, got shape {tuple(logits.shape)}")
        return logits


def multilabel_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(logits, targets)
