"""Token-occlusion attribution. Sensitivity, not causal proof."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .model import TextSequenceEncoder
from .tokenizer import Vocabulary, tokenize
from .utils import LABEL_ORDER


@torch.no_grad()
def occlude_tokens(
    model: TextSequenceEncoder,
    vocab: Vocabulary,
    text: str,
    device: torch.device,
    max_tokens: int = 40,
    label_order: list[str] | None = None,
) -> dict[str, Any]:
    order = label_order or LABEL_ORDER
    model.eval()
    tokens = tokenize(text)[: vocab.max_seq_len]
    ids, length, _ = vocab.encode(text)
    base_logits = _logits(model, ids, length, device)
    base_prob = _sigmoid(base_logits)
    attributions = []
    n = min(len(tokens), max_tokens)
    for i in range(n):
        occluded = tokens.copy()
        occluded[i] = vocab.unk_token
        occluded_text = " ".join(occluded)
        occ_ids, occ_len, _ = vocab.encode(occluded_text)
        occ_prob = _sigmoid(_logits(model, occ_ids, occ_len, device))
        delta = base_prob - occ_prob
        attributions.append(
            {
                "position": i,
                "token": tokens[i],
                "delta_prob": {name: float(delta[j]) for j, name in enumerate(order)},
            }
        )
    return {
        "text": text,
        "tokens": tokens,
        "base_probabilities": {name: float(base_prob[i]) for i, name in enumerate(order)},
        "attributions": attributions,
        "method": "token_occlusion_replace_with_unk",
        "limitations": [
            "Occlusion measures local sensitivity of the trained model, not causality.",
            "Correlated tokens can share credit or hide effects.",
            "Attention weights were not used and would not constitute an explanation by themselves.",
        ],
    }


def _logits(model: TextSequenceEncoder, ids: list[int], length: int, device: torch.device) -> np.ndarray:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    leng = torch.tensor([length], dtype=torch.long, device=device)
    return model(x, leng).squeeze(0).detach().cpu().numpy()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
