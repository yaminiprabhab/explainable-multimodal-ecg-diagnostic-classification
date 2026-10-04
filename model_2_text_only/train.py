"""Training and validation loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import TextSequenceEncoder, multilabel_bce_with_logits
from .utils import LABEL_ORDER, ensure_dir


def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    weights = []
    for i in range(y.shape[1]):
        pos = float(y[:, i].sum())
        neg = float(len(y) - pos)
        if pos == 0 or neg == 0:
            weights.append(1.0)
        else:
            weights.append(neg / pos)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: TextSequenceEncoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor | None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, lengths)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits during training")
        loss = multilabel_bce_with_logits(logits, labels, pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        bs = input_ids.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(model: TextSequenceEncoder, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["lengths"].to(device))
        prob = torch.sigmoid(logits).cpu().numpy()
        probs.append(prob)
        labels.append(batch["labels"].numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def train_model(
    model: TextSequenceEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    pos_weight: torch.Tensor | None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    history = []
    best_val = float("inf")
    best_path = output_dir / "best_model.pt"
    stale = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, pos_weight)
        val_loss = run_epoch(model, val_loader, device, optimizer=None, pos_weight=pos_weight)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "label_order": LABEL_ORDER,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    return {
        "history": history,
        "best_val_loss": best_val if history else None,
        "epochs_run": len(history),
        "checkpoint": str(best_path),
    }
