"""Training and inference for Model 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import ECG1DCNN


def run_epoch(model, loader, device, optimizer, loss_fn) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for batch in loader:
        waveform = batch["waveform"].to(device)
        labels = batch["labels"].to(device)
        logits = model(waveform)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Model 1 loss")
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        size = len(labels)
        total += float(loss.item()) * size
        count += size
    return total / max(count, 1)


@torch.no_grad()
def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        logits = model(batch["waveform"].to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def train_model(
    model: ECG1DCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    pos_weight: torch.Tensor,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    best_loss = float("inf")
    stale = 0
    history = []
    checkpoint = output_dir / "best_model.pt"
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, loss_fn)
        val_loss = run_epoch(model, val_loader, device, None, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
    return {"history": history, "best_val_loss": best_loss, "epochs_run": len(history), "checkpoint": str(checkpoint)}
