"""BCE-with-logits training shared by the CNN and fusion models.

No softmax: five independent sigmoids. Positive class weights are computed
from the training matrix only. Early-stopping uses validation loss; the
best checkpoint is reloaded before threshold selection.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import LABEL_ORDER
from .evaluate import evaluate_multilabel, metrics_to_rows, per_label_rows
from .io_utils import ensure_dir, save_json
from .thresholds import apply_thresholds, select_thresholds

ForwardFn = Callable[[torch.nn.Module, dict[str, Any], torch.device], torch.Tensor]


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


def multilabel_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(logits, targets)


def default_forward(model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    if hasattr(model, "logits_from_batch"):
        return model.logits_from_batch(batch, device)
    raise TypeError("Model must implement logits_from_batch(batch, device)")


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor | None,
    forward_fn: ForwardFn | None = None,
) -> float:
    train = optimizer is not None
    model.train(train)
    forward = forward_fn or default_forward
    total = 0.0
    n = 0
    for batch in loader:
        logits = forward(model, batch, device)
        labels = batch["labels"].to(device)
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
        bs = labels.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_fn: ForwardFn | None = None,
    extra_keys: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    model.eval()
    forward = forward_fn or default_forward
    probs = []
    labels = []
    extras: dict[str, list[Any]] = {k: [] for k in extra_keys}
    for batch in loader:
        logits = forward(model, batch, device)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
        for key in extra_keys:
            extras[key].append(batch[key].numpy() if hasattr(batch[key], "numpy") else np.asarray(batch[key]))
    out = {
        "prob": np.concatenate(probs, axis=0),
        "true": np.concatenate(labels, axis=0),
    }
    for key, values in extras.items():
        out[key] = np.concatenate(values, axis=0)
    return out


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    pos_weight: torch.Tensor | None,
    forward_fn: ForwardFn | None = None,
    param_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    if param_groups is not None:
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    history = []
    best_val = float("inf")
    best_path = output_dir / "best_model.pt"
    stale = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, pos_weight, forward_fn)
        val_loss = run_epoch(model, val_loader, device, None, pos_weight, forward_fn)
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


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
    n_studies: int,
    n_subjects: int,
    forward_fn: ForwardFn | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    pred = predict_proba(model, loader, device, forward_fn)
    y_pred = apply_thresholds(pred["prob"], thresholds)
    metrics = evaluate_multilabel(
        pred["true"],
        pred["prob"],
        y_pred,
        thresholds,
        n_studies=n_studies,
        n_subjects=n_subjects,
    )
    return metrics, pred["true"], pred["prob"], y_pred


def select_and_save_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    strategy: str,
    fallback: float,
    min_val_positives: int,
) -> list[float]:
    thresh_out = select_thresholds(
        y_true,
        y_prob,
        strategy=strategy,
        fallback=fallback,
        min_val_positives=min_val_positives,
    )
    save_json(thresh_out, output_dir / "thresholds.json")
    return thresh_out["thresholds"]


def write_metric_tables(metrics: dict[str, Any], output_dir: Path) -> None:
    import csv

    import pandas as pd

    with (output_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(metrics_to_rows(metrics))
    pd.DataFrame(per_label_rows(metrics)).to_csv(output_dir / "test_per_label_metrics.csv", index=False)
    save_json(metrics, output_dir / "test_metrics.json")
