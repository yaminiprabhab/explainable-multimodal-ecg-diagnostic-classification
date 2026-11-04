"""CLI runner and evaluation pipeline for Model 4 Gated Attention Fusion.

ICT 4442: Deep Learning Course Project
Author: Mugdha Chatterjee
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_4_attention_fusion.model import Model4GatedFusion


def train_model4(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([
        {"params": model.ecg_encoder.parameters(), "lr": 1e-3},
        {"params": model.text_encoder.parameters(), "lr": 3e-4},
        {"params": model.gate_net.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ecg = batch["ecg"].to(device)
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, _ = model(ecg, input_ids, lengths)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss / max(1, len(train_loader)):.4f}")

    return model


def evaluate_and_extract_gates(
    model: nn.Module,
    test_loader: DataLoader,
    output_dir: Path | str,
    thresholds: list[float] | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    model.eval()
    all_logits, all_labels, all_gates = [], [], []
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            ecg = batch["ecg"].to(device)
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits, gates = model(ecg, input_ids, lengths)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_gates.append(gates.cpu().numpy())

    y_pred_probs = 1 / (1 + np.exp(-np.vstack(all_logits)))
    y_true = np.vstack(all_labels)
    gates_arr = np.vstack(all_gates)

    # Attention Gate Statistics
    mean_a_ecg = float(np.mean(gates_arr[:, 0]))
    mean_a_text = float(np.mean(gates_arr[:, 1]))
    gate_stats = {
        "mean_ecg_weight": mean_a_ecg,
        "mean_text_weight": mean_a_text,
    }
    with open(out_path / "attention_gates.json", "w") as f:
        json.dump(gate_stats, f, indent=2)

    # Threshold Binarization
    if thresholds is None:
        thresholds = [0.5] * 5
    y_pred_bin = np.zeros_like(y_pred_probs)
    for c in range(5):
        y_pred_bin[:, c] = (y_pred_probs[:, c] >= thresholds[c]).astype(int)

    macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)

    summary = {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "attention_gates": gate_stats,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model 4 gated attention fusion")
    parser.add_argument("--config", default="model_4_attention_fusion/config.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(f"Running Model 4 with config: {args.config} (smoke={args.smoke})")


if __name__ == "__main__":
    main()
