"""ECG occlusion: zero time windows or whole leads, record Δ probability.

This is local sensitivity of a trained model, not a clinical explanation and
not a causal claim. Gate/attention weights from Model 4 must not be treated
as attribution by themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .constants import LABEL_ORDER, LEAD_ORDER, N_LEADS, TARGET_LENGTH_SAMPLES


@torch.no_grad()
def occlude_ecg(
    model: torch.nn.Module,
    signal: np.ndarray,
    device: torch.device,
    window: int = 250,
    max_windows: int = 20,
    label_order: list[str] | None = None,
    extra_batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    order = label_order or LABEL_ORDER
    model.eval()
    x = np.asarray(signal, dtype=np.float32)
    if x.shape != (N_LEADS, TARGET_LENGTH_SAMPLES):
        raise ValueError(f"Expected ECG shape {(N_LEADS, TARGET_LENGTH_SAMPLES)}, got {x.shape}")
    base = _prob(model, x, device, extra_batch)
    time_attr = []
    n_windows = min(max_windows, TARGET_LENGTH_SAMPLES // window)
    for w in range(n_windows):
        start = w * window
        occ = x.copy()
        occ[:, start : start + window] = 0.0
        delta = base - _prob(model, occ, device, extra_batch)
        time_attr.append(
            {
                "window_start": start,
                "window_end": start + window,
                "delta_prob": {name: float(delta[j]) for j, name in enumerate(order)},
            }
        )
    lead_attr = []
    for i, lead in enumerate(LEAD_ORDER):
        occ = x.copy()
        occ[i] = 0.0
        delta = base - _prob(model, occ, device, extra_batch)
        lead_attr.append(
            {
                "lead": lead,
                "delta_prob": {name: float(delta[j]) for j, name in enumerate(order)},
            }
        )
    return {
        "base_probabilities": {name: float(base[i]) for i, name in enumerate(order)},
        "time_occlusion": time_attr,
        "lead_occlusion": lead_attr,
        "method": "zero_out_time_windows_and_leads",
        "limitations": [
            "Occlusion measures local sensitivity of the trained model, not causality.",
            "Zeroing a window is not a clinically realistic perturbation.",
            "Attention or gate weights are not explanations by themselves.",
        ],
    }


def _prob(
    model: torch.nn.Module,
    signal: np.ndarray,
    device: torch.device,
    extra_batch: dict[str, Any] | None,
) -> np.ndarray:
    batch: dict[str, Any] = {
        "ecg": torch.tensor(signal[None, ...], dtype=torch.float32),
        "labels": torch.zeros(1, 5, dtype=torch.float32),
    }
    if extra_batch:
        batch.update(extra_batch)
    logits = model.logits_from_batch(batch, device)
    prob = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
    return prob


def save_occlusion_heatmap(occ: dict[str, Any], path, label: str = "CD") -> None:
    """Bar chart of time-window occlusion deltas. No patient identifiers."""
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = occ.get("time_occlusion") or []
    if not windows:
        return
    deltas = [w["delta_prob"].get(label, 0.0) for w in windows]
    starts = [w["window_start"] for w in windows]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(deltas)), deltas, color="#3b6d9c")
    ax.set_xticks(range(len(starts)))
    ax.set_xticklabels(starts, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Window start (samples)")
    ax.set_ylabel(f"Δ P({label})")
    ax.set_title(f"Time-window occlusion ({label})")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
