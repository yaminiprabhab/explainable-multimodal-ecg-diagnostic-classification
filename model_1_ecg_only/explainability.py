"""Input sensitivity explanations for ECG waveforms."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def gradient_saliency(model, waveform: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    signal = waveform.detach().clone().to(device).requires_grad_(True)
    logits = model(signal)
    probabilities = torch.sigmoid(logits)
    saliency = []
    for label in range(logits.shape[1]):
        model.zero_grad(set_to_none=True)
        if signal.grad is not None:
            signal.grad.zero_()
        logits[0, label].backward(retain_graph=label < logits.shape[1] - 1)
        saliency.append(signal.grad.detach().abs()[0].cpu().numpy())
    values = np.asarray(saliency)
    return {
        "method": "absolute_input_gradient",
        "probabilities": probabilities.detach().cpu().numpy()[0].tolist(),
        "per_label_lead_importance": values.mean(axis=2).tolist(),
        "per_label_time_importance": values.mean(axis=1).tolist(),
        "limitations": [
            "Gradients measure local sensitivity, not causal clinical explanations.",
            "Attributions are reported for the normalized waveform supplied to the model.",
        ],
    }
