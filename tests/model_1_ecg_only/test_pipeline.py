"""Synthetic tests for Model 1. Smoke metrics are not experimental findings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model_1_ecg_only.model import EcgOnlyModel
from model_1_ecg_only.run_pipeline import run_pipeline
from shared.constants import N_LEADS, TARGET_LENGTH_SAMPLES
from shared.io_utils import repo_root
from shared.waveforms import reorder_and_fix_length


ROOT = repo_root()


def test_cnn_output_shape() -> None:
    model = EcgOnlyModel(channels=[8, 16, 32, 32], embedding_dim=16, dropout=0.0)
    x = torch.randn(2, N_LEADS, TARGET_LENGTH_SAMPLES)
    logits = model(x)
    assert logits.shape == (2, 5)
    z = model.encode(x)
    assert z.shape == (2, 16)


def test_reorder_pads_short_signal() -> None:
    arr = np.ones((12, 100), dtype=np.float32)
    fixed, reason = reorder_and_fix_length(arr, None, target_length=5000)
    assert fixed.shape == (12, 5000)
    assert reason == "padded"


def test_end_to_end_smoke(tmp_path: Path) -> None:
    summary = run_pipeline(
        ROOT / "configs" / "model_1_ecg_only.yaml",
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "outputs",
        waveform_root=tmp_path / "waveforms",
        smoke=True,
    )
    # smoke=True writes its own artifact dirs; still must finish training
    assert summary["stopped_before_training"] is False
    assert summary["n_train"] > 0
    assert summary["test_metrics"]["macro_f1"] is not None
    assert (Path(summary["output_dir"]) / "test_metrics.json").exists()
