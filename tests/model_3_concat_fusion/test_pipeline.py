"""Synthetic tests for Model 3 concat fusion. Not a MIMIC result."""

from __future__ import annotations

from pathlib import Path

import torch

from model_3_concat_fusion.run_pipeline import run_pipeline
from shared.constants import N_LEADS, TARGET_LENGTH_SAMPLES
from shared.fusion import ConcatFusionModel
from shared.io_utils import repo_root

ROOT = repo_root()


def test_concat_logits_shape() -> None:
    model = ConcatFusionModel(
        vocab_size=20,
        pad_id=0,
        text_cfg={"embedding_dim": 8, "hidden_size": 8, "num_layers": 1, "dropout": 0.0, "bidirectional": True},
        ecg_cfg={"channels": [8, 16, 16, 16], "embedding_dim": 16, "dropout": 0.0},
        dropout=0.0,
    )
    ecg = torch.randn(2, N_LEADS, TARGET_LENGTH_SAMPLES)
    ids = torch.randint(0, 20, (2, 6))
    lengths = torch.tensor([6, 4])
    logits = model(ecg, ids, lengths)
    assert logits.shape == (2, 5)


def test_end_to_end_smoke(tmp_path: Path) -> None:
    summary = run_pipeline(
        ROOT / "configs" / "model_3_concat_fusion.yaml",
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "outputs",
        waveform_root=tmp_path / "waveforms",
        smoke=True,
    )
    assert summary["fusion"] == "concat"
    assert summary["stopped_before_training"] is False
    assert summary["test_metrics"]["macro_f1"] is not None
