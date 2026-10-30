"""Synthetic tests for Model 4 gated fusion. Not a MIMIC result."""

from __future__ import annotations

from pathlib import Path

import torch

from model_4_attention_fusion.run_pipeline import run_pipeline
from shared.constants import N_LEADS, TARGET_LENGTH_SAMPLES
from shared.fusion import GatedAttentionFusionModel
from shared.io_utils import repo_root

ROOT = repo_root()


def test_gated_logits_and_gates() -> None:
    model = GatedAttentionFusionModel(
        vocab_size=20,
        pad_id=0,
        text_cfg={"embedding_dim": 8, "hidden_size": 8, "num_layers": 1, "dropout": 0.0, "bidirectional": True},
        ecg_cfg={"channels": [8, 16, 16, 16], "embedding_dim": 16, "dropout": 0.0},
        dropout=0.0,
        fused_dim=8,
    )
    ecg = torch.randn(2, N_LEADS, TARGET_LENGTH_SAMPLES)
    ids = torch.randint(0, 20, (2, 6))
    lengths = torch.tensor([6, 4])
    logits = model(ecg, ids, lengths)
    assert logits.shape == (2, 5)
    _fused, gates = model.fuse(ecg, ids, lengths)
    assert gates.shape == (2, 2)
    assert torch.allclose(gates.sum(dim=1), torch.ones(2), atol=1e-5)


def test_end_to_end_smoke(tmp_path: Path) -> None:
    summary = run_pipeline(
        ROOT / "configs" / "model_4_attention_fusion.yaml",
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "outputs",
        waveform_root=tmp_path / "waveforms",
        smoke=True,
    )
    assert summary["fusion"] == "gated"
    assert (Path(summary["output_dir"]) / "attention_gates.json").exists()
    assert summary["test_metrics"]["macro_f1"] is not None
