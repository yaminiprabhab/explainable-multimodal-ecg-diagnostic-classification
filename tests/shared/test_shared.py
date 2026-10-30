"""Unit tests for shared waveform indexing, encoder size, and comparison tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_2_text_only.model import TextSequenceEncoder
from shared.compare import compare_output_dirs
from shared.constants import LABEL_ORDER, N_LEADS, TARGET_LENGTH_SAMPLES
from shared.smoke import build_smoke_cohort, synthetic_ecg
from shared.waveforms import (
    EcgNormalizer,
    attach_usable_waveforms,
    relative_record_dir,
    reorder_and_fix_length,
    write_download_lists,
)


def test_text_encoder_representation() -> None:
    model = TextSequenceEncoder(vocab_size=20, embedding_dim=8, hidden_size=8, dropout=0.0, num_labels=5, pad_id=0)
    x = torch.randint(0, 20, (3, 5))
    lengths = torch.tensor([5, 4, 2])
    z = model.encode(x, lengths)
    logits = model(x, lengths)
    assert z.shape == (3, 16)
    assert logits.shape == (3, 5)


def test_smoke_waveforms_index(tmp_path: Path) -> None:
    built = build_smoke_cohort(tmp_path, n_studies=12, seed=0)
    frame = built["frame"]
    usable, index, stats = attach_usable_waveforms(frame, built["waveform_root"])
    assert stats["n_usable"] == 12
    assert index["usable"].all()
    arr = synthetic_ecg([1, 0, 0, 0, 0], seed=1)
    assert arr.shape == (N_LEADS, TARGET_LENGTH_SAMPLES)
    fixed, reason = reorder_and_fix_length(arr.T, None)
    assert reason in {"ok", "padded", "truncated"}
    assert fixed.shape[0] == 12


def test_download_list_is_frozen_split_only(tmp_path: Path) -> None:
    manifest = tmp_path / "patient_split_manifest.csv"
    pd.DataFrame(
        {"subject_id": [10000032, 10000032], "study_id": [40689238, 40689238], "split": ["train", "train"]}
    ).to_csv(manifest, index=False)
    stats = write_download_lists(manifest, tmp_path)
    assert stats["n_studies"] == 1
    assert stats["n_urls"] == 2
    rel = relative_record_dir(10000032, 40689238).as_posix()
    assert rel == "files/p1000/p10000032/s40689238"
    urls = Path(stats["url_list"]).read_text(encoding="utf-8")
    assert "40689238.hea" in urls
    assert "40689238.dat" in urls


def test_normalizer_train_only_stats() -> None:
    a = np.ones((12, 5000), dtype=np.float32)
    b = np.ones((12, 5000), dtype=np.float32) * 3
    norm = EcgNormalizer().fit([a, b])
    out = norm.transform(a)
    assert out.shape == a.shape
    assert np.isfinite(out).all()


def test_compare_common_cohort(tmp_path: Path) -> None:
    studies = [1, 2, 3, 4]
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=(4, 5))
    for name in ("a", "b"):
        out = tmp_path / name
        out.mkdir()
        noise = 0.1 if name == "a" else 0.4
        prob = np.clip(y + rng.normal(0, noise, y.shape), 0, 1)
        frame = pd.DataFrame({"study_id": studies})
        for i, lab in enumerate(LABEL_ORDER):
            frame[f"true_{lab}"] = y[:, i]
            frame[f"prob_{lab}"] = prob[:, i]
            frame[f"pred_{lab}"] = (prob[:, i] >= 0.5).astype(int)
        frame.to_csv(out / "test_predictions.csv", index=False)
        (out / "thresholds.json").write_text('{"thresholds": [0.5, 0.5, 0.5, 0.5, 0.5]}', encoding="utf-8")
    report = compare_output_dirs({"a": tmp_path / "a", "b": tmp_path / "b"}, tmp_path / "cmp")
    assert report["n_common_test_studies"] == 4
    assert (tmp_path / "cmp" / "comparison.md").exists()
    assert (tmp_path / "cmp" / "comparison_table.csv").exists()
