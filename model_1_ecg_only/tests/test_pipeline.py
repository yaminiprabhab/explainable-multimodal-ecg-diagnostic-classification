from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model_1_ecg_only.model import ECG1DCNN
from model_1_ecg_only.run_pipeline import run_pipeline, write_smoke_dataset
from model_1_ecg_only.waveform_loader import discover_wfdb_records, load_waveform, load_waveform_index


def test_model_output_shape() -> None:
    model = ECG1DCNN()
    output = model(torch.zeros(2, 12, 5000))
    assert output.shape == (2, 5)


def test_waveform_loader_transposes_and_pads(tmp_path: Path) -> None:
    path = tmp_path / "signal.npz"
    np.savez(path, waveform=np.ones((20, 12), dtype=np.float32))
    with pytest.raises(ValueError, match="exactly 30"):
        load_waveform(path, 12, 30, 500)


def test_wfdb_leads_are_mapped_by_name(tmp_path: Path) -> None:
    import wfdb

    path = tmp_path / "record"
    names = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
    signal = np.vstack([np.full(5000, float(i)) for i in range(12)]).T
    import os
    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        wfdb.wrsamp(
            "record",
            fs=500,
            units=["mV"] * 12,
            sig_name=names,
            p_signal=signal,
            fmt=["16"] * 12,
        )
    finally:
        os.chdir(previous)
    output = load_waveform(path, 12, 5000, 500)
    assert np.all(output[4] == 5.0)
    assert np.all(output[5] == 4.0)


def test_index_rejects_duplicate_studies(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    pd.DataFrame(
        {"subject_id": [1, 1], "study_id": [1, 1], "waveform_path": ["a", "b"]}
    ).to_csv(path, index=False)
    try:
        load_waveform_index(path)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate study IDs must be rejected")


def test_provided_train_test_groups_are_not_mixed(tmp_path: Path) -> None:
    root = tmp_path / "ECG"
    for split, subject, study in [("Train", 1, 10), ("Test", 2, 20)]:
        record = root / split / f"p{subject}" / f"s{study}" / str(study)
        record.parent.mkdir(parents=True)
        (record.with_suffix(".hea")).write_text(
            f"{study} 12 500 5000\n" + "\n".join(
                f"{study}.dat 16 200(0)/mV 16 0 0 0 0 L{i}" for i in range(12)
            ),
            encoding="utf-8",
        )
        (record.with_suffix(".dat")).write_bytes(b"")
    index = discover_wfdb_records(root)
    assert set(index["split"]) == {"Train", "Test"}
    assert set(index["subject_id"]) == {1, 2}


def test_end_to_end_smoke(tmp_path: Path) -> None:
    smoke_index = write_smoke_dataset(tmp_path / "waveforms", 60, 5000, 12)
    config = Path(__file__).parents[1] / "config.yaml"
    summary = run_pipeline(config, waveform_index_path=smoke_index, smoke=True, output_dir=tmp_path / "out")
    assert summary["stopped_before_training"] is False
    assert summary["n_train"] > 0
    assert (tmp_path / "out" / "best_model.pt").exists()
    assert (tmp_path / "out" / "test_metrics.json").exists()
