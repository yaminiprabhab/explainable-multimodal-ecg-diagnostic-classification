"""Synthetic-data unit tests for Model 2. These fixtures are not experimental data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

from model_2_text_only.data_loader import load_measurements
from model_2_text_only.data_splitting import split_patients
from model_2_text_only.data_validation import missing_required_columns, resolve_duplicate_studies, validate_measurements
from model_2_text_only.evaluate import _safe_auroc
from model_2_text_only.label_generation import LabelMapper
from model_2_text_only.leakage_audit import audit_leakage
from model_2_text_only.model import TextSequenceEncoder, multilabel_bce_with_logits
from model_2_text_only.text_preprocessing import build_clinical_text, combine_statements
from model_2_text_only.thresholds import apply_thresholds, select_thresholds
from model_2_text_only.tokenizer import Vocabulary
from model_2_text_only.utils import LABEL_ORDER

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "artifacts" / "label_mapping.yaml"


@pytest.fixture
def mapper() -> LabelMapper:
    return LabelMapper.from_path(MAPPING)


def test_missing_required_columns() -> None:
    assert missing_required_columns(["report_0"]) == ["subject_id", "study_id"]
    assert missing_required_columns(["subject_id", "study_id", "report_0"]) == []


def test_load_measurements_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"report_0": ["Sinus rhythm"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required identifier"):
        load_measurements(path)


def test_duplicate_identifiers_detected() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "study_id": [10, 10, 11],
            "ecg_time": ["2000-01-01", "2000-01-02", "2000-01-03"],
            "report_0": ["a", "b", "c"],
        }
    )
    frame.attrs["report_columns"] = ["report_0"]
    stats = validate_measurements(frame)
    assert stats["duplicate_study_id_rows"] == 1
    deduped, log = resolve_duplicate_studies(frame)
    assert len(deduped) == 2
    assert log["conflicting_duplicate_studies"] == 1


def test_empty_report_fields(mapper: LabelMapper) -> None:
    result = mapper.label_statements([])
    assert result.unmapped
    assert result.labels == {k: 0 for k in LABEL_ORDER}


def test_deterministic_text_construction() -> None:
    a = combine_statements(["Sinus rhythm", "Normal ECG"])
    b = combine_statements(["Sinus rhythm", "Normal ECG"])
    assert a == b == "Sinus rhythm | Normal ECG"


def test_five_label_order(mapper: LabelMapper) -> None:
    assert mapper.label_order == ["NORM", "HYP", "STTC", "MI", "CD"]


def test_norm_does_not_override_abnormality(mapper: LabelMapper) -> None:
    result = mapper.label_statements(["Normal ECG", "Left ventricular hypertrophy"])
    assert result.labels["HYP"] == 1
    assert result.labels["NORM"] == 0
    assert result.ambiguous


def test_norm_except_for_rate(mapper: LabelMapper) -> None:
    result = mapper.label_statements(["Sinus tachycardia", "Normal ECG except for rate"])
    assert result.labels["NORM"] == 1
    assert result.labels["HYP"] == 0


def test_isolated_sinus_rhythm_is_not_norm(mapper: LabelMapper) -> None:
    result = mapper.label_statements(["Sinus rhythm"])
    assert result.labels["NORM"] == 0
    assert result.unmapped


def test_ischemia_is_sttc_not_mi(mapper: LabelMapper) -> None:
    result = mapper.label_statements(
        ["Extensive T wave changes may be due to myocardial ischemia"]
    )
    assert result.labels["STTC"] == 1
    assert result.labels["MI"] == 0
    assert "STTC" in result.uncertain_labels


def test_infarct_assigns_mi(mapper: LabelMapper) -> None:
    result = mapper.label_statements(["Inferior infarct - age undetermined"])
    assert result.labels["MI"] == 1
    assert result.labels["STTC"] == 0


def test_negation_blocks_mi(mapper: LabelMapper) -> None:
    result = mapper.label_statements(["No evidence of inferior infarct"])
    assert result.labels["MI"] == 0


def test_multilabel_hyp_and_sttc(mapper: LabelMapper) -> None:
    result = mapper.label_statements(["LVH with secondary repolarization abnormality"])
    assert result.labels["HYP"] == 1
    assert result.labels["STTC"] == 1
    assert result.labels["NORM"] == 0


def test_possible_lvh_uncertain_positive(mapper: LabelMapper) -> None:
    result = mapper.label_statements(["Possible left ventricular hypertrophy"])
    assert result.labels["HYP"] == 1
    assert result.ambiguous


def test_pvc_not_cd_or_sttc(mapper: LabelMapper) -> None:
    result = mapper.label_statements(["Sinus rhythm with PVC(s)"])
    assert result.labels["CD"] == 0
    assert result.labels["STTC"] == 0


def test_rbbb_is_cd(mapper: LabelMapper) -> None:
    result = mapper.label_statements(["Right bundle branch block"])
    assert result.labels["CD"] == 1


def test_leakage_mask_removes_trigger(mapper: LabelMapper) -> None:
    labels = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "study_id": 1,
                "raw_report_text": "Sinus rhythm | Left ventricular hypertrophy | Abnormal ECG",
                "trigger_statements": "Left ventricular hypertrophy",
                "NORM": 0,
                "HYP": 1,
                "STTC": 0,
                "MI": 0,
                "CD": 0,
            }
        ]
    )
    out = build_clinical_text(
        labels,
        mapper,
        summary_phrases=["abnormal ecg", "borderline ecg", "normal ecg"],
    )
    text = out.loc[0, "clinical_text"].lower()
    assert "hypertrophy" not in text
    assert "abnormal ecg" not in text
    assert "sinus rhythm" in text
    audit = audit_leakage(out, mapper)
    assert "experiment_type" in audit
    assert audit["fraction_reconstructable_from_masked_text"] == 0


def test_patient_level_split_isolation() -> None:
    rows = []
    labels_cycle = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    study = 1
    for subj in range(1, 41):
        for _ in range(2):
            rec = {"subject_id": subj, "study_id": study}
            rec.update({n: labels_cycle[(subj - 1) % 5][i] for i, n in enumerate(LABEL_ORDER)})
            rows.append(rec)
            study += 1
    frame = pd.DataFrame(rows)
    manifest, _ = split_patients(frame, seed=0)
    overlap = manifest.groupby("subject_id")["split"].nunique()
    assert (overlap == 1).all()
    assert set(manifest["split"]) == {"train", "val", "test"}


def test_vocabulary_fit_on_train_only() -> None:
    vocab = Vocabulary(min_freq=1, max_seq_len=8)
    vocab.fit(["alpha beta gamma"], max_seq_len=8)
    assert "alpha" in vocab.token_to_id
    encoded = vocab.encode("alpha unknownword")
    ids, length, truncated = encoded
    assert length == 2
    assert not truncated
    assert ids[1] == vocab.unk_id
    assert "unknownword" not in vocab.token_to_id


def test_padding_and_truncation() -> None:
    vocab = Vocabulary(min_freq=1)
    vocab.fit(["one two three four five"], max_seq_len=3, cap=3)
    ids, length, truncated = vocab.encode("one two three four")
    assert truncated
    assert length == 3
    assert len(ids) == 3
    ids2, length2, truncated2 = vocab.encode("one")
    assert not truncated2
    assert length2 == 1
    assert ids2[1:] == [vocab.pad_id, vocab.pad_id]


def test_model_output_dimensions() -> None:
    model = TextSequenceEncoder(vocab_size=20, num_labels=5, pad_id=0)
    x = torch.randint(0, 20, (4, 6))
    lengths = torch.tensor([6, 5, 4, 3])
    logits = model(x, lengths)
    assert logits.shape == (4, 5)


def test_multilabel_loss_shape() -> None:
    logits = torch.zeros(3, 5)
    targets = torch.tensor(
        [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 1]], dtype=torch.float32
    )
    loss = multilabel_bce_with_logits(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_validation_only_threshold_selection() -> None:
    rng = np.random.RandomState(0)
    y_val = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]] * 3)
    y_prob = np.clip(y_val + rng.normal(0, 0.1, y_val.shape), 0, 1)
    selected = select_thresholds(y_val, y_prob, min_val_positives=3)
    assert len(selected["thresholds"]) == 5
    pred = apply_thresholds(y_prob, selected["thresholds"])
    assert pred.shape == y_val.shape


def test_single_class_auroc_undefined() -> None:
    y = np.zeros(10)
    p = np.linspace(0.1, 0.9, 10)
    assert _safe_auroc(y, p) is None
    # sklearn may warn and return nan instead of raising, depending on version.
    with pytest.warns(UndefinedMetricWarning):
        score = roc_auc_score(y, p)
    assert score != score or score is None


def test_end_to_end_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from model_2_text_only.run_pipeline import run_pipeline, write_smoke_csv

    smoke = write_smoke_csv(tmp_path / "smoke.csv")
    config_src = ROOT / "model_2_text_only" / "config.yaml"
    summary = run_pipeline(
        config_src,
        smoke_csv=smoke,
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "outputs",
    )
    assert "stopped_before_training" in summary
    assert summary.get("n_text_cohort", 0) > 0 or summary["stopped_before_training"]
