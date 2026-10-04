"""Synthetic-data unit tests for Model 2B. These fixtures are not experimental data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_2_text_only.utils import LABEL_ORDER
from model_2b_measurement_only.classifiers import ConstantLabelFallback, fit_binary_models, rf_factory
from model_2b_measurement_only.cohort_builder import align_labels, apply_label_eligibility
from model_2b_measurement_only.data_loader import load_measurements, missing_required_columns
from model_2b_measurement_only.data_splitting import assign_split, load_or_create_split
from model_2b_measurement_only.data_validation import resolve_duplicate_studies, source_has_not_been_written
from model_2b_measurement_only.evaluate import evaluate_multilabel
from model_2b_measurement_only.explainability import permutation_importance_by_label
from model_2b_measurement_only.feature_engineering import (
    apply_timing_validity,
    default_validity_config,
    derived_interval,
    engineer_features,
)
from model_2b_measurement_only.preprocessing import fit_preprocessor
from model_2b_measurement_only.run_pipeline import run_pipeline, write_smoke_csv
from model_2b_measurement_only.thresholds import apply_thresholds, select_thresholds
from model_2b_measurement_only.utils import assert_feature_allowlist, forbidden_columns

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "model_2b_measurement_only" / "config.yaml"


def _feature_cfg() -> dict:
    return {
        "primary": [
            "rr_interval",
            "p_onset",
            "p_end",
            "qrs_onset",
            "qrs_end",
            "t_end",
            "p_axis",
            "qrs_axis",
            "t_axis",
        ],
        "derived": ["pr_interval", "qrs_duration", "qt_interval"],
        "circular_axes": True,
    }


def _minimal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3],
            "study_id": [10, 11, 20, 30],
            "cart_id": [1, 1, 1, 1],
            "ecg_time": ["2000-01-01"] * 4,
            "report_0": ["Normal ECG", "Left ventricular hypertrophy", "Inferior infarct", "RBBB"],
            "bandwidth": ["0.5-40"] * 4,
            "filtering": ["60"] * 4,
            "rr_interval": [800, 820, 29999, 900],
            "p_onset": [40, 40, 40, 40],
            "p_end": [160, 150, 32767, 155],
            "qrs_onset": [200, 210, 200, 205],
            "qrs_end": [280, 290, 285, 270],
            "t_end": [520, 530, 540, 500],
            "p_axis": [30, 40, 32767, 20],
            "qrs_axis": [10, 0, 45, -10],
            "t_axis": [20, 15, 30, 25],
            "NORM": [1, 0, 0, 0],
            "HYP": [0, 1, 0, 0],
            "STTC": [0, 0, 0, 0],
            "MI": [0, 0, 1, 0],
            "CD": [0, 0, 0, 1],
        }
    )


def test_missing_required_columns() -> None:
    assert missing_required_columns(["rr_interval"]) == ["subject_id", "study_id"]
    assert missing_required_columns(["subject_id", "study_id", "rr_interval"]) == []


def test_load_measurements_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"rr_interval": [800]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required identifier"):
        load_measurements(path)


def test_input_csv_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "machine_measurements.csv"
    frame = _minimal_frame().drop(columns=LABEL_ORDER)
    frame.to_csv(path, index=False)
    mtime = path.stat().st_mtime
    size = path.stat().st_size
    loaded = load_measurements(path)
    loaded["rr_interval"] = 0
    assert source_has_not_been_written(path, mtime, size)
    again = pd.read_csv(path)
    assert int(again.loc[0, "rr_interval"]) == 800


def test_conflicting_duplicate_studies_are_excluded() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "study_id": [10, 10, 11],
            "ecg_time": ["2000-01-01", "2000-01-02", "2000-01-03"],
            "rr_interval": [800, 900, 850],
        }
    )
    kept, log = resolve_duplicate_studies(frame)
    assert log["conflicting_duplicate_studies"] == 1
    assert 10 not in set(kept["study_id"])
    assert list(kept["study_id"]) == [11]


def test_exact_duplicate_studies_collapsed() -> None:
    row = {"subject_id": 1, "study_id": 10, "rr_interval": 800}
    frame = pd.DataFrame([row, row, {"subject_id": 2, "study_id": 11, "rr_interval": 900}])
    kept, log = resolve_duplicate_studies(frame)
    assert log["exact_duplicate_rows"] == 1
    assert log["conflicting_duplicate_studies"] == 0
    assert len(kept) == 2


def test_feature_allowlist_rejects_ids_and_reports() -> None:
    leaked = forbidden_columns(["rr_interval", "subject_id", "report_0", "NORM", "bandwidth"])
    assert "subject_id" in leaked
    assert "report_0" in leaked
    assert "NORM" in leaked
    assert "bandwidth" in leaked
    with pytest.raises(ValueError, match="Leakage"):
        assert_feature_allowlist(["rr_interval", "study_id"], ["rr_interval", "study_id"])


def test_engineer_features_excludes_forbidden_columns() -> None:
    frame = _minimal_frame()
    out, cols, audit = engineer_features(frame, _feature_cfg())
    assert "report_0" not in cols
    assert "subject_id" not in cols
    assert "bandwidth" not in cols
    assert "NORM" not in cols
    assert "pr_interval" in cols
    assert "p_axis_sin" not in cols
    assert audit["units"]["p_onset"].startswith("milliseconds")


def test_sentinel_values_become_missing() -> None:
    cfg = default_validity_config()
    series = pd.Series([800, 29999, 32767, 65535, "not-a-number", np.nan])
    cleaned, counts = apply_timing_validity(series, "rr_interval", cfg)
    assert counts["sentinel"] == 3
    assert counts["failed_conversions"] == 1
    assert cleaned.isna().sum() == 5
    assert float(cleaned.iloc[0]) == 800


def test_derived_intervals_require_compatible_valid_inputs() -> None:
    left = pd.Series([160.0, 150.0, np.nan, 90.0])
    right = pd.Series([40.0, np.nan, 40.0, 100.0])
    delta = derived_interval(left, right, require_positive=True)
    assert float(delta.iloc[0]) == 120.0
    assert pd.isna(delta.iloc[1])
    assert pd.isna(delta.iloc[2])
    assert pd.isna(delta.iloc[3])


def test_imputation_fitted_on_training_only() -> None:
    train = pd.DataFrame({"rr_interval": [100.0, np.nan, 300.0], "p_axis": [10.0, 20.0, 30.0]})
    val = pd.DataFrame({"rr_interval": [np.nan, 9999.0], "p_axis": [np.nan, 40.0]})
    prep, report = fit_preprocessor(train, ["rr_interval", "p_axis"], missing_indicator_min_rate=0.0, circular_axes=False)
    assert report["imputer_statistics_train_only"]["rr_interval"] == 200.0
    transformed = prep.transform(val)
    assert transformed.loc[0, "rr_interval"] == 200.0
    assert transformed.loc[0, "rr_interval_missing"] == 1
    assert transformed.loc[1, "rr_interval"] == 9999.0


def test_circular_axes_use_imputed_angles() -> None:
    train = pd.DataFrame(
        {
            "p_axis": [0.0, np.nan],
            "qrs_axis": [90.0, 90.0],
            "t_axis": [0.0, 0.0],
        }
    )
    prep, _ = fit_preprocessor(
        train, ["p_axis", "qrs_axis", "t_axis"], missing_indicator_min_rate=1.1, circular_axes=True
    )
    out = prep.transform(train)
    assert "p_axis_sin" in prep.model_columns
    assert abs(out.loc[0, "p_axis_cos"] - 1.0) < 1e-6
    assert abs(out.loc[1, "p_axis"] - 0.0) < 1e-6
    assert abs(out.loc[1, "p_axis_cos"] - 1.0) < 1e-6


def test_entirely_missing_train_feature_raises() -> None:
    train = pd.DataFrame({"rr_interval": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="entirely missing"):
        fit_preprocessor(train, ["rr_interval"], circular_axes=False)


def test_patient_level_split_has_no_overlap() -> None:
    frame = _minimal_frame()
    manifest, _ = load_or_create_split(
        frame, split_path=None, use_shared=False, seed=42, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
    )
    overlap = manifest.groupby("subject_id")["split"].nunique()
    assert (overlap == 1).all()
    assigned, log = assign_split(frame, manifest)
    assert log["n_not_in_manifest"] == 0
    assert set(assigned["split"]).issubset({"train", "val", "test"})


def test_label_order_is_model_2_order() -> None:
    assert LABEL_ORDER == ["NORM", "HYP", "STTC", "MI", "CD"]


def test_multilabel_targets_retained() -> None:
    labels = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "NORM": [0],
            "HYP": [1],
            "STTC": [1],
            "MI": [0],
            "CD": [0],
        }
    )
    meas = pd.DataFrame({"subject_id": [1], "study_id": [10], "rr_interval": [800]})
    merged, log = align_labels(meas, labels)
    assert log["studies_missing_labels"] == 0
    assert merged.loc[0, ["HYP", "STTC"]].tolist() == [1, 1]


def test_class_imbalance_single_class_fallback() -> None:
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 3))
    y = np.zeros((20, 5), dtype=int)
    y[:, 1] = 1
    y[:8, 2] = 1
    model = fit_binary_models(X, y, rf_factory({"n_estimators": 5, "n_jobs": 1}, 0), {}, "rf")
    proba = model.predict_proba(X)
    assert proba.shape == (20, 5)
    assert np.allclose(proba[:, 0], 0.0)
    assert np.allclose(proba[:, 1], 1.0)
    assert isinstance(model.estimators[0], ConstantLabelFallback)


def test_classifiers_return_five_probabilities() -> None:
    rng = np.random.RandomState(1)
    X = rng.normal(size=(30, 4))
    y = (rng.rand(30, 5) > 0.6).astype(int)
    for i in range(5):
        if y[:, i].sum() == 0:
            y[0, i] = 1
        if y[:, i].sum() == 30:
            y[0, i] = 0
    model = fit_binary_models(X, y, rf_factory({"n_estimators": 8, "n_jobs": 1, "max_depth": 3}, 1), {}, "rf")
    proba = model.predict_proba(X[:7])
    assert proba.shape == (7, 5)
    assert np.all((proba >= 0) & (proba <= 1))


def test_thresholds_use_validation_only() -> None:
    y_val = np.array([[1, 0, 1, 0, 1], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 0, 1]])
    p_val = np.array(
        [
            [0.9, 0.1, 0.8, 0.2, 0.7],
            [0.2, 0.8, 0.7, 0.1, 0.2],
            [0.8, 0.6, 0.2, 0.9, 0.3],
            [0.1, 0.2, 0.9, 0.8, 0.6],
            [0.7, 0.3, 0.1, 0.2, 0.8],
        ]
    )
    out = select_thresholds(y_val, p_val, min_val_positives=2)
    assert len(out["thresholds"]) == 5
    pred = apply_thresholds(p_val, out["thresholds"])
    assert pred.shape == (5, 5)


def test_single_class_evaluation_is_undefined() -> None:
    y_true = np.zeros((6, 5), dtype=int)
    y_true[:, 0] = 1
    y_prob = np.linspace(0.1, 0.9, 30).reshape(6, 5)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate_multilabel(y_true, y_prob, y_pred, [0.5] * 5, n_studies=6, n_subjects=3)
    assert metrics["per_label"]["HYP"]["auroc"] is None
    assert metrics["per_label"]["HYP"]["auprc"] is None
    assert metrics["per_label"]["NORM"]["auroc"] is None
    assert metrics["per_label"]["NORM"]["positive_support"] == 6
    assert metrics["per_label"]["HYP"]["negative_support"] == 6


def test_model_bundle_reload(tmp_path: Path) -> None:
    import joblib

    rng = np.random.RandomState(2)
    X = rng.normal(size=(25, 3))
    y = (rng.rand(25, 5) > 0.5).astype(int)
    for i in range(5):
        y[i, i] = 1
        y[(i + 1) % 25, i] = 0
    model = fit_binary_models(X, y, rf_factory({"n_estimators": 6, "n_jobs": 1, "max_depth": 3}, 2), {}, "rf")
    path = tmp_path / "bundle.joblib"
    joblib.dump({"model": model, "thresholds": [0.5] * 5}, path)
    loaded = joblib.load(path)
    proba = loaded["model"].predict_proba(X[:4])
    assert proba.shape == (4, 5)


def test_feature_importance_output_when_supported() -> None:
    rng = np.random.RandomState(3)
    X = rng.normal(size=(40, 4))
    y = (rng.rand(40, 5) > 0.45).astype(int)
    for i in range(5):
        if len(np.unique(y[:, i])) < 2:
            y[0, i] = 0
            y[1, i] = 1
    model = fit_binary_models(X, y, rf_factory({"n_estimators": 8, "n_jobs": 1, "max_depth": 3}, 3), {}, "rf")
    table = permutation_importance_by_label(model, X, y, ["a", "b", "c", "d"], n_repeats=1, seed=3, dataset_name="validation")
    assert set(table["label"]) <= set(LABEL_ORDER)
    assert "importance_mean" in table.columns


def test_unmapped_records_are_excluded() -> None:
    frame = _minimal_frame()
    frame["unmapped"] = [False, False, False, True]
    frame["exclude_reason"] = ""
    kept, log = apply_label_eligibility(frame, require_at_least_one_label=True)
    assert log["n_excluded_unmapped"] == 1
    assert 30 not in set(kept["study_id"])


def test_nan_exclude_reason_does_not_drop_mapped_studies() -> None:
    frame = _minimal_frame()
    frame["unmapped"] = False
    frame["exclude_reason"] = None
    kept, log = apply_label_eligibility(frame, require_at_least_one_label=True)
    assert log["n_excluded_unmapped"] == 0
    assert len(kept) == 4


def test_source_distribution_audit_records_sentinel_spikes() -> None:
    from model_2b_measurement_only.data_validation import validate_source_table

    frame = _minimal_frame()
    report = validate_source_table(frame)
    rr = report["primary_measurement_distributions"]["rr_interval"]
    assert rr["sentinel_candidate_counts"]["29999"] == 1
    assert "p50" in rr["quantiles"]
    assert rr["n_unique_numeric"] >= 1


def test_missing_primary_measurement_columns_raise() -> None:
    frame = _minimal_frame().drop(columns=["rr_interval"])
    with pytest.raises(ValueError, match="Missing approved measurement columns"):
        engineer_features(frame, _feature_cfg())


def test_end_to_end_smoke(tmp_path: Path) -> None:
    smoke_csv = tmp_path / "smoke.csv"
    write_smoke_csv(smoke_csv)
    original = smoke_csv.read_bytes()
    summary = run_pipeline(
        CONFIG,
        smoke_csv=smoke_csv,
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "outputs",
    )
    assert smoke_csv.read_bytes() == original
    assert summary["stopped_before_training"] is False
    assert summary["n_train"] > 0 and summary["n_val"] > 0 and summary["n_test"] > 0
    assert "random_forest" in summary["models"]
    rf_metrics = summary["models"]["random_forest"]["test_metrics"]
    assert rf_metrics["n_studies"] == summary["n_test"]
    assert Path(tmp_path / "outputs" / "random_forest_bundle.joblib").exists()
    assert Path(tmp_path / "outputs" / "random_forest_test_metrics.json").exists()
    assert Path(tmp_path / "outputs" / "random_forest_confusion_matrices.csv").exists()
    assert Path(tmp_path / "outputs" / "labels_used.csv").exists()
    leaked = forbidden_columns(summary["feature_columns"])
    assert leaked == []
    import joblib

    bundle = joblib.load(tmp_path / "outputs" / "random_forest_bundle.joblib")
    X = np.zeros((2, len(bundle["columns"])))
    proba = bundle["model"].predict_proba(X)
    assert proba.shape == (2, 5)
