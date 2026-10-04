"""End-to-end Model 2B pipeline: numerical ECG measurements only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from model_2_text_only.utils import (
    LABEL_ORDER,
    ensure_dir,
    load_yaml,
    resolve_path,
    save_json,
    save_yaml,
    set_seed,
)

from .classifiers import logistic_factory, select_on_validation
from .cohort_builder import (
    align_labels,
    apply_label_eligibility,
    apply_measurement_eligibility,
    load_or_generate_labels,
)
from .compare_with_model_2 import compare_reports, evaluate_model_2_on_studies, write_markdown_report
from .data_loader import load_measurements, missing_required_columns
from .data_splitting import assign_split, load_or_create_split
from model_2_text_only.data_splitting import sample_working_cohort
from .data_validation import drop_missing_identifiers, resolve_duplicate_studies, validate_source_table
from .evaluate import evaluate_multilabel
from .explainability import impurity_importance, permutation_importance_by_label, plot_top_features
from .feature_engineering import engineer_features
from .preprocessing import fit_preprocessor
from .thresholds import apply_thresholds, select_thresholds
from .train_random_forest import train_random_forest
from .train_xgboost import train_xgboost, xgboost_available
from .utils import assert_feature_allowlist, forbidden_columns, package_versions


def run_pipeline(
    config_path: Path,
    smoke_csv: Path | None = None,
    artifacts_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    cfg = load_yaml(config_path)
    root = config_path.parent.parent
    set_seed(int(cfg["random_seed"]))
    artifacts_dir = ensure_dir(Path(artifacts_dir) if artifacts_dir else resolve_path(root, cfg["artifacts_dir"]))
    output_dir = ensure_dir(Path(output_dir) if output_dir else resolve_path(root, cfg["output_dir"]))
    smoke = smoke_csv is not None

    input_csv = Path(smoke_csv) if smoke_csv else resolve_path(root, cfg["input_csv"])
    original_mtime = input_csv.stat().st_mtime
    original_size = input_csv.stat().st_size

    exclusion_log: dict[str, Any] = {"steps": []}
    frame = load_measurements(input_csv)
    validation = validate_source_table(frame)
    exclusion_log["input_validation"] = validation
    _log_step(exclusion_log, "loaded", len(frame))

    missing = missing_required_columns(list(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame, id_log = drop_missing_identifiers(frame)
    exclusion_log["identifier_filter"] = id_log
    _log_step(exclusion_log, "after_missing_id_drop", len(frame))

    frame, dup_log = resolve_duplicate_studies(frame)
    exclusion_log["duplicate_studies"] = dup_log
    _log_step(exclusion_log, "after_study_dedup", len(frame))

    labels, label_info = load_or_generate_labels(
        frame,
        labels_path=resolve_path(root, cfg["shared_labels_path"]),
        mapping_path=resolve_path(root, cfg["label_mapping_path"]),
        use_shared=bool(cfg["data"].get("use_shared_labels", True)) and not smoke,
    )
    if (not smoke) and (not label_info.get("reused_shared_artifact")):
        labels_path = resolve_path(root, cfg["shared_labels_path"])
        if not labels_path.exists():
            compact = [
                c
                for c in ["subject_id", "study_id", *LABEL_ORDER, "unmapped", "poor_quality", "ambiguous"]
                if c in labels.columns
            ]
            labels[compact].to_csv(labels_path, index=False)
            label_info["wrote_shared_compact_labels"] = str(labels_path)
    _save_label_reference(labels, label_info, output_dir)
    labeled, align_log = align_labels(frame, labels)
    exclusion_log["label_alignment"] = align_log
    if "exclude_reason" not in labeled.columns:
        labeled["exclude_reason"] = ""
    labeled["exclude_reason"] = labeled["exclude_reason"].fillna("")
    missing_lab = labeled[LABEL_ORDER].isna().any(axis=1)
    labeled.loc[missing_lab, "exclude_reason"] = "missing_target_labels"
    labeled = labeled[~missing_lab].copy()
    _log_step(exclusion_log, "after_label_alignment", len(labeled))

    labeled, elig_log = apply_label_eligibility(
        labeled,
        require_at_least_one_label=bool(cfg["data"].get("require_at_least_one_label", True)),
        exclude_poor_quality_unmapped=bool(cfg["data"].get("exclude_poor_quality_unmapped", True)),
    )
    exclusion_log["label_eligibility"] = elig_log
    _log_step(exclusion_log, "after_label_eligibility", len(labeled))

    split_cfg = cfg["splitting"]
    use_shared_split = bool(cfg["data"].get("use_shared_split", True)) and not smoke
    split_info: dict[str, Any] = {"reused_shared_artifact": False}
    manifest: pd.DataFrame | None = None
    if use_shared_split:
        manifest, split_info = load_or_create_split(
            labeled,
            split_path=resolve_path(root, cfg["shared_split_path"]),
            use_shared=True,
            seed=int(cfg["random_seed"]),
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
        )
        if split_info.get("reused_shared_artifact"):
            labeled, assign_log = assign_split(labeled, manifest)
            exclusion_log["split_assignment"] = assign_log
            _log_step(exclusion_log, "after_shared_split_inner_join", len(labeled))
        else:
            use_shared_split = False
            manifest = None

    engineered, feature_cols, feature_audit = engineer_features(
        labeled, cfg["features"], cfg.get("validity")
    )
    cohort = labeled.copy()
    for col in engineered.columns:
        cohort[col] = engineered[col].to_numpy()

    cohort, meas_log = apply_measurement_eligibility(
        cohort, min_valid_primary_features=int(cfg["data"].get("min_valid_primary_features", 1))
    )
    exclusion_log["measurement_eligibility"] = meas_log
    _log_step(exclusion_log, "after_measurement_eligibility", len(cohort))

    if split_info.get("reused_shared_artifact"):
        assigned = cohort
        assign_log = exclusion_log.get("split_assignment") or {}
        assign_log["n_after_measurement_filter"] = int(len(assigned))
        assign_log["split_study_counts_after_measurement_filter"] = assigned["split"].value_counts().to_dict()
        exclusion_log["split_assignment"] = assign_log
    else:
        max_studies = int(cfg["data"].get("working_cohort_max_studies") or 0)
        if (not smoke) and max_studies > 0 and len(cohort) > max_studies:
            sampled = sample_working_cohort(
                cohort, max_studies=max_studies, seed=int(cfg["random_seed"])
            )
            exclusion_log["working_cohort_sample"] = {
                "n_before": int(len(cohort)),
                "n_after": int(len(sampled)),
                "max_studies": max_studies,
                "method": "model_2_sample_working_cohort",
            }
            cohort = sampled
            _log_step(exclusion_log, "after_working_cohort_sample", len(cohort))
        manifest, split_info = load_or_create_split(
            cohort,
            split_path=None,
            use_shared=False,
            seed=int(cfg["random_seed"]),
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
        )
        assigned, assign_log = assign_split(cohort, manifest)
        exclusion_log["split_assignment"] = assign_log
        manifest.to_csv(artifacts_dir / "patient_split_manifest.csv", index=False)
        _log_step(exclusion_log, "after_created_split", len(assigned))

    train_df = assigned[assigned["split"] == "train"].copy()
    val_df = assigned[assigned["split"] == "val"].copy()
    test_df = assigned[assigned["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("A split is empty after measurement eligibility; cannot train honestly")

    expected_features = list(cfg["features"].get("primary") or []) + list(cfg["features"].get("derived") or [])
    assert_feature_allowlist(feature_cols, expected_features)
    leaked = forbidden_columns(feature_cols)
    if leaked:
        raise ValueError(f"Forbidden columns in features: {leaked}")

    circular = bool(cfg["features"].get("circular_axes", True))
    preprocessor, prep_report = fit_preprocessor(
        train_df,
        feature_cols,
        missing_indicator_min_rate=float(cfg["features"].get("missing_indicator_min_rate", 0.05)),
        scale=False,
        circular_axes=circular,
    )
    scaled_prep, scaled_report = fit_preprocessor(
        train_df,
        feature_cols,
        missing_indicator_min_rate=float(cfg["features"].get("missing_indicator_min_rate", 0.05)),
        scale=True,
        circular_axes=circular,
    )
    X_train = preprocessor.transform(train_df, scale=False).to_numpy(dtype=float)
    X_val = preprocessor.transform(val_df, scale=False).to_numpy(dtype=float)
    X_test = preprocessor.transform(test_df, scale=False).to_numpy(dtype=float)
    X_train_s = scaled_prep.transform(train_df, scale=True).to_numpy(dtype=float)
    X_val_s = scaled_prep.transform(val_df, scale=True).to_numpy(dtype=float)
    X_test_s = scaled_prep.transform(test_df, scale=True).to_numpy(dtype=float)
    y_train = train_df[LABEL_ORDER].to_numpy(dtype=int)
    y_val = val_df[LABEL_ORDER].to_numpy(dtype=int)
    y_test = test_df[LABEL_ORDER].to_numpy(dtype=int)
    model_columns = preprocessor.model_columns
    assert X_train.shape[1] == len(model_columns)

    joblib.dump(preprocessor, output_dir / "imputer.joblib")
    joblib.dump(scaled_prep, output_dir / "imputer_and_scaler.joblib")

    rf_grid, xgb_grid, lr_grid = _grids(cfg, smoke)
    selection_metric = str(cfg["selection"]["metric"])
    fitted: dict[str, Any] = {}

    rf_extra = {
        "class_weight": cfg["models"]["random_forest"].get("class_weight", "balanced"),
        "n_jobs": 1 if smoke else cfg["models"]["random_forest"].get("n_jobs", -1),
    }
    rf_model, rf_info = train_random_forest(
        X_train, y_train, X_val, y_val, rf_grid, int(cfg["random_seed"]), selection_metric, rf_extra
    )
    fitted["random_forest"] = _finalize_model(
        rf_model, rf_info, X_val, y_val, X_test, y_test, test_df, cfg, output_dir, "random_forest",
        model_columns, preprocessor, train_df, val_df,
    )

    available, xgb_version = xgboost_available()
    if cfg["models"]["xgboost"].get("enabled", True) and available:
        xgb_model, xgb_info = train_xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            xgb_grid,
            int(cfg["random_seed"]),
            selection_metric,
            extra_params={"n_jobs": 1} if smoke else None,
        )
        fitted["xgboost"] = _finalize_model(
            xgb_model, xgb_info, X_val, y_val, X_test, y_test, test_df, cfg, output_dir, "xgboost",
            model_columns, preprocessor, train_df, val_df,
        )
    else:
        save_json(
            {
                "skipped": True,
                "reason": "xgboost_not_installed" if not available else "disabled_in_config",
                "xgboost_version": xgb_version,
            },
            output_dir / "xgboost_skipped.json",
        )

    if cfg["models"]["logistic_regression"].get("enabled", True):
        lr_extra = {
            "class_weight": cfg["models"]["logistic_regression"].get("class_weight", "balanced"),
            "max_iter": cfg["models"]["logistic_regression"].get("max_iter", 2000),
        }

        def lr_builder(params: dict[str, Any]):
            return logistic_factory({**lr_extra, **params}, int(cfg["random_seed"]))

        lr_model, lr_info = select_on_validation(
            X_train_s, y_train, X_val_s, y_val, lr_builder, lr_grid, "logistic_regression", selection_metric
        )
        fitted["logistic_regression"] = _finalize_model(
            lr_model, lr_info, X_val_s, y_val, X_test_s, y_test, test_df, cfg, output_dir, "logistic_regression",
            model_columns, scaled_prep, train_df, val_df, scaled=True,
        )

    n_repeats = int(cfg["explainability"]["permutation_n_repeats"])
    if smoke:
        n_repeats = int(cfg.get("smoke", {}).get("permutation_n_repeats", 1))
    importance_tables = {}
    for name, payload in fitted.items():
        model = payload["model"]
        Xv = X_val_s if name == "logistic_regression" else X_val
        perm = permutation_importance_by_label(
            model, Xv, y_val, model_columns, n_repeats, int(cfg["random_seed"]), "validation"
        )
        perm_path = output_dir / f"{name}_permutation_importance.csv"
        perm.to_csv(perm_path, index=False)
        plot_top_features(perm, output_dir / f"{name}_permutation_importance.png", int(cfg["explainability"]["top_k"]))
        imp = impurity_importance(model, model_columns)
        if len(imp):
            imp.to_csv(output_dir / f"{name}_native_importance.csv", index=False)
        importance_tables[name] = perm_path.name
        payload.pop("model", None)

    if input_csv.stat().st_mtime != original_mtime or input_csv.stat().st_size != original_size:
        raise RuntimeError("Input CSV was modified; aborting")

    model_2_metrics = _load_json(resolve_path(root, cfg["model_2_output_dir"]) / "test_metrics.json")
    common_ids = test_df["study_id"].tolist()
    common_m2 = None
    if not smoke:
        common_m2 = evaluate_model_2_on_studies(
            common_ids,
            resolve_path(root, cfg["model_2_output_dir"]),
            root / "model_2_text_only" / "config.yaml",
            artifacts_dir / "cohort_manifest.csv",
        )
    cohort_note = _cohort_comparison_note(
        manifest,
        test_df,
        assign_log,
        split_info,
        model_2_metrics,
        label_info,
        common_m2,
    )
    comparison = compare_reports(
        model_2_metrics,
        fitted,
        common_m2,
        cohort_note,
    )
    save_json(comparison, output_dir / "comparison_with_model_2.json")
    write_markdown_report(comparison, output_dir / "comparison_with_model_2.md")

    pd.DataFrame(
        {
            "study_id": assigned["study_id"],
            "subject_id": assigned["subject_id"],
            "split": assigned["split"],
            **{name: assigned[name] for name in LABEL_ORDER},
            "n_valid_primary_features": assigned["n_valid_primary_features"],
        }
    ).to_csv(output_dir / "model_2b_cohort.csv", index=False)

    save_json(feature_audit, output_dir / "feature_validity_audit.json")
    save_yaml(
        {
            "primary": list(cfg["features"].get("primary") or []),
            "derived": list(cfg["features"].get("derived") or []),
            "circular_axes": circular,
            "excluded": list(cfg["features"].get("excluded") or []),
            "feature_columns_before_imputation": feature_cols,
            "model_columns": model_columns,
            "formulas": {
                "pr_interval": "p_end - p_onset",
                "qrs_duration": "qrs_end - qrs_onset",
                "qt_interval": "t_end - qrs_onset",
                "axis_sin": "sin(deg2rad(imputed_axis))",
                "axis_cos": "cos(deg2rad(imputed_axis))",
            },
        },
        output_dir / "feature_engineering_config.yaml",
    )
    save_yaml({"feature_columns": feature_cols, "model_columns": model_columns}, output_dir / "feature_list.yaml")
    save_json(
        {
            "input_rows": exclusion_log.get("input_validation", {}).get("n_rows"),
            "steps": exclusion_log.get("steps"),
            "identifier_filter": exclusion_log.get("identifier_filter"),
            "duplicate_studies": exclusion_log.get("duplicate_studies"),
            "label_alignment": exclusion_log.get("label_alignment"),
            "label_eligibility": exclusion_log.get("label_eligibility"),
            "measurement_eligibility": exclusion_log.get("measurement_eligibility"),
            "split_assignment": exclusion_log.get("split_assignment"),
            "final_train": int(len(train_df)),
            "final_val": int(len(val_df)),
            "final_test": int(len(test_df)),
            "final_test_patients": int(test_df["subject_id"].nunique()),
            "note": (
                "When the Model 2 patient_split_manifest is reused, Model 2B is "
                "restricted to that working cohort (text-eligible studies after "
                "leakage masking), then further restricted by measurement eligibility."
            ),
        },
        output_dir / "inclusion_exclusion_report.json",
    )
    save_json(prep_report, output_dir / "imputation_report.json")
    save_json(scaled_report, output_dir / "scaler_report.json")
    save_json(exclusion_log, output_dir / "exclusion_log.json")
    save_json(split_info, output_dir / "split_info.json")
    save_json(
        {
            "shared_split_reused": bool(split_info.get("reused_shared_artifact")),
            "train": _split_label_stats(train_df),
            "val": _split_label_stats(val_df),
            "test": _split_label_stats(test_df),
        },
        output_dir / "split_stats.json",
    )
    save_json(
        {
            "package_versions": package_versions(),
            "seed": cfg["random_seed"],
            "config_path": str(config_path),
            "input_csv": str(input_csv),
            "label_info": label_info,
            "split_info": split_info,
            "label_mapping_version": label_info.get("mapping_version"),
            "weak_labels": True,
            "clinically_verified": False,
        },
        output_dir / "reproducibility.json",
    )
    save_yaml(
        load_yaml(resolve_path(root, cfg["label_mapping_path"])),
        output_dir / "label_mapping_used.yaml",
    )
    _write_data_quality_report(output_dir, validation, feature_audit, exclusion_log, train_df, feature_cols)

    summary = {
        "stopped_before_training": False,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_test_subjects": int(test_df["subject_id"].nunique()),
        "feature_columns": feature_cols,
        "model_columns": model_columns,
        "models": {k: {"test_metrics": v.get("test_metrics"), "selection": v.get("selection")} for k, v in fitted.items()},
        "exclusion_log": exclusion_log,
        "label_info": label_info,
        "importance_tables": importance_tables,
        "output_dir": str(output_dir),
    }
    save_json(summary, output_dir / "run_summary.json")
    return summary


def _split_label_stats(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_studies": int(len(frame)),
        "n_subjects": int(frame["subject_id"].nunique()) if len(frame) else 0,
        "label_prevalence": {
            name: {
                "n_pos": int(frame[name].sum()),
                "prevalence": float(frame[name].mean()) if len(frame) else None,
            }
            for name in LABEL_ORDER
            if name in frame.columns
        },
    }


def _cohort_comparison_note(
    manifest: pd.DataFrame | None,
    test_df: pd.DataFrame,
    assign_log: dict[str, Any],
    split_info: dict[str, Any],
    model_2_metrics: dict[str, Any] | None,
    label_info: dict[str, Any],
    common_m2: dict[str, Any] | None,
) -> dict[str, Any]:
    reused = bool(split_info.get("reused_shared_artifact"))
    m2b_test_ids = set(pd.to_numeric(test_df["study_id"], errors="raise").astype("int64").tolist())
    manifest_test_ids: set[int] = set()
    n_manifest_test_missing_from_2b = None
    if reused and manifest is not None and "split" in manifest.columns:
        test_manifest = manifest[manifest["split"].eq("test")]
        manifest_test_ids = set(
            pd.to_numeric(test_manifest["study_id"], errors="raise").astype("int64").tolist()
        )
        n_manifest_test_missing_from_2b = int(len(manifest_test_ids - m2b_test_ids))
    common_n = int(len(test_df))
    if common_m2 and common_m2.get("n_studies") is not None:
        common_n = int(common_m2["n_studies"])
    return {
        "model_2_original_test_studies": (model_2_metrics or {}).get("n_studies"),
        "model_2b_test_studies": int(len(test_df)),
        "model_2b_test_patients": int(test_df["subject_id"].nunique()),
        "model_2_manifest_test_studies": int(len(manifest_test_ids)) if reused else None,
        "n_model_2_test_studies_absent_from_model_2b": n_manifest_test_missing_from_2b,
        "n_measurement_studies_not_in_shared_manifest": int(assign_log.get("n_not_in_manifest", 0)),
        "common_test_studies": common_n,
        "common_cohort_evaluated": common_m2 is not None,
        "shared_split_reused": bool(split_info.get("reused_shared_artifact")),
        "label_source": label_info,
        "note": (
            "Direct modality comparison requires the same patient-level test studies. "
            "If Model 2B dropped measurement-ineligible records, original Model 2 test "
            "metrics and common-cohort Model 2 metrics are reported separately."
        ),
    }


def _finalize_model(
    model,
    info: dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
    name: str,
    model_columns: list[str],
    preprocessor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scaled: bool = False,
) -> dict[str, Any]:
    val_prob = model.predict_proba(X_val)
    thresh_out = select_thresholds(
        y_val,
        val_prob,
        strategy=str(cfg["thresholds"]["strategy"]),
        fallback=float(cfg["thresholds"]["fallback"]),
        min_val_positives=int(cfg["thresholds"]["min_val_positives"]),
    )
    save_json(thresh_out, output_dir / f"{name}_thresholds.json")
    thresholds = thresh_out["thresholds"]
    test_prob = model.predict_proba(X_test)
    test_pred = apply_thresholds(test_prob, thresholds)
    metrics = evaluate_multilabel(
        y_test,
        test_prob,
        test_pred,
        thresholds,
        n_studies=len(test_df),
        n_subjects=int(test_df["subject_id"].nunique()),
    )
    save_json(metrics, output_dir / f"{name}_test_metrics.json")
    _metrics_to_csv(metrics, output_dir / f"{name}_test_metrics.csv")
    _per_label_csv(metrics, output_dir / f"{name}_test_per_label_metrics.csv")
    _confusion_csv(metrics, output_dir / f"{name}_confusion_matrices.csv")
    joblib.dump(
        {"model": model, "preprocessor": preprocessor, "thresholds": thresholds, "columns": model_columns, "scaled": scaled},
        output_dir / f"{name}_bundle.joblib",
    )
    save_json(
        {
            "name": name,
            "selection": info,
            "label_order": LABEL_ORDER,
            "class_imbalance": {
                label: {
                    "train_positives": int(train_df[label].sum()),
                    "train_prevalence": float(train_df[label].mean()),
                    "val_positives": int(val_df[label].sum()),
                    "val_prevalence": float(val_df[label].mean()),
                }
                for label in LABEL_ORDER
            },
        },
        output_dir / f"{name}_training_config.json",
    )
    error_rows = []
    for i, label in enumerate(LABEL_ORDER):
        yt, yp = y_test[:, i], test_pred[:, i]
        error_rows.append(
            {
                "label": label,
                "false_positives": int(((yp == 1) & (yt == 0)).sum()),
                "false_negatives": int(((yp == 0) & (yt == 1)).sum()),
            }
        )
    pd.DataFrame(error_rows).to_csv(output_dir / f"{name}_error_counts.csv", index=False)
    return {
        "model": model,
        "test_metrics": metrics,
        "selection": info,
        "thresholds": thresh_out,
    }


def _grids(cfg: dict[str, Any], smoke: bool) -> tuple[dict, dict, dict]:
    rf = dict(cfg["models"]["random_forest"]["grid"])
    xgb = dict(cfg["models"]["xgboost"]["grid"])
    lr = dict(cfg["models"]["logistic_regression"]["grid"])
    if smoke:
        smoke_cfg = cfg.get("smoke") or {}
        rf = {"n_estimators": [int(smoke_cfg.get("rf_n_estimators", 8))], "max_depth": [4], "min_samples_leaf": [1]}
        xgb = {"n_estimators": [int(smoke_cfg.get("xgb_n_estimators", 8))], "max_depth": [2], "learning_rate": [0.3]}
        lr = {"C": [1.0]}
    return rf, xgb, lr


def _log_step(log: dict[str, Any], name: str, n: int, extra: dict[str, Any] | None = None) -> None:
    rec = {"step": name, "n": int(n)}
    if extra:
        rec.update(extra)
    log["steps"].append(rec)
    print(f"[{name}] n={n}")


def _metrics_to_csv(metrics: dict[str, Any], path: Path) -> None:
    rows = [{"metric": k, "value": v} for k, v in metrics.items() if k not in {"per_label", "label_order", "thresholds"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _per_label_csv(metrics: dict[str, Any], path: Path) -> None:
    rows = [{"label": label, **vals} for label, vals in metrics["per_label"].items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def _confusion_csv(metrics: dict[str, Any], path: Path) -> None:
    rows = []
    for label, vals in metrics["per_label"].items():
        rows.append(
            {
                "label": label,
                "tp": vals.get("tp"),
                "fp": vals.get("fp"),
                "tn": vals.get("tn"),
                "fn": vals.get("fn"),
                "positive_support": vals.get("positive_support"),
                "negative_support": vals.get("negative_support"),
                "threshold": vals.get("threshold"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_label_reference(labels: pd.DataFrame, label_info: dict[str, Any], output_dir: Path) -> None:
    keep = [c for c in ["subject_id", "study_id", *LABEL_ORDER, "unmapped", "poor_quality"] if c in labels.columns]
    labels[keep].to_csv(output_dir / "labels_used.csv", index=False)
    save_json(
        {
            **label_info,
            "n_label_rows": int(len(labels)),
            "weak_labels": True,
            "clinically_verified": False,
            "note": (
                "Targets are reused from Model 2 mapping version 1.0.0 when the shared "
                "artifact exists; otherwise they are regenerated with the same LabelMapper. "
                "Report text is not used as a Model 2B feature."
            ),
        },
        output_dir / "label_source.json",
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_data_quality_report(
    output_dir: Path,
    validation: dict[str, Any],
    feature_audit: dict[str, Any],
    exclusion_log: dict[str, Any],
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    missing_train = {c: float(train_df[c].isna().mean()) for c in feature_cols}
    save_json(
        {
            "source_validation": validation,
            "feature_audit": feature_audit,
            "exclusion_steps": exclusion_log.get("steps"),
            "train_missingness_before_imputation": missing_train,
            "high_missingness_train_features": {
                c: rate for c, rate in missing_train.items() if rate >= 0.20
            },
            "label_prevalence_train": {
                name: float(train_df[name].mean()) for name in LABEL_ORDER if name in train_df.columns
            },
        },
        output_dir / "data_quality_report.json",
    )


def write_smoke_csv(path: Path) -> Path:
    rng = np.random.RandomState(0)
    rows = []
    phrases = [
        ["Sinus rhythm", "Normal ECG"],
        ["Sinus rhythm", "Left ventricular hypertrophy", "Abnormal ECG"],
        ["Sinus tachycardia", "Nonspecific T abnormalities, diffuse leads", "Abnormal ECG"],
        ["Sinus rhythm", "Inferior infarct - age undetermined", "Abnormal ECG"],
        ["Sinus rhythm", "Right bundle branch block", "Abnormal ECG"],
        ["Atrial fibrillation", "Possible left ventricular hypertrophy"],
        ["Sinus bradycardia", "Lateral ST-T changes are nonspecific"],
        ["Ventricular pacing", "Pacemaker rhythm - no further analysis"],
        ["Sinus rhythm", "Possible anterior infarct - age undetermined"],
        ["Sinus rhythm", "Normal ECG except for rate"],
        ["Sinus rhythm", "No evidence of inferior infarct", "Normal ECG"],
        ["Sinus rhythm", "Extensive T wave changes may be due to myocardial ischemia"],
    ]
    sid = 1
    subj = 1
    for i in range(48):
        stmt = phrases[i % len(phrases)]
        rr = int(rng.randint(600, 1100))
        p_on = 40
        p_end = p_on + int(rng.randint(80, 120))
        q_on = p_end + int(rng.randint(40, 80))
        q_end = q_on + int(rng.randint(70, 110))
        t_end = q_end + int(rng.randint(200, 320))
        if i % 11 == 0:
            p_end = 29999
        if i % 13 == 0:
            p_axis = 32767
        else:
            p_axis = int(rng.randint(-30, 90))
        row = {
            "subject_id": subj,
            "study_id": sid,
            "cart_id": 1,
            "ecg_time": "2000-01-01",
            **{f"report_{k}": stmt[k] if k < len(stmt) else None for k in range(18)},
            "bandwidth": "0.5-40",
            "filtering": "60",
            "rr_interval": rr,
            "p_onset": p_on,
            "p_end": p_end,
            "qrs_onset": q_on,
            "qrs_end": q_end,
            "t_end": t_end,
            "p_axis": p_axis,
            "qrs_axis": int(rng.randint(-40, 90)),
            "t_axis": int(rng.randint(-20, 80)),
        }
        rows.append(row)
        sid += 1
        if i % 2 == 1:
            subj += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model 2B machine-measurement-only baseline")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke_csv = None
    artifacts_dir = None
    output_dir = None
    if args.smoke:
        smoke_csv = Path(__file__).resolve().parent / "tests" / "fixtures" / "smoke_measurements.csv"
        write_smoke_csv(smoke_csv)
        artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts" / "model_2b_smoke"
        output_dir = Path(__file__).resolve().parent.parent / "outputs" / "model_2b_measurement_only_smoke"
    summary = run_pipeline(Path(args.config), smoke_csv=smoke_csv, artifacts_dir=artifacts_dir, output_dir=output_dir)
    print("Run complete.")
    models = summary.get("models") or {}
    for name, payload in models.items():
        metrics = payload.get("test_metrics") or {}
        print(name, "macro_f1", metrics.get("macro_f1"), "micro_f1", metrics.get("micro_f1"))


if __name__ == "__main__":
    main()
