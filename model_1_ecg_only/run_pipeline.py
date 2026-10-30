"""End-to-end Model 1 ECG signal-only pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_2_text_only.data_splitting import sample_working_cohort, split_patients
from model_2_text_only.label_generation import LabelMapper, generate_labels
from model_2_text_only.thresholds import apply_thresholds, select_thresholds
from model_2_text_only.utils import LABEL_ORDER, load_yaml, resolve_path, save_json, set_seed
from model_2b_measurement_only.data_splitting import load_or_create_split
from .dataset import ECGWaveformDataset
from .explainability import gradient_saliency
from .model import ECG1DCNN, compute_pos_weight
from .train import predict_proba, train_model
from .waveform_loader import compute_normalization, discover_wfdb_records, load_waveform_index


def run_pipeline(
    config_path: Path,
    waveform_index_path: Path | None = None,
    smoke: bool = False,
    output_dir: Path | None = None,
    records_root: Path | None = None,
    machine_measurements_path: Path | None = None,
    record_list_path: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    cfg = load_yaml(config_path)
    root = config_path.parent.parent
    set_seed(int(cfg["random_seed"]))
    out = (output_dir or resolve_path(root, cfg["output_dir"])).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg["data"]
    n_leads = int(data_cfg["n_leads"])
    n_samples = int(round(float(data_cfg["duration_seconds"]) * int(data_cfg["sampling_frequency"])))
    if smoke:
        index_path = write_smoke_dataset(out / "smoke_waveforms", int(cfg["smoke"]["n_studies"]),
                                         int(cfg["smoke"]["n_samples"]), n_leads)
        index = load_waveform_index(index_path)
    else:
        if records_root is not None:
            index = discover_wfdb_records(records_root.resolve())
            index_path = out / "waveform_index.csv"
            index.to_csv(index_path, index=False)
        else:
            index_path = (waveform_index_path or resolve_path(root, cfg["waveform_index"])).resolve()
            index = load_waveform_index(index_path)
    index["waveform_path"] = index["waveform_path"].map(
        lambda value: str((index_path.parent / str(value)).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )
    if len(index) < 20:
        raise ValueError("Model 1 requires at least 20 indexed studies for a meaningful split")
    if machine_measurements_path is not None:
        cfg["_machine_measurements_path"] = str(machine_measurements_path.resolve())
    matched_measurements = None
    if "header_valid" not in index.columns:
        index["header_valid"] = True
    if "header_exclusion_reason" not in index.columns:
        index["header_exclusion_reason"] = ""
    cohort_audit = {
        "discovered_waveform_records": int(len(index)),
        "header_valid_records": int(index.get("header_valid", pd.Series(True, index=index.index)).sum()),
        "header_exclusions": index.loc[
            ~index.get("header_valid", pd.Series(True, index=index.index)).astype(bool),
            ["subject_id", "study_id", "header_exclusion_reason"],
        ].to_dict(orient="records"),
    }
    if not smoke:
        matched_measurements, source_audit = _match_external_tables(
            index,
            record_list_path,
            machine_measurements_path,
            root,
            cfg,
        )
        cohort_audit["source_matching"] = source_audit
        save_json(cohort_audit, out / "cohort_audit.json")
    labels = _load_labels(root, index, cfg, smoke, matched_measurements)
    index_features = index.drop(columns=[c for c in LABEL_ORDER if c in index.columns])
    frame = index_features.merge(labels, on=["subject_id", "study_id"], how="inner", validate="one_to_one")
    before_label_filter = frame.copy()
    frame = frame[frame[LABEL_ORDER].sum(axis=1) > 0].copy()
    cohort_audit["label_counts"] = {
        label: int(frame[label].sum()) for label in LABEL_ORDER
    }
    cohort_audit["unmapped_studies"] = before_label_filter.loc[
        before_label_filter[LABEL_ORDER].sum(axis=1).eq(0),
        ["subject_id", "study_id"],
    ].to_dict(orient="records")
    cohort_audit["label_eligible_studies"] = int(len(frame))
    cohort_audit["label_excluded_studies"] = int(len(before_label_filter) - len(frame))
    if frame.empty:
        raise ValueError("No waveform studies remain after label alignment")
    if not smoke and "split" in frame.columns:
        frame, manifest = _respect_provided_split(frame, int(cfg["random_seed"]))
        manifest.to_csv(out / "patient_split_manifest.csv", index=False)
    elif not smoke:
        raise ValueError(
            "Real Model 1 data must provide Train/Test split information in the waveform "
            "index or records-root layout; refusing to create a random test split."
        )
    else:
        frame, _ = _create_split(frame, int(cfg["random_seed"]))
    train_df, val_df, test_df = [frame[frame["split"].eq(name)].copy() for name in ("train", "val", "test")]
    if min(map(len, (train_df, val_df, test_df))) == 0:
        raise RuntimeError("A Model 1 split is empty")
    cohort_audit["split_counts"] = {
        split: {
            "studies": int(part["study_id"].nunique()),
            "patients": int(part["subject_id"].nunique()),
            "label_counts": {label: int(part[label].sum()) for label in LABEL_ORDER},
        }
        for split, part in (("train", train_df), ("validation", val_df), ("test", test_df))
    }
    frame.to_csv(out / "cohort_manifest.csv", index=False)
    pd.DataFrame(cohort_audit.get("unmapped_studies", [])).to_csv(
        out / "label_exclusions.csv", index=False
    )
    save_json(cohort_audit, out / "cohort_audit.json")
    paths = [Path(p) for p in train_df["waveform_path"]]
    mean, std = compute_normalization(paths, n_leads, n_samples, int(data_cfg["sampling_frequency"]))
    np.savez(out / "normalization.npz", mean=mean, std=std)
    prep = cfg["preprocessing"]
    common = dict(n_leads=n_leads, n_samples=n_samples, sampling_frequency=int(data_cfg["sampling_frequency"]),
                  mean=mean, std=std, clip_value=float(prep["clip_value"]))
    datasets = [ECGWaveformDataset(part, **common) for part in (train_df, val_df, test_df)]
    loaders = [DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=i == 0,
                          num_workers=int(cfg["training"]["num_workers"])) for i, ds in enumerate(datasets)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECG1DCNN(n_leads=n_leads, channels=list(cfg["model"]["channels"]),
                     kernels=list(cfg["model"]["kernels"]), dropout=float(cfg["model"]["dropout"]),
                     num_labels=5).to(device)
    training = cfg["training"]
    history = train_model(model, loaders[0], loaders[1], device, out, int(training["max_epochs"]),
                          float(training["learning_rate"]), float(training["weight_decay"]),
                          int(training["early_stopping_patience"]), compute_pos_weight(datasets[0].labels))
    save_json(history, out / "training_history.json")
    val_prob, val_true = predict_proba(model, loaders[1], device)
    threshold_info = select_thresholds(val_true, val_prob, **cfg["thresholds"])
    save_json(threshold_info, out / "thresholds.json")
    test_prob, test_true = predict_proba(model, loaders[2], device)
    test_pred = apply_thresholds(test_prob, threshold_info["thresholds"])
    from model_2_text_only.evaluate import evaluate_multilabel
    metrics = evaluate_multilabel(test_true, test_prob, test_pred, threshold_info["thresholds"],
                                  len(test_df), int(test_df["subject_id"].nunique()))
    save_json(metrics, out / "test_metrics.json")
    explanations = []
    for i in range(min(int(cfg["explainability"]["n_examples"]), len(datasets[2]))):
        batch = datasets[2][i]["waveform"].unsqueeze(0)
        explanations.append({"example_id": i, **gradient_saliency(model, batch, device)})
    save_json({"examples": explanations, "label_order": LABEL_ORDER}, out / "explainability.json")
    summary = {"stopped_before_training": False, "n_train": len(train_df), "n_val": len(val_df),
               "n_test": len(test_df), "device": str(device), "label_order": LABEL_ORDER,
               "test_metrics": metrics, "output_dir": str(out), "waveform_index": str(index_path)}
    save_json(summary, out / "run_summary.json")
    return summary


def _load_labels(
    root: Path,
    index: pd.DataFrame,
    cfg: dict[str, Any],
    smoke: bool,
    matched_measurements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if set(LABEL_ORDER).issubset(index.columns):
        return index[["subject_id", "study_id", *LABEL_ORDER]].copy()
    shared = resolve_path(root, cfg["shared_labels_path"])
    if matched_measurements is not None:
        mapper = LabelMapper.from_path(resolve_path(root, cfg["label_mapping_path"]))
        labels = generate_labels(matched_measurements, mapper)
    elif shared.exists() and not smoke:
        labels = pd.read_csv(shared)
    else:
        source = Path(cfg.get("_machine_measurements_path") or resolve_path(root, cfg["input_csv"]))
        if not source.exists():
            raise FileNotFoundError("Need artifacts/labels.csv or machine_measurements.csv to obtain labels")
        mapper = LabelMapper.from_path(resolve_path(root, cfg["label_mapping_path"]))
        study_ids = set(index["study_id"].tolist())
        header = pd.read_csv(source, nrows=0)
        report_columns = [column for column in header.columns if column.startswith("report_")]
        usecols = ["subject_id", "study_id", *report_columns]
        chunks = []
        for chunk in pd.read_csv(source, usecols=usecols, chunksize=50000, low_memory=False):
            selected = chunk[chunk["study_id"].isin(study_ids)]
            if not selected.empty:
                chunks.append(selected)
        if not chunks:
            raise ValueError("No waveform study IDs were found in the machine measurements file")
        labels = generate_labels(pd.concat(chunks, ignore_index=True), mapper)
    required = ["subject_id", "study_id", *LABEL_ORDER]
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise ValueError(f"Labels missing columns: {missing}")
    return labels[required].drop_duplicates("study_id")


def _match_external_tables(
    index: pd.DataFrame,
    record_list_path: Path | None,
    machine_measurements_path: Path | None,
    root: Path,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match the selected records to both source tables without silent drops."""
    audit: dict[str, Any] = {}
    if record_list_path is None:
        raise ValueError("record_list.csv is required for real cohort verification")
    if machine_measurements_path is None:
        raise ValueError("machine_measurements.csv is required for real cohort verification")
    record_list_path = record_list_path.resolve()
    machine_measurements_path = machine_measurements_path.resolve()
    record_list = pd.read_csv(record_list_path, usecols=["subject_id", "study_id", "file_name", "path"])
    if record_list["study_id"].duplicated().any():
        raise ValueError("record_list.csv contains duplicate study_id values")
    selected_ids = set(index["study_id"].tolist())
    record_matches = record_list[record_list["study_id"].isin(selected_ids)].copy()
    missing_record_list = sorted(selected_ids - set(record_matches["study_id"]))
    audit["record_list_rows"] = int(len(record_list))
    audit["record_list_matches"] = int(len(record_matches))
    audit["missing_in_record_list"] = missing_record_list
    audit["record_list_id_conflicts"] = record_matches.loc[
        ~record_matches["subject_id"].isin(index["subject_id"]),
        ["subject_id", "study_id"],
    ].to_dict(orient="records")
    machine_header = pd.read_csv(machine_measurements_path, nrows=0)
    report_columns = [column for column in machine_header.columns if column.startswith("report_")]
    usecols = ["subject_id", "study_id", *report_columns]
    chunks = []
    for chunk in pd.read_csv(machine_measurements_path, usecols=usecols, chunksize=50000, low_memory=False):
        selected = chunk[chunk["study_id"].isin(selected_ids)]
        if not selected.empty:
            chunks.append(selected)
    measurements = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    if measurements["study_id"].duplicated().any():
        raise ValueError("machine_measurements.csv contains duplicate selected study_id values")
    missing_measurements = sorted(selected_ids - set(measurements["study_id"]))
    audit["machine_measurement_matches"] = int(len(measurements))
    audit["missing_in_machine_measurements"] = missing_measurements
    merged = index.merge(
        record_matches[["subject_id", "study_id", "file_name", "path"]],
        on="study_id",
        how="left",
        suffixes=("", "_record_list"),
    )
    subject_mismatch_record = merged[
        merged["subject_id_record_list"].notna()
        & (merged["subject_id"] != merged["subject_id_record_list"])
    ]
    audit["record_list_subject_mismatches"] = subject_mismatch_record[
        ["subject_id", "study_id", "subject_id_record_list"]
    ].to_dict(orient="records")
    meas_ids = measurements.set_index("study_id")["subject_id"].to_dict()
    audit["measurement_subject_mismatches"] = [
        {"subject_id": int(row.subject_id), "study_id": int(row.study_id), "measurement_subject_id": int(meas_ids[row.study_id])}
        for row in index.itertuples()
        if row.study_id in meas_ids and int(meas_ids[row.study_id]) != int(row.subject_id)
    ]
    if missing_record_list or missing_measurements or audit["record_list_subject_mismatches"] or audit["measurement_subject_mismatches"]:
        raise ValueError(f"Cross-source cohort conflicts detected: {audit}")
    return measurements, audit


def _create_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest, _ = split_patients(frame, seed=seed)
    return frame.merge(manifest, on=["subject_id", "study_id"], how="inner"), manifest


def _respect_provided_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the supplied Test patients held out and split only supplied Train patients."""
    values = {str(value).lower() for value in frame["split"].dropna().unique()}
    if not {"train", "test"}.issubset(values):
        raise ValueError(f"Expected Train/Test recording groups, found {sorted(values)}")
    normalized = frame["split"].astype(str).str.lower()
    train = frame[normalized.eq("train")].copy()
    test = frame[normalized.eq("test")].copy()
    train_subjects = train["subject_id"].drop_duplicates().to_numpy()
    test_subjects = test["subject_id"].drop_duplicates().to_numpy()
    overlap = set(train_subjects).intersection(set(test_subjects))
    if overlap:
        raise RuntimeError(f"Provided Train/Test groups share patients: {sorted(overlap)}")
    if len(train_subjects) < 3:
        raise ValueError("At least three training patients are required to create train/validation splits")
    from sklearn.model_selection import train_test_split

    train_ids, val_ids = train_test_split(
        train_subjects, test_size=max(1, int(round(len(train_subjects) * 0.2))), random_state=seed
    )
    assigned = frame.copy()
    assigned["split"] = normalized.map(
        lambda value: "test" if value == "test" else value
    )
    assigned.loc[assigned["subject_id"].isin(train_ids), "split"] = "train"
    assigned.loc[assigned["subject_id"].isin(val_ids), "split"] = "val"
    manifest = assigned[["subject_id", "study_id", "split"]].copy()
    if manifest["split"].isna().any() or manifest["split"].eq("split").any():
        raise RuntimeError("Failed to assign every waveform to a split")
    if manifest.groupby("subject_id")["split"].nunique().gt(1).any():
        raise RuntimeError("Patient appears in multiple Model 1 splits")
    return assigned, manifest


def write_smoke_dataset(directory: Path, n_studies: int, n_samples: int, n_leads: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    rows = []
    for study in range(1, n_studies + 1):
        subject = (study - 1) // 2 + 1
        signal = rng.normal(0, 0.05, (n_leads, n_samples)).astype(np.float32)
        t = np.linspace(0, 10, n_samples, endpoint=False)
        signal += np.sin(2 * np.pi * (1 + study % 3 / 5) * t)[None, :] * (0.1 + (study % 5) / 100)
        path = directory / f"study_{study}.npz"
        np.savez_compressed(path, waveform=signal)
        label = LABEL_ORDER[(study - 1) % len(LABEL_ORDER)]
        rows.append({"subject_id": subject, "study_id": study, "waveform_path": str(path),
                     **{name: int(name == label) for name in LABEL_ORDER}})
    index = pd.DataFrame(rows)
    path = directory / "waveform_index.csv"
    index.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model 1 ECG signal-only baseline")
    parser.add_argument("--config", default="model_1_ecg_only/config.yaml")
    parser.add_argument("--waveform-index")
    parser.add_argument("--records-root")
    parser.add_argument("--machine-measurements")
    parser.add_argument("--record-list")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run_pipeline(
        Path(args.config),
        waveform_index_path=Path(args.waveform_index) if args.waveform_index else None,
        smoke=args.smoke,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        records_root=Path(args.records_root) if args.records_root else None,
        machine_measurements_path=Path(args.machine_measurements) if args.machine_measurements else None,
        record_list_path=Path(args.record_list) if args.record_list else None,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
