# Model 1: ECG Signal-Only Baseline

This package trains a 12-lead, waveform-only 1D CNN for the fixed multi-label
target order `NORM, HYP, STTC, MI, CD`. It reuses the Model 2 weak labels and
patient-level split artifacts. Reports, machine measurements, demographics, and
identifiers are never model inputs.

## Dataset contract

Raw MIMIC-IV-ECG files are not included in this repository. After obtaining
authorized data access, create `waveform_index.csv` with:

```csv
subject_id,study_id,waveform_path
1,123,/path/to/record
```

`waveform_path` may point to a WFDB record base path, `.npy`, or `.npz` file.
NPZ files must contain a `waveform` array. The loader requires 12 leads, a
500 Hz sampling rate, and exactly 5000 samples for WFDB records. Leads are
reordered by their header names to the fixed project order; invalid or
non-finite records are excluded and reported rather than silently altered.

Do not commit waveform files or restricted patient data.

## Running

Install dependencies from the repository root:

```text
pip install -r requirements.txt
```

Run the synthetic end-to-end smoke pipeline:

```text
python -m model_1_ecg_only.run_pipeline --config model_1_ecg_only/config.yaml --smoke
```

Run real data directly from the extracted Train/Test folders:

```text
python -m model_1_ecg_only.run_pipeline --config model_1_ecg_only/config.yaml --records-root D:\ECG\ECG --machine-measurements D:\ECG_Project\data\machine_measurements.csv --output-dir D:\ECG_Project\outputs\model_1_ecg_only
```

For the verified pilot cohort, also provide the source record list and use a
separate output directory:

```text
python -m model_1_ecg_only.run_pipeline --config model_1_ecg_only/config.yaml --records-root D:\ECG\ECG --record-list D:\ECG_Project\data\record_list.csv --machine-measurements D:\ECG_Project\data\machine_measurements.csv --output-dir D:\ECG_Project\outputs\model_1_ecg_only_pilot
```

The real run discovers WFDB files, matches them by `subject_id` and `study_id`
to the existing project's `LabelMapper` rules, and keeps the supplied Test
patients completely held out. It splits only the supplied Train patients into
train and validation groups. It does not use the synthetic smoke split artifact
and does not create a random test split. The pilot run reports cross-source
matching and exclusions in `cohort_audit.json`; its 20-patient subset is
descriptive and is not directly comparable with the larger Model 2/2B cohorts.

Outputs include the checkpoint, training history, normalization statistics,
metrics, thresholds, run summary, and gradient-saliency explanations.
