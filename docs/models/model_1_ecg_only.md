# Model 1: ECG-only 1D CNN

Predicts the same five weak labels as Model 2 from **raw 12-lead waveforms** only. It reuses the frozen Model 2 mapping (`artifacts/label_mapping.yaml` v1.0.0) and `patient_split_manifest.csv`. It does not use report text or machine measurements.

## Run

From the repo root, with the venv active (`source .venv/bin/activate`). First-time setup: [running.md](../running.md). Run **Model 2 first** so labels and the patient split exist. Put waveforms under `waveforms/` (or `waveform_root` in `configs/model_1_ecg_only.yaml`).

```text
# Train/evaluate the ECG-only 1D CNN on the frozen Model 2 split (needs local waveforms).
python -m model_1_ecg_only.run_pipeline --config configs/model_1_ecg_only.yaml

# Same pipeline on synthetic waveforms (CI / wiring check; not clinical).
python -m model_1_ecg_only.run_pipeline --config configs/model_1_ecg_only.yaml --smoke

# Run Model 1 tests only.
pytest tests/model_1_ecg_only
```

Do **not** download the full ~800k MIMIC dump. After Model 2, run `python scripts/list_waveform_downloads.py` and fetch **only** those study folders (default working cohort is ~8000; you can use fewer). Missing files are dropped from the CNN cohort; the frozen split is not rebuilt. Details: [running.md](../running.md).

Place MIMIC-IV-ECG WFDB records under `waveform_root` using the PhysioNet layout `files/pXXXX/p{subject_id}/s{study_id}/`, or `.npy` arrays named `{study_id}.npy` with shape `(12, 5000)`.
