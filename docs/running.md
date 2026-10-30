# Running this project

All commands below are from the **repository root**. Packages live in `src/`; configs in `configs/`. Do not commit `machine_measurements.csv`, waveforms, or generated CSVs.

This is a **research prototype**, not a clinical tool. `--smoke` and `pytest` use synthetic data and are not experimental results.

## 1. One-time environment

Python 3.10+ (the local `.venv` was created with 3.13). Create and enter the venv, then install the repo in editable mode so `python -m model_*` works:

```text
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

`pip install -e ".[dev]"` reads `pyproject.toml`: it puts `src/` on the import path and installs pandas, numpy, torch, scikit-learn, xgboost, wfdb, matplotlib, pytest. The first install can take a long time if PyTorch pulls CUDA wheels.

Check:

```text
which python                       # should be .../explainable-multimodal-ecg-diagnostic-classification/.venv/bin/python
python -c "import torch, model_2_text_only; print('ok')"
```

Every later terminal: `source .venv/bin/activate` first (prompt shows `(.venv)`).

Without an editable install you can instead `export PYTHONPATH=src` after activating the venv, then `pip install -r requirements.txt`.

## 2. Data you need

| File | Used by | Where |
| --- | --- | --- |
| `machine_measurements.csv` | Models 2 and 2B (and label generation) | Repo root. Controlled-access MIMIC-IV-ECG. **Not in git.** |
| 12-lead waveforms | Models 1, 3, 4 | `waveforms/` (config `waveform_root`). PhysioNet layout `files/pXXXX/p{subject_id}/s{study_id}/` or `{study_id}.npy` shape `(12, 5000)`. **Not in git.** Do **not** download all ~800k records — see below. |
| `artifacts/label_mapping.yaml` | All models | Already in the repo (v1.0.0). Do not retune after seeing test metrics. |

Model 2 writes (local, gitignored CSVs): `artifacts/labels.csv`, `patient_split_manifest.csv`, `cohort_manifest.csv`, `leakage_audit.json`. Later models reuse those. Run Model 2 before 1 / 2B / 3 / 4 on real data.

### You do not need 800k waveform files

MIMIC-IV-ECG has ~800,035 studies. This project never trains on all of them.

1. **Model 2 and 2B need no waveforms.** They only use `machine_measurements.csv`.
2. **Training is already a subset.** After labels are built on the full table, Model 2 samples a patient-level **working cohort** (`working_cohort_max_studies`, default **8000**) and freezes `artifacts/patient_split_manifest.csv`. That list is what Models 1/3/4 look for.
3. **If 8000 files is still too many**, set `working_cohort_max_studies` lower (e.g. 500–2000) **before** you freeze the split. Do not rebuild the split after looking at test metrics.
4. **Download only those study folders.** After Model 2:

```text
python scripts/list_waveform_downloads.py
```

That writes `artifacts/waveform_relative_dirs.txt` and `artifacts/waveform_physionet_urls.txt` (~2 URLs per study: `.hea` and `.dat`). With a PhysioNet account:

```text
mkdir -p waveforms
wget -r -N -c -np -nH --cut-dirs=3 --user YOUR_PHYSIONET_USER --ask-password \
  -i artifacts/waveform_physionet_urls.txt -P waveforms
```

Put files so a record lands at `waveforms/files/pXXXX/p{subject_id}/s{study_id}/{study_id}.hea` (and `.dat`), which is the PhysioNet layout.

5. **A partial download still runs.** Models 1/3/4 index `waveform_root`, keep studies whose files exist and look like 12×5000, and **drop the rest**. They do not rebuild the patient split. Comparison then uses the inner join of studies that have both residual text and a usable waveform. Fewer files → smaller CNN/fusion cohort, not a crash (unless zero files match).
6. **No real waveforms yet:** `--smoke` / `pytest` use synthetic arrays. Those numbers are not experiment results.

## 3. Full experiment (real files)

With the venv active:

```text
# 1) Labels, leakage mask, frozen 70/15/15 patient split, text LSTM
python -m model_2_text_only.run_pipeline --config configs/model_2_text_only.yaml

# 2) Tabular extra baseline on machine measurements (same split/labels)
python -m model_2b_measurement_only.run_pipeline --config configs/model_2b_measurement_only.yaml

# 3) Waveform-only 1D CNN (needs waveforms/)
python -m model_1_ecg_only.run_pipeline --config configs/model_1_ecg_only.yaml

# 4) Concat fusion of ECG + residual text
python -m model_3_concat_fusion.run_pipeline --config configs/model_3_concat_fusion.yaml

# 5) Gated attention fusion of the same encoders
python -m model_4_attention_fusion.run_pipeline --config configs/model_4_attention_fusion.yaml

# 6) Common-cohort tables (needs test_predictions.csv from the models you ran)
python scripts/compare_models.py --dest outputs/comparison
```

Omitting `--config` is fine; each CLI defaults to `configs/<package>.yaml`.

Outputs go under `outputs/<model>/` (checkpoints, metrics, `test_predictions.csv`, error analysis). Comparison writes `outputs/comparison/comparison.md`. None of that is committed.

## 4. Smoke tests (no MIMIC files)

```text
source .venv/bin/activate
python -m model_2_text_only.run_pipeline --smoke
python -m model_2b_measurement_only.run_pipeline --smoke
python -m model_1_ecg_only.run_pipeline --smoke
python -m model_3_concat_fusion.run_pipeline --smoke
python -m model_4_attention_fusion.run_pipeline --smoke
pytest
```

`pytest.ini` sets `pythonpath = src` and `testpaths = tests`.

## 5. What “done” looks like

- **Model 2 / 2B:** real-data pipelines exist; documented runs in `docs/model2.md` and `docs/models/model_2b_measurement_only.md` (local artifacts, not in git).
- **Models 1 / 3 / 4:** code is ready; real numbers need local waveforms, then step 6 above.
- Metrics are agreement with **weak report-derived labels**, not clinical accuracy.
