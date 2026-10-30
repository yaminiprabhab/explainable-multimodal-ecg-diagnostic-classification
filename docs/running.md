# Running this project

All commands below are from the **repository root**. Packages live in `src/`; configs in `configs/`.

This branch **includes the laptop slice**: `machine_measurements.csv` (2500 rows), `waveforms/` (~69 records), `artifacts/` labels/split, and `outputs/` (metrics, comparison table, explanation PNGs). That is **not** the full MIMIC dump. Cite [PhysioNet MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/). Still a research prototype, not a clinical tool.

`--smoke` and `pytest` use synthetic data and are not experimental results.

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

Keep this dumb-simple. **This checkout already has the slice.**

1. `machine_measurements.csv` at repo root → Models **2** and **2B**.
2. `waveforms/files/...` → Models **1 / 3 / 4** (missing files skipped; need train, val, and test).
3. After runs: `outputs/<model>/` and `outputs/comparison/comparison.md`.

Do not download the full ~800k dump. Mapping: `artifacts/label_mapping.yaml`.

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

Look in `outputs/<model>/` (`test_metrics.csv`, `test_predictions.csv`) and `outputs/comparison/comparison.md` after a compare run. Metrics are agreement with **weak report-derived labels**, not clinical accuracy. Laptop-slice numbers and a report checklist live in [study.md](study.md).
