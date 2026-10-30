# Model comparison protocol

This file describes how to compare Models 1–4 (plus the extra Model 2B tabular baseline) after each pipeline has written `test_predictions.csv`. Generated numeric tables belong in `outputs/comparison/` and are not committed.

## Frozen experimental controls

- Labels: `artifacts/label_mapping.yaml` **v1.0.0** (weak labels from machine reports; not clinically verified).
- Split: `artifacts/patient_split_manifest.csv` (patient-level 70/15/15, seed 42). Do not rebuild after seeing test metrics.
- Text input for Models 2, 3, and 4: leakage-controlled residual `clinical_text` from `artifacts/cohort_manifest.csv`.
- Comparison set: inner join of test `study_id`s that have residual text **and** a usable 12-lead waveform (and measurements, if Model 2B is included).
- Thresholds: chosen on **validation** F1 only, then frozen. Common-cohort metrics reuse each model's own frozen thresholds.
- Identifiers are never features. Public error reports use `example_id` only.

## How to run

Environment (venv, install, data files): [running.md](running.md). From the repo root, with `.venv` activated:

```text
source .venv/bin/activate

# Model 2 (run first): weak labels, frozen split, residual text, then train/eval the text LSTM.
python -m model_2_text_only.run_pipeline --config configs/model_2_text_only.yaml

# Extra tabular baseline on machine measurements (same labels/split as Model 2).
python -m model_2b_measurement_only.run_pipeline --config configs/model_2b_measurement_only.yaml

# ECG-only 1D CNN on raw 12-lead waveforms (needs local waveform_root files).
python -m model_1_ecg_only.run_pipeline --config configs/model_1_ecg_only.yaml

# Concatenate ECG + text embeddings, then a linear classifier.
python -m model_3_concat_fusion.run_pipeline --config configs/model_3_concat_fusion.yaml

# Project both embeddings, softmax-gate them, then classify.
python -m model_4_attention_fusion.run_pipeline --config configs/model_4_attention_fusion.yaml

# Inner-join test predictions and write metric tables under outputs/comparison/.
python scripts/compare_models.py --dest outputs/comparison
```

Waveforms: put MIMIC-IV-ECG WFDB records under `waveform_root` (`files/pXXXX/p{subject_id}/s{study_id}/`) or `{study_id}.npy` arrays of shape `(12, 5000)`. Do not commit them.

Smoke (synthetic, not clinical):

```text
source .venv/bin/activate

# Tiny synthetic ECG-only run (no MIMIC files; not clinical results).
python -m model_1_ecg_only.run_pipeline --smoke

# Tiny synthetic concat-fusion run.
python -m model_3_concat_fusion.run_pipeline --smoke

# Tiny synthetic gated-attention fusion run.
python -m model_4_attention_fusion.run_pipeline --smoke

# Execute the pytest suite under tests/ (pythonpath includes src/).
pytest
```

## Fusion formulas

- **Model 3:** `logits = Linear(concat(z_ecg, z_text))`.
- **Model 4:** `P` projects each encoder to dimension `d`; `a = softmax(W [P(z_ecg); P(z_text)])`; `z = a_ecg P(z_ecg) + a_text P(z_text)`. Gate weights are inspectable and are **not** causal explanations.

Default training: finetune both encoders; text encoder uses a lower learning rate. Checkpoints from Models 1 and 2 are loaded when present (`strict=False`).

## Interpretation

Scores are agreement with rule-based weak labels. Masking does not remove synonym or template leakage in residual text. Machine measurements and machine reports may share a device-algorithm origin. These models are research prototypes and are not for clinical use.
