# Model 2: Clinical-Text-Only Baseline

This directory contains the first complete model pipeline for the project. It creates reusable weak labels, a patient-level split, and a leakage-controlled LSTM text classifier.

## What this experiment is

Labels are **rule-based weak targets** derived from MIMIC-IV-ECG machine report lines (`report_0` … `report_17`). They are **not** independently adjudicated cardiologist diagnoses.

The classifier is trained on **leakage-controlled residual text**: diagnostic statements that fired labeling rules are removed, along with summary phrases such as `Normal ECG` / `Abnormal ECG`. Remaining text is mostly rhythm, rate, technical notes, and other non-trigger statements.

**Do not interpret test metrics as leakage-free diagnostic performance.** Masking trigger phrases does not remove synonyms, correlated rhythm language, or report structure that can still reveal the weak labels. The `leakage_audit.json` artifact records how much text remains and whether rules still fire after masking.

If masking leaves almost no usable text, training is skipped.

## Five-label mapping (v1.0.0)

Fixed order: `NORM, HYP, STTC, MI, CD` (multi-label).

| Label | Assigned when (examples) | Not assigned when |
| ----- | ------------------------ | ----------------- |
| NORM | Explicit `Normal ECG` / `within normal limits` / `Normal ECG except for rate`, **and** no HYP/STTC/MI/CD | Isolated `Sinus rhythm`; `Borderline ECG`; `normal variant`; any other target is positive |
| HYP | Ventricular hypertrophy, LVH/RVH, atrial enlargement/abnormality | Bare `enlargement` or `overload` without chamber context |
| STTC | ST/T changes, T-wave abnormalities, ST elevation/depression, `myocardial ischemia` as ST/T language | Isolated PVC/PAC; prolonged QT (this version) |
| MI | `infarct` / `infarction` / STEMI / `myocardial infarction` | Isolated `ischemia` without infarct language |
| CD | AV block, bundle-branch/fascicular block, IVCD, pacing/pacemaker, WPW, axis deviation, prolonged/short PR | Isolated PAC/PVC |

Uncertainty (`possible`, `probable`, `consider`, `cannot exclude`, `cannot rule out`, `may be`, `borderline` on a finding): **positive weak label** plus an `ambiguous` / `uncertain_labels` flag.

Negation (`no evidence of`, `ruled out`, …): the finding is **not** assigned.

Studies with no mapped class are **unmapped** and excluded from the classification cohort. They remain in `artifacts/labels_full.csv`.

Authoritative patterns live in `artifacts/label_mapping.yaml`. Do not change mapping rules after looking at test metrics.

## How to run

From the repository root, with `machine_measurements.csv` present:

```text
python -m model_2_text_only.run_pipeline --config model_2_text_only/config.yaml
```

Synthetic smoke run (does not use the real CSV):

```text
python -m model_2_text_only.run_pipeline --config model_2_text_only/config.yaml --smoke
pytest model_2_text_only/tests
```

## Working cohort

The full table has 800,035 unique studies. Label generation runs on **all** rows. Training uses a **patient-sampled working cohort** (`working_cohort_max_studies`, default 8000) so Models 1, 3, and 4 can share a computationally manageable split. Increase that setting when more compute is available; regenerate the split only if you intentionally replace the shared cohort.

## Architecture

Clinical text → train-only tokenizer → embedding (64) → bidirectional LSTM (hidden 64) → dropout → 5 logits.

Loss: `BCEWithLogitsLoss` (no softmax). Padding is excluded via packed sequences. Per-label positive weights come from **training** counts only. Thresholds are chosen on **validation** F1, then frozen for test.

## Shared artifacts (reuse in Models 1, 3, 4)

| Path | Role |
| ---- | ---- |
| `artifacts/label_mapping.yaml` | Frozen mapping rules |
| `artifacts/labels_full.csv` | Weak labels for every study |
| `artifacts/labels.csv` | Compact label matrix |
| `artifacts/cohort_manifest.csv` | Working-cohort studies, text, labels, evidence, split |
| `artifacts/patient_split_manifest.csv` | `study_id` / `subject_id` / `split` |
| `artifacts/leakage_audit.json` | Leakage quantification |

`outputs/model_2_text_only/` holds checkpoints, vocabulary, thresholds, metrics, and error analysis. Public error reports use `example_id` rather than `subject_id`.

## Identifiers

`subject_id` is used only for grouping and splits. Identifiers and machine measurements are never model features.
