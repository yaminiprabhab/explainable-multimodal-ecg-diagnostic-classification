# Model 2B: Machine-Measurement-Only ECG Classification

Model 2B is a **tabular baseline**. It predicts the same five weak diagnostic labels as Model 2 using **numerical ECG summary measurements only**. It does **not** load raw ECG waveforms, reconstruct signals, or use `report_*` text as predictive input.

This model is **not** a replacement for the planned ECG-only 1D CNN (Model 1). It does **not** test multimodal fusion. Those questions require Models 1, 3, and 4.

## Research motivation

Model 2 learns from leakage-controlled clinical report text. Model 2B asks a complementary question:

> Can numerical ECG summary measurements (timing and electrical axes) predict the five diagnostic categories associated with an ECG study, and how does that performance compare with the clinical-text-only baseline?

The comparison is informative only if both models use the **same label definitions** and the **same patient-level split**. Target labels may still be **report-derived weak labels**, not independently verified diagnoses. If the machine measurements and the machine report were produced by the same ECG cart software, Model 2B may partly recover the device interpretation from the device measurements. Metrics must be read in that light.

## Five labels (same order as Model 2)

`NORM`, `HYP`, `STTC`, `MI`, `CD` — multi-label, five independent probabilities, **no softmax**.

Mapping version: `artifacts/label_mapping.yaml` version **1.0.0**, status `weak_labels_from_machine_reports`, `clinically_verified: false`.

## Input features

Approved primary measurements (milliseconds for timing/fiducials; degrees for axes):

| Feature | Interpretation used here |
| --- | --- |
| `rr_interval` | R–R cycle length (ms) |
| `p_onset`, `p_end` | P-wave time positions in the ECG window (ms), not precomputed durations |
| `qrs_onset`, `qrs_end` | QRS time positions (ms) |
| `t_end` | T-wave end time position (ms) |
| `p_axis`, `qrs_axis`, `t_axis` | Frontal-plane electrical axes (degrees) |

Derived intervals, computed only when **both** source values are valid and the result is strictly positive:

* `pr_interval` = `p_end - p_onset`
* `qrs_duration` = `qrs_end - qrs_onset`
* `qt_interval` = `t_end - qrs_onset` (QT proxy)

After **training-only** median imputation of the axes, circular features `*_sin` and `*_cos` are added so angles wrap correctly at ±180°.

Missingness indicators are added for features whose **training** missing rate is at least `missing_indicator_min_rate` (default 0.05).

### Explicitly excluded

* `subject_id`, `study_id`, `cart_id`, `ecg_time`
* all `report_*` columns and any label-generation evidence
* `bandwidth`, `filtering` (acquisition/processing metadata, reserved for a future sensitivity analysis)
* target columns and split metadata

The pipeline asserts that the feature matrix contains only the allowlist.

## Invalid-value handling

Timing/fiducial sentinels treated as missing: `29999`, `32767`, `65534`, `65535`.

Axis sentinels: those values plus `-32767`, `-32768`.

Additional documented bounds (not performance-tuned cutoffs):

* fiducials: 0–10000 ms (10-second recording window)
* `rr_interval`: > 0 and ≤ 10000 ms
* axes: −180 to 180 degrees

`29999` is a common MUSE/unavailable ECG measurement code; `32767` / `65535` match 16-bit sentinel encodings. These conventions are documented and unit-tested. They are **not** a claim that PhysioNet published an exhaustive sentinel table for this file.

Raw CSV values are never overwritten. Cleaned values exist only in derived tables and fitted transformers.

## Dataset and cohort

Expected input (path from `config.yaml`, not a hard-coded machine path):

```text
machine_measurements.csv
```

Unit of analysis: one `study_id` after resolving duplicates. Patient grouping uses `subject_id`.

Duplicate policy:

* exact duplicate rows are collapsed
* **conflicting** duplicate `study_id` rows are **excluded** (Model 2 keeps the first sorted row after logging conflicts). The current MIMIC-IV-ECG table has **no duplicate study IDs**, so the practical cohorts match.

Eligibility:

1. Drop missing identifiers.
2. Reuse Model 2 labels (`artifacts/labels.csv` / `labels_full.csv`) generated with mapping v1.0.0.
3. Exclude unmapped studies (no positive target class), matching Model 2.
4. Inner-join the shared `artifacts/patient_split_manifest.csv` when present (Model 2 working cohort after leakage text filtering: 7378 studies in the completed Model 2 run).
5. Keep studies with at least one valid primary measurement (`min_valid_primary_features: 1`). Partial missingness is allowed; values are imputed from **training** medians.

Do not compare Model 2B test metrics with Model 2 test metrics if the study sets differ. The comparison report includes a **common-cohort** Model 2 evaluation when Model 2 artifacts are available.

## Leakage prevention

Report fields may have been used to **generate labels**. They are never used as Model 2B features, for feature selection, or for hyperparameter tuning.

## Patient-level splits

If `artifacts/patient_split_manifest.csv` exists, it is reused exactly (seed 42, 70/15/15 patient split from Model 2). All studies from one `subject_id` stay in one split.

If the manifest is missing, Model 2B samples a working cohort with the same helper and seed as Model 2 (`working_cohort_max_studies: 8000`) and writes a new manifest. It will **not** overwrite an existing Model 2 manifest.

Imputation, scaling (logistic regression only), hyperparameters, and thresholds are fit or chosen on **training / validation only**.

## Classifiers

One binary classifier per label:

* **Random Forest** (`class_weight='balanced'`), small validation grid over trees/depth/leaf size
* **XGBoost** (`scale_pos_weight` from training counts) if installed
* **Logistic regression** reference with training-fitted `StandardScaler` and `class_weight='balanced'`

Selection metric: validation **macro AUROC** over labels that have both classes.

Thresholds: per-label F1 grid on validation probabilities; fallback 0.5 if a label has fewer than 5 validation positives.

Labels with only one class in training use a constant fallback predictor and are reported as having limited evaluability.

## Explainability

Per-label **permutation importance** (AUROC drop) on the **validation** set. Impurity/gain importance is saved when the estimator provides it, with a bias warning. Importance is not a causal claim and is not used to change the frozen test evaluation.

## How to run

From the repository root, with `machine_measurements.csv` present:

```text
python -m model_2b_measurement_only.run_pipeline --config model_2b_measurement_only/config.yaml
```

Synthetic smoke run (does not use the real CSV; results are not clinical findings):

```text
python -m model_2b_measurement_only.run_pipeline --config model_2b_measurement_only/config.yaml --smoke
pytest model_2b_measurement_only/tests
```

## Output artifacts

Directory: `outputs/model_2b_measurement_only/`

| File | Contents |
| --- | --- |
| `data_quality_report.json` | Source validation, missingness, quantiles, and sentinel-value spikes |
| `split_stats.json` | Study/patient counts and label prevalence by split |
| `inclusion_exclusion_report.json` | Cohort counts by stage |
| `feature_validity_audit.json` | Sentinel and range audit |
| `feature_engineering_config.yaml` / `feature_list.yaml` | Feature definitions |
| `label_source.json` / `labels_used.csv` | Label mapping version and the binary matrix used |
| `imputer.joblib` / `imputer_and_scaler.joblib` | Training-fitted preprocessors |
| `*_bundle.joblib` | Classifier, preprocessor, frozen thresholds |
| `*_thresholds.json` | Validation-selected thresholds |
| `*_test_metrics.json` / `.csv` | Test metrics |
| `*_confusion_matrices.csv` | Per-label TP/FP/TN/FN |
| `*_permutation_importance.csv` / `.png` | Validation importance |
| `comparison_with_model_2.json` / `.md` | Text vs measurements |
| `reproducibility.json` | Seed and package versions |

Public plots do not include patient identifiers.

## Known limitations

1. Targets are weak labels from machine reports, not adjudicated diagnoses.
2. Machine measurements and machine reports may share a device-algorithm origin.
3. Comparison with Model 2 is on Model 2’s computationally reduced working cohort (and any further measurement-eligibility subset), not the full 800,035-study table.
4. Model 2 still uses residual report text after masking; residual text leakage can inflate Model 2 scores relative to a purely numerical model.
5. Sentinel semantics are documented from encoding conventions and empirical spikes, not from an official per-field codebook in this repository.
6. This experiment does not evaluate raw waveforms or fusion models.
7. Shared Model 2 artifacts (`artifacts/labels.csv`, `artifacts/patient_split_manifest.csv`, `outputs/model_2_text_only/`) are not committed. If they are present locally, Model 2B reuses them. If labels are missing, the pipeline regenerates them with Model 2's `LabelMapper` (mapping v1.0.0) and writes a compact `artifacts/labels.csv` for reuse. If the split manifest is missing, Model 2B creates a patient-level 70/15/15 split on its measurement-eligible working cohort and writes `artifacts/patient_split_manifest.csv` without overwriting an existing Model 2 file. In that fallback, comparison with a previously published Model 2 test table is not a same-cohort modality comparison.
