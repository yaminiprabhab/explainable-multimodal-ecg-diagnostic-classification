# Model 2 — Leakage-Controlled Clinical-Text-Only Baseline

## 1. Purpose and scope

Model 2 is the project's **clinical-text-only baseline**. It predicts five broad ECG diagnostic categories from the machine-generated report text attached to each ECG study. It does **not** use ECG waveforms, numeric machine measurements, `subject_id`, `study_id`, or any other non-text feature.

The model is deliberately a *residual-text* experiment. The target labels are themselves generated from the report text, so directly training on the original report would let the model read the phrases that created its targets. Before training, all label-triggering statements and report-summary phrases are removed. The model therefore learns from the leftover text (for example, rhythm and rate statements), not from the direct diagnostic wording.

This is a research baseline, not a clinical diagnostic system. Its labels are weak, rule-derived labels from machine reports, not cardiologist-adjudicated ground truth. Even after masking, residual correlated language can leak information about labels; results must not be presented as leakage-free clinical diagnostic performance.

## 2. Inputs and task definition

### Source data

The completed run used `machine_measurements.csv` from the MIMIC-IV-ECG matched subset. The file contained **800,035 rows**, each a unique ECG study, for **161,352 patients**. It had 33 columns:

- identifiers: `subject_id`, `study_id`, `cart_id`, and `ecg_time`;
- 18 report-text fields: `report_0` through `report_17`;
- ECG machine-measurement columns such as RR interval, onset/end measurements, and axes.

Only the 18 `report_*` fields are used to create labels and model text. The report fields are concatenated in numeric order using ` | ` as a separator. The identifiers are used solely for record validation and patient-level splitting.

### Input data checks

The pipeline validates the CSV before any modelling. In this run:

| Check | Observed value |
| --- | ---: |
| Rows / unique studies | 800,035 / 800,035 |
| Unique patients | 161,352 |
| Missing `subject_id` | 0 |
| Missing `study_id` | 0 |
| Exact duplicate rows | 0 |
| Duplicate study-ID rows | 0 |
| Rows with no usable report text | 1 |

Consequently, no rows were removed for missing identifiers or duplicate studies.

### Multi-label targets

One study can have zero or more of the following labels. The fixed output order is always `NORM, HYP, STTC, MI, CD`.

| Code | Meaning | Examples of report concepts mapped to the label |
| --- | --- | --- |
| NORM | Normal ECG | `Normal ECG`, `Normal EKG`, `within normal limits` |
| HYP | Hypertrophy / chamber enlargement | LVH, RVH, ventricular hypertrophy, atrial enlargement/abnormality |
| STTC | ST/T change | ST-T change, T-wave abnormality, ST elevation/depression, myocardial ischemia |
| MI | Myocardial infarction | infarct/infarction, STEMI, acute MI, Q waves suggesting infarct |
| CD | Conduction disorder | bundle/fascicular/AV block, IVCD, pacing, WPW, axis deviation, prolonged/short PR |

The task uses five independent binary outputs with sigmoid activations and binary cross-entropy loss; it is not a mutually exclusive softmax classification task.

## 3. Weak-label generation

Labels were generated for every input row with the frozen mapping in `artifacts/label_mapping.yaml` (version **1.0.0**, status `weak_labels_from_machine_reports`). For each statement in `report_0`–`report_17`, the mapper stores the label evidence and the statements that fired a rule.

Important mapping decisions are:

- **NORM is explicit and exclusive.** A normal label requires an explicit normal summary; it is set to 0 if HYP, STTC, MI, or CD is positive. “Sinus rhythm” alone does not mean NORM.
- **MI requires infarct language.** Isolated ischemia is STTC, not MI.
- **STTC excludes isolated PAC/PVC and prolonged QT** in this mapping version.
- **CD excludes isolated PAC/PVC.** Pacing is considered CD, including a pacing statement within “no further analysis”.
- Generic “enlargement” or “overload” without a chamber/hypertrophy context does not create HYP.
- Negated findings (for example, “no evidence of infarct” or “ruled out”) do not create labels.
- Uncertain wording such as “possible,” “probable,” “consider,” “cannot exclude,” or “borderline” is retained as a **positive weak label** and is flagged in `ambiguous` / `uncertain_labels`.

The full regular expressions, negation patterns, uncertainty patterns, and residual-leakage clue list are documented in the mapping file rather than duplicated here. The mapping was frozen before evaluating test performance; it must not be tuned after observing test metrics.

### Full-data label EDA

The mapper ran on all **800,035** studies. The table below describes label prevalence over the full input table, before cohort sampling and text filtering.

| Label | Positive studies | Negative studies | Prevalence |
| --- | ---: | ---: | ---: |
| NORM | 156,572 | 643,463 | 19.57% |
| HYP | 112,208 | 687,827 | 14.03% |
| STTC | 322,930 | 477,105 | 40.36% |
| MI | 179,638 | 620,397 | 22.45% |
| CD | 290,081 | 509,954 | 36.26% |

Studies with no mapped target class are retained in `artifacts/labels_full.csv` for traceability but excluded from the classification cohort. This removed **94,974** studies and left **705,061 label-eligible studies**.

## 4. Cohort construction and leakage control

Training on the full eligible dataset was not necessary for this baseline. The pipeline selected a manageable patient-sampled working cohort, with a configured maximum of 8,000 studies. Because entire patients are selected, the final sample can slightly exceed the nominal cap: this run sampled **8,002 studies**.

### Why masking is necessary

The labels and raw model text have the same origin. For example, if “left ventricular hypertrophy” creates HYP and the model sees that same phrase, high performance would largely measure rule recovery rather than prediction. The pipeline uses `mask_trigger_statements`:

1. Build raw report text from all non-empty report fields.
2. Identify statements that fired any label rule.
3. Remove those statements from the model input.
4. Also remove summary phrases: `abnormal ecg`, `borderline ecg`, `normal ecg`, `within normal limits`, and `summary:`.
5. Re-run the label mapper on each surviving statement. If it still matches any target rule, remove it too.
6. Drop studies with no text remaining after masking.

The code calls the surviving input `clinical_text`; the original unmasked text remains in the cohort manifest only for audit purposes. The mask token is reserved by the tokenizer, but this implementation drops triggering statements rather than inserting `[MASKED]` into the final text.

### Cohort flow

| Pipeline stage | Studies remaining | Change |
| --- | ---: | ---: |
| Loaded / validated source table | 800,035 | — |
| After missing-ID removal | 800,035 | 0 removed |
| After duplicate-study resolution | 800,035 | 0 removed |
| After requiring at least one mapped label | 705,061 | 94,974 removed |
| Patient-sampled working cohort | 8,002 | computational cohort |
| After leakage text filter | 7,378 | 624 removed for empty residual text |

### Leakage-audit results

The leakage audit is encouraging for **direct rule exposure**, but it is not proof that all leakage has been removed.

| Audit quantity | Value |
| --- | ---: |
| Sampled studies audited | 8,002 |
| Studies with remaining text | 7,378 (92.20%) |
| Empty after masking | 624 |
| Mean / median remaining tokens | 4.04 / 2 |
| Studies where a primary label rule still fires | 0 |
| Fraction directly reconstructable by primary rules | 0.00% |

Residual clue checks found correlated, non-primary cues: HYP 2 hits (2 among HYP positives), STTC 53 (14), MI 272 (28), CD 57 (23), and NORM 0 (0). Examples of text left for training include “Sinus tachycardia | Low QRS voltages in limb leads”, “Sinus rhythm with PACs | Low QRS voltages in limb leads”, and “Atrial fibrillation with uncontrolled ventricular response | Low QRS voltages in limb leads”.

Thus Model 2 should be interpreted precisely as **residual-text prediction of weak labels**. Synonyms, rhythm context, co-occurring conditions, report templates, and label correlations may still make labels predictable.

## 5. Patient-level split

The 7,378 residual-text studies were split by `subject_id`, so no patient appears in more than one partition. The requested study design was 70% train, 15% validation, and 15% test; splitting used multi-label patient-combination stratification, with rare label combinations folded into a `rare` stratum. The actual patient ratios were 69.92% / 15.04% / 15.04%.

| Split | Studies | Patients | NORM | HYP | STTC | MI | CD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 5,219 | 967 | 1,391 (26.65%) | 1,009 (19.33%) | 2,035 (38.99%) | 1,354 (25.94%) | 1,474 (28.24%) |
| Validation | 1,057 | 208 | 293 (27.72%) | 211 (19.96%) | 375 (35.48%) | 239 (22.61%) | 303 (28.67%) |
| Test | 1,102 | 208 | 270 (24.50%) | 201 (18.24%) | 445 (40.38%) | 338 (30.67%) | 283 (25.68%) |

The split manifest is saved as `artifacts/patient_split_manifest.csv`, and the final text/label/split cohort is `artifacts/cohort_manifest.csv`.

## 6. Text preparation and tokenization

Tokenization is intentionally simple and reproducible:

- lower-case tokens matching alphabetic words, numeric values, or `[MASKED]`;
- vocabulary fitted on **training text only**, preventing validation/test vocabulary leakage;
- `<pad>` ID 0, `<unk>` ID 1, `[masked]` ID 2;
- minimum training token frequency: 2;
- sequences padded to a fixed length, while real lengths are retained for packed LSTM processing.

Observed training-text EDA:

| Quantity | Value |
| --- | ---: |
| Training documents | 5,219 |
| Vocabulary size | 202 |
| Mean / median length | 4.16 / 2 tokens |
| 95th percentile / maximum length | 12 / 38 tokens |
| Chosen maximum sequence length | 12 tokens |
| Truncation rate: train / validation / test | 4.33% / 4.26% / 4.45% |

The chosen length is calculated from the 95th percentile of training lengths (subject to a minimum of 8 and cap of 64). Because the residual text is short, the effective input for many examples is only a few rhythm or technical terms.

## 7. Model architecture

The trained encoder is a bidirectional LSTM. Its forward path is:

```text
leakage-controlled text
  → token IDs (length 12, padded)
  → trainable 64-dimensional embedding
  → 1-layer bidirectional LSTM (64 hidden units per direction)
  → concatenate final forward and backward hidden states (128 features)
  → dropout (0.30)
  → linear layer (128 → 5 logits)
  → sigmoid probability for each label
```

Padding is excluded from recurrence through `pack_padded_sequence`. The model produces five independent logits; no softmax is used.

## 8. Training configuration

| Setting | Value |
| --- | --- |
| Random seed | 42 |
| Device | CPU |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss | `BCEWithLogitsLoss` |
| Class imbalance handling | Training-only `pos_weight = negatives / positives` |
| Gradient clipping | L2 norm capped at 5.0 |
| Maximum epochs | 8 |
| Early-stopping patience | 3 validation epochs |

Approximate positive weights derived only from the training split were NORM 2.752, HYP 4.172, STTC 1.565, MI 2.855, and CD 2.541. This increases the loss contribution from the less common labels, especially HYP.

The best validation-loss checkpoint occurred at epoch 6 and was restored before threshold selection and test evaluation.

| Epoch | Training loss | Validation loss |
| ---: | ---: | ---: |
| 1 | 0.9534 | 0.9285 |
| 2 | 0.9223 | 0.9184 |
| 3 | 0.9113 | 0.9153 |
| 4 | 0.9059 | 0.9121 |
| 5 | 0.9005 | 0.9135 |
| 6 | 0.8989 | **0.9078** |
| 7 | 0.8970 | 0.9098 |
| 8 | 0.8943 | 0.9129 |

All 8 configured epochs ran; early stopping did not end the run before the epoch limit.

## 9. Threshold selection

Probabilities were converted to binary outputs using **validation-only**, per-label thresholds. For each label, the pipeline searched 0.05 to 0.95 in 0.05 increments and chose the threshold with the best validation F1. These thresholds were then frozen and applied once to the held-out test set.

| Label | Validation positives | Selected threshold | Validation F1 at selected threshold |
| --- | ---: | ---: | ---: |
| NORM | 293 | 0.65 | 0.5767 |
| HYP | 211 | 0.35 | 0.3394 |
| STTC | 375 | 0.45 | 0.5326 |
| MI | 239 | 0.45 | 0.4033 |
| CD | 303 | 0.40 | 0.4557 |

No label required the 0.50 fallback threshold because every validation label had at least the required five positive cases.

## 10. Held-out test results

The test set has **1,102 studies from 208 unseen patients**. Reported metrics use the restored best-validation-loss checkpoint and the frozen validation thresholds above.

### Overall metrics

| Metric | Value |
| --- | ---: |
| Macro F1 | **0.4433** |
| Micro F1 | **0.4335** |
| Macro precision | 0.3765 |
| Macro recall | 0.6802 |
| Micro precision | 0.3299 |
| Micro recall | 0.6318 |
| Macro AUROC | 0.6326 |
| Macro AUPRC | 0.3981 |

Precision is notably lower than recall. This is partly expected because thresholds were selected to maximize F1 under class-weighted training, but it also means the model makes many positive calls that are not supported by the weak labels.

### Label-wise metrics and confusion counts

| Label | Support | Threshold | Precision | Recall | F1 | AUROC | AUPRC | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NORM | 270 | 0.65 | 0.3525 | 0.9074 | 0.5078 | 0.6848 | 0.3430 | 245 | 450 | 382 | 25 |
| HYP | 201 | 0.35 | 0.1934 | 0.9900 | 0.3236 | 0.6207 | 0.2779 | 199 | 830 | 71 | 2 |
| STTC | 445 | 0.45 | 0.5430 | 0.4539 | 0.4945 | 0.5863 | 0.4966 | 202 | 170 | 487 | 243 |
| MI | 338 | 0.45 | 0.4804 | 0.5089 | 0.4943 | 0.6687 | 0.4883 | 172 | 186 | 578 | 166 |
| CD | 283 | 0.40 | 0.3129 | 0.5406 | 0.3964 | 0.6027 | 0.3845 | 153 | 336 | 483 | 130 |

### Interpretation of the observed results

- NORM has the highest F1 (0.5078) and very high recall (90.74%), but 450 false positives show that residual rhythm language is not sufficient to reliably distinguish all normal studies.
- HYP has almost perfect recall (199/201) but the lowest precision (19.34%) and **830 false positives**. Its low 0.35 threshold and broad residual correlations make its positive decisions especially overinclusive.
- STTC has the highest precision (54.30%) but misses 243 of 445 positive cases, producing lower recall (45.39%).
- MI has the highest AUROC (0.6687), yet its thresholded F1 remains modest (0.4943), showing that ranking ability does not automatically imply accurate binary decisions.
- CD has modest discrimination and substantial false positives (336).

These findings describe the model's agreement with weak report-derived labels after direct triggering text has been removed. They do not establish clinical diagnostic accuracy.

## 11. Error analysis and explainability

The pipeline saves an error report with 12 representative test examples. It stores a split-local `example_id` rather than patient identifiers. For each label, the aggregate error counts are:

| Label | False positives | False negatives |
| --- | ---: | ---: |
| NORM | 450 | 25 |
| HYP | 830 | 2 |
| STTC | 170 | 243 |
| MI | 186 | 166 |
| CD | 336 | 130 |

Text explanations use **token occlusion**. For each selected text, one token at a time is replaced with `<unk>` and the change in predicted probability is recorded. A positive delta means removing that token lowered the original probability, so the token locally supported the prediction; a negative delta means it locally opposed it.

For example, for residual text `Sinus rhythm`, the model predicted NORM 0.6502 and HYP 0.4796. Replacing `sinus` lowered the NORM probability by 0.1263, while replacing `rhythm` lowered it by 0.0502. The same input produced a HYP false positive because its HYP probability exceeded the 0.35 threshold. This illustrates both the model's use of residual rhythm language and the risk of a low threshold.

Occlusion is a local sensitivity analysis, not causal proof or clinical reasoning. Correlated terms can share or hide importance. No attention weights are used in Model 2.

## 12. Reproducing the run

From the repository root, install the dependencies in `requirements.txt`, place `machine_measurements.csv` in the root, then run:

```powershell
python -m model_2_text_only.run_pipeline --config model_2_text_only/config.yaml
```

For a synthetic, non-clinical smoke test:

```powershell
python -m model_2_text_only.run_pipeline --config model_2_text_only/config.yaml --smoke
pytest model_2_text_only/tests
```

The recorded environment for the completed real-data run was Python 3.11.9, pandas 2.3.1, NumPy 2.2.6, PyYAML 6.0.2, PyTorch 2.8.0+cpu, and scikit-learn 1.7.1, with random seed 42.

## 13. Saved outputs

| Path | Contents |
| --- | --- |
| `artifacts/label_mapping.yaml` | Frozen weak-label rules and leakage-clue definitions |
| `artifacts/labels_full.csv` | All studies, weak labels, evidence, and label flags |
| `artifacts/labels.csv` | Compact label matrix and identifiers |
| `artifacts/cohort_manifest.csv` | Final 7,378-study cohort: split, raw text, residual text, labels, and evidence |
| `artifacts/patient_split_manifest.csv` | Study-to-patient split assignment |
| `artifacts/leakage_audit.json` | Direct-rule and residual-clue leakage audit |
| `outputs/model_2_text_only/best_model.pt` | Best validation-loss LSTM checkpoint (epoch 6) |
| `outputs/model_2_text_only/vocabulary.json` | Train-only tokenizer vocabulary and length settings |
| `outputs/model_2_text_only/training_history.json` | Per-epoch train/validation loss |
| `outputs/model_2_text_only/thresholds.json` | Validation-selected thresholds |
| `outputs/model_2_text_only/test_metrics.json` | Full aggregate and label-wise test metrics |
| `outputs/model_2_text_only/test_metrics.csv` | Aggregate test metrics in tabular form |
| `outputs/model_2_text_only/test_per_label_metrics.csv` | Label-wise metrics and confusion counts |
| `outputs/model_2_text_only/error_analysis.json` | Representative errors/correct examples with token occlusion |
| `outputs/model_2_text_only/exclusion_log.json` | Validation and cohort-flow counts |
| `outputs/model_2_text_only/run_summary.json` | Consolidated run summary |

## 14. Limitations and next use

1. Targets are weak labels created from the same kind of text used as model input. Direct trigger statements are removed, but masking cannot eliminate synonym, context, template, or label-correlation leakage.
2. The residual text is extremely short (median two tokens). This restricts semantic information and makes model behaviour sensitive to recurring phrases such as rhythm descriptions.
3. The working cohort is a 8,002-study patient-level sample, not all 705,061 label-eligible studies. Results can change when the cohort size, seed, or mapping version changes.
4. The model is a baseline for controlled comparison with Models 1, 3, and 4. It should reuse the shared label mapping and patient split when compared with those models.
5. Thresholds optimize validation F1, which can yield high recall and poor precision. Threshold choice should be revisited only through a new validation protocol—not by adjusting thresholds on the held-out test results.
6. This work is for research only and must not be used for patient care or clinical decision-making.
