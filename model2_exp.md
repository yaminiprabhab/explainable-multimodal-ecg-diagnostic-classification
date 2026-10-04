# Model 2 Experiment: Data Acquisition and Preprocessing

## Purpose

This document describes only the data-acquisition and preprocessing stages of Model 2, the leakage-controlled clinical-text-only baseline. It explains how the raw MIMIC-IV-ECG machine-report table becomes the final text, label, and patient-split inputs used by the LSTM.

Model 2 predicts five weak, multi-label ECG categories: `NORM`, `HYP`, `STTC`, `MI`, and `CD`. It does not use ECG waveforms or numerical machine measurements as model features.

## 1. Dataset acquisition

### Dataset required

The pipeline expects the MIMIC-IV-ECG matched machine-measurement table, saved locally as:

```text
machine_measurements.csv
```

Place this file in the project root:

```text
explainable-multimodal-ecg-diagnostic-classification/
├── machine_measurements.csv
├── model_2_text_only/
└── artifacts/
```

MIMIC-IV-ECG data are controlled-access clinical data. Obtain access through the dataset's official distribution process and follow its data-use agreement. Do not commit the source CSV, report text, patient identifiers, or generated patient-level artifacts to a public repository.

### Expected input fields

The two required fields are:

| Column | Use |
| --- | --- |
| `subject_id` | Groups all studies belonging to the same patient for leakage-safe splitting |
| `study_id` | Unique ECG-study identifier and unit of analysis |

The pipeline discovers all columns beginning with `report_`, sorts them numerically, and uses them as text fields. In the completed run, the input contained `report_0` through `report_17` (18 report fields).

The remaining columns, such as RR interval, onset/end intervals, and ECG axes, are retained in the source table but are **not** passed to Model 2.

### Load the source table

The implementation in `model_2_text_only/data_loader.py` loads the CSV and validates that IDs and report columns exist.

```python
import pandas as pd

REQUIRED_ID_COLUMNS = ["subject_id", "study_id"]

def detect_report_columns(columns: list[str], prefix: str = "report_") -> list[str]:
    report_cols = [c for c in columns if c.startswith(prefix)]

    def sort_key(name: str) -> int:
        suffix = name[len(prefix):]
        return int(suffix) if suffix.isdigit() else 10**9

    return sorted(report_cols, key=sort_key)

def load_measurements(path, report_prefix: str = "report_") -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing_ids = [c for c in REQUIRED_ID_COLUMNS if c not in frame.columns]
    if missing_ids:
        raise ValueError(f"Missing required identifier columns: {missing_ids}")
    report_cols = detect_report_columns(list(frame.columns), prefix=report_prefix)
    if not report_cols:
        raise ValueError(f"No report columns found with prefix {report_prefix!r}")
    frame.attrs["report_columns"] = report_cols
    return frame
```

## 2. Input validation and record cleaning

Before labels or text are generated, the pipeline checks row count, column names/types, missing IDs, report-text availability, duplicate rows, and duplicate study IDs. Rows with missing `subject_id` or `study_id` are dropped. If a study appears more than once, the pipeline inspects conflicts and retains one deterministic row per `study_id`.

```python
def drop_missing_identifiers(frame: pd.DataFrame):
    before = len(frame)
    cleaned = frame.dropna(subset=["subject_id", "study_id"])
    return cleaned.reset_index(drop=True), {
        "rows_before_id_filter": int(before),
        "rows_missing_ids": int(before - len(cleaned)),
        "rows_after_id_filter": int(len(cleaned)),
    }

def resolve_duplicate_studies(frame: pd.DataFrame):
    ordered = frame.sort_values(["study_id", "ecg_time"], kind="mergesort")
    deduped = ordered.drop_duplicates(subset=["study_id"], keep="first")
    return deduped.reset_index(drop=True)
```

### Observed input EDA for this run

| Check | Value |
| --- | ---: |
| Rows | 800,035 |
| Columns | 33 |
| Unique studies | 800,035 |
| Unique patients | 161,352 |
| Missing `subject_id` / `study_id` | 0 / 0 |
| Exact duplicate rows | 0 |
| Duplicate study-ID rows | 0 |
| Rows with no report text | 1 |

No row was removed for missing IDs or duplicate studies in this run.

## 3. Report-text construction

Every non-empty report field is normalized for whitespace and concatenated in report-column order with ` | `.

```python
import re

def combine_statements(statements, separator: str = " | ") -> str:
    cleaned = []
    for item in statements:
        text = re.sub(r"\s+", " ", str(item).strip())
        if text:
            cleaned.append(text)
    return separator.join(cleaned)
```

The initial concatenated text is called `raw_report_text`. It is used for rule-based weak-label generation and audit storage. It is **not** directly used as the final model input because it contains diagnostic statements that create the labels.

## 4. Weak-label generation

### Label source

Labels are created from report text using the frozen rule mapping in:

```text
artifacts/label_mapping.yaml
```

The mapping version is `1.0.0`; its status is `weak_labels_from_machine_reports`, and `clinically_verified` is `false`. This means the targets are machine-report-derived weak labels, not confirmed clinical diagnoses.

### Rules in brief

| Label | Main matching concepts |
| --- | --- |
| `NORM` | Explicit `normal ecg`, `normal ekg`, or `within normal limits`; removed if any other target is positive |
| `HYP` | LVH/RVH, ventricular hypertrophy, atrial enlargement/abnormality, voltage criteria for LVH |
| `STTC` | ST-T changes, T-wave abnormality, ST elevation/depression, myocardial ischemia/injury |
| `MI` | infarct/infarction, STEMI, acute MI, myocardial infarction, Q waves suggesting infarct |
| `CD` | bundle/fascicular/AV block, IVCD, pacing, WPW, axis deviation, abnormal PR interval |

Negated phrases, such as “no evidence of” or “ruled out”, do not create labels. Uncertain phrases, such as “possible”, “probable”, “consider”, “cannot exclude”, and “borderline”, create a positive weak label but also set ambiguity flags.

The core label-generation logic applies patterns statement by statement, records evidence, and suppresses NORM if another target is present:

```python
def label_statements(self, statements: list[str]) -> LabelResult:
    labels = {name: 0 for name in self.label_order}
    evidence = {name: [] for name in self.label_order}
    uncertain_labels = []

    for statement in statements:
        normalized = _normalize(statement)
        if not normalized:
            continue
        negated = bool(self.negation_re and self.negation_re.search(normalized))
        uncertain = bool(self.uncertainty_re and self.uncertainty_re.search(normalized))
        if negated:
            continue
        for name, patterns in self.rules.items():
            if any(pattern.search(normalized) for pattern in patterns):
                labels[name] = 1
                evidence[name].append(statement)
                if uncertain:
                    uncertain_labels.append(name)

    if any(labels[name] for name in ("HYP", "STTC", "MI", "CD")):
        labels["NORM"] = 0
        evidence["NORM"] = []

    return LabelResult(labels=labels, evidence=evidence,
                       uncertain_labels=sorted(set(uncertain_labels)),
                       ambiguous=bool(uncertain_labels), ...)
```

The pipeline saves all generated labels, evidence, ambiguity fields, and source text to `artifacts/labels_full.csv`.

### Full-table label prevalence

Weak labels were generated for all 800,035 studies before sampling or masking.

| Label | Positive studies | Prevalence |
| --- | ---: | ---: |
| NORM | 156,572 | 19.57% |
| HYP | 112,208 | 14.03% |
| STTC | 322,930 | 40.36% |
| MI | 179,638 | 22.45% |
| CD | 290,081 | 36.26% |

Studies without any of the five mapped labels are marked `unmapped` and excluded from the classification cohort. This removed 94,974 rows, leaving **705,061 label-eligible studies**.

## 5. Patient-sampled working cohort

The complete eligible cohort is large, so Model 2 uses a reproducible patient-level working cohort. The configuration sets `working_cohort_max_studies: 8000` and `random_seed: 42`.

Sampling occurs by patient—not individual study—to preserve all of a selected patient's studies. Patient label combinations are used as strata; selected patients are drawn round-robin across strata. Because all studies from a patient are retained, the final study count can slightly exceed the target.

```python
def sample_working_cohort(frame, max_studies: int, seed: int):
    patient_y = frame.groupby("subject_id")[["NORM", "HYP", "STTC", "MI", "CD"]].max()
    keys = patient_y.astype(int).astype(str).agg("".join, axis=1)
    rng = np.random.RandomState(seed)
    # Shuffle patients within label-combination strata, then add whole patients.
    # Stop once the study target is reached or exceeded.
    ...
    return frame[frame["subject_id"].isin(set(chosen))].copy()
```

The completed run selected **8,002 studies**.

## 6. Leakage-controlled text preprocessing

### Problem being controlled

The same report fields create both the weak target and the candidate text feature. If text such as “inferior infarct” remains in the input after it produces an MI label, a text model can simply reproduce the rule. This is target leakage.

### Preprocessing procedure

The configured leakage mode is `mask_trigger_statements`. In practice, it removes—not merely hides—the following statements from final `clinical_text`:

1. Statements that fired a label rule.
2. Statements containing summary phrases: `abnormal ecg`, `borderline ecg`, `normal ecg`, `within normal limits`, or `summary:`.
3. Any remaining statement that still triggers a label rule when the label mapper is re-applied.

```python
def mask_statements(statements, trigger_statements, mapper,
                    also_mask_summary=True, summary_phrases=None):
    trigger_set = {re.sub(r"\s+", " ", t.strip()) for t in trigger_statements}
    kept, masked = [], []

    for stmt in statements:
        normalized = re.sub(r"\s+", " ", stmt.strip())
        drop = normalized in trigger_set
        if also_mask_summary and _contains_phrase(normalized, summary_phrases or []):
            drop = True
        if not drop:
            relabeled = mapper.label_statements([normalized])
            if any(relabeled.labels[name] == 1 for name in mapper.label_order):
                drop = True
        if drop:
            masked.append(normalized)
        else:
            kept.append(normalized)
    return kept, masked

def build_clinical_text(labels_frame, mapper, separator=" | ", ...):
    out = labels_frame.copy()
    for _, row in out.iterrows():
        statements = [s.strip() for s in row["raw_report_text"].split(" | ") if s.strip()]
        triggers = [s.strip() for s in row["trigger_statements"].split(" || ") if s.strip()]
        kept, masked = mask_statements(statements, triggers, mapper, ...)
        # `clinical_text` is the only text supplied to the LSTM.
        ...
    return out
```

If no residual text remains, the study is excluded from Model 2 training, validation, and testing.

### Leakage-audit code

After masking, `audit_leakage()` checks whether the primary mapping rules can still recover labels from the residual text. It also counts configured residual clues, such as `q wave`, `poor r wave progression`, `voltage criteria`, `ischemia`, `paced`, and `aberrant`.

```python
audit = audit_leakage(cohort, mapper)
save_json(audit, artifacts_dir / "leakage_audit.json")

if cfg["leakage"].get("drop_study_if_no_remaining_text", True):
    text_cohort = cohort[cohort["has_usable_text"]].copy()
```

### Observed leakage-preprocessing results

| Quantity | Value |
| --- | ---: |
| Patient-sampled cohort before masking | 8,002 studies |
| No residual text after masking | 624 studies |
| Final residual-text cohort | **7,378 studies** |
| Studies retaining residual text | 92.20% |
| Mean / median residual tokens | 4.04 / 2 |
| Studies where a primary rule still fires | 0 |
| Fraction directly reconstructable by primary rules | 0.00% |

Examples of final model text include `Sinus rhythm`, `Sinus bradycardia`, and `Sinus tachycardia | Low QRS voltages in limb leads`.

Direct label-rule exposure was removed, but residual leakage remains possible. The audit found clue hits for HYP (2), STTC (53), MI (272), and CD (57), while NORM had none. Rhythms, report templates, synonyms, and co-occurring findings may still correlate with weak labels.

## 7. Patient-level train/validation/test split

Only the 7,378 studies with residual text are split. A patient must be entirely in one partition. The pipeline aggregates each patient's labels with `max`, stratifies on the five-label combination where possible, and treats combinations occurring fewer than three times as `rare`.

```python
def patient_label_matrix(frame):
    return frame.groupby("subject_id")[["NORM", "HYP", "STTC", "MI", "CD"]].max()

patient_y = patient_label_matrix(frame)
keys = np.array(["".join(str(int(v)) for v in row)
                 for row in patient_y.to_numpy()])
rare = set(pd.Series(keys).value_counts().loc[lambda x: x < 3].index)
strat = np.array(["rare" if key in rare else key for key in keys])

rest_idx, test_idx = train_test_split(..., test_size=0.15,
                                      random_state=42, stratify=strat)
train_idx, val_idx = train_test_split(..., test_size=0.15 / 0.85,
                                      random_state=42, stratify=rest_strat)
```

The code verifies that no `subject_id` occurs in more than one split. The final split was:

| Split | Studies | Patients |
| --- | ---: | ---: |
| Train | 5,219 | 967 |
| Validation | 1,057 | 208 |
| Test | 1,102 | 208 |

The requested patient split was 70% / 15% / 15%; the actual patient ratios were 69.92% / 15.04% / 15.04%.

## 8. Tokenization and final model-ready data

Vocabulary fitting uses **only training text**, preventing test/validation token leakage. Tokens are lowercased alphabetic strings, numeric values, or `[MASKED]`; terms appearing fewer than twice in training become `<unk>`. Padding uses ID 0 and unknown uses ID 1.

```python
TOKEN_RE = re.compile(r"\[MASKED\]|[A-Za-z]+|\d+(?:\.\d+)?")

def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))] if text else []

vocab.fit(
    train_df["clinical_text"].tolist(),
    max_seq_len=None,
    percentile=95,
    cap=64,
)
```

For this run, the training-only vocabulary had 202 tokens. The median residual-text length was two tokens; the 95th percentile was 12, so the chosen sequence length was 12. The train/validation/test truncation rates were 4.33%, 4.26%, and 4.45%, respectively.

The resulting tensors are:

```text
input_ids : integer token IDs, shape [number_of_studies, 12]
lengths   : actual non-padding lengths, shape [number_of_studies]
labels    : five binary weak targets, shape [number_of_studies, 5]
```

## 9. Run the complete acquisition-to-preprocessing pipeline

From the project root:

```powershell
python -m model_2_text_only.run_pipeline --config model_2_text_only/config.yaml
```

The configuration controlling this experiment is `model_2_text_only/config.yaml`. Key values are:

```yaml
random_seed: 42
data:
  report_prefix: report_
  text_separator: " | "
  working_cohort_max_studies: 8000
  require_at_least_one_label: true
leakage:
  mode: mask_trigger_statements
  also_mask_summary_phrases: true
  drop_study_if_no_remaining_text: true
splitting:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
tokenizer:
  min_token_freq: 2
  max_seq_len_percentile: 95
  max_seq_len_cap: 64
```

## 10. Preprocessing outputs

| Output | Description |
| --- | --- |
| `artifacts/labels_full.csv` | All source studies, weak labels, flags, evidence, and raw report text |
| `artifacts/labels.csv` | Compact labels and identifiers |
| `artifacts/cohort_manifest.csv` | Final residual-text cohort with labels and split assignment |
| `artifacts/patient_split_manifest.csv` | Patient-safe study split assignment |
| `artifacts/leakage_audit.json` | Rule-recovery and residual-clue audit |
| `outputs/model_2_text_only/exclusion_log.json` | Data validation and exclusion counts |
| `outputs/model_2_text_only/label_prevalence_full.csv` | Full-table weak-label prevalence |
| `outputs/model_2_text_only/split_stats.json` | Study/patient counts and label prevalence by split |
| `outputs/model_2_text_only/vocabulary.json` | Training-only vocabulary and tokenization settings |
| `outputs/model_2_text_only/tokenizer_stats.json` | Sequence-length and truncation statistics |

For Model 2 architecture, training, validation thresholds, evaluation, and error analysis, see [model2.md](model2.md).
