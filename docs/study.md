# Study Guide — Explainable Multimodal ECG Diagnostic Classification

> **Living document.** Agents must update this file after **any** code, config, artifact, or docs change in the repo. Add a short entry under [Changelog](#changelog) and revise the sections that the change affects.
>
> Goal of this file: help humans understand the project, answer common questions, and stay aligned with what the code actually does.

---

## One-sentence summary

We ask whether **12-lead ECG waveforms + clinical report text** (fused simply or with attention) beat **waveform-only** or **text-only** models at predicting five weak diagnostic labels from MIMIC-IV-ECG — with leakage controls, patient-level splits, metrics, error analysis, and explainability.

---

## The research question (plain English)

An ECG is a short recording of the heart’s electrical activity (here: **12 leads**, ~**10 seconds**, **500 Hz** → shape `(12, 5000)`). Hospitals also store a **machine-written report** for that ECG (rhythm notes, morphology phrases, etc.).

This project checks three layered questions:

1. How well can we predict five broad diagnostic buckets from **waveform alone** vs **report text alone**?
2. Does **combining** waveform + text help?
3. Is a fancier **attention/gated fusion** better than just **concatenating** the two embeddings?

We also care about **why** a model fired (attribution on ECG regions / text tokens) and about **honest evaluation** (no patient leakage across splits; don’t let the text model cheat by reading the exact phrases that created the labels).

---

## Labels (what we predict)

Fixed multi-label order (never reorder):

| Code | Meaning | Rough idea |
| ---- | ------- | ---------- |
| NORM | Normal | Explicit “normal ECG”-style statements and no HYP/STTC/MI/CD |
| HYP | Hypertrophy | LVH/RVH, atrial enlargement, etc. |
| STTC | ST/T changes | ST depression/elevation, T-wave abnormalities, ischemia-as-STT language |
| MI | Myocardial infarction | Infarct / STEMI / myocardial infarction language |
| CD | Conduction disorder | Blocks, pacing, WPW, axis deviation, abnormal PR, etc. |

**Important:** these are **rule-based weak labels** mined from `report_*` machine-report lines (`artifacts/label_mapping.yaml` **v1.0.0**). They are **not** cardiologist-adjudicated diagnoses. Metrics measure agreement with those rules, not clinical truth.

A study can have **multiple** positives at once → we use **five independent sigmoid logits** and `BCEWithLogitsLoss` (no softmax).

Unmapped studies (no target class fires) are excluded from the classification cohort but kept in `labels_full.csv`.

---

## Models at a glance

| ID | Folder | Input | Core idea |
| -- | ------ | ----- | --------- |
| **1** | `src/model_1_ecg_only/` | Raw 12-lead waveform | 1D CNN → 5 logits |
| **2** | `src/model_2_text_only/` | Leakage-controlled residual clinical text | Bidirectional LSTM/GRU → 5 logits |
| **2B** | `src/model_2b_measurement_only/` | Tabular machine measurements (intervals/axes) | RF / XGBoost baselines (extra, not the CNN) |
| **3** | `src/model_3_concat_fusion/` | ECG + residual text | `Linear(concat(z_ecg, z_text))` |
| **4** | `src/model_4_attention_fusion/` | ECG + residual text | Project both to `d`, softmax gates, weighted sum |

Shared glue lives in `src/shared/` (cohort join, waveforms, training loop, thresholds, metrics, fusion heads, smoke data, compare helpers). Configs are in `configs/`.

Comparison protocol: `docs/comparison.md`. Runner: `scripts/compare_models.py` → writes under `outputs/comparison/` (not committed).

---

## Critical experimental controls (memorize these)

1. **Frozen labels** — `artifacts/label_mapping.yaml` v1.0.0. Do not change rules after peeking at test metrics.
2. **Frozen patient split** — `artifacts/patient_split_manifest.csv` (seed **42**, ~70/15/15 by `subject_id`). All studies of one patient stay in one split.
3. **Text for Models 2/3/4** — residual `clinical_text` from `artifacts/cohort_manifest.csv` after masking label-trigger phrases (and summary Normal/Abnormal ECG lines). Masking ≠ “leakage-free.”
4. **Thresholds** — chosen on **validation** per-label F1 only, then frozen for test.
5. **IDs are never features** — `subject_id` / `study_id` only for joins and grouping. Public error reports use `example_id`.
6. **Common-cohort comparison** — fair model comparison uses the inner join of test studies that have residual text **and** a usable waveform (and measurements if 2B is included).

Model 2 is the **source of truth** for labels + split + residual text. Models 1/3/4/2B reuse those artifacts via `src/shared/cohort.py`.

---

## Getting the data (keep it simple)

[PhysioNet MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/). Follow their license.

**On this branch the laptop slice is in git** (not the full ~800k records):

| Path | What |
| --- | --- |
| `machine_measurements.csv` | 2500-row table → Models **2** and **2B** |
| `waveforms/files/...` | ~69 WFDB records → **1 / 3 / 4** |
| `artifacts/labels.csv`, `patient_split_manifest.csv`, `cohort_manifest.csv` | Frozen slice labels/split/residual text |
| `outputs/comparison/comparison.md` | Fair 15-study table |
| `outputs/model_1_ecg_only/explanations/*.png` | ECG occlusion plots |
| `outputs/model_3_concat_fusion/explanations/*.png` | Concat fusion occlusion |
| `outputs/model_4_attention_fusion/explanations/*.png` | Gated fusion occlusion |
| `outputs/model_2b_measurement_only/xgboost_permutation_importance.png` | 2B importance plot |

`--smoke` still needs no extra files. Commands: [running.md](running.md).

---

## How a run hangs together (flow)

```text
PhysioNet (credentialed)
  ├─ machine_measurements.csv  ──►  repo root
  └─ selected WFDB .hea/.dat   ──►  waveforms/files/pXXXX/p{subject}/s{study}/
                                        (list from frozen split; not the full dump)

machine_measurements.csv
        │
        ▼
 Model 2 pipeline  (run this first on real data)
   • load reports + IDs
   • rule-based labels     → artifacts/labels.csv  (+ labels_full.csv)
   • mask trigger phrases  → residual clinical_text
   • sample working cohort → config cap (500 here; seed 42)
   • patient split 70/15/15→ artifacts/patient_split_manifest.csv
   • cohort + residual text→ artifacts/cohort_manifest.csv
   • train text LSTM       → outputs/model_2_text_only/
        │
        ├─ Model 2B  same CSV, numeric columns only, same split/labels
        ├─ Model 1   join split+labels, index waveforms/, 1D CNN
        ├─ Model 3   join split+labels+residual text + waveforms, concat fusion
        └─ Model 4   same pair as 3, gated attention fusion
                │
                ▼
        each model writes test_predictions.csv
                │
                ▼
        scripts/compare_models.py
          inner-join study_ids that have residual text AND a usable waveform
          → outputs/comparison/
```

Join key everywhere: **`subject_id` + `study_id`**. IDs are never model features. Paths are resolved from the repo root (`src/shared/io_utils.py:repo_root()`).

---

## How each model uses the data

The same ECG study can contribute **three** views. They are not mixed inside a unimodal model.

| Column / file | Model 2 (text) | Model 2B (tabular) | Model 1 (CNN) | Models 3 & 4 (fusion) |
| --- | --- | --- | --- | --- |
| `subject_id`, `study_id` | join / split groups only | same | same | same |
| `report_*` lines | concatenate → **labels** from mapping v1.0.0; then **mask** those phrases → `clinical_text` → LSTM | **forbidden** as features (still used only if labels must be regenerated) | unused | same residual `clinical_text` as Model 2 |
| Numeric cart fields (`rr_interval`, fiducials, axes, …) | unused as features | **only** input (RF/XGBoost/logistic) | unused | unused |
| WFDB / `{study_id}.npy` | unused | unused | **only** input `(12, 5000)` | CNN encoder on waveform **plus** LSTM on residual text |

**Shared protocol after Model 2:** `src/shared/cohort.py` inner-joins `patient_split_manifest.csv` + `labels.csv` (+ residual text if the model needs it). Model 1 sets `require_text=False` so a study can train the CNN even if residual text is empty. Fusion sets `require_text=True`. `src/shared/waveforms.py` indexes `waveform_root`, keeps 12-lead finite records of length 5000 (pad/truncate), writes `artifacts/waveform_index.csv`, and drops the rest.

**Comparison:** a study counts in the fair table only if every compared model produced a test row for that `study_id`. In practice that means residual text **and** a file on disk (and measurement eligibility if 2B is included).

**Normalization / leakage (short):** ECG z-scores fit on **train waveforms only**. Text vocab fits on **train residual text only**. Thresholds fit on **validation** F1. Machine measurements and machine reports may share a device-algorithm origin — treat 2B vs 2 as a correlated-source caveat, not independent clinical truth.

---

## Architecture details (study notes)

### Model 1 — ECG CNN

- Input tensor `(B, 12, 5000)`.
- Stack of Conv1d → BatchNorm → ReLU → MaxPool blocks, then adaptive average pool → projection to embedding (default 128) → linear head to 5 logits.
- Encoder class `EcgCNNEncoder` is reused by Models 3 and 4.

### Model 2 — Text LSTM

- Train-only tokenizer/vocab; embedding (64) → bidirectional LSTM (hidden 64) → 5 logits.
- Padding handled with packed sequences.
- Positive class weights from **train** counts only.
- Leakage audit artifact: `artifacts/leakage_audit.json`.

### Model 2B — Measurements

- Timing/fiducials + axes; derived PR/QRS/QT when valid; sin/cos for axes after train-median imputation.
- Never uses `report_*` as features (labels may still come from reports).
- Same frozen split/labels as Model 2 for comparable cohorts.

### Model 3 — Concat fusion

```text
logits = Linear( concat(z_ecg, z_text) )
```

Default: finetune both encoders; lower LR on text; optional init from Model 1/2 checkpoints (`strict=False`).

### Model 4 — Gated modality attention

```text
P projects each encoder → dim d
a = softmax( W [P(z_ecg); P(z_text)] )   # two gates
z = a_ecg * P(z_ecg) + a_text * P(z_text)
logits = Linear(z)
```

Gate weights are **inspectable**, not causal explanations.

---

## Repo map

Canonical layout (do not put packages back at the repo root):

```text
configs/     one YAML per model (paths relative to repo root)
src/         Python packages (on PYTHONPATH / pip install -e .)
  shared/              protocol: labels, split, metrics, waveforms, fusion
  model_1_ecg_only/    CNN
  model_2_text_only/   residual-text LSTM
  model_2b_measurement_only/  tabular extra baseline
  model_3_concat_fusion/
  model_4_attention_fusion/
tests/       pytest; synthetic smoke only
scripts/     compare_models.py, list_waveform_downloads.py
docs/        this file + comparison protocol + Model 2 write-ups + per-model notes
artifacts/   frozen mapping (committed); labels/split CSVs local (gitignored)
outputs/     checkpoints, metrics, predictions (gitignored)
waveforms/   local WFDB or .npy (gitignored)
pyproject.toml, pytest.ini, requirements.txt
```

| Path | Role |
| ---- | ---- |
| `README.md` | Research proposal, layout, implementation checklist |
| `docs/study.md` | **This file** — living understanding + Q&A |
| `docs/comparison.md` | Frozen comparison protocol |
| `docs/model2.md` / `docs/model2_exp.md` | Completed Model 2 experiment notes |
| `docs/models/*.md` | Per-model run notes |
| `configs/*.yaml` | Hyperparameters and data paths |
| `src/shared/` | Shared cohort, waveforms, train/eval, fusion, smoke |
| `scripts/compare_models.py` | Common-cohort comparison |
| `scripts/list_waveform_downloads.py` | PhysioNet URL list for the **frozen split only** |
| `tests/` | Unit + smoke tests (`pytest.ini` sets `pythonpath = src`) |

### Shared modules (`src/shared/`)

| Module | Purpose |
| ------ | ------- |
| `constants.py` | `LABEL_ORDER`, lead names, 5000 samples @ 500 Hz |
| `io_utils.py` | `repo_root()`, `default_config()`, JSON/YAML, seeds |
| `cohort.py` | Load frozen split/labels/text; join shared cohort |
| `waveforms.py` | Index/load/filter/normalize 12-lead signals |
| `training.py` | Generic multi-label train loop, pos weights |
| `thresholds.py` | Val-only per-label F1 threshold search |
| `evaluate.py` | Macro/micro F1, precision/recall, safe AUROC/AUPRC |
| `predictions.py` | Write `test_predictions.csv` for comparison |
| `fusion.py` | `ConcatFusionModel`, `GatedAttentionFusionModel` |
| `fusion_pipeline.py` | Shared train/eval path for Models 3 & 4 |
| `multimodal_dataset.py` | Paired ECG + text batches |
| `ecg_explainability.py` | Time-window / lead occlusion + heatmap PNGs |
| `error_analysis.py` | Public error reports (`example_id` only) |
| `smoke.py` | Synthetic data for smoke tests |
| `compare.py` | Inner-join comparison tables |

---

## How to run (quick)

Canonical steps (venv, data, full vs smoke): **[running.md](running.md)**.

From the **repository root**. Use `.venv` (gitignored). Packages live under `src/`.

```text
# One-time: python3 -m venv .venv
source .venv/bin/activate

# Make src/ importable and install torch/pytest/etc. Alternative: export PYTHONPATH=src
pip install -e ".[dev]"

# Create frozen labels/split/residual text and train the text-only LSTM (do this first).
python -m model_2_text_only.run_pipeline --config configs/model_2_text_only.yaml

# Tabular RF/XGBoost/logistic on machine measurements; reuses Model 2 artifacts.
python -m model_2b_measurement_only.run_pipeline --config configs/model_2b_measurement_only.yaml

# Train the waveform-only 1D CNN (needs 12-lead files under waveform_root).
python -m model_1_ecg_only.run_pipeline --config configs/model_1_ecg_only.yaml

# Train concat fusion: Linear(concat(z_ecg, z_text)).
python -m model_3_concat_fusion.run_pipeline --config configs/model_3_concat_fusion.yaml

# Train gated attention fusion of the same two encoders.
python -m model_4_attention_fusion.run_pipeline --config configs/model_4_attention_fusion.yaml

# Build common-cohort comparison tables from each model's test_predictions.csv.
python scripts/compare_models.py --dest outputs/comparison

# Synthetic (non-clinical) end-to-end checks; do not report these metrics as results.
python -m model_2_text_only.run_pipeline --smoke          # fake text LSTM pipeline
python -m model_1_ecg_only.run_pipeline --smoke           # fake ECG CNN pipeline
python -m model_3_concat_fusion.run_pipeline --smoke      # fake concat fusion
python -m model_4_attention_fusion.run_pipeline --smoke   # fake gated fusion

# Run unit + smoke tests under tests/.
pytest
```

Typical order: **Model 2 first** (creates frozen artifacts), then 2B / 1 / 3 / 4, then compare. Config defaults already point at `configs/<model>.yaml` if you omit `--config`.

---

## Metrics we care about

- **Macro F1** — average F1 across the five labels (equal weight per label; good with imbalance).
- **Micro F1** — pool all TP/FP/FN then F1 (instance-decision aggregate).
- Precision / recall (macro and micro).
- **AUROC / AUPRC** per label when defined (implementation returns `None` if undefined — e.g. one class missing).

Thresholding strategy is part of the result story: thresholds come from validation F1, then frozen.

---

## Leakage & honesty (exam-style reminders)

| Risk | Mitigation in this repo |
| ---- | ----------------------- |
| Labels mined from the same text we feed the model | Mask trigger phrases → residual text; audit remaining firings |
| Patient appears in train and test | Patient-level split by `subject_id` |
| Tuning on test | Thresholds/hparams on val only; freeze mapping/split |
| Synonyms still leak after masking | Documented limitation; do not claim “leakage-free” |
| Model 2B vs machine reports from same cart | Interpret 2B as possibly recovering device interpretation from measurements |
| IDs as shortcuts | Never use IDs as features |

**Bottom line for papers/talks:** scores = agreement with weak machine-report rules under these controls — research prototypes, **not clinical use**.

---

## Explainability (what “explainable” means here)

- **ECG (Models 1/3/4):** zero out time windows and whole leads; record Δ probability (`src/shared/ecg_explainability.py`). A few example heatmaps go under `outputs/.../explanations/` (no patient IDs).
- **Text (Model 2):** token occlusion (replace with `<unk>`) in `src/model_2_text_only/explainability.py`.
- **Fusion:** same ECG + text occlusions on the fused model; Model 4 also saves softmax **gates** (`attention_gates.json`, columns on `test_predictions.csv`).

Attribution shows **sensitivity**, not proven clinical causality. Attention/gates ≠ explanation of reasoning. Integrated Gradients are listed in the README as a possible method; the implemented default is occlusion.

---

## Q&A — common project questions

### How do I get the data, and how do models use it?

Full story: [Getting the data](#getting-the-data-how-it-actually-arrives) and [How each model uses the data](#how-each-model-uses-the-data). Short version: PhysioNet MIMIC-IV-ECG → put **one** `machine_measurements.csv` at the repo root for Models 2/2B → run Model 2 → download **only** the frozen-split WFDB folders into `waveforms/` for Models 1/3/4. Never the full ~800k waveform dump.

### How do I run the project?

See [running.md](running.md). Short version: `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -e ".[dev]"`, then run Model 2 first (`python -m model_2_text_only.run_pipeline`). Smoke checks: `--smoke` or `pytest`.

### Why five labels and multi-label?

ECGs often show more than one finding type at once (e.g. conduction block + ST/T changes). Multi-label with independent sigmoids matches that better than mutually exclusive classes.

### Why start with Model 2?

Model 2 creates the **shared weak labels**, **patient split**, and **residual text** that every later model must reuse for a fair comparison.

### Why is Model 2B not “Model 1”?

Model 1 is the **raw waveform CNN**. Model 2B is a **tabular** baseline on machine summary numbers. Different scientific question; useful contrast to text, but not a substitute for waveform learning or fusion.

### What is “residual text”?

After label rules fire, those diagnostic statements (and Normal/Abnormal ECG summary phrases) are removed from the report. What’s left — rhythm, rate, technical notes, other non-trigger lines — is residual text. If almost nothing remains, training may be skipped.

### Why freeze the split and mapping?

Changing them after seeing test numbers would invalidate the comparison and invite overfitting to the holdout. Treat them like a sealed envelope.

### Do Models 3 and 4 need Models 1 and 2 trained first?

They need Model 2 **artifacts** (labels/split/text). Checkpoints from Models 1 and 2 are **optional** warm starts when present.

### What does “working cohort” mean?

Full MIMIC-IV-ECG has on the order of **~800k** studies. You do **not** need all of them. This checkout keeps a **small local CSV slice** and `working_cohort_max_studies: **500**`. Raise that only if you want a larger experiment; do not regenerate the shared split after seeing test metrics.

### What is `--smoke`?

A tiny synthetic end-to-end path for tests/CI. Never report smoke metrics as clinical results.

### Where do I put real waveforms?

Under `waveform_root` from config (default `waveforms/` at repo root): PhysioNet-style `files/pXXXX/p{subject_id}/s{study_id}/` or `{study_id}.npy` of shape `(12, 5000)`. Keep them off git. **Do not download all ~800k records.** Model 2/2B need none. After Model 2, `python scripts/list_waveform_downloads.py` lists only the frozen working cohort. Models 1/3/4 skip missing files. See [running.md](running.md).

### Why did packages move under `src/`?

So configs, tests, docs, and code are separate. Imports stay `from model_2_text_only...` and `from shared...` because `src/` is on the path. Do not recreate `model_*` or `shared/` at the repository root.

### How should we interpret high text-model F1?

Suspect residual leakage (synonyms, templates, correlated rhythm language) before claiming “text understands cardiology.” Always read `leakage_audit.json` and error analysis.

---

## Laptop slice (2026-10-31) — numbers you can cite locally

Not the published ~8k Model 2 write-up. Local files only (gitignored).

- CSV: 2500 rows → Model 2 working cohort ~500 studies (test n=81 for text-only metrics).
- Waveforms: ~69 usable records spanning train/val/test.
- Fair table: **15** common test studies with text **and** a waveform (`outputs/comparison/comparison.md`).

| Model | Macro F1 (own test) | Macro F1 (common 15) |
| --- | ---: | ---: |
| 2 text LSTM | 0.429 (n=81) | 0.392 |
| 2B XGBoost / RF / logistic | RF 0.460, XGB 0.472, LR 0.428 | 2B 0.463 |
| 1 ECG CNN | 0.325 | 0.325 |
| 3 concat | 0.264 | 0.264 |
| 4 gated | 0.348 | 0.348 |

**Do not** treat this as a ranking of architectures. n=15, weak labels, residual leakage, 2B may echo the same cart as the reports. Fusion skipped the Model 2 text checkpoint here because vocab size differed on the waveform subset.

---

## What to put in the report (and what to leave out)

**Put in**

1. Research question: does ECG + residual text beat ECG-only or text-only on five weak labels; concat vs gated fusion.
2. Data: MIMIC-IV-ECG v1.0. One CSV (`report_*` + measurements). Optional 12-lead WFDB. Patient-level split, seed 42. **State the cohort size you actually used** (this laptop: 2500 CSV rows / ~500 train cohort / 15 common test studies).
3. Labels: five multi-label weak rules from reports (`label_mapping.yaml` v1.0.0), **not** cardiologist diagnoses.
4. Leakage control: mask trigger phrases; residual text still can leak; IDs never features; thresholds on val only.
5. Models in one table: 1 CNN, 2 LSTM, 2B tabular (extra), 3 concat, 4 gates.
6. Metrics: macro/micro F1, maybe AUROC. Use the **common-cohort** table for comparing models.
7. Explainability: occlusion (ECG windows/leads, text tokens). Gates ≠ causal explanation.
8. Limitations: small slice, weak labels, possible 2B/report correlation, no clinical use.

**Leave out**

- Smoke F1 numbers.
- The full 800k download story.
- Claiming fusion “won” or clinical accuracy from n=15.
- Patient IDs in figures (use `example_id`).

Longer Model 2 write-up (larger historical run): [model2.md](model2.md). Protocol: [comparison.md](comparison.md).

---

## Current implementation status (snapshot)

*Last snapshot: 2026-10-31.*

| Area | Status |
| ---- | ------ |
| Modular layout | `src/`, `configs/`, `tests/`, `scripts/`, `docs/` |
| Label mapping v1.0.0 | Committed at `artifacts/label_mapping.yaml` |
| Model 2 text LSTM | Historical ~7k run in `docs/model2.md`; **laptop slice** 2026-10-31 also ran |
| Model 2B tabular | Laptop slice on the same frozen split |
| Model 1 / 3 / 4 | Laptop slice on ~69 local waveforms (15 common test studies) |
| Comparison + plots | Tracked under `outputs/` on this branch (see paths above) |
| Local venv | `.venv/`; `source .venv/bin/activate` |
| Final slides | Not started |
| Clinical deployment | Out of scope |

---

## Agent maintenance rules

When an agent changes **anything** in this repo:

1. Update the relevant sections of this file so they still match reality.
2. Append a bullet under [Changelog](#changelog) with date, what changed, and why it matters for understanding.
3. Prefer short, concrete notes over rewriting the whole guide.
4. If a question was answered in chat that future readers should know, add it to the Q&A section.

---

## Changelog

- **2026-10-31 (slice in git)** — Tracked the laptop CSV, waveforms, artifacts (not smoke bundles), and `outputs/` (comparison + plots + metrics). Documented exact report file paths. Still not the full MIMIC dump.
- **2026-10-31 (laptop slice + simple data)** — Ran 2, 2B, 1, 3, 4 on the small CSV + ~69 waveforms; comparison n=15. Fusion skips a Model 2 checkpoint when vocab size differs. Running/study data sections shortened to: CSV in root, optional waveforms folder, same `python -m` commands. Added report checklist.
- **2026-10-31 (numpy shuffle)** — `sample_working_cohort` copies patient-id arrays before `RandomState.shuffle` (NumPy 2 / pandas 3 mark index arrays read-only). Needed for the small real CSV run.
- **2026-10-31 (small local slice)** — Stopped the full 174 MB CSV download. Kept ~2500 complete rows in gitignored `machine_measurements.csv` and set `working_cohort_max_studies: 500` so a laptop run is enough. Not the published ~8k Model 2 cohort.
- **2026-10-31 (data flow in study)** — Added how data is obtained (PhysioNet DUA, `machine_measurements.csv` vs subset WFDB), the end-to-end flow, and a table of what each model actually reads. This was previously only sketched as a diagram and split across running.md / model2_exp.md.
- **2026-10-31 (waveforms subset)** — Documented that ~800k WFDB files are not required: working cohort (~8000 by default, configurable *before* freeze), `scripts/list_waveform_downloads.py` for PhysioNet URL lists, and Models 1/3/4 dropping missing files. See [running.md](running.md).
- **2026-10-30 (running docs)** — Added `docs/running.md` as the how-to-run guide (venv, `pip install -e ".[dev]"`, data files, Model 2→2B→1→3→4→compare, smoke/pytest). Linked it from README, docs index, comparison, Model 2 write-up, per-model notes, and this file.
- **2026-10-30 (docs commands)** — Annotated every runnable command in README, comparison, study, Model 2 write-ups, and per-model notes so each line says what the command does.
- **2026-10-30 (venv)** — Added local `.venv/` (gitignored). Activate with `source .venv/bin/activate` before `pip install -e ".[dev]"` / `pytest`.
- **2026-10-30 (layout)** — Packages moved to `src/`, YAML to `configs/`, tests to `tests/`, write-ups to `docs/`, comparison CLI to `scripts/compare_models.py`. Paths use `repo_root()`. Documented `pip install -e ".[dev]"` / `PYTHONPATH=src`. Status table now distinguishes Model 2/2B real-data results from Model 1/3/4 smoke-only code. Explainability section matches implemented occlusion (not IG).
- **2026-10-30** — Created `study.md` as the living study guide: research question, labels, all five model tracks, frozen controls, data flow, shared module map, run commands, leakage/metrics/explainability notes, Q&A, and agent update rules.
