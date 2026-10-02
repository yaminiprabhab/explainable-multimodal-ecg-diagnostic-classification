# Explainable Multimodal ECG Diagnostic Classification Using ECG Signals and Clinical Text

## Overview

Electrocardiograms (ECGs) are widely used to identify cardiac abnormalities by analyzing the electrical activity of the heart. Traditional automated ECG classification systems primarily learn from waveform data to predict diagnostic categories. However, clinical ECG reports contain complementary information about rhythm, waveform morphology, conduction patterns, and other diagnostic observations.

This project investigates whether combining **12-lead ECG signals with associated clinical text** improves multi-label diagnostic classification compared with using either modality independently. It also compares conventional feature-concatenation fusion with attention-based multimodal fusion and examines which waveform regions and textual features contribute to model predictions.

The project uses a controlled, patient-level experimental design and evaluates four progressively more complex architectures, from unimodal baselines to attention-based multimodal learning.

> **Research focus:** Does multimodal learning improve ECG diagnostic classification, and does attention-based fusion provide measurable benefits over simple feature concatenation?

## Objectives

* Evaluate ECG-only and clinical-text-only diagnostic classification baselines.
* Investigate whether combining ECG waveforms and clinical text improves classification performance.
* Compare feature-concatenation fusion with attention-based multimodal fusion.
* Predict five broad diagnostic categories using a multi-label classification framework.
* Apply leakage controls to reduce the risk of models learning diagnostic labels directly from the text used to generate them.
* Generate interpretable explanations identifying influential ECG regions and clinical-text features.
* Perform error analysis to understand model behavior across diagnostic categories and identify limitations.

## Dataset

**Dataset:** MIMIC-IV-ECG Diagnostic Electrocardiogram Matched Subset, version 1.0.

The dataset contains diagnostic ECG recordings linked to associated clinical information. The project uses a manageable subset of records with corresponding textual information to accommodate available computational resources and the project timeline.

### Dataset characteristics

* **ECG modality:** Raw 12-lead ECG waveforms.
* **Sampling frequency:** 500 Hz.
* **Recording duration:** Approximately 10 seconds per ECG.
* **Signal format:** WFDB.
* **Text modality:** Clinical report fields associated with the corresponding ECG examination.
* **Record linkage:** `subject_id` and `study_id`.
* **Data selection:** A filtered subset of ECG examinations with usable waveform and text data.

### Diagnostic categories

The task is formulated as multi-label classification across five broad diagnostic categories.

| Label | Category              | Description                                           |
| ----- | --------------------- | ----------------------------------------------------- |
| NORM  | Normal                | ECGs classified under the normal category             |
| HYP   | Hypertrophy           | ECG findings associated with cardiac hypertrophy      |
| STTC  | ST/T Changes          | ST-segment and T-wave changes                         |
| MI    | Myocardial Infarction | ECG findings associated with myocardial infarction    |
| CD    | Conduction Disorder   | ECG findings associated with conduction abnormalities |

A single ECG may be associated with more than one diagnostic category. Therefore, the task is treated as multi-label classification rather than mutually exclusive multi-class classification.

### Label generation

The five diagnostic labels are generated from clinical text using a rule-based labeling approach adapted from prior MEDBind/MIMIC-ECG work.

Because the same text is also used as a model input, this setup introduces a potential source of **target leakage**. The project therefore incorporates explicit text preprocessing or masking intended to prevent the text model from directly exploiting the phrases used to generate the labels.

The effectiveness of these controls must be verified during implementation and evaluation.

## Proposed Model Architectures

The project compares four architectures under a controlled experimental setup.

### Model 1: ECG-Only Baseline

**Architecture:** One-dimensional Convolutional Neural Network (1D CNN)

The model processes the raw 12-lead ECG waveform to learn temporal and morphological features relevant to diagnostic classification.

**Purpose:**

* Establish a waveform-only baseline.
* Learn discriminative ECG representations.
* Identify the diagnostic performance achievable without clinical text.

### Model 2: Clinical-Text-Only Baseline

**Architecture:** Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU)

The model processes tokenized clinical text and learns contextual representations from the associated ECG report.

**Purpose:**

* Establish a text-only baseline.
* Evaluate the diagnostic information contained in clinical language.
* Examine the limitations introduced by text quality and leakage controls.

### Model 3: Basic Multimodal Fusion

**Architecture:** 1D CNN + LSTM/GRU with feature concatenation

The ECG encoder and text encoder independently learn modality-specific representations. Their representations are concatenated and passed to a fully connected classification head.

**Purpose:**

* Evaluate whether combining independently learned ECG and text features improves performance.
* Establish a baseline for multimodal feature fusion.
* Determine whether simple concatenation captures useful complementary information.

### Model 4: Attention-Based Multimodal Fusion

**Architecture:** 1D CNN + LSTM/GRU with attention-based fusion

The model uses the ECG and text encoders to obtain modality representations, which are combined through an attention mechanism designed to learn their relative importance for diagnostic classification.

**Purpose:**

* Investigate whether learned modality weighting or interaction improves upon feature concatenation.
* Examine how the model uses the two modalities across diagnostic categories.
* Compare the benefits of attention against its additional architectural complexity.

The exact attention formulation will be documented alongside the implementation.

### Model comparison at a glance

| Model   | Input         | Architecture   | Fusion                 |
| ------- | ------------- | -------------- | ---------------------- |
| Model 1 | ECG waveform  | 1D CNN         | None                   |
| Model 2 | Clinical text | LSTM/GRU       | None                   |
| Model 3 | ECG + text    | CNN + LSTM/GRU | Feature concatenation  |
| Model 4 | ECG + text    | CNN + LSTM/GRU | Attention-based fusion |

All four models are intended to use the same patient-level data split, target labels, and comparable evaluation procedures to support a controlled comparison.

## Experimental Methodology

The project follows a staged experimental workflow.

1. **Data acquisition:** Identify the relevant ECG records and associated clinical text.
2. **Data quality filtering:** Remove or handle unusable waveforms, missing reports, invalid records, and other data-quality issues.
3. **Record matching:** Associate each ECG waveform with its corresponding clinical text using available record identifiers.
4. **Label generation:** Derive the five diagnostic targets through the rule-based labeling procedure.
5. **Leakage-aware preprocessing:** Mask or transform label-generating phrases in the text input and verify that preprocessing does not expose target information.
6. **Patient-level splitting:** Partition the dataset into training, validation, and test sets so that records from the same patient do not cross split boundaries.
7. **Modality-specific preprocessing:** Prepare ECG signals for the CNN and tokenize clinical text for the recurrent encoder.
8. **Model training:** Train the ECG-only, text-only, concatenation-fusion, and attention-fusion models.
9. **Evaluation:** Compare model performance using consistent multi-label classification metrics.
10. **Error analysis:** Examine category-specific errors and cases where multimodal models differ from unimodal baselines.
11. **Explainability:** Identify influential ECG regions and text features for selected predictions.
12. **Documentation:** Record experimental configurations, results, limitations, and reproducibility information.

## Data Leakage and Evaluation Integrity

Leakage prevention is a central methodological consideration in this project.

### Text-derived labels

The target labels are generated from clinical text that is also provided as a model input. Consequently, a model could achieve apparently strong performance by learning the wording of the labeling rules instead of learning clinically meaningful associations.

The project addresses this risk through explicit text masking or preprocessing. The implementation should document:

* Which phrases, terms, or patterns are used to generate labels.
* Which corresponding text features are masked, removed, or transformed.
* Whether the preprocessing preserves clinically useful information unrelated to the labeling rules.
* How residual leakage is assessed.
* Whether label generation and preprocessing are applied consistently across all data splits.

**Important limitation:** Masking alone does not guarantee the absence of leakage. Synonyms, abbreviations, indirect references, and other correlated textual cues may still reveal the target labels. Results should therefore be interpreted in light of the implemented controls.

### Patient-level data splitting

All four models use an identical patient-level split. ECG records belonging to a given patient must remain within a single partition.

This reduces the risk of overly optimistic results caused by patient overlap between training and evaluation data.

A proposed split is 70% training, 15% validation, and 15% testing, subject to the final implementation and dataset constraints.

## Evaluation Metrics

The models are evaluated using multiple complementary metrics.

| Metric         | Purpose                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| Macro F1-score | Calculates F1 separately for each label and averages across labels, giving each label equal weight    |
| Micro F1-score | Aggregates true positives, false positives, and false negatives across labels before calculating F1   |
| Precision      | Measures the proportion of positive predictions that are correct                                      |
| Recall         | Measures the proportion of actual positive instances identified                                       |
| AUROC          | Measures the ability to distinguish positive from negative instances across classification thresholds |

Macro F1 is particularly relevant when diagnostic categories have different frequencies. Micro F1 provides an aggregate view of performance across all label-instance decisions.

For a meaningful comparison, the final report should also specify the classification thresholding strategy, label-wise results, class frequencies, and the method used to aggregate AUROC across labels.

## Explainability

Explainability is a core component of the project. In addition to evaluating predictive performance, the project investigates which parts of each modality contribute to individual predictions.

### ECG explainability

The ECG branch aims to identify waveform regions that influence a prediction.

Potential approaches include:

* Gradient-based attribution methods.
* Integrated Gradients.
* Occlusion-based sensitivity analysis.
* Lead-wise and temporal-region attribution visualizations.

These methods can help identify influential portions of the signal, including relevant leads and time intervals.

### Clinical-text explainability

The text branch aims to identify words or phrases that influence predictions.

Potential approaches include:

* Token-level attribution.
* Integrated Gradients for text representations.
* Perturbation-based word or phrase importance analysis.
* Comparison of predictions before and after selected text features are removed.

### Multimodal explainability

For the fusion models, the project can examine the contributions of the ECG and text representations and compare explanations across modalities.

The analysis should distinguish between:

* **Feature attribution:** Which input features influence a prediction?
* **Attention weights:** Which representations receive greater weight under the implemented attention mechanism?
* **Clinical validity:** Whether the highlighted information is clinically meaningful.

Attention weights alone should not be treated as definitive explanations of model reasoning. Explanations are intended to support model inspection and error analysis, not to establish clinical correctness or replace professional interpretation.

## Error Analysis

Error analysis will investigate model behavior beyond aggregate metrics.

Areas of investigation include:

* Diagnostic categories with comparatively low precision, recall, or F1-score.
* Differences between ECG-only and text-only predictions.
* Cases where multimodal fusion improves or worsens predictions.
* Differences between feature concatenation and attention-based fusion.
* False positives and false negatives for each diagnostic category.
* Effects of class imbalance and missing or noisy clinical text.
* Potential residual leakage and inconsistencies in rule-based labels.
* Whether highlighted ECG regions and text features provide plausible explanations for predictions.

Where feasible, the analysis will include label-wise confusion statistics and representative correctly and incorrectly classified examples.

## Literature Review and Research Motivation

The project is motivated by research demonstrating the potential of multimodal learning to combine physiological information with clinical semantics.

The literature reviewed for the project covers the following directions:

* **ECG–clinical-report retrieval:** Learning relationships between ECGs and associated reports to support retrieval of clinically similar records.
* **Multimodal representation learning:** Aligning ECG signals with clinical text and, in broader frameworks, chest X-rays.
* **Cross-modal interaction:** Using reconstruction and other interaction mechanisms to transfer semantic information between modalities.
* **Local and global signal representations:** Identifying clinically meaningful waveform regions alongside broader ECG context.
* **Structured clinical semantics:** Mapping free-text findings to standardized cardiac terminology.
* **ECG-to-text generation:** Using learned ECG representations to support clinical report generation and cross-domain evaluation.

These research directions motivate an empirical comparison of simpler and more expressive fusion mechanisms.

The proposed study focuses on a narrower, controlled question: whether the addition of clinical text improves five-category multi-label ECG classification and whether attention-based fusion offers benefits over feature concatenation under the same evaluation protocol.

## Expected Outcomes

The project aims to produce:

* Four comparable diagnostic classification models.
* A reproducible data preprocessing and patient-level splitting workflow.
* An empirical comparison of unimodal and multimodal classification.
* An evaluation of feature concatenation versus attention-based fusion.
* Quantitative results using macro F1, micro F1, precision, recall, and AUROC.
* Error analysis across diagnostic categories and model architectures.
* ECG waveform and clinical-text attribution visualizations.
* A discussion of data leakage, class imbalance, interpretability, and methodological limitations.

These are intended deliverables rather than claims of completed experiments. The final conclusions will depend on the observed results.

## Limitations

The project has several important limitations:

1. **Rule-based labels:** Automatically generated labels may not fully capture clinical diagnostic complexity.
2. **Potential target leakage:** Clinical text may contain information correlated with the rules used to generate labels.
3. **Subset selection:** Findings from a computationally manageable subset may not generalize to the complete dataset.
4. **Class imbalance:** Differences in diagnostic prevalence can affect training and evaluation.
5. **Fusion complexity:** Attention-based fusion may introduce additional parameters and computational costs without necessarily improving performance.
6. **Explanation limitations:** Feature attribution indicates model sensitivity or contribution, not necessarily causal relationships or clinically valid reasoning.
7. **External generalization:** Performance on one dataset does not establish robustness across hospitals, patient populations, devices, or reporting practices.
8. **Clinical use:** The models are research prototypes and are not intended to provide standalone diagnoses or replace clinical judgment.

## Reproducibility

To support reproducibility, the repository should document:

* Dataset version, access requirements, and subset-selection criteria.
* Patient-level split assignments and random seeds.
* ECG preprocessing and text tokenization procedures.
* Label-generation rules and leakage-control implementation.
* Model architectures and hyperparameters.
* Training configurations and classification thresholds.
* Metric computation procedures.
* Software dependencies and execution instructions.
* Explainability methods and representative analysis outputs.

The dataset may require separate access approval. Users should follow the dataset's applicable access conditions and avoid committing restricted patient data, credentials, or sensitive clinical records to the repository.

## Implementation Plan

* [ ] Dataset acquisition and subset selection
* [ ] ECG–text record matching and data quality checks
* [ ] Rule-based diagnostic label generation
* [ ] Leakage-aware clinical-text preprocessing
* [ ] Patient-level train, validation, and test split
* [ ] ECG-only CNN baseline
* [ ] Clinical-text-only LSTM/GRU baseline
* [ ] Feature-concatenation fusion model
* [ ] Attention-based fusion model
* [ ] Evaluation using all specified metrics
* [ ] Error analysis and label-wise comparison
* [ ] ECG waveform attribution
* [ ] Clinical-text attribution
* [ ] Comparison of multimodal explanations
* [ ] Final documentation and presentation

## Conclusion

This project investigates the contribution of clinical text to ECG diagnostic classification through a controlled comparison of four model architectures. By establishing unimodal baselines before evaluating concatenation and attention-based fusion, it aims to distinguish the contribution of additional clinical information from the contribution of more complex fusion mechanisms.

Combining quantitative evaluation with waveform and text attribution provides a way to investigate not only whether the models make accurate predictions, but also which input features influence those predictions.

The central objective is to assess whether multimodal ECG classification provides a measurable benefit under leakage-aware evaluation and whether that benefit justifies the additional complexity of attention-based fusion.

---

**Project title:** Explainable Multimodal ECG Diagnostic Classification Using ECG Signals and Clinical Text

**Dataset:** MIMIC-IV-ECG Diagnostic Electrocardiogram Matched Subset v1.0

**Task:** Five-category multi-label ECG diagnostic classification

**Core methods:** 1D CNN, LSTM/GRU, feature-concatenation fusion, attention-based fusion, and input-feature attribution

**Project status:** Research and implementation in progress
