# Model 2 vs Model 2B comparison

Labels are weak, report-derived targets. This is not independent clinical validation.

## Cohort

```json
{
  "model_2_original_test_studies": 81,
  "model_2b_test_studies": 81,
  "model_2b_test_patients": 15,
  "model_2_manifest_test_studies": 81,
  "n_model_2_test_studies_absent_from_model_2b": 0,
  "n_measurement_studies_not_in_shared_manifest": 1722,
  "common_test_studies": 81,
  "common_cohort_evaluated": true,
  "shared_split_reused": true,
  "label_source": {
    "label_order": [
      "NORM",
      "HYP",
      "STTC",
      "MI",
      "CD"
    ],
    "mapping_version": "1.0.0",
    "source": "/home/arushi/tech/explainable-multimodal-ecg-diagnostic-classification/artifacts/labels.csv",
    "weak_labels": true,
    "clinically_verified": false,
    "uncertain_policy": "positive_weak",
    "reused_shared_artifact": true
  },
  "note": "Direct modality comparison requires the same patient-level test studies. If Model 2B dropped measurement-ineligible records, original Model 2 test metrics and common-cohort Model 2 metrics are reported separately."
}
```

## Overall metrics

| Model | Macro F1 | Micro F1 | Macro AUROC | Macro AUPRC | N studies | N patients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model 2 original test | 0.4292 | 0.4427 | 0.5581 | 0.3331 | 81 | 15 |
| Model 2 common test | 0.4292 | 0.4427 | 0.5581 | 0.3331 | 81 | 15 |
| Model 2B random_forest | 0.4600 | 0.5351 | 0.6672 | 0.5397 | 81 | 15 |
| Model 2B xgboost | 0.4719 | 0.5053 | 0.6718 | 0.5106 | 81 | 15 |
| Model 2B logistic_regression | 0.4279 | 0.5310 | 0.6802 | 0.5229 | 81 | 15 |

## Per-label F1

| Model | NORM | HYP | STTC | MI | CD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model 2 original test | 0.3291 | 0.4118 | 0.5789 | 0.2857 | 0.5405 |
| Model 2 common test | 0.3291 | 0.4118 | 0.5789 | 0.2857 | 0.5405 |
| Model 2B random_forest | 0.2857 | 0.2222 | 0.6984 | 0.1429 | 0.9508 |
| Model 2B xgboost | 0.1739 | 0.3774 | 0.7059 | 0.1842 | 0.9180 |
| Model 2B logistic_regression | 0.1739 | 0.3333 | 0.6866 | 0.1333 | 0.8125 |

## Notes

- Both models are scored against the same report-derived weak labels.
- Model 2 uses leakage-controlled residual report text; Model 2B uses numerical measurements only.
- Do not attribute differences to modality if the test populations differ.
- These results do not evaluate raw-waveform CNNs or multimodal fusion.
