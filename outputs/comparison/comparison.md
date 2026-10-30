# Common-cohort model comparison

Metrics measure agreement with weak report-derived labels, not clinical diagnostic accuracy. Models 2/3/4 use residual text after masking; residual leakage can remain. Model 2B measurements may correlate with the same cart software that wrote the reports.

Common test studies: **15**.

| Model | Macro F1 | Micro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| model_2_text | 0.392 | 0.418 | 0.648 | 0.558 |
| model_2b_measurements | 0.463 | 0.600 | 0.748 | 0.692 |
| model_1_ecg | 0.325 | 0.433 | 0.580 | 0.491 |
| model_3_concat | 0.264 | 0.444 | 0.507 | 0.393 |
| model_4_gated | 0.348 | 0.444 | 0.591 | 0.498 |

## Per-label F1

| Model | NORM | HYP | STTC | MI | CD |
| --- | ---: | ---: | ---: | ---: | ---: |
| model_2_text | 0.556 | 0.333 | 0.571 | 0.000 | 0.500 |
| model_2b_measurements | 0.571 | 0.000 | 0.909 | 0.000 | 0.833 |
| model_1_ecg | 0.500 | 0.353 | 0.571 | 0.000 | 0.200 |
| model_3_concat | 0.500 | 0.250 | 0.571 | 0.000 | 0.000 |
| model_4_gated | 0.526 | 0.333 | 0.571 | 0.000 | 0.308 |

## Pairwise Hamming wins

- `model_2_text` vs `model_2b_measurements` (n=15): left better 1, right better 14, ties 0; right corrects left 5, right introduces error 0.
- `model_2_text` vs `model_1_ecg` (n=15): left better 0, right better 11, ties 4; right corrects left 1, right introduces error 0.
- `model_2_text` vs `model_3_concat` (n=15): left better 0, right better 13, ties 2; right corrects left 0, right introduces error 0.
- `model_2_text` vs `model_4_gated` (n=15): left better 0, right better 9, ties 6; right corrects left 0, right introduces error 0.
- `model_2b_measurements` vs `model_1_ecg` (n=15): left better 11, right better 3, ties 1; right corrects left 1, right introduces error 5.
- `model_2b_measurements` vs `model_3_concat` (n=15): left better 9, right better 2, ties 4; right corrects left 0, right introduces error 5.
- `model_2b_measurements` vs `model_4_gated` (n=15): left better 12, right better 1, ties 2; right corrects left 0, right introduces error 5.
- `model_1_ecg` vs `model_3_concat` (n=15): left better 4, right better 8, ties 3; right corrects left 0, right introduces error 1.
- `model_1_ecg` vs `model_4_gated` (n=15): left better 6, right better 1, ties 8; right corrects left 0, right introduces error 1.
- `model_3_concat` vs `model_4_gated` (n=15): left better 11, right better 2, ties 2; right corrects left 0, right introduces error 0.
