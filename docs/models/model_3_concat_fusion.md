# Model 3: Concatenation fusion

ECG 1D CNN representation concatenated with Model 2 LSTM residual-text representation, then a linear head. Uses the frozen patient split, weak labels v1.0.0, and leakage-controlled `clinical_text`.

Default: finetune both encoders, lower learning rate on the text encoder. Optional initialization from Model 1 / Model 2 checkpoints if those files exist.

Run from the repo root with the venv on (`source .venv/bin/activate`). Setup: [running.md](../running.md). Needs Model 2 artifacts plus local waveforms.

```text
# Train concat fusion: ECG CNN + text LSTM embeddings concatenated, then a linear head.
python -m model_3_concat_fusion.run_pipeline --config configs/model_3_concat_fusion.yaml

# Synthetic wiring check (no MIMIC waveforms/text; not clinical).
python -m model_3_concat_fusion.run_pipeline --config configs/model_3_concat_fusion.yaml --smoke
```
