# Model 4: Gated modality-attention fusion

Same encoders as Model 3. After projecting `z_ecg` and `z_text` to a shared size `d`,

`a = softmax(W [P(z_ecg); P(z_text)])`

`z = a_ecg * P(z_ecg) + a_text * P(z_text)`

Gates are saved per test study. They are **not** causal explanations.

Run from the repo root with the venv on (`source .venv/bin/activate`). Setup: [running.md](../running.md). Needs Model 2 artifacts plus local waveforms.

```text
# Train gated attention fusion: softmax weights over projected ECG and text embeddings.
python -m model_4_attention_fusion.run_pipeline --config configs/model_4_attention_fusion.yaml

# Synthetic wiring check (no MIMIC waveforms/text; not clinical).
python -m model_4_attention_fusion.run_pipeline --config configs/model_4_attention_fusion.yaml --smoke
```
