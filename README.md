# embeddings-sparse-autoencoders

## Polarity Dial Experiment (Protocol A)

### Signed Symmetric Gated SAE (Ablation)
To run the Protocol A ablation promised in the paper (symmetric magnitude calibration), set the following in the model config:

- `sign_aware: true`
- `symmetric_magnitude: true` (ties the negative magnitude scale to the positive one, so effectively $r^- = r^+$)
- optionally `single_activation_offset: true` if you also want symmetric thresholds ($\delta^- = \delta^+$)

Example (inside a model entry in the Protocol A config JSON/YAML):
```json
{
  "sae_class": "GatedSAE",
  "sign_aware": true,
  "symmetric_magnitude": true,
  "single_activation_offset": true
}
```
- Entry point: `python -m sgnsae.scripts.run_polarity_dial_experiment --config-path <config>`.
- Outputs: `<output_dir>/polarity_dial/<run_id>/` with config, data exports, per-model checkpoints/metrics/plots, and aggregated comparisons.
- Default device selection uses CUDA when available unless overridden in the config.

### Paper reproduction configs
- Configs live under `configs/` with exact parameters from the draft:
  - `python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed0.json`
  - `python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed1.json`
  - `python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed2.json`
- Aggregate the metrics in `comparison_summary.json` across the three runs per model to match paper-reported numbers.

### Notes
- Requirements now include `matplotlib` and `scipy` for plotting and Hungarian alignment.
- Requirements now include `PyYAML` for YAML experiment configs.
- Optional Weights & Biases logging activates when both `wandb_project` and `wandb_api_key` are provided in the experiment config.
