# embeddings-sparse-autoencoders

## Polarity Dial Experiment (Protocol A)
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
- Optional Weights & Biases logging activates when both `wandb_project` and `wandb_api_key` are provided in the experiment config.
