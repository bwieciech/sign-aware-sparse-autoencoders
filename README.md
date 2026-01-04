# embeddings-sparse-autoencoders

## Polarity Dial Experiment (Protocol A)

This experiment trains SAEs on a synthetic dataset where each ground-truth feature can appear with either sign, but with asymmetric sign frequency and asymmetric magnitude distributions (LogNormal for positives, Exponential for negatives). The goal is to reproduce the Protocol A results in the paper by running a small set of predefined configs (seeds 0, 1, 2), then confirming that the expected artifacts were produced and that the sign-aware gated model behaves correctly.

### 0) Prerequisites

1) Install dependencies:
```bash
.venv/bin/pip install -r requirements.txt
```

2) Sanity check the repository (recommended before long runs):
```bash
.venv/bin/python -m unittest
```

### 1) What you will run (exact order)

You will run three configs, one per seed:

1) `configs/polarity_dial_paper_seed0.json`
2) `configs/polarity_dial_paper_seed1.json`
3) `configs/polarity_dial_paper_seed2.json`

Each config trains three models:

- `relu_sae` (SAE, non-gated baseline)
- `gated_sae` (GatedSAE, sign_aware=false)
- `sign_aware_gated_sae` (GatedSAE, sign_aware=true)

All runs write into:

- `output_dir` from the config (defaults in the provided configs: `assets/polarity_dial_paper`)
- a unique run directory under: `<output_dir>/polarity_dial/<run_id>/`

Notes:
- `device: "auto"` uses CUDA if available, otherwise CPU.
- If you need CPU only, set `"device": "cpu"` in the config.

### 2) Run the experiments

Run seed 0:
```bash
.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed0.json
```

Run seed 1:
```bash
.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed1.json
```

Run seed 2:
```bash
.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed2.json
```

For each run, the script prints the output directory at the end:
```
Run complete. Artifacts written to: <run_dir>
```
Copy that path somewhere. You will use it in the verification steps below.

### 3) Verify outputs (required artifacts checklist)

For each `<run_dir>` produced above, verify the following artifacts exist.

#### 3.1 Run-level artifacts (in `<run_dir>/`)

- `config.json` (resolved config with secrets redacted)
- `comparison_summary.json` (aggregated evaluation summaries for all models in this run)
- `gamma_diff_bar_comparison.png` (bar chart comparing gamma calibration error across models)

#### 3.2 Data export artifacts (in `<run_dir>/data/`)

- `generator_config.json` (the dataset generator parameters)
- `feature_directions.pt` (ground-truth feature direction matrix)
- `coefficients_stats.json` (summary statistics of generated coefficients)
- `sampled_data.pt` (small preview of samples and coefficients)

#### 3.3 Per-model artifacts (for each model `relu_sae`, `gated_sae`, `sign_aware_gated_sae`)

Each model has a directory:
`<run_dir>/models/<model_name>/`

Inside, verify:

- `model.pt` (trained model checkpoint)
- `model_config.json` (exact model hyperparameters used)
- `training_metrics.json` (loss curves)
- `alignment.json` (latent-to-ground-truth matching)
- `calibration.json` (per-pair gamma metrics and summary)

Plots (in `<run_dir>/models/<model_name>/plots/`):
- `calibration_scatter_<model_name>.png`
- `gamma_hist_<model_name>.png`
- `gamma_plus_minus_strip_<model_name>.png`
- `training_loss_<model_name>.png`

If any of these artifacts are missing, treat the run as incomplete and fix the underlying error before proceeding.

### 4) What results to check (quick interpretation guide)

Open `<run_dir>/comparison_summary.json`. This file contains, for each model, a `summary` object derived from `calibration.json`. At minimum, confirm:

- `sign_aware_gated_sae` produces meaningful calibration on both sides (both `gamma_plus` and `gamma_minus` are present for many matched pairs).
- `gated_sae` and `relu_sae` are expected to struggle on the negative side unless they effectively allocate separate atoms for the two directions (non-negative latents cannot directly represent negative coefficients along the same axis).

Then open `<run_dir>/gamma_diff_bar_comparison.png` and confirm it visually reflects the above behavior for that seed.

Repeat for seeds 0, 1, and 2.

### 5) Run the Signed Symmetric Gated SAE ablation (paper baseline)

The paper describes an ablation that keeps sign-aware gating but enforces symmetric magnitude scaling across polarities (effectively $r^- = r^+$). This is implemented via a config flag on `GatedSAE`:

- `symmetric_magnitude: true`

For paper-quality results, run this ablation for all three seeds (0, 1, 2), so results are not dependent on a lucky initialization or dataset draw.

Recommended approach (easiest to compare): create three ablation configs by copying each seed config and adding one extra model entry.

1) Copy configs:
```bash
cp configs/polarity_dial_paper_seed0.json configs/polarity_dial_paper_seed0_ablation.json
cp configs/polarity_dial_paper_seed1.json configs/polarity_dial_paper_seed1_ablation.json
cp configs/polarity_dial_paper_seed2.json configs/polarity_dial_paper_seed2_ablation.json
```

2) In each `*_ablation.json`, add the model entry below to the `"models"` list (fourth model). Do not change the `"data"` section so the dataset stays identical for that seed.

Example model entry:
```json
{
  "name": "sign_aware_gated_sae_symmetric_mag",
  "sae_class": "GatedSAE",
  "sign_aware": true,
  "symmetric_magnitude": true,
  "single_activation_offset": false,
  "reset_activation_offsets_on_resample": true,
  "d_hidden": 128,
  "sparsity_coeff": 0.001,
  "learning_rate": 0.0001,
  "num_epochs": 50,
  "batch_size": 1024,
  "adam_betas": [0.9, 0.999]
}
```

3) Run the ablation configs (same order as the base runs):

Seed 0 ablation:
`bash
.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed0_ablation.json
`

Seed 1 ablation:
`bash
.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed1_ablation.json
`

Seed 2 ablation:
`bash
.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed2_ablation.json
`

Expected outputs: the run should produce the same artifacts as the base runs, plus one additional model directory:
`<run_dir>/models/sign_aware_gated_sae_symmetric_mag/`

When verifying results, compare (within the same seed) the calibration summaries and plots for:
- `sign_aware_gated_sae` (full model, independent $r^+, r^-$)
- `sign_aware_gated_sae_symmetric_mag` (ablation, forced $r^- = r^+$)

This paired comparison per seed is the intended evidence for the claim that independent $(r^+, r^-)$ improves calibration under asymmetric magnitude distributions.

### 6) Troubleshooting

- If you see NaNs during training, the training loop will terminate early and mark `training_metrics.json` with `terminated_early`. Reduce the learning rate or increase `noise_std` slightly to stabilize the dataset.
- If you expected CUDA but the run used CPU, confirm `device` is `"auto"` or `"cuda"` and that `torch.cuda.is_available()` is true in your environment.
- If `sign_aware: true` is set, the config must specify `reset_activation_offsets_on_resample` and `single_activation_offset` (even if you do not plan to resample)._

### Paper reproduction configs
- Configs live under `configs/` with exact parameters from the draft:
  - `.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed0.json`
  - `.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed1.json`
  - `.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_seed2.json`
- Aggregate the metrics in `comparison_summary.json` across the three runs per model to match paper-reported numbers.

### Notes
- Requirements now include `matplotlib` and `scipy` for plotting and Hungarian alignment.
- Requirements now include `PyYAML` for YAML experiment configs.
- Optional Weights & Biases logging activates when both `wandb_project` and `wandb_api_key` are provided in the experiment config.

---

## Directional Deviation Detection Experiment (Protocol B)

This experiment trains SAE variants under a latent budget constraint (`d_hidden < num_channels`) and evaluates direction-aware anomaly detection (“too high” vs “too low”) at a fixed false positive rate.

### 0) Quickstart (tiny run, sanity check only)

1) Run the tiny config:
```bash
.venv/bin/python -m sgnsae.scripts.run_directional_deviation_experiment --config-path configs/directional_deviation_tiny.json
```

2) Find the run directory printed at the end:
```
Run complete. Artifacts written to: <run_dir>
```

This tiny run is intended to verify the pipeline and artifact generation, but it is not the run you should use to produce paper results.

### 1) Paper run (Protocol B results)

This repository includes a paper-oriented config that matches the Protocol B setup in `main.tex`:
- `N=1024` channels with standard normal data (`x ~ N(0, I)`),
- directional anomalies as `±Δ` shifts on single channels,
- undercomplete width (`H < N`) swept over multiple budgets,
- Directional Recall @ fixed `FPR=1%`.

Run the paper config:
```bash
.venv/bin/python -m sgnsae.scripts.run_directional_deviation_experiment --config-path configs/directional_deviation_paper.json
```

This single command runs a width sweep and a seed sweep (configured in the JSON). It writes outputs under:
`assets/directional_deviation_paper/directional_deviation/<run_id>/`. If you instead override `output_dir` to `assets/directional_deviation`, the runner now places artifacts directly under `assets/directional_deviation/<run_id>` instead of nesting another `directional_deviation` folder.

The run directory contains:
- One subdirectory per width: `<run_dir>/width_<H>/`
- Within each width, one subdirectory per seed: `<run_dir>/width_<H>/seed_<seed>/`

For paper plots and numbers, use:
- `<run_dir>/plots/recall_vs_width.png` and `<run_dir>/plots/coverage_vs_width.png`
- `<run_dir>/per-width metrics.json` (mean/std metrics per width, aggregated over seeds)

### 2) What the config controls

- Dataset: `data.num_channels` (`N`), normal sample counts, anomaly counts (`num_test_high_per_channel`, `num_test_low_per_channel`), anomaly streaming/materialization (`data.anomaly_eval_mode`), and symmetry-breaking normal modes (`data.normal_mode`, `data.channel_stds`, `data.non_gaussian_dist`).
- Anomalies: `anomaly.delta` (`Δ`), `anomaly.shift_mode` (`single_channel`, `subset`, or `feature_direction`). For feature directions: `anomaly.direction_source`, `anomaly.num_directions`, `anomaly.reference_model_name`, `anomaly.direction_selection`, `anomaly.direction_latent_indices`, `anomaly.normalize_directions`, and per-direction counts (`anomaly.num_test_high_per_direction`, `anomaly.num_test_low_per_direction`).
- Calibration: `thresholds.latent_quantile` (per-latent thresholds), `thresholds.score_fpr` (fixed-FPR score thresholds).
- Scoring: `scoring.pooling` (`sum` / `max` / `topk`), `scoring.topk_k` (for `topk`), and `scoring.target_restricted_scoring` (single-channel only).
- Models: `models[]` (each has `name`, `sae_class`, `sign_aware`, `d_hidden`, training hyperparameters).
- Sweeps:
  - Width sweep: add `"width_sweep": [32, 64, 128]` at the run root to run separate `width_<H>/` subruns.
  - Seed sweep: set `data.seeds: [0, 1, 2]` to run separate `seed_<seed>/` subruns.

### 3) Outputs to expect

Run root (`<run_dir>/`):
- `config.json`
- `comparison_summary.json`
- `plots/` (PNG plots; leaf runs include `leaf_summary.png`)

Data export (`<run_dir>/data/`):
- `generator_config.json`
- `anomaly_manifest.json`
- `anomaly_directions.pt` (only for `shift_mode="feature_direction"`)
- `sampled_data.pt`

Per model (`<run_dir>/models/<model_name>/`):
- `model.pt`
- `model_config.json`
- `training_metrics.json`
- `decoder_alignment.json` (latent-to-channel alignment diagnostics)
- `thresholds/thresholds_metadata.json`
- `thresholds/score_thresholds.json`
- `evaluation.json`
- Baseline models also write per-latent direction assignments to `thresholds/latent_directions.pt`
- Sign-aware models also write `thresholds/tau_plus.pt` and `thresholds/tau_minus.pt`

Width sweep only:
- Each width subrun directory (`<run_dir>/width_<H>/`) includes `metrics.json` and its own full set of run artifacts.
- `<run_dir>/per-width metrics.json` collates per-width `metrics.json` (and uses mean/std when a seed sweep is enabled).
- `plots/recall_vs_width.png`, `plots/coverage_vs_width.png`, and `plots/unique_channels_vs_width.png` summarize performance and alignment vs width from `comparison_summary.json`.

### 4) Verify outputs (required artifacts checklist)

For the paper run `<run_dir>`:
- `config.json`
- `comparison_summary.json`
- `per-width metrics.json`
- `plots/recall_vs_width.png`
- `plots/coverage_vs_width.png`

For each width subrun (`<run_dir>/width_<H>/`):
- `config.json`
- `comparison_summary.json` (seed sweep summary, if `data.seeds` is enabled)
- `metrics.json`

For each leaf run (`<run_dir>/width_<H>/seed_<seed>/`):
- `data/generator_config.json`
- `data/anomaly_manifest.json`
- `data/anomaly_directions.pt` (only for `shift_mode="feature_direction"`)
- `data/sampled_data.pt`
- `comparison_summary.json`
- `plots/leaf_summary.png`
- `plots/score_histograms.png`
- `plots/decoder_purity_hist.png`
- `models/<model_name>/model.pt`
- `models/<model_name>/training_metrics.json`
- `models/<model_name>/thresholds/score_thresholds.json`
- `models/<model_name>/evaluation.json`
- `models/<model_name>/decoder_alignment.json`
