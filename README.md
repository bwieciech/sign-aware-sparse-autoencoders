# sign-aware-sparse-autoencoders

Code for the *[Sign-Aware Gated Sparse Autoencoders: Modeling Anticorrelated Features with Bi-Jump-ReLU Activations](https://arxiv.org/abs/2605.28149)* paper (arXiv:2605.28149).

Please cite as:

```bibtex
@misc{wieciech2026signaware,
  author        = {Wieciech, Bartosz and Awrahman, Zmnako and Czelej, Marcin and
                   Jaramillo Velasquez, Victor Hugo and Stobieniecka, Wioletta},
  title         = {Sign-Aware Gated Sparse Autoencoders: Modeling Anticorrelated Features with {Bi-Jump-ReLU} Activations},
  year          = {2026},
  eprint        = {2605.28149},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2605.28149},
  url           = {https://arxiv.org/abs/2605.28149}
}
```

## Polarity Dial Experiment (Protocol A)

This experiment trains SAEs on a synthetic dataset where each ground-truth feature can appear with either sign, but with asymmetric sign frequency and asymmetric magnitude distributions (LogNormal for positives, Exponential for negatives). The goal is to reproduce the Protocol A results in the paper by running the paper config (16 seeds, 0–15), then confirming that the expected artifacts were produced and that evaluation is axis-aware (so non-negative baselines that learn both \(+u_j\) and \(-u_j\) are credited correctly).

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

Run the paper config once:

- `configs/polarity_dial_paper.json`

This config is already paper-grade for the **Protocol A table** (fixed operating point at `sparsity_coeff = 1e-3`) and runs **16 seeds** (0–15) via `data.seed: [0, ..., 15]`.

It trains the following model variants (this is the default set used for paper-quality Protocol A comparisons):

- `relu_sae` (SAE, non-gated baseline, width = num_features)
- `relu_sae_2x` (SAE, width doubled to allow separate +u_j and -u_j atoms)
- `gated_sae` (GatedSAE, sign_aware=false, width = num_features)
- `gated_sae_2x` (GatedSAE, sign_aware=false, width doubled)
- `signed_soft_threshold` (SignedSoftThresholdSAE, signed-latent baseline, width = num_features)
- `sign_aware_gated_sae` (GatedSAE, sign_aware=true, width = num_features)

If you only want the “core three” models (relu_sae, gated_sae, sign_aware_gated_sae), remove the *_2x variants and signed_soft_threshold from the config.

All runs write into:

- `output_dir` from the config (defaults in the provided configs: `assets/polarity_dial_paper`)
- a unique run directory under: `<output_dir>/polarity_dial/<run_id>/`

Notes:
- `device: "auto"` uses CUDA if available, otherwise CPU.
- If you need CPU only, set `"device": "cpu"` in the config.

- Axis-aware evaluation uses `evaluation.min_axis_abs_cos` (default 0.9) to prevent spurious axis aggregation: any +/- atom below the threshold is treated as absent for that axis.
- For reproducibility, each model is trained with a deterministic per-model seed derived from (data.seed, model.name), so results do not depend on model ordering; this is logged in `training_metrics.json`.

**Important (Protocol A sweep figure):** `paper/generate_figures.py` expects the Protocol A run directories to contain numeric subrun folders (`0/`, `1/`, …) corresponding to a **sparsity-coefficient grid**. The provided `configs/polarity_dial_paper.json` is the fixed operating point used for the table; it will **not** create those numeric subruns.

To regenerate `paper/protocol_a_sweep_tradeoff.png` from scratch, create a sweep config by copying `polarity_dial_paper.json` and changing each model's `sparsity_coeff` from a scalar to a list of 64 log-spaced values in `[1e-5, 1e-2]` (as described in `paper/main.tex`).

Example helper to print a JSON-ready list:
```bash
python - <<'PY'
import numpy as np
grid = np.logspace(-5, -2, 64)
print("[" + ", ".join(f"{x:.8g}" for x in grid) + "]")
PY
```

Then run the sweep config; each per-seed run will contain `0/`, `1/`, … subruns with `comparison_summary.json` files.

**Note (Signed Soft-Threshold seeds):** the paper reports Signed Soft-Threshold SAE metrics aggregated over seeds 0–7 (see `paper/main.tex`). The provided config trains it over all 16 seeds; to match paper-reported numbers exactly, compute aggregates using only seeds 0–7 (or run a dedicated `signed_soft_threshold`-only config with `data.seed: [0, ..., 7]`).

### 2) Run the experiments

Run the paper config:
```bash
.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper.json
```

At the end, the script prints where artifacts were written.

- If `data.seed` is a single integer, you will see:
`
Run complete. Artifacts written to: <run_dir>
`

- If `data.seed` is a list (as in the paper config), you will see:
```
Runs complete. Artifacts written to:
- <run_dir_seed0>
- <run_dir_seed1>
- ...
```

For each run, the script prints the output directory at the end:
```
Run complete. Artifacts written to: <run_dir>
```
Copy that path somewhere. You will use it in the verification and figure regeneration steps.

### 3) Verify outputs (required artifacts checklist)

For each `<run_dir>` produced above, verify the following artifacts exist.

#### 3.1 Run-level artifacts (in `<run_dir>/`)

- `config.json` (resolved config with secrets redacted)
- `environment.json` (Python, torch, CUDA, and key dependency versions for reproducibility)
- `comparison_summary.json` (aggregated evaluation summaries for all models in this run)
- `gamma_diff_bar_comparison.png` (bar chart comparing gamma calibration error across models)

#### 3.2 Data export artifacts (in `<run_dir>/data/`)

- `generator_config.json` (the dataset generator parameters)
- `feature_directions.pt` (ground-truth feature direction matrix)
- `coefficients_stats.json` (summary statistics of generated coefficients)
- `sampled_data.pt` (small preview of samples and coefficients)

#### 3.3 Per-model artifacts (for each model listed in the config)

Each model has a directory:
`<run_dir>/models/<model_name>/`

Inside, verify:

- `model.pt` (trained model checkpoint)
- `model_config.json` (exact model hyperparameters used)
- `training_metrics.json` (loss curves)
- `alignment.json` (latent-to-ground-truth matching)
- `axis_alignment.json` (axis-aware mapping, including two-atom aggregation info for non-negative models)
- `calibration.json` (per-pair gamma metrics and summary)

Plots (in `<run_dir>/models/<model_name>/plots/`):
- `calibration_scatter_<model_name>.png`
- `gamma_hist_<model_name>.png`
- `gamma_plus_minus_strip_<model_name>.png`
- `training_loss_<model_name>.png`

If any of these artifacts are missing, treat the run as incomplete and fix the underlying error before proceeding.

If sweeps are enabled in the config, verify these run-level artifacts exist:
- `mse_vs_l0_tradeoff.json` and `mse_vs_l0_tradeoff.png` (sparsity-fidelity sweep)
- `asymmetry_sweep_summary.json` and `asymmetry_sweep_calibration.png` (asymmetry sweep)

### 4) What results to check (quick interpretation guide)

Open `<run_dir>/comparison_summary.json`. This file contains, for each model, a `summary` object derived from `calibration.json`. At minimum, confirm:

- `sign_aware_gated_sae` produces meaningful calibration on both sides (both `gamma_plus` and `gamma_minus` are present for many matched pairs).
- `gated_sae` and `relu_sae` may struggle on the negative side at fixed width unless they effectively allocate separate atoms for the two directions.
- When `relu_sae_2x` or `gated_sae_2x` are enabled, they should improve negative-regime calibration if they learn both \(+u_j\) and \(-u_j\), and the axis-aware evaluation should reflect that via `axis_alignment.json`.
- Use the per-regime diagnostics in `calibration.json` (`coef_mse_pos/neg`, `pearson_r_pos/neg`, `r2_pos/neg`, and `n_pos/n_neg`) to detect ill-conditioned fits.

Then open `<run_dir>/gamma_diff_bar_comparison.png` and confirm it visually reflects the above behavior for that seed.

Repeat for seeds 0, 1, and 2.

### 5) Run the Signed Symmetric Gated SAE ablation (paper baseline)

The paper describes an ablation that forces symmetric magnitude scaling across polarities (effectively r^- = r^+) while keeping sign-aware gating. This is implemented in the code via:

- `symmetric_magnitude: true` in the `GatedSAE` config.

For paper-quality results, run this ablation for the full seed sweep (0–15), so results are not dependent on a lucky initialization or dataset draw.

Recommended approach (easiest to compare): create one ablation config by copying the paper config and adding one extra model entry.

1) Copy config:
```bash
cp configs/polarity_dial_paper.json configs/polarity_dial_paper_ablation.json
```

2) In `configs/polarity_dial_paper_ablation.json`, add the model entry below as an additional entry in the `"models"` list (seventh model if you keep the default six). Do not change the `"data"` section so the dataset stays identical for each seed.

(Optional but recommended) Set a unique `"run_id"` at the top level (e.g. `"polarity_dial_paper_ablation"`) so the ablation outputs are easy to distinguish from the base run.

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

3) Run the ablation config:

```bash
.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper_ablation.json
```

Expected outputs: each run directory will contain an additional model directory:
`<run_dir>/models/sign_aware_gated_sae_symmetric_mag/`

When verifying results, compare (within the same seed) the calibration summaries and plots for:
- `sign_aware_gated_sae` (full model, independent $r^+, r^-$)
- `sign_aware_gated_sae_symmetric_mag` (ablation, forced $r^- = r^+$)

This paired comparison per seed is the intended evidence for the claim that independent $(r^+, r^-)$ improves calibration under asymmetric magnitude distributions.

### 6) Troubleshooting

- If you see NaNs during training, the training loop will terminate early and mark `training_metrics.json` with `terminated_early`. Reduce the learning rate or increase `noise_std` slightly to stabilize the dataset.
- If you expected CUDA but the run used CPU, confirm `device` is `"auto"` or `"cuda"` and that `torch.cuda.is_available()` is true in your environment.
- If `sign_aware: true` is set, the config must specify `reset_activation_offsets_on_resample` and `single_activation_offset` (even if you do not plan to resample).

### Paper reproduction configs

- Paper operating point (Protocol A table, fixed `sparsity_coeff = 1e-3`):
  - `.venv/bin/python -m sgnsae.scripts.run_polarity_dial_experiment --config-path configs/polarity_dial_paper.json`
  - This runs seeds 0–15 and writes one run directory per seed.
  - Aggregate metrics across the per-seed `comparison_summary.json` files to match paper-reported means/std (with the Signed Soft-Threshold caveat noted above).

- Protocol A sweep figure (`paper/protocol_a_sweep_tradeoff.png`):
  - Create a sweep config as described in Section 1 (list-valued `sparsity_coeff` grid) and run it.
  - The figure script expects per-seed run directories containing numeric subruns `0/`, `1/`, … with `comparison_summary.json`.

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

Protocol B in `paper/main.tex` includes two related benchmarks:

- **Fig. 5a (coordinate anomalies):** single-channel shifts under symmetry-breaking normals.
- **Fig. 5b / Table (feature-direction anomalies):** shifts along decoder-derived feature directions, with localized pooling variants.

The repository provides separate paper configs for these variants:

- `configs/directional_deviation_paper_5a_v2.json` (Fig. 5a: symmetry-breaking coordinate anomalies)
- `configs/directional_deviation_paper_5b_v2_max.json` (Fig. 5b/Table: revised protocol + max pooling)
- `configs/directional_deviation_paper_5b_sum.json` (Fig. 5b comparison: legacy sum pooling)
- Optional: `configs/directional_deviation_paper_5b_v2_topk.json` (revised protocol + top-k pooling)

Run one or more of:
```bash
.venv/bin/python -m sgnsae.scripts.run_directional_deviation_experiment --config-path configs/directional_deviation_paper_5a_v2.json

.venv/bin/python -m sgnsae.scripts.run_directional_deviation_experiment --config-path configs/directional_deviation_paper_5b_v2_max.json

.venv/bin/python -m sgnsae.scripts.run_directional_deviation_experiment --config-path configs/directional_deviation_paper_5b_sum.json
```

Each config runs a width sweep and a seed sweep (configured in the JSON). It writes outputs under:
`<output_dir>/directional_deviation/<run_id>/`

For the provided configs, `output_dir` already includes the paper variant name, e.g.
`assets/directional_deviation_paper/5b_feature_direction_v2_max/`, so the full run path is:
`assets/directional_deviation_paper/5b_feature_direction_v2_max/directional_deviation/<run_id>/`

The run directory contains:
- One subdirectory per width: `<run_dir>/width_<H>/`
- Within each width, one subdirectory per seed: `<run_dir>/width_<H>/seed_<seed>/`

For paper plots and numbers, use:
- `<run_dir>/plots/recall_vs_width.png` and `<run_dir>/plots/coverage_vs_width.png`
- `<run_dir>/per-width metrics.json` (mean/std metrics per width, aggregated over seeds)

Note: The min-recall CDF plot in `paper/generate_figures.py` expects **both** `5b_feature_direction_sum` and `5b_feature_direction_v2_max` to exist under a common `data_root/directional_deviation_paper/` hierarchy.

### 2) What the config controls

- Dataset: `data.num_channels` (`N`), normal sample counts, anomaly counts (`num_test_high_per_channel`, `num_test_low_per_channel`), anomaly streaming/materialization (`data.anomaly_eval_mode`), and symmetry-breaking normal modes (`data.normal_mode`, `data.channel_stds`, `data.non_gaussian_dist`).
- Anomalies: `anomaly.delta` (`Δ`), `anomaly.shift_mode` (`single_channel`, `subset`, or `feature_direction`). For feature directions: `anomaly.direction_source`, `anomaly.num_directions`, `anomaly.reference_model_name`, `anomaly.direction_selection`, `anomaly.direction_latent_indices`, `anomaly.normalize_directions`, and per-direction counts (`anomaly.num_test_high_per_direction`, `anomaly.num_test_low_per_direction`).
- Calibration: `thresholds.latent_quantile` (per-latent thresholds), `thresholds.score_fpr` (fixed-FPR score thresholds).
- Scoring: `scoring.pooling` (`sum` / `max` / `topk`), `scoring.topk_k` (for `topk`), and `scoring.target_restricted_scoring` (single-channel only).
- Models: `models[]` (each has `name`, `sae_class`, `sign_aware`, `d_hidden`, training hyperparameters).
- Sweeps:
  - Width sweep: add `"width_sweep": [32, 64, 128]` at the run root to run separate `width_<H>/` subruns.
  - Seed sweep: set `data.seed: [0, 1, 2]` to run separate `seed_<seed>/` subruns.

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
- `comparison_summary.json` (seed sweep summary, if `data.seed` is a list)
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

---

## Toy Model Geometry Validation Experiment (Protocol C)

This experiment trains baseline SAEs and the Sign-Aware Gated SAE on a synthetic 2D dataset with `k` antipodal feature pairs. It produces geometry-first diagnostics (unit-circle decoder plots), signed activation histograms, and a Pair Consolidation Rate sweep over within-pair correlation.

### 0) Prerequisites

1) Install dependencies:
```bash
.venv/bin/pip install -r requirements.txt
```

2) (Recommended) Sanity check before long runs:
```bash
.venv/bin/python -m unittest
```

Notes:
- Run commands from the repository root (so `sgnsae/` is importable).
- Budget at least ~45 minutes wall time per run, depending on hardware.

### 1) Paper runs (exact steps)

All commands below write artifacts into `output_dir` specified in each config (defaults to `assets/tmp_toy_model_geometry_validation_runs`), under:
`<output_dir>/toy_model_geometry_validation/<run_id>/`

**Important:** the provided paper configs already sweep **16 seeds** via `data.seed: [0, ..., 15]`. Some configs also sweep additional axes (e.g. `active_pairs_per_sample`). You should run each config **once**; the runner will create one run directory per seed (and per sweep setting) and will print a list of output paths at the end.

#### 1.1) k=8, correlation = -1 (qualitative + quantitative, 16 seeds)

```bash
.venv/bin/python -m sgnsae.scripts.run_toy_model_geometry_validation_experiment \
  --config-path configs/toy_model_geometry_validation_paper_k8_corrm1.json
```

#### 1.2) k=8, correlation sweep (16 seeds)

Note: this config was renamed to `configs/toy_model_geometry_validation_paper_sweep_k8_identifiable_lognormal.json`.

```bash
.venv/bin/python -m sgnsae.scripts.run_toy_model_geometry_validation_experiment \
  --config-path configs/toy_model_geometry_validation_paper_sweep_k8_identifiable_lognormal.json
```

#### 1.3) Superposition tolerance (k=16, active pairs per sample, 16 seeds × 4 settings)

This config sweeps `data.active_pairs_per_sample: [1, 2, 3, 4]` and will therefore produce **64** runs total (16 seeds × 4 active-pair settings):

```bash
.venv/bin/python -m sgnsae.scripts.run_toy_model_geometry_validation_experiment \
  --config-path configs/toy_model_geometry_validation_paper_superposition_tolerance_k16.json
```

Each command prints the final artifact directory (or a list of directories):
```
Run complete. Artifacts written to: <run_dir>
```

or:

```
Runs complete. Artifacts written to:
 - <run_dir_1>
 - <run_dir_2>
 - ...
```

### 2) Verify outputs (required artifacts checklist)

Run root (`<run_dir>/`):
- `config.json`
- `environment.json`
- `comparison_summary.json`
- Sweep only: `pair_consolidation_sweep.json` and `plots/pair_consolidation_sweep.png`

Data export (`<run_dir>/data/`):
- `generator_config.json`
- `feature_directions.pt`
- `coeffs_train.pt`, `coeffs_val.pt`, `coeffs_test.pt`
- `pair_active_mask_train.pt`, `pair_active_mask_val.pt`, `pair_active_mask_test.pt`
- `sampled_data.pt`

Per model (`<run_dir>/models/<model_name>/`):
- `model.pt`
- `model_config.json`
- `training_metrics.json`
- `evaluation.json`
- `plots/training_loss_<model_name>.png`
- `plots/unit_circle_<model_name>.png`
- Sign-aware only: `plots/bi_jump_hist_<model_name>_latent<i>.png`

## LLM SAE Benchmark

This section describes the general workflow for the real-model SAE benchmark: how a run is
configured, how the stage pipeline is driven with `--stages`, where artifacts land, and how to
aggregate results. It deliberately does not prescribe particular config files — the same process
applies to every config under `configs/`.

### 0) Prerequisites

1) Install dependencies from the repository root:
```bash
.venv/bin/pip install -r requirements.txt
```

2) Sanity check the repository before long runs:
```bash
.venv/bin/python -m unittest
```

3) Optional inputs:
- `datasets` is required for `data.backend: "hf"`.
- `zstandard` is required for reading `.jsonl.zst` shards directly.

4) Hardware: full-scale runs assume a CUDA device (`device: "cuda"`). `"cpu"` and `"auto"` are
accepted and are only practical for smoke runs; requesting CUDA on a machine without it fails fast
at device resolution rather than silently falling back.

Always launch the runner from the repository root. `output_dir` is used verbatim, so relative values
such as `./experiments/...` resolve against the current working directory. `./experiments/` is
gitignored, so run artifacts never pollute `git status`.

### 1) The single entrypoint

Every benchmark run — text or image, smoke or full scale — goes through one script:
```bash
.venv/bin/python -m sgnsae.scripts.run_llm_sae_benchmark <path/to/config.json>
```

- The config path may be passed positionally **or** via `--config-path`, but not both.
- JSON and YAML are both accepted.
- The runner echoes the fully resolved config, executes the requested stages, and ends with
  `Run complete. Artifacts written to: <output_dir>`.

`output_dir` is the run root: every artifact the run produces, including the activation cache, is
written underneath it. Two runs should only share an `output_dir` when you intend them to share
artifacts (see step 4).

### 2) How configs are organised

Configs live under `configs/<hookpoint>-<layer>/<backbone>/`, one directory per cell of the
benchmark grid (hook point × backbone). Each cell contains two kinds of config:

| Kind           | `variants`         | Purpose                                                          |
|----------------|--------------------|------------------------------------------------------------------|
| cache config   | empty (`[]`)       | builds the activation cache for that cell once                   |
| variant config | one entry          | trains and evaluates one SAE family across a sparsity sweep      |

A variant entry declares the SAE class, width, and the sweep axis: `sparsity_coeff` (a list gives a
`lambda` sweep) or `topk_k` (a list gives a `k` sweep). Every (variant, seed, sweep value) triple
becomes one subrun directory, so a config with 1 variant × 3 seeds × 8 sparsity values expands to 24
trained SAEs.

The schema is defined by the pydantic models in `sgnsae/experiments/llm_benchmark/config.py` and
rejects unknown keys, so the fastest way to author a run is to copy the closest existing config in
the cell you care about and edit these fields:

- `output_dir`, `device`, `seed` (a scalar or a list of seeds)
- `llm.model_id` / `llm.torch_dtype`, or the `vision` block when `modality: "image"`
- `data.*` — dataset source (step 3)
- `hook.hook_name`, `hook.layer_selector`, `hook.module_path` — which activations to cache
- `cache.*` — sequence length, number of sequences, shard size, train/val/test split
- `training.*` — steps, batch size, learning rate, dead-latent resampling
- `variants[]` — the SAE(s) and their sweep values
- `loss_recovered`, `feature_ablation`, `latent_examples` — optional evaluation extras

Local paths inside the `data` block (`data_files`, `data_files_glob`, `data_dir`) are resolved
relative to the config file. `output_dir` is not — it is relative to the working directory.

### 3) Dataset options

Option A (recommended): Hugging Face streaming
- `data.backend: "hf"` with `data.dataset_id`, `data.split`, `data.text_field`, and
  `data.streaming: true`.

Option B: local JSONL shards
- `data.backend: "local_jsonl"` with `data.data_files` or `data.data_files_glob` pointing at
  `.jsonl` or `.jsonl.zst` files.
- Prepare shards with:
```bash
.venv/bin/python scripts/prepare_openwebtext2_shards.py convert \
  --input-glob "/path/to/corpus/*.jsonl.zst" \
  --output-dir "/path/to/corpus_jsonl" \
  --records-per-shard 100000 \
  --text-field text
```

Image modality: set `modality: "image"` and supply a `vision` block instead of `llm`. The hook must
be `hook_name: "hidden_state"` with `module_path` unset, the data/cache blocks must be the image
variants, and `loss_recovered` must be disabled. For gated Hugging Face image datasets, accept the
dataset terms on the account you will use, authenticate with `huggingface-cli login`, and verify read
access before committing to a long run.

Fully offline execution (model weights and tokenizer already cached):
```bash
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/path/to/hf_cache
```

### 4) Run the stages: `--stages`

The pipeline has seven stages:

```
validate → cache → train → eval → examples → plot → summarize
```

Rules that matter in practice:
- The config's own `stages` list is the default; `--stages a,b,c` overrides it. Most configs in
  `configs/` ship with a minimal `stages` value, so **expect to pass `--stages` explicitly**.
- Stages execute in the order you list them; the runner does not reorder them. If one fails, the
  remaining requested stages are marked blocked in `run_status.json` and the run stops.
- `validate` must be the only stage when present. `--validate-only` is shorthand for it and cannot be
  combined with `--stages`.
- `train`, `eval`, `examples`, `plot` and `summarize` require at least one variant, so they cannot be
  run against a cache-only config.

**Step 1 — validate the config.** Cheap: writes `config.json`, `environment.json` and
`run_status.json` into the run root, then exits successfully.
```bash
.venv/bin/python -m sgnsae.scripts.run_llm_sae_benchmark <config> --validate-only
```

**Step 2 — build the activation cache once per cell**, using that cell's cache config:
```bash
.venv/bin/python -m sgnsae.scripts.run_llm_sae_benchmark <cache-config> --stages cache
```
This writes the activation shards and `activations_manifest.json` into
`<output_dir>/activations/`. This is the expensive, model-bound part of the benchmark; everything
downstream reads the cache instead of the backbone.

**Step 3 — point every variant run at that cache.** The `train` stage resolves the manifest strictly
as `<output_dir>/activations/activations_manifest.json` and otherwise fails with
`Missing activation manifest required for training`. A variant run therefore must either use the
same `output_dir` as the cache run, or have the cache linked into its own run root:
```bash
mkdir -p <variant-output-dir>
ln -s /abs/path/to/<cache-output-dir>/activations <variant-output-dir>/activations
```
Do this once per variant run root. Skipping it makes each variant re-cache activations from the
backbone.

**Step 4 — train and evaluate**, one variant config at a time:
```bash
.venv/bin/python -m sgnsae.scripts.run_llm_sae_benchmark <variant-config> --stages train

.venv/bin/python -m sgnsae.scripts.run_llm_sae_benchmark <variant-config> \
  --stages eval,examples,plot,summarize
```
or as a single invocation:
```bash
.venv/bin/python -m sgnsae.scripts.run_llm_sae_benchmark <variant-config> \
  --stages train,eval,examples,plot,summarize
```
Splitting `train` from the evaluation stages is the practical choice for long runs: evaluation,
example extraction and plotting can then be re-run and iterated on without retraining.

Repeat steps 3–4 for each variant config in the cell, then move to the next cell.

Resume and overwrite semantics:
- `resume: true` (with `overwrite: false`) reuses completed subruns and continues partially trained
  ones. `resume` and `overwrite` are mutually exclusive.
- The run root records a config fingerprint, so a changed config will not silently reuse stale
  artifacts; the comparison is reported under `resume_check` in `run_status.json`.
- With both `resume: false` and `overwrite: false`, re-running an already completed subrun is an
  error rather than a silent skip. Set `overwrite: true` to force regeneration.
- `cache.cache_overwrite` controls the activation cache independently of the run-level flags.

`examples` stage notes:
- Works for both text and image modalities when `latent_examples.enabled: true`.
- Writes a per-subrun `latent_examples.json` plus a run-level `latent_examples_index.json`.

### 5) Expected outputs to check

Run root (`<output_dir>/`):
- `run_status.json`
- `config.json`
- `environment.json`
- `comparison_summary.json`
- `results_summary.json`
- `activations/activations_manifest.json` (cache runs, or the symlink from step 4)
- `plots/` canonical run-level plots

Subrun directories (`<output_dir>/models/<variant_slug>/seed_<seed>/<lambda|k>_<value>/`):
- `model.pt`
- `model_config.json`
- `training_metrics.json`
- `evaluation.json`
- `gamma_metrics.json`
- `loss_recovered.json` (if enabled)
- `feature_ablation.json` (if enabled)
- `latent_examples.json` (if `latent_examples.enabled: true`)

Run-level optional artifact:
- `latent_examples_index.json` (if `latent_examples.enabled: true`)

View/decode `latent_examples.json`:
- Text mode stores highlighted strings using configurable markers (default `|>` and `<|`).
- Image mode stores each exemplar as `[full_png_b64, patch_png_b64]`.

For `modality: "text"`, `examples` is an object with `kind`, `latents`, `duplicate_pairs` and
`messages`. Sign-aware variants populate `latents` (each entry has `latent_index`,
`positive_examples` and `negative_examples`); non-negative baselines instead populate
`duplicate_pairs` with anti-correlated decoder pairs. Print a quick preview:
```bash
.venv/bin/python - <<'PY'
import json
path = "experiments/<hookpoint>/<backbone>/<variant>/models/<variant_slug>/seed_0/lambda_1e-03/latent_examples.json"
with open(path, "r") as file:
    payload = json.load(file)
print("modality:", payload["modality"], "kind:", payload["examples"]["kind"])
for entry in payload["examples"]["latents"][:3]:
    print("latent", entry["latent_index"])
    for example in entry["positive_examples"][:1] + entry["negative_examples"][:1]:
        print(f"  {example['activation']:+.2f}: {example['context'][:120]}")
PY
```

For `modality: "image"` the run falls back to the legacy per-latent payload, where `examples` is
keyed by latent index. Decode one image exemplar pair to PNG files:
```bash
.venv/bin/python - <<'PY'
import base64
import json
path = "experiments/<cell>/<variant>/models/<variant_slug>/seed_0/lambda_1e-03/latent_examples.json"
with open(path, "r") as file:
    payload = json.load(file)
latent_key = sorted(payload["examples"].keys())[0]
full_b64, patch_b64 = payload["examples"][latent_key][0]
with open("full_image.png", "wb") as full_file:
    full_file.write(base64.b64decode(full_b64))
with open("patch_image.png", "wb") as patch_file:
    patch_file.write(base64.b64decode(patch_b64))
print("wrote full_image.png and patch_image.png for latent", latent_key)
PY
```

### 6) Optional: SAE overhead microbenchmark

Measure per-step SAE compute overhead independently of a benchmark run:
```bash
.venv/bin/python -m sgnsae.scripts.benchmark_sae_overhead \
  --run-dir ./experiments/<cell>/<variant> \
  --device auto \
  --out-dir ./experiments/overhead \
  --plot
```
Widths and input dimension are read from the run directory when `--run-dir` is given, or can be set
explicitly with `--d-in` and `--widths`.

### 7) Troubleshooting

- `Missing activation manifest required for training` — the `cache` stage has not run for this
  `output_dir`, or the `activations/` symlink from step 4 is missing.
- `At least one variant must be provided for stages: [...]` — a cache-only config (empty `variants`)
  was used with `train`/`eval`/`examples`/`plot`/`summarize`.
- `If 'validate' is present in stages, it must be the only stage.` — drop the other stages, or use
  `--validate-only`.
- `--validate-only and --stages cannot both be provided.` — pick one.
- `resume=true and overwrite=true are mutually exclusive.` — set only one in the config.
- Refusal to overwrite a completed subrun — set `resume: true` to reuse it, or `overwrite: true` to
  recompute it.
- `Encountered legacy pre-status training_metrics.json` — the run directory predates the training-status
  schema. Mark trusted historical outputs as completed with
  `.venv/bin/python scripts/migrate_llm_training_status.py <output_dir>`, or rerun with
  `overwrite: true` to regenerate them.
- Config validation errors name the offending field path (for example `variants[].sae_class`); the
  authoritative field list is `sgnsae/experiments/llm_benchmark/config.py`.
