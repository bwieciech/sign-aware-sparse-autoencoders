# Polarity Dial Experiment Design (Protocol A)

## Overview

This document describes the design of Protocol A ("Polarity Dial") for evaluating the Sign-Aware Gated Sparse Autoencoder (SA-GSAE) against baselines (ReLU SAE and standard Gated SAE) on synthetic data with asymmetric, signed features. The design focuses on:
- A synthetic data generator that produces inputs with signed feature coefficients whose positive and negative regimes follow different magnitude distributions.
- A unified experiment runner that trains multiple model variants under identical data and hyperparameters.
- A calibration and alignment evaluation module that computes separate scaling factors for positive and negative regimes along each ground-truth axis, using an axis-aware alignment that is fair to non-negative baselines that may allocate two latents per axis.
- A plotting layer that produces clear, publication-quality visualizations of calibration behavior.
- A structured output directory and file format for storing configurations, metrics, and plots.
- A sweep runner that produces (1) sparsity-fidelity trade-off curves (MSE vs \(L_0\)) over sparsity strengths and (2) an asymmetry sweep over generator parameters (the "dial").

The implementation will be added to the existing `sgnsae` package as a self-contained experiment module.

## Architecture

The experiment is split into four main subsystems:

1. Data generation:
   - Module: `sgnsae.experiments.polarity_dial.data`
   - Responsible for generating synthetic datasets with asymmetric signed features and storing auxiliary ground-truth information (feature directions, coefficients, signs).

2. Training and model orchestration:
   - Module: `sgnsae.experiments.polarity_dial.training`
   - Provides a thin wrapper around the existing training utilities (`train_sparse_autoencoder` components) adapted to in-memory synthetic data instead of JSON embeddings.
   - Instantiates specific model variants (`SAE`, `GatedSAE`, sign-aware `GatedSAE`), trains each, and records training metrics.

3. Evaluation and calibration:
   - Module: `sgnsae.experiments.polarity_dial.evaluation`
   - Implements axis-aware alignment, axis-level coefficient extraction, split-scale calibration estimation (\(\gamma_+\), \(\gamma_-\)), robustness diagnostics (correlation, coefficient MSE, \(R^2\)), and summary statistics.

4. Plotting and reporting:
   - Module: `sgnsae.experiments.polarity_dial.plotting`
   - Generates plots using `matplotlib` (and optionally `seaborn` if desired later) and writes summary reports and comparison artifacts.

A single CLI entry point coordinates these subsystems:
- Script: `sgnsae/scripts/run_polarity_dial_experiment.py`
- Uses `argparse` to load an experiment configuration (YAML or JSON), triggers data generation, trains specified model variants, runs evaluation, and exports artifacts into a structured output directory.

## Components and Interfaces

### Configuration Data Models

Configurations are modeled as dataclasses (or Pydantic models if strict validation is desired) that can be serialized to/from JSON. At minimum:

- `PolarityDialDataConfig`:
  - `seed: int`
  - `num_train: int`
  - `num_val: int`
  - `num_test: int`
  - `input_dim: int` (\(d\))
  - `num_features: int` (\(k\); number of underlying feature directions)
  - `activation_prob: float` (\(\rho\), probability each feature is active per sample)
  - `positive_sign_prob: float` (\(p_+\))
  - `lognormal_mu: float` (e.g., `0.0`)
  - `lognormal_sigma: float` (e.g., `0.5`)
  - `exponential_lambda: float` (e.g., `1.5`)
  - `noise_std: float` (\(\sigma\); standard deviation of Gaussian noise added to each sample)

- `PolarityDialModelVariantConfig`:
  - `name: str` (unique run name, used for directory naming)
  - `sae_class: str` (`"SAE"`, `"GatedSAE"`, `"SignedSoftThresholdSAE"`)
  - `sign_aware: bool` (for GatedSAE only)
  - `symmetric_magnitude: bool` (optional ablation for sign-aware GatedSAE)
  - `single_activation_offset: bool` (for sign-aware GatedSAE only)
  - `reset_activation_offsets_on_resample: bool` (for sign-aware GatedSAE only)
  - `resample_latents_epochs: Optional[int]` (optional; resample “dead” latents every N epochs)
  - `d_hidden: int` (dictionary width)
  - `sparsity_coeff: float` (lambda for sparsity)
  - `learning_rate: float`
  - `num_epochs: int`
  - `batch_size: int`
  - `adam_betas: Tuple[float, float]`
  - NOTE: “2x width baselines” are represented as separate model entries with `d_hidden = 2 * num_features`.

- `PolarityDialEvaluationConfig`:
  - `alignment_mode: str` (e.g., `"axis_aware_abs_cos"`)
  - `axis_aggregation_for_nonnegative: bool` (whether to aggregate +/- latents for non-negative models)
  - `normalize_decoder_for_eval: bool` (whether to use normalized decoder columns in cosine computations)
  - `min_axis_abs_cos: float` (minimum absolute cosine required to accept a latent as representing an axis in axis-aggregation mode, default `0.9`)
  - `min_regime_samples: int` (minimum samples in a regime to compute \(\gamma_+\) or \(\gamma_-\))
  - `max_points_per_regime_scatter: int` (subsample points for scatter plots)

- `PolarityDialSweepConfig`:
  - `sparsity_coeff_grid: List[float]` (optional; if provided, run MSE vs \(L_0\) sweeps)
  - `p_plus_grid: List[float]` (optional; if provided, run asymmetry sweeps)
  - `lognormal_sigma_grid: List[float]` (optional)
  - `exponential_lambda_grid: List[float]` (optional)
  - `sweep_mode: str` (one of: `"none"`, `"sparsity"`, `"asymmetry"`, `"both"`)

- `PolarityDialExperimentConfig`:
  - `output_dir: str`
  - `device: str` (`"auto"`, `"cpu"`, `"cuda"`)
  - `data: PolarityDialDataConfig`
  - `models: List[PolarityDialModelVariantConfig]`
  - `evaluation: PolarityDialEvaluationConfig`
  - `sweeps: PolarityDialSweepConfig`

### Data Generation Component

Module: `sgnsae.experiments.polarity_dial.data`

Function:

```python
def generate_polarity_dial_dataset(config: PolarityDialDataConfig, device: str) -> Dict[str, torch.Tensor]:
    ...
```

Outputs:
- `x_train: Tensor` of shape `(num_train, input_dim)`
- `x_val: Tensor` of shape `(num_val, input_dim)`
- `x_test: Tensor` of shape `(num_test, input_dim)`
- `feature_directions: Tensor` of shape `(num_features, input_dim)` (rows are unit-norm `u_j`)
- `coefficients_train: Tensor` of shape `(num_train, num_features)` with signed magnitudes `c_j(x) = s_j m_j`
- `coefficients_val: Tensor` of shape `(num_val, num_features)`
- `coefficients_test: Tensor` of shape `(num_test, num_features)`

Data generation procedure (per sample):
1. Sample a sign-agnostic feature activation mask:
   - For each feature \(j \in \{1,\dots,k\}\), sample `active_j ~ Bernoulli(activation_prob)`.
2. For each active feature, determine sign:
   - `s_j ~ Bernoulli(positive_sign_prob)`; map to \(+1\) if true, \(-1\) if false.
3. For each active feature, sample magnitude:
   - If \(s_j = +1\): `m_j ~ LogNormal(mu=lognormal_mu, sigma=lognormal_sigma)`.
   - If \(s_j = -1\): `m_j ~ Exponential(lambda=exponential_lambda)`.
4. Compute signed coefficient: `c_j = s_j * m_j`; inactive features have `c_j = 0`.
5. Generate ground-truth feature directions:
   - Sample `u_j` independently from `N(0, I_d)`, then normalize to unit length.
   - Stack into `feature_directions[j] = u_j`.
6. Form the noiseless signal:
   - `x_clean = sum_j c_j * u_j`, implemented as matrix multiplication: `x_clean = coefficients @ feature_directions`.
7. Add isotropic noise:
   - `eps ~ N(0, noise_std^2 I_d)`; `x = x_clean + eps`.

Random seeds:
- A single `seed` controls:
  - Feature directions.
  - All Bernoulli draws (`active_j`, `s_j`).
  - All magnitude draws and noise.

- Additionally, each *model variant* is trained with a deterministic *per-model seed* derived from `(data.seed, model_config.name)`:
  - This prevents “model ordering” effects when multiple models are trained in the same run (RNG state should not leak across models).
  - Recommended implementation: `model_seed = stable_hash(f\"{data.seed}:{model_name}\") % 2**31`.
  - The runner reseeds (Python, NumPy, Torch CPU, Torch CUDA) immediately before instantiating and training each model.
  - The derived `model_seed` is persisted in per-model artifacts (`training_metrics.json`).

Splits:
- Generation uses disjoint random draws for train, validation, and test sets.
- All splits share the same `feature_directions` and distributional parameters, but have independent coefficients and noise.

### Training and Model Orchestration Component

Module: `sgnsae.experiments.polarity_dial.training`

Responsibilities:
- Wrap the synthetic data in `TensorDataset`/`DataLoader` instances.
- Instantiate SAE variants using `PolarityDialModelVariantConfig`.
- Reuse `instantiate_model`, `run_training_step`, and related utilities from `sgnsae.scripts.train_sparse_autoencoder` where possible, but operate directly on tensors instead of JSON embeddings.
- Ensure deterministic DataLoader shuffling per model by passing an explicit `generator` seeded with `model_seed`.
- Persist `model_seed` so the run can be reproduced without relying on global RNG state.

Key function:

```python
def run_training_for_variant(
    experiment_config: PolarityDialExperimentConfig,
    data: Dict[str, torch.Tensor],
    model_config: PolarityDialModelVariantConfig,
) -> Dict[str, Any]:
    ...
```

Outputs per variant:
- Trained model instance.
- Training history (loss curves, sparsity statistics) as plain Python structures ready for JSON serialization.

Model widths:
- For fair comparison under the Protocol A focus on calibration (not width efficiency), recommended defaults:
  - `d_hidden` identical for all variants, e.g., `d_hidden = num_features` or a small multiple (e.g., `2 * num_features`) to avoid strong undercomplete constraints.
- Sign-aware settings:
  - For ReLU SAE: `sign_aware=False`.
  - For standard Gated SAE: `sae_class="GatedSAE"`, `sign_aware=False`.
  - For Sign-Aware Gated SAE: `sae_class="GatedSAE"`, `sign_aware=True`, `single_activation_offset=False` (dual offsets to emulate Bi-Jump dead zone), `reset_activation_offsets_on_resample=True`.

  - For SignedSoftThreshold SAE (signed-latent baseline):
    - `sae_class="SignedSoftThresholdSAE"` (or equivalent).
    - The implementation MUST be scale-invariant: use a normalized decoder `D_norm` in forward pass and compute `z = D_norm^T (x - b_dec) + b_enc` (or an equivalent tied parameterization) before soft-thresholding.

### Evaluation and Calibration Component

Module: `sgnsae.experiments.polarity_dial.evaluation`

Responsibilities:
- Align decoder columns with ground-truth feature directions in an axis-aware way.
- Extract axis-level scalar coefficient estimates \(\hat{c}_j(x)\) for each ground-truth axis \(u_j\), in a way that credits baselines that allocate two opposite-direction atoms.
- Compute split-scale calibration factors \(\gamma_+\) and \(\gamma_-\) at the axis level.
- Compute robust per-regime diagnostics (correlation, coefficient MSE, \(R^2\)).

Decoder preprocessing for evaluation:
1. Extract decoder directions as columns `D[:, i]`.
2. If `normalize_decoder_for_eval=true`, normalize each column to unit norm for evaluation only:
   - `D_eval[:, i] = D[:, i] / (||D[:, i]|| + eps)`.
3. Record in `alignment.json` whether normalization was applied.

Axis-aware alignment:
1. Compute signed cosine similarity:
   - `S_signed[i, j] = cos(D_eval[:, i], u_j)`.
2. Compute axis similarity:
   - `S_axis[i, j] = abs(S_signed[i, j])`.
3. Apply Hungarian matching on `-S_axis` to produce an initial one-to-one mapping from latents to axes:
   - Store `(latent_index, axis_index=j, signed_cos=S_signed[i,j], axis_cos=abs(S_signed[i,j]))`.
4. Derive an axis-level mapping for non-negative models when `axis_aggregation_for_nonnegative=true`:
   - For each axis `j`, select up to two latents:
     - `i_pos`: the best latent with `S_signed[i, j] >= 0` (aligned with `+u_j`),
     - `i_neg`: the best latent with `S_signed[i, j] < 0` (aligned with `-u_j`),
     - subject to `evaluation.min_axis_abs_cos` (default `0.9`).
   - This mapping is many-to-one: two latents may map to the same axis.
   - Store this mapping in `axis_alignment.json` (per axis: `latent_pos`, `latent_neg`, and their cosine values).

Coefficient estimation:
We compute axis-level predicted coefficients \(\hat{c}_j(x)\) on a held-out split (default test).

1. Ground-truth coefficients:
   - `c_true[:, j]` is taken directly from `coefficients_test` (signed, zero when inactive).

2. Predicted coefficients for sign-aware models:
   - For each axis `j`, choose its matched latent `i` from the one-to-one axis matching.
   - Let `a_i(x)` be the scalar activation for latent `i`.
   - Define:
     - `hat_c_j(x) = a_i(x) * S_signed[i, j]`
     - This corrects for cases where the decoder column is aligned to \(-u_j\) rather than \(+u_j\).
   - Record the sign correction factor per axis in `axis_alignment.json`.

3. Predicted coefficients for non-negative models (axis aggregation mode):
   - For each axis `j`, use `(i_pos, i_neg)` from `axis_alignment.json` when available.
   - Compute:
     - `hat_c_j(x) = a_{i_pos}(x) * S_signed[i_pos, j] + a_{i_neg}(x) * S_signed[i_neg, j]`
   - If only one of `i_pos` or `i_neg` is available, treat the missing side as zero.
   - This credits baselines that allocate separate atoms to \(+u_j\) and \(-u_j\).

4. Optional projection-based validation (debug mode):
   - Compute `x_hat = model(x)` and validate that projecting `x_hat` onto `u_j` correlates with `hat_c_j(x)`; record diagnostics to detect implementation errors.

Calibration factors:
For each axis `j`:
1. Split samples into regimes using `c_true[:, j]`:
   - positive regime: `c_true > 0`
   - negative regime: `c_true < 0`
2. For each regime, if `count < min_regime_samples`, set \(\gamma\) and regime metrics to `null`.
3. Otherwise compute closed-form least squares:
   - `gamma = (c_true * c_pred).sum() / (c_pred.pow(2).sum() + eps)`
4. Compute per-regime robust metrics on the same subset:
   - `coef_mse = mean((c_pred - c_true)^2)`
   - `pearson_r` between `c_pred` and `c_true`
   - `r2` (explained variance) with an `eps` guard for near-constant targets

Aggregate metrics:
- For each model:
  - Per-axis \(\gamma_+\), \(\gamma_-\), and derived metrics:
    - `gamma_diff = gamma_plus - gamma_minus`
    - `rel_error_plus = |gamma_plus - 1|`
    - `rel_error_minus = |gamma_minus - 1|`
  - Global summary:
    - Mean and standard deviation over matched pairs.
    - Percentiles (e.g., 25th, 50th, 75th) of `rel_error_plus` and `rel_error_minus`.
    - Mean and standard deviation over axes for `coef_mse_pos/neg`, `pearson_r_pos/neg`, and `r2_pos/neg`.

### Plotting and Reporting Component

Module: `sgnsae.experiments.polarity_dial.plotting`

Library:
- Use `matplotlib` for all plots.

General style:
- Background: white.
- Grid: light gray grid (`#e0e0e0`) on y-axis for readability.
- Fonts: default `matplotlib` sans-serif.
- Positive regime color: blue (`#1f77b4`).
- Negative regime color: red (`#d62728`).
- Ground-truth ideal or reference lines: dark gray (`#555555`).
- Sign-aware model highlight: green (`#2ca02c`) where applicable.

Plots to produce:

1. Calibration scatter plots (per model):
   - Figure: `calibration_scatter_[model_name].png`
   - Axes:
     - x-axis: predicted coefficient `hat_c_i(x)`.
     - y-axis: ground-truth coefficient `c_j(x)`.
   - Visual encoding:
     - Positive regime points (`c_j(x) > 0`): blue markers (`marker='o'`, `alpha=0.4`).
     - Negative regime points (`c_j(x) < 0`): red markers (`marker='x'`, `alpha=0.4`).
   - Reference:
     - Plot y = x line in dark gray as ideal calibration.
   - Sampling:
     - For readability, subsample points (e.g., at most ~5000 per regime) uniformly at random.

2. Gamma distribution plots (per model):
   - Figure: `gamma_hist_[model_name].png`
   - Subplots:
     - Top panel: histogram of \(\gamma_+\) across matched pairs.
     - Bottom panel: histogram of \(\gamma_-\) across matched pairs.
   - Colors:
     - \(\gamma_+\): blue bars with `alpha=0.7`.
     - \(\gamma_-\): red bars with `alpha=0.7`.
   - Reference:
     - Vertical dashed line at 1.0 indicating ideal calibration.

3. Gamma difference bar plot (multi-model comparison):
   - Figure: `gamma_diff_bar_comparison.png`
   - For each model variant, plot:
     - Mean `rel_error_plus` and `rel_error_minus` as two bars.
   - Colors:
     - For each model:
       - `rel_error_plus`: blue.
       - `rel_error_minus`: red.
     - Use grouped bars per model with slight horizontal offsets.
   - Error bars:
     - Use standard deviation or standard error as vertical error bars.

4. Latent-wise gamma comparison strip plot (optional):
   - Figure: `gamma_plus_minus_strip_[model_name].png`
   - For each matched pair, plot:
     - \(\gamma_+\) and \(\gamma_-\) as two points connected by a gray line.
   - Colors:
     - \(\gamma_+\): blue dot.
     - \(\gamma_-\): red dot.
   - Purpose:
     - Visually emphasize the asymmetry between positive and negative calibration per latent.

5. Training curves (per model):
   - Figure: `training_loss_[model_name].png`
   - Axes:
     - x-axis: epochs.
     - y-axis: loss (total loss and optionally reconstruction loss).
   - Colors:
     - Total loss: dark blue.
     - Reconstruction loss: light blue or orange.

6. Sparsity-fidelity trade-off curves (multi-model, sweep):
   - Figure: `mse_vs_l0_tradeoff.png`
   - For each model variant, plot points from a `sparsity_coeff_grid` sweep:
     - x-axis: average \(L_0\) on test
     - y-axis: test reconstruction MSE
   - Annotate points by sparsity coefficient or include a legend mapping to \(\lambda\).

7. Asymmetry sweep plots (multi-model, sweep):
   - Figure: `asymmetry_sweep_calibration.png`
   - Plot a calibration error summary (e.g., mean `rel_error_plus` and `rel_error_minus`) as a function of:
     - `p_plus` (and optionally separate panels for `exponential_lambda` or `lognormal_sigma`).

All plots should:
- Include titles indicating the model and variant (e.g., `"Sign-Aware Gated SAE: Calibration"`).
- Label axes with units where applicable.
- Include legends where multiple colors/markers are used.
- Use `bbox_inches="tight"` when saving to avoid clipped labels.

## Data Models (Metrics and Outputs)

The output directory for a run is:

```
[output_dir]/polarity_dial/[run_id]/
```

Where:
- `[output_dir]` is taken from `PolarityDialExperimentConfig.output_dir`.
- `[run_id]` is a timestamp-based or UUID-based identifier (e.g., `2025-07-15T12-30-00Z`).

Structure:

1. Configuration:
   - File: `config.json`
   - Contents:
     - Serialized `PolarityDialExperimentConfig`, including nested `data`, `models`, `evaluation`, and optional `sweeps` sections.

2. Environment metadata:
   - File: `environment.json`
   - Contents:
     - Python version, OS, torch version, CUDA availability, and key library versions (numpy, scipy, matplotlib, einops).

2. Data summary:
   - Directory: `data/`
   - Files:
     - `generator_config.json`: copy of `PolarityDialDataConfig`.
     - `feature_directions.pt`: Torch tensor of shape `(num_features, input_dim)` (float32).
     - `coefficients_stats.json`: summary statistics of ground-truth coefficients:
       - Mean, std, min, max per feature and per sign regime.
       - Overall sparsity (fraction of zeros).
     - `sampled_data.pt`: optional small subset of `(x, coefficients)` pairs for inspection, e.g., first 1024 training samples.

3. Per-model artifacts:
   - Directory: `models/[model_name]/`
   - Files:
     - `model_config.json`: serialized `PolarityDialModelVariantConfig`.
     - `model.pt`: serialized model (`torch.save`).
     - `training_metrics.json`:
       - `model_seed`: int (derived seed used for this model variant)
       - `epoch`: list[int]
       - `loss_total`: list[float]
       - `loss_reconstruction`: list[float]
       - `loss_sparsity`: list[float]
       - Any additional SAE-specific metrics (e.g., mean active latents).
    - `alignment.json`:
      - `alignment_mode`: e.g., `"axis_aware_abs_cos"`
      - `pairs`: list of `{latent_index, feature_index, signed_cosine}`
    - `axis_alignment.json`:
      - `axes`: list of objects with:
        - `feature_index: int`
        - `latent_pos: int | null`
        - `latent_neg: int | null`
        - `signed_cos_pos: float | null`
        - `signed_cos_neg: float | null`
        - `axis_cos_pos: float | null`
        - `axis_cos_neg: float | null`
      - `eval_decoder_normalized: bool`
      - `min_axis_abs_cos`: float (threshold used to accept a latent for an axis side)
      - `coverage_summary`: object with counts/fractions for pos/neg/both/neither coverage
    - `calibration.json`:
      - `per_pair`: list of objects with:
        - `latent_index: int`
        - `feature_index: int`
        - `gamma_plus: float`
        - `gamma_minus: float`
        - `rel_error_plus: float`
        - `rel_error_minus: float`
        - `n_pos: int`
        - `n_neg: int`
        - `coef_mse_pos: float | null`
        - `coef_mse_neg: float | null`
        - `pearson_r_pos: float | null`
        - `pearson_r_neg: float | null`
        - `r2_pos: float | null`
        - `r2_neg: float | null`
      - `summary`:
        - `mean_gamma_plus: float`
        - `mean_gamma_minus: float`
        - `mean_rel_error_plus: float`
        - `mean_rel_error_minus: float`
        - `std_rel_error_plus: float`
        - `std_rel_error_minus: float`
        - `num_pairs: int`
        - `mean_coef_mse_pos: float | null`
        - `mean_coef_mse_neg: float | null`
        - `mean_pearson_r_pos: float | null`
        - `mean_pearson_r_neg: float | null`
        - `mean_r2_pos: float | null`
        - `mean_r2_neg: float | null`
     - `plots/`:
       - `calibration_scatter_[model_name].png`
       - `gamma_hist_[model_name].png`
       - `gamma_plus_minus_strip_[model_name].png`
       - `training_loss_[model_name].png`

4. Cross-model comparison:
   - File: `comparison_summary.json`
   - Contents:
     - `models`: list of model names.
     - For each model:
       - The summary calibration metrics (copied from each model’s `calibration.json["summary"]`).
     - Optionally:
       - Differences relative to a chosen baseline (e.g., ReLU SAE).
   - File: `gamma_diff_bar_comparison.png`
     - Bar chart as described above.
   - File: `mse_vs_l0_tradeoff.json` (if sparsity sweep enabled)
     - Contains per-model sweep points (sparsity coeff, test MSE, test \(L_0\)).
   - File: `mse_vs_l0_tradeoff.png` (if sparsity sweep enabled)
   - File: `asymmetry_sweep_summary.json` (if asymmetry sweep enabled)
     - Contains per-model results keyed by asymmetry parameters.
   - File: `asymmetry_sweep_calibration.png` (if asymmetry sweep enabled)

All JSON files use snake_case keys for consistency with the Python config dataclasses.

## Error Handling

- Configuration validation:
  - Invalid or inconsistent configurations (e.g., negative `activation_prob`, `num_features > input_dim`) should raise `ValueError` with clear messages before any computation.
  - Missing required config fields should be caught when parsing the configuration file.

- Dependency validation:
  - Fail fast if required optional dependencies are missing (e.g., `einops.einsum` if used), with a clear error message pointing to installing `requirements.txt`.
- Data generation:
  - If `num_train`, `num_val`, or `num_test` is zero or negative, raise `ValueError`.
  - If `noise_std < 0`, raise `ValueError`.
- Training:
  - If the model diverges numerically (e.g., NaNs in loss), stop training early, log an error message, and still persist partial training metrics with a flag `{"terminated_early": true}` in `training_metrics.json`.
- Evaluation:
  - If a matched pair has too few samples in a regime (e.g., fewer than 5 positive or negative instances), skip computing \(\gamma_+\) or \(\gamma_-\) for that pair and record `null` in `calibration.json` for the missing value.
  - Alignment failures (e.g., due to incorrect shapes) should raise `RuntimeError` with diagnostic details.
- File I/O:
  - Ensure output directories are created with `os.makedirs(..., exist_ok=True)`.
  - Use atomic writes where feasible (e.g., write to a temp file and rename) to reduce risk of corrupted JSON on failure.

## Testing Strategy

Tests will be implemented using `pytest` in the `tests/` package, focusing on:

1. Data generation tests:
   - Verify that generated tensors have the expected shapes and device.
   - Check that feature directions are unit norm (within numerical tolerance).
   - Empirically verify distributional properties:
     - Positive coefficients approximate a log-normal distribution (mean and variance within expected ranges).
     - Negative coefficients approximate an exponential distribution when signs are flipped.
   - Confirm that setting the same `seed` reproduces identical data (bitwise equality).

2. Calibration and alignment tests:
   - Construct small toy datasets with known feature directions and coefficients and confirm that the alignment procedure recovers the expected mapping when decoder columns exactly equal ground-truth directions.
   - For a trivial identity-model case (decoder equals ground-truth directions and latent activations equal coefficients), check that \(\gamma_+ \approx 1\) and \(\gamma_- \approx 1\) within tight tolerances.

3. End-to-end smoke tests:
   - Run a very small experiment (tiny dataset, few epochs) through the CLI entry point and assert that:
     - `config.json`, `data/generator_config.json`, per-model `model_config.json`, `training_metrics.json`, `alignment.json`, and `calibration.json` are created.
     - Plots are generated and non-empty.

4. Error handling tests:
   - Ensure invalid configurations raise the expected exceptions with informative messages.
   - Ensure that evaluation gracefully handles regimes with too few samples by emitting `null` for missing gamma values.

5. Reproducibility tests (order independence):
   - Run the same tiny config twice with the model list permuted.
   - Assert per-model summaries match within tolerances.

6. Scale-invariance tests for SignedSoftThreshold baseline:
   - Scale the raw decoder weights by a constant factor.
   - Assert reconstructions and calibration metrics are unchanged (within tolerance) when using `D_norm`.

7. Environment artifact tests:
   - Assert `environment.json` exists and contains expected keys.

This testing strategy is designed to give high confidence that the Protocol A implementation correctly generates data, trains models, computes calibration metrics, and writes outputs in the expected structure.
