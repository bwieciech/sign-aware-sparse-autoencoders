# Parallel Multi-Sparsity Training for Protocol A (Polarity Dial)

## Overview

This design adds list-valued `sparsity_coeff` support to the Protocol A (Polarity Dial) experiment runner so that, for a fixed seed and model variant, multiple sparsity settings can be trained in parallel using the existing `n_instances` “slicing” semantics of `sgnsae/models`.

Key properties:
- Backward compatible when `sparsity_coeff` is a single float.
- No global constraints on list length: each model may use either a float or a list, and lists may have different lengths.
- Seeds still run sequentially; parallelism is within a seed by using `n_instances`.
- Evaluation and artifact generation are performed sequentially after training by slicing each trained multi-instance model into `n_instances=1` single-instance models.

## Configuration Semantics

### Model sparsity coefficient

Each `PolarityDialModelVariantConfig` already supports:
- `sparsity_coeff: float` (scalar sparsity mode), or
- `sparsity_coeff: List[float]` (list sparsity mode).

Validation remains:
- Float must be numeric and `>= 0`.
- List must be non-empty, numeric entries, each `>= 0`.
- Mixed usage across models is allowed.

### Coefficient subruns under a seed

For a single seed run:
- Define `subrun_count = max(list_len(model) for model in models)` where:
  - `list_len(model) = len(model.sparsity_coeff)` if list mode
  - `list_len(model) = 1` if scalar mode

The seed run directory contains `subrun_count` subdirectories named by integer index:
`0/`, `1/`, ..., `subrun_count - 1/`.

For a given coefficient index `i`:
- A model variant is included in subrun `i` if:
  - it is in list mode and `i < len(sparsity_coeff_list)`, using `sparsity_coeff = sparsity_coeff_list[i]`, OR
  - it is in scalar mode, using the same scalar value for every subrun index.

Models that are list mode but `i >= len(list)` are omitted from that subrun.

## Training Design

### Core idea

Training is modified to support `n_instances > 1` so one training loop can optimize multiple independent instances with different `sparsity_coeff` values in parallel.

For each model variant:
- If scalar mode: instantiate model with `n_instances = 1` and `sparsity_coeffs = [value]`.
- If list mode: instantiate model with `n_instances = len(value_list)` and `sparsity_coeffs = value_list`.

The input batch is shared across instances (broadcasted along the instance dimension), and `calculate_losses()` returns per-instance loss tensors. The training loop uses `losses.sum().backward()`; because parameters are sliced by instance, gradients should not mix across instances (validated by dedicated gradient tests).

### Initialization across instances (list mode)

In list mode, instance comparisons are most interpretable if differences are driven primarily by the sparsity coefficient, not by different random initializations. Therefore:
- Instantiate once with `n_instances = n`.
- Immediately after instantiation, copy the instance-0 parameter slice into all instances for every parameter that has a leading `n_instances` dimension.

This keeps starting weights identical across coefficient indices while preserving full independence during optimization.

### Metrics collection

The current training loop aggregates losses by taking `.mean()` over all instances. For list mode, this is changed to track per-instance metrics.

For each epoch, for each metric tensor returned by `calculate_losses()`:
- Accumulate an `epoch_total` tensor of shape `(n_instances,)` over batches.
- Record `epoch_mean = epoch_total / steps_per_epoch` as a Python list of floats.

Storage format:
- Scalar mode: keep the current format (`loss_total`, `loss_reconstruction`, `loss_sparsity`, `mean_active_latents`, optional `loss_aux` as `List[float]`).
- List mode: store per-instance metrics as `List[List[float]]` aligned by coefficient index:
  - `loss_total_by_instance: List[List[float]]` where outer length is `n_instances`, inner length is epochs.
  - similarly for reconstruction/sparsity/aux/mean_active_latents.

Additionally, store:
- `model_seed` (existing behavior),
- `sparsity_coeff_list` for list mode (the exact list used during training).

### Dead latent resampling bookkeeping

The training loop’s inactivity tracking is updated to be per instance:
- `inactive_latents_count` has shape `(n_instances, d_hidden)`.
- Updates and `resample_dead_latents()` operate on this full tensor, maintaining per-instance resampling decisions.

## Artifact Generation Design

### High-level structure

In scalar mode, behavior is unchanged: artifacts are written directly under `<seed_run_dir>/` as today.

If at least one model uses list mode, the seed run directory acts as a container:
- Root (`<seed_run_dir>/`) contains:
  - the original config (including list-valued `sparsity_coeff`) and environment metadata,
  - the shared data export artifacts for that seed,
  - optional sweep artifacts (same placement as today),
  - per-model multi-sparsity summary plots (Requirements 5 and 6).
- Each coefficient subrun (`<seed_run_dir>/<i>/`) contains a “normal” scalar-mode run layout:
  - `config.json`, `environment.json`, run-level comparison artifacts,
  - `data/` exports (copied from the seed root),
  - per-model artifacts under `models/<model_name>/...`.

### Slicing a trained multi-instance model into single-instance models

Evaluation and artifact writing are currently implemented for `n_instances=1` models (e.g., decoder extraction uses instance 0). To avoid touching evaluation code:
- For any trained model with `n_instances > 1`, create a new model object of the same class with `n_instances=1`.
- Copy parameter values for instance slice `i` into the new model’s instance 0 parameters for every parameter that has a leading instance dimension.
- Run evaluation, calibration, and plotting on the sliced model exactly as in scalar mode.

This produces per-coefficient subruns whose artifacts are indistinguishable from training that coefficient in scalar mode, up to the fact that they were trained in parallel.

### Models with scalar `sparsity_coeff` in list-mode runs

Scalar-mode models are trained once per seed and then included in every coefficient subrun.

To satisfy the “full artifact set per subrun” requirement:
- For each coefficient subrun directory, reuse the trained scalar model and its evaluation outputs.
- Either:
  - re-run evaluation for that model into the subrun directory, or
  - copy its per-model artifact directory into each subrun.

The design prefers reusing the trained model and re-running evaluation into each subrun directory to avoid file-copy edge cases and to keep subrun directories self-contained with consistent JSON/plot generation code paths.

## Seed-level Multi-Sparsity Plots

For each model in list mode, write one set of plots under:
`<seed_run_dir>/plots/multi_sparsity/<model_name>/`

Plots:
- `sparsity_loss_vs_sparsity_coeff.png`
- `sparsity_loss_vs_log_sparsity_coeff.png`
- `reconstruction_loss_vs_sparsity_coeff.png`
- `reconstruction_loss_vs_log_sparsity_coeff.png`
- `aux_loss_vs_sparsity_coeff.png` (only if the model provides aux loss)
- `aux_loss_vs_log_sparsity_coeff.png` (only if applicable)
- `mean_active_latents_vs_sparsity_coeff.png`
- `mean_active_latents_vs_log_sparsity_coeff.png`

Y-values:
- Use the final epoch’s per-instance mean value for the corresponding metric.

Log-x handling:
- Since `sparsity_coeff` may be `0.0`, log plots use `log10(max(coeff, eps))` with `eps = 1e-12`.

## Sweeps Compatibility

Sweeps remain enabled and continue to run exactly as today.

Placement:
- Sweep artifacts remain at the seed run root (current behavior), e.g.:
  - `<seed_run_dir>/mse_vs_l0_tradeoff.json`
  - `<seed_run_dir>/mse_vs_l0_tradeoff.png`
  - `<seed_run_dir>/asymmetry_sweep_summary.json`
  - `<seed_run_dir>/asymmetry_sweep_calibration.png`

Rationale:
- Sweep results are conceptually attached to the seed run, not to a coefficient index.
- Duplicating sweep outputs into each coefficient subrun would create redundant and potentially confusing artifacts without adding information.

## Testing Strategy

### Gradient independence tests

Extend the existing instance gradient tests to verify slice independence for:
- all supported model classes,
- both sign-aware and non-sign-aware modes,
- both single and dual offset modes for sign-aware `SAE`,
- symmetric magnitude mode for sign-aware `GatedSAE`,
- and for each loss component tensor (`losses`, `reconstruction_losses`, `sparsity_losses`, optional `aux_loss`).

The tests check that:
- Backpropagating only instance `i` yields zero gradients for other instance slices.
- The instance `i` gradient matches the corresponding slice of the gradient obtained from backpropagating the summed loss over all instances.

### Runner/artifact tests

Add Protocol A tests for list-mode runs:
- Verify that a list-mode config creates `<seed_run_dir>/<i>/` subdirectories for `i` in `[0, max_len)`.
- Verify each subrun contains the expected run-level and per-model artifacts.
- Verify scalar models appear in every subrun; list models appear only where index exists.
- Verify per-model multi-sparsity plots exist under `<seed_run_dir>/plots/multi_sparsity/...`.

Backward compatibility tests remain unchanged and must still pass.

## Error Handling and Validation

- If a model’s `sparsity_coeff` list is empty, config validation already fails.
- In list mode, if `subrun_count` resolves to `0` (should be impossible due to validation), raise `RuntimeError`.
- When generating plots on the log scale, avoid `log(0)` by clamping with `eps`.
