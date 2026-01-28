# Parallel Multi-Sparsity Training for Protocol A (Polarity Dial)

## Overview

This design adds list-valued `sparsity_coeff` support to the Protocol A (Polarity Dial) experiment runner so that, for a fixed seed and model variant, multiple sparsity settings can be trained in parallel using the existing `n_instances` “slicing” semantics of `sgnsae/models`.

This design also specifies sweep semantics when list-valued `sparsity_coeff` is present. In particular, asymmetry sweeps must evaluate every list instance. Otherwise the implementation can train multiple instances but silently record only instance 0 in sweep outputs.

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

The input batch is shared across instances (broadcasted along the instance dimension), and `calculate_losses()` returns per-instance loss tensors. The training loop uses `losses.sum().backward()`. Because parameters are sliced by instance, gradients should not mix across instances (validated by dedicated gradient tests).

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

Important nuance: gradient independence is necessary but not sufficient for strict equivalence to independent scalar runs. Dead-latent resampling can consume randomness differently depending on which instances resample. See “Dead-latent resampling RNG independence” below.

## Artifact Generation Design

### High-level structure

In scalar mode, behavior is unchanged: artifacts are written directly under `<seed_run_dir>/` as today.

If at least one model uses list mode, the seed run directory acts as a container:
- Root (`<seed_run_dir>/`) contains:
  - the original config (including list-valued `sparsity_coeff`) and environment metadata,
  - the shared data export artifacts for that seed,
  - sweep artifacts (see “Sweeps in list-sparsity runs”),
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

This produces per-coefficient subruns whose artifacts are indistinguishable from training that coefficient in scalar mode, except for the RNG nuance described below.

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

### Raw plot data persistence (regeneration policy)

The raw x/y data required to regenerate the multi-sparsity plots is considered persisted if it can be reconstructed from the coefficient subruns.

For a list-mode model `M` with list length `k`:

1. For each coefficient index `i` in `[0, k)`:
   - x is read from `<seed_run_dir>/<i>/config.json` for model `M` via the scalar `sparsity_coeff` field.
   - y is read from `<seed_run_dir>/<i>/models/M/training_metrics.json` as the final element of:
     - `loss_sparsity[-1]`
     - `loss_reconstruction[-1]`
     - optional `loss_aux[-1]`
     - `mean_active_latents[-1]`

2. Sort points by x to match plotting behavior.

If an implementation change breaks this reconstructability, then persist an explicit JSON per model containing the plot data (x list and y lists) alongside the plots.

## Sweeps in list-sparsity runs

Sweeps remain enabled and are written at the seed run root (not duplicated into coefficient subruns).

However, list mode requires explicit semantics because the existing evaluation code for Protocol A assumes `n_instances=1` and reads instance 0 internally. Without additional handling, a sweep can train a multi-instance model but silently report only instance 0.

### Sweep type 1: Sparsity sweep (`sweep_mode` includes `"sparsity"`)

Semantics (unchanged):
- The sparsity sweep is driven by `experiment_config.sweeps.sparsity_coeff_grid`.
- For each grid value `λ`, each model is trained and evaluated using `sparsity_coeff = λ` (scalar), regardless of whether the base model config specified a list.

Rationale:
- The entire purpose of a sparsity sweep is to vary sparsity coefficient externally.
- Multiplying the sweep by an internal list would turn it into a 2D sweep (`λ_grid × λ_list`) and break existing behavior.

Implementation sketch:
```python
for sparsity_coeff in sparsity_coeff_grid:
    for model_config in experiment_config.models:
        model_config_for_point = replace(model_config, sparsity_coeff=float(sparsity_coeff))
        train_and_eval(model_config_for_point)
```

### Sweep type 2: Asymmetry sweep (`sweep_mode` includes `"asymmetry"`)

Semantics (extended for list mode):
- At every sweep point (for example a specific `positive_sign_prob` and a specific tail parameter):
  1. Train each base model config exactly once.
     - Scalar mode uses `n_instances=1`.
     - List mode uses `n_instances=len(list)`.
  2. Record metrics for:
     - scalar models under their base model name, and
     - list-mode models under expanded sweep model names, one per coefficient index.

Expanded sweep model names:
- Use a deterministic naming scheme that is collision-resistant. Recommended format:
  - `{base}__sparsity={coeff:.6g}__idx={i}`
- Including `idx` avoids collisions if the list contains duplicates.

Mapping metadata:
- In addition to the flat `models` list used for plotting, persist an explicit mapping (dictionary) from base model name to:
  - expanded name, coefficient value, and index.
This avoids downstream tools needing to parse strings.

Implementation approach (recommended): slice per instance.
- Train a list-mode model once (multi-instance).
- For each coefficient index `i`, call the existing “slice to single instance” utility to create an `n_instances=1` model.
- Evaluate the sliced model using the existing evaluation functions unchanged.
This is consistent with the approach used for coefficient subruns.

## Dead-latent resampling RNG independence (Remark 3.1)

### Why gradients are not the whole story

Even when gradient independence holds, two runs can diverge due to randomness in dead-latent resampling:
- resampling is triggered per instance based on that instance’s activity,
- a shared RNG stream means the order and number of random draws depends on which instances resample.

This can break strict bitwise equivalence between:
- training instance 0 in a multi-instance run, and
- training the same configuration as a single-instance run.

### Option A: document and accept

If the project does not require strict equivalence to independent scalar runs, document:

- Parallel multi-instance training is gradient independent but not guaranteed bitwise-identical to independent scalar runs when dead-latent resampling occurs.

### Option B: per-instance RNG (more correct)

Implement per-instance RNG streams for resampling.

Design sketch:
1. Update `resample_dead_latents()` to accept either:
   - no generators (old behavior), or
   - a list of `torch.Generator` objects, one per instance.
2. In training (list mode):
   - create `generators[i] = torch.Generator(device=device).manual_seed(model_seed + i)`
   - pass `generators` into `model.resample_dead_latents(...)`
3. Resample instance-by-instance using its own generator:
   - RNG draws for instance `i` are independent of whether other instances resampled.

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

### Sweep tests (new)

Add tests that specifically cover asymmetry sweeps with list-mode models:
- When a base model has `sparsity_coeff=[λ0, λ1]`, the sweep output must contain two expanded model names for that base model at every sweep point.
- The asymmetry sweep summary JSON must include:
  - the expanded `models` list used for plotting,
  - and the mapping metadata (so downstream scripts do not need to parse names).
- Scalar-only configs must preserve the existing sweep output schema.

### RNG resampling tests (optional, only if Option B is implemented)

If per-instance RNG is implemented, add a regression-style test that checks:
- RNG draws for one instance do not depend on whether other instances resampled.
This can be done by forcing deterministic resampling conditions and comparing instance-0 resampled weights between `n_instances=1` and `n_instances=2` runs with the same seed.

Backward compatibility tests remain unchanged and must still pass.

## Error Handling and Validation

- If a model’s `sparsity_coeff` list is empty, config validation already fails.
- In list mode, if `subrun_count` resolves to `0` (should be impossible due to validation), raise `RuntimeError`.
- When generating plots on the log scale, avoid `log(0)` by clamping with `eps`.
