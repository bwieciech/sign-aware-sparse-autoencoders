# Parallel Multi-Sparsity Training for Protocol A (Polarity Dial)

## Introduction

Protocol A (Polarity Dial) currently trains each model variant for a single `sparsity_coeff` value per seed. The model implementations in `sgnsae/models` support training multiple independent “instances” in parallel via `n_instances`, where each instance corresponds to a disjoint parameter slice and should receive gradients only from its own loss term.

This spec adds support for training multiple `sparsity_coeff` values in parallel (per seed) by allowing `sparsity_coeff` to be specified as a list in the Protocol A configuration, while preserving full backward compatibility when `sparsity_coeff` is a single float.

This spec also defines sweep semantics when a model variant uses list-valued `sparsity_coeff`. The goal is to prevent the current failure mode where list-mode models are trained as multi-instance, but sweep evaluation silently records only instance 0.

## Goals

- Allow `sparsity_coeff` to be either a single float (current behavior) or a list of floats (new behavior) in Protocol A configs.
- When a list is provided, train the corresponding model instances in parallel within the same seed using `n_instances`.
- Maintain existing multi-seed behavior (seeds run sequentially via `for seed in seeds:`).
- After training, generate per-`sparsity_coeff` artifacts sequentially (evaluation/plots/summaries).
- Keep sweeps enabled, and define unambiguous semantics for list-valued `sparsity_coeff`:
  - sparsity sweeps keep existing behavior (the sweep grid is authoritative),
  - asymmetry sweeps report results for every list instance (expanded model names).
- Add extensive unit tests that verify instance slices do not affect each other by checking gradients.
- Document and optionally address a reproducibility nuance: even with gradient independence, shared RNG in dead-latent resampling can break strict equivalence to independent scalar runs (Remark 3.1).

## Non-goals

- Changing the mathematical definitions of losses, calibration, or evaluation metrics.
- Modifying the public API of model classes in `sgnsae/models` beyond what is required to support `n_instances` usage from Protocol A.
- Turning sweeps into a full 2D sweep over both `sparsity_coeff_grid` and a per-model list. For sparsity sweeps, the grid remains the single source of truth.

## Definitions

- **Scalar sparsity mode**: `isinstance(model.sparsity_coeff, float) == True`.
- **List sparsity mode**: `isinstance(model.sparsity_coeff, list) == True` and the list is non-empty.
- **Coefficient index**: the position `i` in a model’s `sparsity_coeff` list.
- **Seed run directory**: the run directory for a specific seed (current behavior, `.../polarity_dial/<run_id>_seed<seed>/`).
- **Coefficient subrun directory**: in list sparsity mode, a directory under the seed run directory named by coefficient index (`.../<seed_run_dir>/<i>/`).
- **Base model name**: the configured model name (`model_config.name`).
- **Expanded sweep model name**: a deterministic name used in sweep outputs to uniquely identify a particular list instance of a base model. Example: `{base}__sparsity={coeff:.6g}__idx={i}`.

## Requirements

### Requirement 1: Gradient independence is verified (no cross-instance mixing)

**User Story:** As a researcher, I need to trust that training multiple instances in parallel does not mix gradients across instances, so that changing one `sparsity_coeff` cannot influence another instance’s parameter updates.

This requirement is specifically about gradient independence. Strict bitwise equivalence to independent scalar runs is not guaranteed if randomness is consumed differently across instances. See Remark 3.1.

#### Acceptance Criteria

1. WHEN backpropagating the sum of per-instance losses, THEN gradients for parameters in each instance slice SHALL match the gradients obtained when backpropagating only that instance’s loss, and other instance slices SHALL receive zero gradients.
2. WHEN backpropagating a single instance’s loss for instance `i`, THEN ONLY parameter slices for instance `i` (where the parameter has a leading `n_instances` dimension) SHALL have non-zero gradients.
3. The above SHALL be tested for every model class under `sgnsae/models` that is used by Protocol A:
   - `SAE` (non-sign-aware and sign-aware, both single and dual offset modes),
   - `GatedSAE` (non-sign-aware and sign-aware, including symmetric magnitude),
   - `SignedSoftThresholdSAE`.
4. The above SHALL be tested not only for the combined `"losses"` tensor, but also for each loss component tensor exposed by the model’s `calculate_losses` output:
   - `"reconstruction_losses"`,
   - `"sparsity_losses"`,
   - `"aux_loss"` (when present).

### Requirement 2: Config accepts scalar and list sparsity coefficients (no global list-length constraints)

**User Story:** As an experiment author, I want to specify one or multiple `sparsity_coeff` values per model variant without being forced to use the same list length across models.

#### Acceptance Criteria

1. WHEN `sparsity_coeff` is a float, THEN it SHALL be validated as numeric and non-negative (existing behavior).
2. WHEN `sparsity_coeff` is a list, THEN it SHALL be validated as a non-empty list of numeric, non-negative values (existing behavior).
3. The config system SHALL allow mixed usage across models (some models use floats, others lists) and SHALL NOT require lists to have the same length.

### Requirement 3: Parallel training uses `n_instances` to represent multiple `sparsity_coeff` values

**User Story:** As a researcher, I want the Protocol A runner to train multiple sparsity settings in parallel for a given model and seed, so I can reduce wall-clock time without changing the conceptual meaning of the results.

#### Acceptance Criteria

1. WHEN a model variant has `sparsity_coeff` as a list of length `n`, THEN the training code SHALL instantiate the model with `n_instances = n`.
2. WHEN a model variant has `sparsity_coeff` as a float, THEN the training code SHALL keep `n_instances = 1` (backward compatible).
3. During training, each instance slice SHALL use its own `sparsity_coeff` value when computing its per-instance loss.
4. Training metrics SHALL be recorded per instance in list sparsity mode (so that per-coefficient subruns can reproduce scalar-mode artifacts).
5. Existing per-model deterministic seeding behavior (derived from `(data.seed, model_config.name)`) SHALL be maintained in both scalar and list sparsity modes.

### Requirement 4: Artifact structure includes per-coefficient subruns under each seed

**User Story:** As a user inspecting results, I want each seed’s outputs to be split into per-sparsity-coefficient subdirectories, each containing the same artifacts as a normal single-sparsity run.

#### Acceptance Criteria

1. WHEN `sparsity_coeff` is a float for all models (scalar sparsity mode), THEN the output directory structure and artifact set SHALL remain unchanged.
2. WHEN at least one model uses list sparsity mode, THEN each seed run directory SHALL contain coefficient subrun directories:
   - The seed run directory SHALL contain subdirectories named `0`, `1`, ..., `m-1`, where `m` is the maximum list length across models in that seed run (treat floats as length `1`).
   - For a given coefficient index `i`, a model SHALL be included in subrun `i` only if:
     - it has a list with length `> i` (use list value at index `i`), OR
     - it has a float (use that float value in every subrun).
3. Each coefficient subrun directory SHALL contain the full set of artifacts that are produced by a scalar-mode run for the models included in that subrun:
   - run-level: `config.json`, `environment.json`, `comparison_summary.json`, `gamma_diff_bar_comparison.png`, and `data/` exports,
   - per-model: `models/<model_name>/...` including checkpoints, metrics, alignment/calibration JSON, and plots.
4. If sweeps are enabled, THEN sweep artifacts SHALL still be produced and SHALL remain consistent with the semantics defined in Requirement 7. Placement/copying policy for sweep artifacts in list sparsity mode SHALL be defined in the design.

### Requirement 5: Seed-level per-model plots across sparsity coefficients

**User Story:** As a researcher, I want quick per-model summaries of how losses change as a function of `sparsity_coeff` when training multiple coefficients in parallel.

#### Acceptance Criteria

1. WHEN a model uses list sparsity mode, THEN the seed run directory SHALL include, for that model, the following plots under `<seed_run_dir>/plots/multi_sparsity/<model_name>/`:
   - `sparsity_loss_vs_sparsity_coeff.png`
   - `sparsity_loss_vs_log_sparsity_coeff.png`
   - `reconstruction_loss_vs_sparsity_coeff.png`
   - `reconstruction_loss_vs_log_sparsity_coeff.png`
   - `aux_loss_vs_sparsity_coeff.png` (only if the model provides aux loss)
   - `aux_loss_vs_log_sparsity_coeff.png` (only if applicable)
2. The plots SHALL be “one set per model” (i.e., separate files per model, not a single combined plot with all models).
3. The plots SHALL use per-instance training metrics corresponding to the model’s coefficient list.
4. The raw x/y data required to regenerate these plots SHALL be reconstructable from persisted artifacts:
   - x: the scalar `sparsity_coeff` value written into each coefficient subrun’s `config.json`,
   - y: the final-epoch metric value in each coefficient subrun’s `models/<model_name>/training_metrics.json`.
   If this is not satisfied by the implementation, then a dedicated JSON containing the plot data SHALL be persisted alongside the plots.

### Requirement 6: Seed-level per-model mean active latents plots across sparsity coefficients

**User Story:** As a researcher, I want quick per-model summaries of how mean active latents change as a function of `sparsity_coeff` when training multiple coefficients in parallel.

#### Acceptance Criteria

1. WHEN a model uses list sparsity mode, THEN the seed run directory SHALL include, for that model, the following plots under `<seed_run_dir>/plots/multi_sparsity/<model_name>/`:
   - `mean_active_latents_vs_sparsity_coeff.png`
   - `mean_active_latents_vs_log_sparsity_coeff.png`
2. The plots SHALL be “one set per model” (i.e., separate files per model, not a single combined plot with all models).
3. The plots SHALL use per-instance training metrics corresponding to the model’s coefficient list.
4. The raw x/y data required to regenerate these plots SHALL be reconstructable from persisted artifacts using the same policy as Requirement 5.4. If this is not satisfied by the implementation, then a dedicated JSON containing the plot data SHALL be persisted alongside the plots.

### Requirement 7: Sweeps fully support list-valued `sparsity_coeff`

**User Story:** As a researcher, I want sweeps (especially asymmetry sweeps) to be correct and complete even when some models specify `sparsity_coeff` as a list, so I do not silently lose results for the other sparsity settings.

This requirement has two parts because the repo has two sweep types with different semantics.

#### Requirement 7.A: Sparsity sweep behavior (existing semantics, grid is authoritative)

A sparsity sweep already sweeps `sparsity_coeff` using `sparsity_coeff_grid`. Therefore list-valued `sparsity_coeff` in the base experiment config SHALL NOT multiply the sweep output. The sweep grid remains the single source of truth.

Acceptance Criteria:
1. WHEN `sweep_mode` includes `"sparsity"`, THEN for each grid value `λ` in `sparsity_coeff_grid`, each model variant SHALL be trained and evaluated as if `sparsity_coeff = λ` (scalar), regardless of whether its base config used a list.
2. The returned `mse_vs_l0_tradeoff.json` format SHALL remain unchanged compared to scalar-only configs.

#### Requirement 7.B: Asymmetry sweep behavior (extended for list mode)

For asymmetry sweeps, `sparsity_coeff` is part of “the model variant configuration”. If a variant specifies multiple `sparsity_coeff` values, the sweep SHALL report results for each one.

Acceptance Criteria:
1. WHEN `sweep_mode` includes `"asymmetry"` and a model uses list sparsity mode, THEN the asymmetry sweep SHALL report metrics for every coefficient index in that list at every sweep point.
2. For a list-mode base model config with name `M` and coefficient list `[λ0, λ1, ...]`, the sweep output SHALL contain expanded sweep model names that uniquely identify each instance, for example:
   - `M__sparsity=λ0__idx=0`
   - `M__sparsity=λ1__idx=1`
   The exact formatting is design-defined, but must be deterministic and collision-resistant. Including the index is required to avoid collisions when coefficients repeat.
3. Scalar-mode models SHALL appear exactly once (same as today), using their base model name.
4. Efficiency requirement: for each sweep point, each base model config SHALL be trained exactly once.
   - If list mode, train a single multi-instance model and evaluate each instance.
   - The implementation SHALL NOT redundantly train `len(list)` separate models per sweep point.
5. The asymmetry sweep summary JSON (`asymmetry_sweep_summary.json`) SHALL include:
   - the list of expanded model names used for plotting (`models`),
   - and an explicit mapping that allows recovering `(base_model_name, sparsity_coeff, idx)` for each expanded name without string parsing.
6. The asymmetry sweep plot (`asymmetry_sweep_calibration.png`) SHALL render without requiring changes to plotting logic. Expanded names simply produce additional curves.

## Remark 3.1: “Equivalence to independent training” is not strictly true if dead-latent resampling uses shared RNG

Even if gradients do not mix across instances (Requirement 1), training multiple instances in a single process can still differ from running fully independent scalar trainings due to randomness consumption patterns.

### Why this happens

1. Dead-latent resampling is triggered per instance depending on that instance’s activity statistics.
2. If `resample_dead_latents()` draws random numbers from a single global RNG stream and resamples all instances in one call, the sequence of random draws depends on which instances resampled this step and how many latents were resampled per instance.
3. In independent scalar runs, the RNG stream is consumed only by that run’s resampling events.
4. Therefore in parallel multi-instance training, instance 0 can see different random resample values than in a scalar run purely because other instances triggered RNG draws earlier.

This means: parallel training can be gradient-independent but not guaranteed bitwise-identical to independent scalar runs when resampling occurs.

### Option A (simplest): accept and document the nuance

Treat gradient independence as “good enough” for research use and document:

- Parallel multi-instance training is gradient independent but not guaranteed bitwise-identical to independent scalar runs when dead-latent resampling occurs.

### Option B (more correct): make resampling RNG per-instance

If strict equivalence to independent scalar runs is important, update resampling to use a separate RNG stream per instance.

Acceptance Criteria for Option B:
1. `resample_dead_latents()` SHALL support per-instance RNG (for example a list of `torch.Generator` objects), one per instance.
2. In list mode, training SHALL seed per-instance generators deterministically (for example `model_seed + instance_index`) and pass them to resampling.
3. RNG draws for instance `i` SHALL be independent of resampling activity in other instances.
