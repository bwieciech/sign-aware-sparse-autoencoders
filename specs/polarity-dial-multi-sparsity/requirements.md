# Parallel Multi-Sparsity Training for Protocol A (Polarity Dial)

## Introduction

Protocol A (Polarity Dial) currently trains each model variant for a single `sparsity_coeff` value per seed. The model implementations in `sgnsae/models` support training multiple independent “instances” in parallel via `n_instances`, where each instance corresponds to a disjoint parameter slice and should receive gradients only from its own loss term.

This spec adds support for training multiple `sparsity_coeff` values in parallel (per seed) by allowing `sparsity_coeff` to be specified as a list in the Protocol A configuration, while preserving full backward compatibility when `sparsity_coeff` is a single float.

## Goals

- Allow `sparsity_coeff` to be either a single float (current behavior) or a list of floats (new behavior) in Protocol A configs.
- When a list is provided, train the corresponding model instances in parallel within the same seed using `n_instances`.
- Maintain existing multi-seed behavior (seeds run sequentially via `for seed in seeds:`).
- After training, generate per-`sparsity_coeff` artifacts sequentially (evaluation/plots/summaries).
- Add extensive unit tests that verify instance slices do not affect each other by checking gradients.

## Non-goals

- Changing the mathematical definitions of losses, calibration, or evaluation metrics.
- Modifying the public API of model classes in `sgnsae/models` beyond what is required to support `n_instances` usage from Protocol A.
- Reworking sweep implementations beyond ensuring compatibility with list-valued `sparsity_coeff`.

## Definitions

- **Scalar sparsity mode**: `isinstance(model.sparsity_coeff, float) == True`.
- **List sparsity mode**: `isinstance(model.sparsity_coeff, list) == True` and the list is non-empty.
- **Coefficient index**: the position `i` in a model’s `sparsity_coeff` list.
- **Seed run directory**: the run directory for a specific seed (current behavior, `.../polarity_dial/<run_id>_seed<seed>/`).
- **Coefficient subrun directory**: in list sparsity mode, a directory under the seed run directory named by coefficient index (`.../<seed_run_dir>/<i>/`).

## Requirements

### Requirement 1: Gradient independence is verified (no cross-instance mixing)

**User Story:** As a researcher, I need to trust that training multiple instances in parallel is equivalent to training them independently, so that changing one `sparsity_coeff` cannot influence another instance’s parameters.

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

**User Story:** As a researcher, I want the Protocol A runner to train multiple sparsity settings in parallel for a given model and seed, so I can reduce wall-clock time without changing results.

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
4. If sweeps are enabled, THEN sweep artifacts SHALL still be produced and SHALL remain consistent with existing behavior (compatibility requirement). The exact placement/copying policy for sweep artifacts in list sparsity mode SHALL be defined in the design.

### Requirement 5: Seed-level per-model plots across sparsity coefficients

**User Story:** As a researcher, I want quick per-model summaries of how losses change as a function of `sparsity_coeff` when training multiple coefficients in parallel.

#### Acceptance Criteria

1. WHEN a model uses list sparsity mode, THEN the seed run directory SHALL include, for that model, plots of:
   - sparsity loss vs `sparsity_coeff`,
   - sparsity loss vs `log(sparsity_coeff)`,
   - reconstruction loss vs `sparsity_coeff`,
   - reconstruction loss vs `log(sparsity_coeff)`,
   - auxiliary loss vs `sparsity_coeff` (if applicable),
   - auxiliary loss vs `log(sparsity_coeff)` (if applicable).
2. The plots SHALL be “one set per model” (i.e., separate files per model, not a single combined plot with all models).
3. The plots SHALL use per-instance training metrics corresponding to the model’s coefficient list.

