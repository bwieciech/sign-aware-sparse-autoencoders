# Requirements Document

## Introduction

This document specifies Protocol C (“Geometric Validation on Toy Models”) from `main.tex`. The goal is to implement a fully reproducible experiment that:

- Generates a 2D synthetic “toy superposition” dataset with `k` ground-truth **paired features**, where each pair corresponds to two **antipodal** directions on the unit circle (a feature and its anticorrelated opposite).
- Decouples (a) **overall sparsity / superposition level** (how many pairs participate per sample) from (b) **within-pair correlation structure** (how often the two antipodal features co-occur when the pair participates), so the correlation sweep `[-1, 0]` is feasible without forcing every pair to be active in every sample.
- Trains baseline SAE variants (non-negative latents) and the Sign-Aware Gated SAE on this dataset.
- Produces geometry-first diagnostics that visually and quantitatively validate whether the Sign-Aware model consolidates anticorrelated pairs into a single decoder direction and represents the opposite feature via negative activations.

The output of this Protocol C implementation MUST be analogous in structure and quality to Protocol A (`specs/polarity-dial-experiment/*`) and Protocol B (`specs/directional-deviation-detection-experiment/*`): a single CLI entrypoint driven by a JSON config, a deterministic artifact directory layout, and unit tests that lock in correctness and reproducibility.

## Requirements

### Requirement 1: Protocol C configuration and CLI runner

**User Story:** As a junior AI/ML researcher, I want a single config-driven CLI entrypoint for Protocol C, so that I can run the toy experiment and obtain all required artifacts without manual steps.

#### Acceptance Criteria

1. WHEN the Protocol C runner is invoked with `--config-path`, THEN the system SHALL load a JSON configuration and validate it before running any compute.
2. WHEN the configuration is valid, THEN the system SHALL create a unique run directory under `<output_dir>/toy_model_geometry_validation/<run_id>/` and write the resolved config to `<run_dir>/config.json`.
3. WHEN the run starts, THEN the system SHALL write `<run_dir>/environment.json` capturing Python version, OS, torch version, CUDA availability, and key dependency versions (at minimum: numpy, matplotlib, scipy, einops).
4. IF the runner is configured with `device="auto"`, THEN the system SHALL select CUDA when available, otherwise CPU, and record the resolved device in `config.json`.
5. WHEN the runner is configured with multiple model variants, THEN the system SHALL derive and apply an order-independent deterministic per-model seed from `(data.seed, model_variant.name)` and record it in per-model artifacts.

---

### Requirement 2: Toy dataset generation with paired antipodal features

**User Story:** As a researcher, I want a deterministic 2D data generator with paired antipodal features, so that decoder geometry can be inspected on the unit circle and compared to known ground truth.

#### Acceptance Criteria

1. WHEN dataset generation is run with a given `seed`, THEN the generated dataset (including feature directions and splits) SHALL be deterministic.
2. WHEN configured with `num_pairs=k`, THEN the generator SHALL create exactly `2k` ground-truth feature directions in `R^2`, arranged as `k` antipodal pairs:
   - `u_pair[j, 1] = -u_pair[j, 0]`
   - and each direction SHALL have unit norm within tolerance.
3. WHEN `direction_mode="uniform"`, THEN the generator SHALL place the `k` unique pair axes at deterministic angles evenly spaced in `[0, π)`, and record these angles in `data/generator_config.json`.
4. WHEN `direction_mode="random"`, THEN the generator SHALL sample `k` angles uniformly at random in `[0, π)` and record them in `data/generator_config.json`.
5. WHEN the generator produces samples, THEN it SHALL output:
   - `x_train`, `x_val`, `x_test` tensors of shape `(num_samples, 2)`,
   - and per-split ground-truth coefficients `coeffs_*` of shape `(num_samples, 2k)` for the `2k` features.
   - and per-split `pair_active_mask_*` of shape `(num_samples, k)` indicating whether each pair was “eligible/active” for sampling on that example (see Requirement 3).
6. WHEN noise is enabled (`noise_std > 0`), THEN the generator SHALL add isotropic Gaussian noise `N(0, noise_std^2 I_2)` to each sample.
7. WHEN configured with `pair_axis_active_prob=rho`, THEN for each sample and pair the generator SHALL draw `g ~ Bernoulli(rho)` and record it into `pair_active_mask_*`. IF `g=0`, THEN the generator SHALL set both features in that pair to inactive (the corresponding two entries in `coeffs_*` MUST be exactly 0).

---

### Requirement 3: Controllable within-pair correlation sweep (ρ from -1 to 0)

**User Story:** As a researcher, I want to vary the correlation between features in each pair from perfectly anticorrelated to uncorrelated, so that I can measure when consolidation is optimal and detect a phase-transition-like behavior.

#### Acceptance Criteria

1. WHEN configured with `pair_axis_active_prob=rho`, `pair_feature_prob=p`, and `pair_correlation=r`, THEN for each sample and pair the system SHALL:
   - draw a pair-eligibility mask `g ~ Bernoulli(rho)` (and record it into `pair_active_mask_*`), and
   - IF `g=0`, set `(f_plus, f_minus)=(0,0)`, ELSE sample `(f_plus, f_minus)` from a bivariate Bernoulli distribution with:
     - marginal probabilities `P(f_plus=1)=P(f_minus=1)=p`, and
     - Pearson correlation approximately `r` (in expectation, up to sampling error),
   - where the correlation is defined over the binary activations `(f_plus, f_minus)` *restricted to examples with `g=1`* (i.e., the “within-pair correlation when the pair participates”), not over magnitudes.
   - Rationale: this matches `main.tex`’s intent (vary within-pair anticorrelation from `-1` to `0`) while keeping the overall dataset in a sparse/superposition regime suitable for geometric recovery.
2. IF a requested `pair_correlation=r` is infeasible for the configured `p`, THEN the system SHALL raise a clear `ValueError` that includes the feasible correlation interval.
3. WHEN `pair_correlation=-1` is requested, THEN the default configuration SHALL be such that `-1` is feasible (i.e., defaults MUST support the full sweep `[-1, 0]` without changing `p`).
   - Concretely, the default `pair_feature_prob` SHOULD be set to `0.5` so that `pair_correlation=-1` is achievable *within eligible pairs*, and the default `pair_axis_active_prob` SHOULD be < 1 (e.g., `0.05`–`0.2`) so the experiment remains meaningfully sparse at the `-1` endpoint.
4. WHEN a correlation sweep grid is provided (e.g., `pair_correlation_grid`), THEN the runner SHALL run the full experiment for each grid point deterministically and produce an aggregated summary artifact.

---

### Requirement 4: Model variants and training loop for Protocol C

**User Story:** As a researcher, I want to train baseline models and the Sign-Aware Gated SAE on the same toy dataset, so that I can compare learned geometry and activation behavior.

#### Acceptance Criteria

1. WHEN a model variant is configured, THEN the system SHALL support at least the following variants:
   - a non-negative baseline (e.g., `SAE` or `GatedSAE` with `sign_aware=false`),
   - the Sign-Aware Gated SAE (`GatedSAE` with `sign_aware=true`).
2. WHEN configured for the “efficiency comparison” mode, THEN the system SHALL support widths:
   - baseline width `H=2k`,
   - sign-aware width `H=k`,
   and record these widths in `models/<model_name>/model_config.json`.
3. WHEN configured for the “phase transition” mode, THEN the system SHALL support training sign-aware models at width `H=2k` (so the model can choose whether to use one or two latents per pair), and record this choice in `model_config.json`.
4. WHEN training is started, THEN the system SHALL use a single shared training loop abstraction for all variants and persist:
   - `model.pt`,
   - `training_metrics.json` (epoch-wise losses),
   - and `training_plots/` (loss curves).
5. IF training encounters NaNs or divergence, THEN the system SHALL stop training early, mark `terminated_early=true` in `training_metrics.json`, and still write partial artifacts for debugging.

---

### Requirement 5: Unit circle geometry visualization

**User Story:** As a researcher, I want a unit circle plot of decoder directions, so that I can verify that baselines learn antipodal pairs while the Sign-Aware model learns one vector per pair.

#### Acceptance Criteria

1. WHEN training completes for a model, THEN the system SHALL extract the decoder dictionary `D` and compute per-latent 2D directions by normalizing columns to unit length.
2. WHEN the unit circle plot is generated, THEN it SHALL include:
   - ground-truth directions for all `2k` features (or `k` axes, with both antipodes),
   - learned decoder directions for the model’s `H` latents,
   - and a legend or annotation sufficient to interpret the plot.
3. WHEN baseline models are trained at width `H=2k` on anticorrelated data (`r≈-1`), THEN the plot SHALL show learned directions close to antipodal pairs aligned with ground truth (up to permutation).
4. WHEN sign-aware models are trained on anticorrelated data (`r≈-1`), THEN the plot SHALL show approximately one learned direction per pair axis (up to permutation), and SHOULD NOT require separate antipodal atoms for the opposite feature.
5. The plot SHALL be saved under `models/<model_name>/plots/unit_circle_<model_name>.png`.

---

### Requirement 6: Bi-Jump activation histogram visualization

**User Story:** As a researcher, I want to visually confirm the Bi-Jump-ReLU behavior by inspecting a signed latent’s activation histogram, so that I can see a dead zone and positive/negative lobes.

#### Acceptance Criteria

1. WHEN a sign-aware model is trained, THEN the system SHALL select at least one representative latent for histogramming using a deterministic rule (e.g., best-aligned to a chosen ground-truth pair axis).
2. WHEN activations are collected on a held-out split, THEN the histogram SHALL be computed over the signed latent activations `a_i(x)` (not gate logits) and SHALL include:
   - a large mass at exactly zero (or within a near-zero bin), and
   - non-trivial density on both positive and negative sides.
3. The histogram SHALL be saved under `models/<model_name>/plots/bi_jump_hist_<model_name>_latent<i>.png` and accompanied by metadata recording the chosen latent index and its alignment.

---

### Requirement 7: Pair Consolidation Rate metric

**User Story:** As a researcher, I want a quantitative “Pair Consolidation Rate” metric, so that I can measure how often a single latent represents both features in a pair as correlation varies.

#### Acceptance Criteria

1. WHEN evaluation runs, THEN the system SHALL compute Pair Consolidation Rate for each model:
   - `pair_consolidation_rate = (# consolidated pairs) / k`.
2. A ground-truth pair `j` SHALL be considered consolidated if there exists a single latent `i` such that:
   - the decoder direction is aligned to the pair axis (absolute cosine similarity ≥ `min_axis_abs_cos`, because the two ground-truth features are antipodal), AND
   - the latent’s signed activation correlates with the pair’s ground-truth signed coefficient with Pearson `r ≥ min_activation_corr`, AND
   - the latent uses both signs: mean activation is > 0 on samples with positive coefficient and < 0 on samples with negative coefficient, given minimum sample counts per regime.
3. IF a regime has fewer than `min_regime_samples`, THEN the system SHALL mark the pair as “not evaluable” for consolidation (and exclude it from the denominator), and SHALL report both “rate_over_evaluable_pairs” and “evaluable_pair_fraction” in `evaluation.json`.
4. WHEN running a correlation sweep, THEN the system SHALL write an aggregated file `pair_consolidation_sweep.json` containing, for each sweep point:
   - the correlation value,
   - each model’s consolidation rate statistics,
   - and the number of evaluable pairs.
5. WHEN the sweep completes, THEN the system SHALL produce a plot `pair_consolidation_sweep.png` showing consolidation rate vs correlation for all models.

---

### Requirement 8: Artifacts, reproducibility, and minimal diagnostics

**User Story:** As a reproducer, I want all artifacts needed to interpret and rerun Protocol C, so that results can be audited and debugged.

#### Acceptance Criteria

1. The run directory SHALL include:
   - `config.json`, `environment.json`,
   - `data/generator_config.json`,
   - `data/feature_directions.pt` and `data/coeffs_test.pt` (or equivalent),
   - `data/pair_active_mask_test.pt` (or equivalent),
   - `comparison_summary.json`,
   - `plots/` for run-level sweep plots.
2. For each model variant, the run SHALL include:
   - `models/<model_name>/model.pt`,
   - `models/<model_name>/model_config.json`,
   - `models/<model_name>/training_metrics.json`,
   - `models/<model_name>/evaluation.json`,
   - `models/<model_name>/plots/*`.
3. WHEN the run is repeated with the same config and seed on the same hardware, THEN deterministic artifacts (JSON and tensors) SHALL be bitwise identical unless nondeterministic algorithms are explicitly enabled/disabled in config.
4. The system SHALL fail fast with clear errors if required dependencies are missing (torch, numpy, matplotlib, scipy), and SHALL include actionable remediation text (e.g., “install requirements.txt”).

---

### Requirement 9: Tests and coverage

**User Story:** As a maintainer, I want unit tests that lock in Protocol C correctness, so that future refactors do not silently break the experiment.

#### Acceptance Criteria

1. Tests SHALL use the `unittest` framework and follow the repository’s existing style in `tests/`.
2. Tests SHALL cover:
   - determinism of data generation (same seed ⇒ identical outputs),
   - antipodal direction construction and unit norms,
   - empirical correlation sanity checks for at least three `pair_correlation` values (e.g., `-1`, `-0.5`, `0`), computed on examples where `pair_active_mask==1` for the tested pair,
   - correctness of the Pair Consolidation Rate logic on a constructed toy “perfect model” case.
3. The repository SHALL include a small end-to-end smoke test that runs a tiny Protocol C config (very small dataset and epochs) and asserts required artifacts exist.
4. After implementation, coverage MUST be measured with Coverage.py, targeting 100% coverage of newly added/modified Protocol C modules (excluding scripts under `sgnsae/scripts/`), or the implementer MUST document why full coverage is infeasible.
