# Implementation Plan

- [x] 1. Create Protocol C package skeleton and config models
  - Create `sgnsae/experiments/toy_model_geometry_validation/` with `__init__.py`.
  - Implement `sgnsae.experiments.toy_model_geometry_validation.config` dataclasses:
    - `Toy2DDataConfig`, `Toy2DModelVariantConfig`, `Toy2DEvaluationConfig`, `Toy2DExperimentConfig`.
  - Implement JSON load/save helpers (mirror Protocol A/B patterns).
  - Add `__post_init__` validation, including correlation feasibility checks with informative errors.
  - _Requirements: 1.1, 3.1, 3.2, 8.4_

- [x] 2. Implement Protocol C CLI entrypoint
  - Add `sgnsae/scripts/run_toy_model_geometry_validation_experiment.py`:
    - parse `--config-path`,
    - validate dependencies,
    - resolve device,
    - create `<output_dir>/toy_model_geometry_validation/<run_id>/`,
    - write `config.json` and `environment.json`,
    - dispatch to a runner function.
  - Ensure the script prints the final `<run_dir>` path at completion.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 8.1_

- [x] 3. Implement toy 2D data generation (paired antipodal features)
  - Create `sgnsae/experiments/toy_model_geometry_validation/data.py`.
  - Implement `generate_toy2d_dataset(config, device)` returning:
    - `feature_directions` (2k,2),
    - `pair_axis_directions` (k,2),
    - `pair_active_mask_train/val/test` (num,k),
    - `x_train/x_val/x_test`,
    - `coeffs_train/coeffs_val/coeffs_test`.
  - Implement `direction_mode ∈ {uniform, random}` and export angles in `data/generator_config.json`.
  - Implement magnitude sampling (`lognormal` and `constant`) and isotropic noise.
  - Export under `<run_dir>/data/`:
    - `generator_config.json`, `feature_directions.pt`,
    - `pair_active_mask_train.pt`, `pair_active_mask_val.pt`, `pair_active_mask_test.pt`,
    - `coeffs_train.pt`, `coeffs_val.pt`, `coeffs_test.pt`,
    - `sampled_data.pt` preview.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 8.1_

- [x] 4. Implement within-pair correlation sampling and sweep plumbing
  - In `data.py`, implement bivariate Bernoulli sampling with target `pair_correlation`:
    - compute `q = p^2 + r p (1-p)` and validate feasibility.
    - sample `(f_plus, f_minus)` via a 4-way categorical distribution.
  - Add `pair_axis_active_prob` (overall participation / sparsity) to the data config:
    - sample `g ~ Bernoulli(pair_axis_active_prob)` per `(sample, pair)` and record it in `pair_active_mask_*`,
    - apply participation by zeroing both features when `g=0`,
    - define correlation sanity checks and feasibility w.r.t. the conditional `g=1` distribution (so the sweep `[-1,0]` remains meaningful while the dataset stays sparse).
  - Add an optional `pair_correlation_grid` to data config:
    - runner loops over the grid and runs the full pipeline per grid point.
  - Persist per-grid-point data config resolution into each leaf run’s `data/generator_config.json`.
  - _Requirements: 3.1, 3.2, 3.4_

- [x] 5. Implement training orchestration for Protocol C model variants
  - Create `sgnsae/experiments/toy_model_geometry_validation/training.py`.
  - Implement `_instantiate_model(model_config, d_in=2, device)` supporting:
    - `SAE`,
    - `GatedSAE` (sign_aware false/true).
  - Implement `run_training_for_variant(...)`:
    - wrap data in `TensorDataset`/`DataLoader` with deterministic shuffling,
    - train for `num_epochs`,
    - record epoch-wise metrics (recon, sparsity, aux when applicable),
    - detect NaNs and early terminate with `terminated_early=true`.
  - Persist per model:
    - `model.pt`, `model_config.json`, `training_metrics.json`,
    - `plots/training_loss_<model_name>.png`.
  - _Requirements: 1.5, 4.1, 4.4, 4.5, 8.2_

- [x] 6. Implement Protocol C runner and artifact layout
  - Create `sgnsae/experiments/toy_model_geometry_validation/runner.py`.
  - Implement `run_toy_model_geometry_validation(experiment_config) -> str`:
    - call data generator once per sweep point,
    - for each model variant:
      - derive `model_seed` from `(data.seed, model_name)` deterministically,
      - reseed Python/numpy/torch (CPU/CUDA),
      - train, evaluate, plot, persist artifacts.
    - write a run-root `comparison_summary.json` collecting key metrics per model.
  - Ensure run-level layout matches the design doc exactly.
  - _Requirements: 1.2, 1.5, 4.4, 8.1, 8.2_

- [ ] 7. Implement evaluation utilities: decoder extraction, alignment, and consolidation
  - Create `sgnsae/experiments/toy_model_geometry_validation/evaluation.py`.
  - Implement:
    - decoder direction extraction and normalization,
    - axis alignment via `abs(cos)` and deterministic matching (Hungarian optional),
    - signed coefficient extraction per pair: `c_j = coeff_plus - coeff_minus`,
    - Pair Consolidation Rate metric with:
      - `min_axis_abs_cos`,
      - `min_activation_corr`,
      - `min_regime_samples`,
      - evaluable-pair gating.
  - Persist per-model `evaluation.json` including:
    - consolidation metrics,
    - alignment summaries,
    - chosen histogram latent index + its alignment (if sign-aware).
  - _Requirements: 5.1, 7.1, 7.2, 7.3_

- [ ] 8. Implement plotting: unit circle and Bi-Jump histogram
  - Create `sgnsae/experiments/toy_model_geometry_validation/plotting.py`.
  - Implement `plot_unit_circle(...)`:
    - plot ground truth directions and learned decoder directions on a unit circle,
    - save `models/<model_name>/plots/unit_circle_<model_name>.png`.
  - Implement `plot_bijump_histogram(...)` for sign-aware models:
    - histogram signed activations for the selected latent on `x_test`,
    - save `models/<model_name>/plots/bi_jump_hist_<model_name>_latent<i>.png`.
  - _Requirements: 5.2, 5.5, 6.1, 6.2, 6.3_

- [ ] 9. Implement sweep aggregation and run-level sweep plots
  - In `runner.py`, aggregate metrics across `pair_correlation_grid` into:
    - `<run_dir>/pair_consolidation_sweep.json`
    - `<run_dir>/plots/pair_consolidation_sweep.png`
  - Include per-sweep-point evaluable-pair fraction.
  - _Requirements: 7.4, 7.5_

- [ ] 10. Add sample configs for Protocol C reproduction
  - Add at least one minimal config under `configs/`, e.g.:
    - `configs/toy_model_geometry_validation_smoke.json` (tiny, fast),
    - `configs/toy_model_geometry_validation_sweep.json` (small sweep grid).
  - Use the following recommended values (from the design doc) so results are reproducible without tuning:
    - `k=8`, `num_train=2000`, `num_val=512`, `num_test=512`, `num_epochs=3`, `batch_size=256` for the smoke config.
    - `k=64`, `num_train=100000`, `num_val=10000`, `num_test=10000`, `num_epochs=50`, `batch_size=1024` for the sweep config.
    - `pair_axis_active_prob=0.2` (smoke) / `0.05` (sweep),
      `pair_feature_prob=0.5`, `direction_mode="uniform"`, `magnitude_dist="lognormal"`, `lognormal_mu=0.0`, `lognormal_sigma=0.25`, `noise_std=0.01`.
    - `pair_correlation=-1.0` for the smoke config and `pair_correlation_grid=[-1.0, -0.75, -0.5, -0.25, 0.0]` for the sweep config.
    - Models for sweep (capacity parity): baseline `GatedSAE sign_aware=false d_hidden=2k` and sign-aware `GatedSAE sign_aware=true d_hidden=2k`.
    - Models for smoke: include the above plus an additional sign-aware efficiency variant `d_hidden=k` (optional but recommended).
  - Confirm defaults reproduce the paper’s qualitative expectations on `pair_correlation=-1`:
    - baseline shows antipodal pairs,
    - sign-aware shows one vector per pair axis and tri-modal signed activation histogram.
  - _Requirements: 1.1, 3.4, 5.3, 5.4_

- [ ] 11. Add unit tests for data generation and correlation feasibility
  - Add `tests/test_toy_model_geometry_validation_data.py` using `unittest`:
    - determinism by seed,
    - antipodal and unit-norm checks,
    - correlation sanity for `r ∈ {-1, -0.5, 0}` within tolerance, computed on examples where `pair_active_mask==1` for the tested pair,
    - infeasible correlation raises `ValueError` with feasible interval in message.
  - _Requirements: 2.1, 2.2, 3.1, 3.2, 9.1, 9.2_

- [ ] 12. Add unit tests for Pair Consolidation Rate correctness
  - Add `tests/test_toy_model_geometry_validation_consolidation.py`:
    - perfect sign-aware “mock model” case where consolidation rate is 1.0,
    - baseline non-negative case where consolidation cannot hold due to sign condition.
  - _Requirements: 7.2, 9.2_

- [ ] 13. Add end-to-end smoke test and artifact completeness test
  - Add `tests/test_toy_model_geometry_validation_smoke.py`:
    - run the CLI on the smoke config,
    - assert required artifacts exist (run root, data, per-model outputs).
  - _Requirements: 8.1, 8.2, 9.3_

- [ ] 14. Measure coverage for Protocol C modules
  - Run Coverage.py for the newly added Protocol C modules and document coverage results.
  - If any module is not at 100% coverage, add tests or record a justification.
  - _Requirements: 9.4_

- [ ] 15. Update documentation
  - Update `README.md` with Protocol C run instructions and a checklist of expected artifacts/plots.
  - _Requirements: 8.1_
