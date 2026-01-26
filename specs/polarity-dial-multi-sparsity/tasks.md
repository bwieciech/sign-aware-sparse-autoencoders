# Implementation Plan (Tasks)

## 1) Expand gradient-independence unit tests (must happen first)

- [x] 1.1 Extend `tests/test_model_instance_gradient_independence.py` to test each loss component
  - Add coverage for `loss_tensor_key` in:
    - `"losses"`
    - `"reconstruction_losses"`
    - `"sparsity_losses"`
    - `"aux_loss"` (only for models that return it, e.g. `GatedSAE`)
  - For each `loss_tensor_key`, verify for **both** instances:
    - Backprop `losses_dict[loss_tensor_key].sum()` and store grads.
    - Backprop `losses_dict[loss_tensor_key][0]` and assert:
      - instance-0 parameter slices match the stored grads’ instance-0 slices,
      - instance-1 parameter slices are exactly zero.
    - Backprop `losses_dict[loss_tensor_key][1]` and assert the symmetric condition (instance-1 matches, instance-0 zero).
  - Keep the existing parameter filtering rule: only assert slice-level conditions for parameters whose grad has leading dimension `n_instances`.
  - Requirements: 1.1, 1.2, 1.3, 1.4

- [x] 1.2 Add targeted “broadcasted y_true” coverage
  - Add at least one test per model family where:
    - `y_true` is provided as `(batch, d_in)` (broadcasted across instances),
    - gradients still do not mix across instances.
  - Requirements: 1.1, 1.2

## 2) Enable list-valued `sparsity_coeff` parallelism in Protocol A training

- [x] 2.1 Update model instantiation to support `n_instances > 1`
  - Modify `sgnsae/experiments/polarity_dial/training.py`:
    - Plumb `n_instances` into `_instantiate_model(...)`.
    - For list mode, set `n_instances = len(sparsity_coeff_list)`.
    - For scalar mode, keep `n_instances = 1`.
  - Requirements: 3.1, 3.2

- [x] 2.2 Ensure identical initialization across instances in list mode
  - After instantiation in list mode, copy instance-0 parameter values into every instance slice for all parameters with a leading instance dimension.
  - Requirements: 3.1

- [x] 2.3 Build `sparsity_coeffs` tensors correctly for both modes
  - Scalar mode: `sparsity_coeffs.shape == (1,)`.
  - List mode: `sparsity_coeffs.shape == (n_instances,)`, aligned with coefficient index.
  - Requirements: 3.3

- [x] 2.4 Track per-instance training metrics in list mode while preserving scalar-mode output format
  - Keep existing scalar keys and structure unchanged when `n_instances == 1`.
  - In list mode, record per-instance epoch series for:
    - total loss
    - reconstruction loss
    - sparsity loss
    - aux loss (if present)
  - Include `sparsity_coeff_list` (exact list) in training metrics for list mode.
  - Requirements: 3.4, 4.1

- [x] 2.5 Make dead-latent resampling bookkeeping per-instance
  - Ensure `inactive_latents_count` has shape `(n_instances, d_hidden)` and is updated per-instance.
  - Requirements: 3.1

- [x] 2.6 Allow list-valued `sparsity_coeff` in configs
  - Update `PolarityDialModelVariantConfig` validation to accept either:
    - a single non-negative numeric value, or
    - a non-empty list of non-negative numeric values.
  - Ensure mixed float/list usage across models remains valid.
  - Requirements: 2.1, 2.2, 2.3

## 3) Add multi-sparsity “subrun” artifact generation in the Protocol A runner

- [x] 3.1 Detect list mode at the seed-run level and compute `subrun_count`
  - In `sgnsae/experiments/polarity_dial/runner.py`, determine whether any model uses list mode.
  - Compute `subrun_count = max(1, max_list_len_across_models)` where scalar models contribute length 1.
  - Requirements: 4.2

- [x] 3.2 Train each model variant once per seed, using list parallelism when requested
  - List-mode models: train once with `n_instances = len(list)`.
  - Scalar-mode models: train once with `n_instances = 1`.
  - Preserve deterministic per-model seeding.
  - Requirements: 3.1, 3.2, 3.5

- [x] 3.3 Implement “slice model to single instance” utility
  - Add a helper (module location to be chosen during implementation) that:
    - constructs a same-class model with `n_instances = 1`,
    - copies parameter slice `i` into the new model’s instance-0 parameters for all per-instance parameters,
    - preserves non-sliced state (constants/attributes) required for evaluation.
  - This helper is used for generating scalar-like artifacts for coefficient subruns.
  - Requirements: 4.3

- [x] 3.4 Write coefficient subrun directories under each seed
  - For each coefficient index `i` in `[0, subrun_count)`:
    - Create `<seed_run_dir>/<i>/`.
    - Create a scalar-style `config.json` containing only models included at index `i`:
      - list models included only if `i < len(list)`, using `sparsity_coeff = list[i]`,
      - scalar models included always, using their float coefficient.
    - Copy or regenerate `environment.json` into the subrun directory.
    - Copy or regenerate the `data/` export directory into the subrun directory.
    - For each included model:
      - slice or reuse the trained model as `n_instances=1`,
      - write per-model artifacts into `<subrun_dir>/models/<model_name>/...` using existing persistence + plotting code paths.
    - Create `<subrun_dir>/comparison_summary.json` and `<subrun_dir>/gamma_diff_bar_comparison.png` for the models included in that subrun.
  - Requirements: 4.2, 4.3

- [x] 3.5 Define and implement sweep artifact behavior in list mode (no disabling)
  - Ensure sweeps remain enabled and run as they do today.
  - Keep sweep artifacts at the seed-run root (no duplication into coefficient subruns), as per design.
  - Requirements: 4.4

## 4) Add seed-level per-model loss-vs-sparsity plots (list mode)

- [x] 4.1 Add plotting utilities for per-model multi-sparsity summaries
  - Extend `sgnsae/experiments/polarity_dial/plotting.py` with functions that write, per model:
    - sparsity loss vs coefficient
    - sparsity loss vs log(coefficient)
    - reconstruction loss vs coefficient
    - reconstruction loss vs log(coefficient)
    - aux loss vs coefficient (if present)
    - aux loss vs log(coefficient) (if present)
  - Write outputs under `<seed_run_dir>/plots/multi_sparsity/<model_name>/`.
  - Requirements: 5.1, 5.2, 5.3

- [x] 4.2 Invoke these plots from the Protocol A runner in list mode
  - For each list-mode model, use the final-epoch per-instance metrics to populate the plot y-values.
  - Requirements: 5.1, 5.3

## 5) Add Protocol A list-mode end-to-end tests (runner + artifacts)

- [x] 5.1 Add a small end-to-end list-mode smoke test
  - Create a tiny config in a temp directory with:
    - at least one model in list mode with length 2 or 3,
    - at least one model in scalar mode,
    - short training (1–2 epochs) and tiny dataset sizes.
  - Assert:
    - `<seed_run_dir>/0/`, `<seed_run_dir>/1/`, ... exist up to `subrun_count - 1`.
    - each subrun contains run-level artifacts (`config.json`, `environment.json`, `comparison_summary.json`, `gamma_diff_bar_comparison.png`) and `data/` exports.
    - model directories exist only for models included at that coefficient index.
  - Requirements: 4.2, 4.3

- [x] 5.2 Add assertions for seed-level per-model multi-sparsity plot artifacts
  - Assert plots exist for list-mode models under `<seed_run_dir>/plots/multi_sparsity/<model_name>/`.
  - Requirements: 5.1, 5.2

- [x] 5.3 Ensure scalar-mode behavior remains unchanged
  - Keep existing Protocol A scalar-mode smoke and artifact tests passing without modifications to their expectations unless strictly necessary.
  - Requirements: 4.1

## 6) Coverage + regression checks

- [x] 6.1 Run unit tests and verify passing status
  - Run the full `unittest` suite.

- [x] 6.2 Measure coverage for modified/new code
  - Use `coverage.py` to measure coverage for files changed under `sgnsae/experiments/polarity_dial` and `tests/`.
  - Aim for ~100% coverage of added/modified logic (excluding scripts).
