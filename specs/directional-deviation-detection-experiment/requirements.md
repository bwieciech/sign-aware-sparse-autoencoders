# Requirements: Protocol B Extensions (Directional Deviation Detection)

## Context and Motivation

The existing Protocol B pipeline successfully runs end-to-end, but the initial “paper run” on **isotropic** normal data
(`x ~ N(0, I)`) produces **near-chance directional recall** (≈ target FPR) and **0 coverage**. This is expected: isotropic
Gaussians are rotationally symmetric, so *coordinate axes (sensor channels) are not identifiable* and learned latents
will generally not align to channels.

This document updates the requirements to implement the improvements suggested in the analysis:

- **5A — Break rotational symmetry** of the normal data distribution so “channel detectors” are learnable.
- **5B — Feature-space anomalies**: generate anomalies along *learned* feature directions, not just coordinate axes.
- **5C — Localized scoring**: add pooling strategies (max/top‑k) and optional target-restricted scoring for single-channel anomalies.
- **5D — Diagnostics**: log axis-alignment / channel-purity metrics to verify whether “channel semantics” are actually learned.

## Definitions

- `N`: number of input channels (`num_channels`)
- `H`: SAE width / number of latents (`d_hidden`)
- “High” anomaly: positive shift along the target direction (`+Δ`)
- “Low” anomaly: negative shift along the target direction (`-Δ`)
- “Target”: what an anomaly is applied to:
  - `channel` for coordinate anomalies
  - `direction` (index into a direction list) for feature-space anomalies

## Non‑Goals

- Do not change model architectures (SAE/GatedSAE/Sign-Aware Gated SAE).
- Do not change the baseline “no two-sided evidence leakage” rule.
- Do not add external dependencies beyond what is already used (PyTorch, numpy, matplotlib, scipy if already present).

---

## Requirements

### R1. Configuration must support Protocol B variants (5A–5D)

**User story:** As a research engineer, I can select the normal distribution, anomaly type, scoring pooling, and diagnostics
from a single config file and reproduce results.

**Acceptance criteria:**

1. The experiment config SHALL remain loadable from JSON (YAML optional).
2. The config SHALL validate parameters with clear error messages (e.g., missing distribution parameters, invalid ranges).
3. New configuration fields SHALL have safe defaults that reproduce the current behavior when not set.
4. The resolved, fully-expanded config SHALL be written to `<run_dir>/config.json`.

---

### R2. Normal data generation must support symmetry-breaking distributions (5A)

**User story:** As a scientist, I can generate normal data where coordinate axes are identifiable so single-channel anomalies
can be detected and “coverage” is meaningful.

**Normal distribution modes (configurable):**

1. `standard_normal`: `x ~ N(0, I)` (current default; kept for sanity checks).
2. `heteroskedastic_normal`: `x_j ~ N(0, σ_j^2)` with user-provided `channel_stds` (must be length `N`).
3. `iid_non_gaussian`: independent, identically distributed non-Gaussian channels, with distribution:
   - `laplace` (zero mean, unit variance),
   - `logistic` (zero mean, unit variance),
   - `student_t` (df > 2, scaled to unit variance).
4. (Optional but recommended) `non_gaussian_heteroskedastic`: same as `iid_non_gaussian` but additionally scaled by `channel_stds`.

**Acceptance criteria:**

1. WHEN `normal_mode` is non-default, THEN `generator_config.json` SHALL record the mode and all distribution parameters.
2. WHEN `heteroskedastic_normal` is selected, THEN:
   - `channel_stds` MUST be provided and length `N`,
   - all `channel_stds[j] > 0`,
   - the config MAY enforce `all_unique(channel_stds)` if `ensure_unique_channel_stds=true` (default true) to avoid axis ambiguity.
3. WHEN `iid_non_gaussian` or `non_gaussian_heteroskedastic` is selected, THEN:
   - the generator MUST produce approximately zero-mean samples (within tolerance on a large sample),
   - each channel MUST have approximately the configured variance (unit variance before optional `channel_stds` scaling).
4. GIVEN the same seed, mode, and parameters, the generated normal tensors SHALL be identical across runs.

---

### R3. Anomaly generation must support coordinate and feature-space targets (5B)

**User story:** As a scientist, I can test anomaly detection both on coordinate shifts (channels) and on shifts along learned
feature directions.

**Supported anomaly shift modes:**

1. `single_channel` (existing): shift exactly one channel by `±Δ`.
2. `subset` (existing): shift a subset mask of size `subset_size` by `±Δ`.
3. `feature_direction` (new): shift along one of `M` unit directions `v_k ∈ R^N` by `±Δ`.

**Direction sources for `feature_direction`:**

- `reference_model_decoder` (required for the paper): directions are taken from the **decoder columns** of a reference model
  (default reference model: `sign_aware_gated_sae`) trained on the same normal data.
- `fixed_random` (optional): directions are a fixed random orthonormal (or normalized random) set generated from the seed.
- `per_model_decoder` (optional diagnostic): directions are taken from each model’s own decoder (self-evaluation only).

**Acceptance criteria:**

1. For `feature_direction`, the run MUST write:
   - `anomaly_directions.pt` (shape `(M, N)`),
   - `anomaly_manifest.json` including `direction_source`, selection rules, and the mapping from `direction_index -> latent_index` (if applicable).
2. The anomaly generator SHALL support `anomaly_eval_mode="stream"` for memory safety:
   - It must not require materializing all anomalies in memory at once.
3. Labels MUST support both target kinds:
   - `labels_type ∈ {0=normal, 1=high, 2=low}`
   - `labels_target` (channel index for coordinate, direction index for feature)
   - `labels_target_kind ∈ {"channel","direction"}` stored in the manifest (and/or in `evaluation.json`).
4. For `single_channel`, each anomaly sample MUST differ from its base sample in exactly one coordinate by exactly `±Δ` (plus optional noise if configured).

---

### R4. Scoring must support localized pooling and (optional) target-restricted scoring (5C)

**User story:** As a scientist, I can choose scoring aggregation that is appropriate for localized anomalies and reduces noise
from summing over many unrelated latents.

**Pooling modes (configurable):**

- `sum` (current behavior): `Σ_i relu(...)`
- `max`: `max_i relu(...)`
- `topk`: `Σ_{i in TopK} relu(...)` with configurable `k`

**Target-restricted scoring (optional, for `single_channel`):**

- If enabled, the score for channel `j` uses only latents assigned to that channel (via `argmax |D_{j,i}|`), rather than all latents.

**Acceptance criteria:**

1. Pooling mode SHALL apply consistently to:
   - sign-aware scores (`Score_high`, `Score_low` based on `π` and `(τ^+, τ^-)`),
   - baseline scores (one-sided evidence `e` and direction partition).
2. For `topk`, `k` MUST satisfy `1 ≤ k ≤ H`.
3. For target-restricted scoring:
   - If no latents are assigned to a target, the score MUST be defined as 0.0 (not NaN).
4. Score threshold calibration for fixed FPR MUST continue to target `score_fpr` on validation normal data for each direction.

---

### R5. Evaluation must report new metrics and diagnostics (5D)

**User story:** As a scientist, I can tell whether poor recall is due to model failure or due to “no axis alignment.”

**Required metrics:**

1. Directional Recall @ Fixed FPR (`high`, `low`)
2. Achieved FPR sanity check on held-out normal test set
3. Coverage (fraction of targets with both-direction recall ≥ `coverage_recall_target`), for:
   - channels when `shift_mode="single_channel"`
   - directions when `shift_mode="feature_direction"`

**Required diagnostics (decoder alignment):**

Compute and persist (per model):

- `latent_argmax_channel[i] = argmax_j |D[j,i]|`
- `latent_purity[i] = |D[j*,i]| / ||D[:,i]||_2`  (in `[0,1]`)
- `unique_channels_claimed = |{latent_argmax_channel[i]}|`
- Summary statistics: mean/median purity, 10/50/90th percentiles, and a histogram bucketed for plotting.

**Acceptance criteria:**

1. Diagnostics SHALL be written to `models/<model_name>/decoder_alignment.json`.
2. The run root `comparison_summary.json` SHALL include alignment summary alongside recall/coverage.
3. Plotting SHALL include at least:
   - `decoder_purity_hist.png` per model (or a combined figure),
   - `unique_channels_vs_width.png` for width sweeps (optional if no sweep).

---

### R6. Artifacts must be reproducible and minimally sufficient

**Acceptance criteria:**

1. All artifacts required to reproduce evaluation MUST be written under the run directory:
   - full config, generator config, anomaly manifest, models, thresholds, evaluation, diagnostics, plots.
2. Artifacts MUST clearly distinguish:
   - target kind (`channel` vs `direction`),
   - anomaly mode (`single_channel` / `subset` / `feature_direction`),
   - distribution mode (`standard_normal` / …).
3. If any stage fails, the system SHOULD write partial artifacts where safe (`config.json`, logs, partial metrics) with an explicit error field.

---

### R7. Tests must cover 5A–5D

**Acceptance criteria:**

1. Data generation tests for each `normal_mode`:
   - determinism,
   - variance sanity checks,
   - shape checks.
2. Feature-direction anomaly tests:
   - direction selection determinism,
   - anomaly perturbation matches `±Δ v_k`.
3. Scoring tests for pooling modes:
   - `sum`, `max`, `topk` behave as expected on fixed tensors.
4. Decoder alignment diagnostics test:
   - purity is 1.0 for one-hot columns,
   - purity matches expected value for uniform columns.
