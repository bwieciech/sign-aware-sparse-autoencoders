# Requirements Document

## Introduction

This document captures the requirements for implementing Protocol A (the "Polarity Dial" experiment) for the Sign-Aware Gated Sparse Autoencoder in the existing `sgnsae` codebase. The goal is to build a reproducible experimental pipeline that:
- Generates synthetic data with asymmetric, signed feature distributions as described in Protocol A.
- Trains and evaluates baseline models (standard Gated SAE and ReLU SAE) and the Sign-Aware Gated SAE on this data.
- Quantifies calibration quality separately for positive and negative regimes along each learned latent axis (e.g., via the split-scale calibration metrics \(\gamma_+\) and \(\gamma_-\)), using an axis-aware evaluation that is fair to non-negative baselines.
- Produces metrics and artifacts that allow clear comparison between models and verification of the Bi-Jump-ReLU behavior.
- Produces calibration and sparsity-fidelity trade-off curves (MSE vs \(L_0\)) over a sweep of sparsity strengths, and produces an asymmetry sweep (the "dial") over key generator parameters.

## Requirements

### Requirement 1: Synthetic dataset generation for polarity-asymmetric signed features

**User Story:** As a research engineer, I want a configurable synthetic data generator that produces signed, polarity-asymmetric features aligned with random directions, so that I can rigorously evaluate how different SAE variants handle asymmetric positive/negative evidence along shared axes.

#### Acceptance Criteria

1. WHEN the experiment is run with a specified random seed, THEN the system SHALL generate the same synthetic dataset deterministically.
2. WHEN the experiment is configured with parameters \((N, d, k, \rho, p_+)\), THEN the system SHALL generate inputs \(x \in \mathbb{R}^d\) as linear combinations of feature directions with:
   - feature activations drawn with probability \(\rho\),
   - signs drawn from a Bernoulli distribution with parameter \(p_+\),
   - magnitudes following a LogNormal distribution for positive signs and an Exponential distribution for negative signs, as described in Protocol A.
3. WHEN synthetic samples are generated, THEN the system SHALL add configurable Gaussian noise \(\epsilon \sim \mathcal{N}(0, \sigma^2 I)\) to match the protocol specification.
4. IF the user requests a diagnostic export of the synthetic dataset, THEN the system SHALL be able to persist a small sample (e.g., to NumPy or Torch tensor files) together with the generator configuration for inspection.

### Requirement 2: Model variants and training loop for Protocol A

**User Story:** As a research engineer, I want a unified training loop that can train multiple SAE variants (ReLU SAE, standard Gated SAE, Sign-Aware Gated SAE) on the same synthetic dataset, so that I can compare their behavior under matched conditions.

#### Acceptance Criteria

1. WHEN the experiment is configured to use a particular model class (e.g., `SAE`, `GatedSAE`, sign-aware variant), THEN the system SHALL instantiate that model with hyperparameters (width, depth, sign-aware flags, sparsity coefficients, learning rate, etc.) specified in a configuration object or file.
2. WHEN training is started, THEN the system SHALL use a shared training loop abstraction (data loader, optimizer, epochs, batches) that can be reused for each model variant without duplicating core logic.
3. IF a model is sign-aware, THEN the system SHALL correctly initialize and use the additional parameters controlling the Bi-Jump-ReLU-like behavior (e.g., activation offsets / thresholds and separate positive/negative magnitude parameters) according to the existing `sgnsae` implementations.
4. WHEN training completes for a model, THEN the system SHALL persist all relevant training metrics (loss curves, sparsity statistics) and model checkpoints to a designated output directory, tagged with the model variant and configuration identifier.
5. IF a model variant is configured with a different width than other variants (e.g., a `2x_width` baseline), THEN the system SHALL train and evaluate it under the same dataset split and training schedule, while clearly recording its width and parameter count in its output artifacts.

### Requirement 3: Split-scale calibration and axis alignment evaluation

**User Story:** As a research scientist, I want to quantify how well each model estimates feature magnitudes separately for positive and negative regimes along each latent axis, so that I can verify that the Sign-Aware Gated SAE calibrates both sides simultaneously while baselines are evaluated fairly (including baselines that use two latents per axis).

#### Acceptance Criteria

1. WHEN training finishes for a given model, THEN the system SHALL compute a latent–ground-truth alignment that is axis-aware:
   - It SHALL use \(|\cos(D_{:,i}, u_j)|\) (absolute cosine similarity) for matching so that decoder columns aligned to either \(+u_j\) or \(-u_j\) can match axis \(u_j\).
   - It SHALL record the signed alignment (whether \(D_{:,i}\) aligns with \(+u_j\) or \(-u_j\)) for each matched latent.
2. WHEN evaluating a non-negative model (ReLU SAE or non-sign-aware Gated SAE), THEN the system SHALL support an axis-level aggregation mode:
   - It SHALL allow up to two latents to be assigned to a single ground-truth axis (one aligned to \(+u_j\), one aligned to \(-u_j\)), and it SHALL compute an axis-level coefficient estimate \(\hat{c}_j(x)\) by combining both contributions.
   - It SHALL store the per-axis mapping in an artifact (e.g., `axis_alignment.json`) distinct from one-to-one latent matching.
   - It SHALL treat a candidate latent as *unassigned* if its absolute cosine alignment to the axis is below a configurable threshold (`min_axis_abs_cos`, default `0.9`), so that weak/noisy matches cannot create misleading “negative-regime” calibration for non-negative models.
   - It SHALL treat any missing side (no latent above threshold) as a zero contribution during axis-level coefficient aggregation.
3. WHEN an alignment is available, THEN the system SHALL compute, for each ground-truth axis \(j\), the ground-truth scalar coefficients \(c_j(x)\) and the model’s scalar estimates \(\hat{c}_j(x)\) using a scale-consistent method:
   - EITHER normalize decoder columns for evaluation, OR compute coefficients via a projection that is invariant to decoder scaling, and record which method is used.
4. WHEN evaluating calibration, THEN the system SHALL estimate separate least-squares scaling factors \(\gamma_+\) and \(\gamma_-\) for the positive and negative halves of the data, respectively, as defined in Protocol A, at the axis level (not only the latent level).
5. WHEN calibration metrics are computed, THEN the system SHALL also compute robust per-regime goodness-of-fit diagnostics:
   - correlation (Pearson \(r\)) between \(c_j(x)\) and \(\hat{c}_j(x)\) on each regime,
   - coefficient-space MSE on each regime,
   - and \(R^2\) (or explained variance) on each regime.
6. WHEN calibration metrics are aggregated, THEN the system SHALL include:
   - mean and standard deviation over axes for each metric,
   - the number of axes evaluated,
   - and the per-axis counts of positive and negative samples, so reviewers can detect ill-conditioned fits.
7. IF a regime for a given axis has too few samples (configurable threshold, default 50), THEN the system SHALL skip \(\gamma\) fitting for that regime for that axis and record `null` for the missing values, while still reporting other available metrics.
8. IF the user requests diagnostic plots, THEN the system SHALL be able to generate plots that reveal metric fragility (e.g., predicted-vs-true scatter per regime, and histograms of coefficient predictions showing near-zero collapse).

9. WHEN axis-level aggregation is used, THEN the system SHALL report axis coverage diagnostics including the number and fraction of ground-truth axes that have:
   - a valid positive-side latent, a valid negative-side latent, both, and neither (at the configured `min_axis_abs_cos` threshold).

### Requirement 4: Experiment configuration, reproducibility, and CLI interface

**User Story:** As a practitioner running experiments, I want a simple command-line interface and configuration mechanism for Protocol A, so that I can easily reproduce runs and sweep over key parameters without modifying code.

#### Acceptance Criteria

1. WHEN the user invokes the Protocol A experiment entry point from the command line with a configuration file path or CLI flags, THEN the system SHALL load all relevant hyperparameters and dataset settings without requiring code changes.
2. WHEN a run is started, THEN the system SHALL log the full configuration (including synthetic data parameters, model hyperparameters, training settings, and random seeds) to the output directory.
3. IF the same configuration (including seed) is used in a subsequent run, THEN the system SHALL reproduce identical synthetic data and training behavior within numerical tolerances of the underlying frameworks.
4. WHEN multiple model variants are requested in a single run (e.g., baseline vs. sign-aware model), THEN the system SHALL train and evaluate each variant under the same dataset and configuration, writing results to separate, clearly named subdirectories.
5. IF the configuration requests sweeps (over sparsity strengths and/or data asymmetry parameters), THEN the system SHALL run the sweep deterministically and produce a single aggregated summary artifact that includes all sweep points.

6. WHEN training multiple model variants in the same run, THEN the system SHALL reseed all relevant RNGs deterministically per model (Python, NumPy, Torch CPU, Torch CUDA) using a stable function of the dataset seed and the model name, so that a model’s results do not depend on the ordering of models in the config. The derived per-model seed SHALL be recorded in per-model artifacts.

### Requirement 5: Metrics reporting and comparison for Protocol A

**User Story:** As a research scientist, I want clear, structured outputs that highlight differences in calibration behavior between models on the Polarity Dial experiment, so that I can draw conclusions about the benefits of the Sign-Aware Gated SAE.

#### Acceptance Criteria

1. WHEN an experiment run finishes, THEN the system SHALL produce a summary report artifact (e.g., JSON or a small text/markdown file) that includes:
   - Core training statistics (losses, sparsity),
   - Split-scale calibration metrics \(\gamma_+\) and \(\gamma_-\),
   - Key configuration details (model type, width, data parameters).
2. WHEN multiple model variants are trained under the same configuration, THEN the system SHALL create a comparison artifact that juxtaposes their key metrics (e.g., calibration errors, alignment quality) in a single structure for easy downstream analysis.
3. IF a logging backend (such as Weights & Biases) is enabled, THEN the system SHALL log the experiment configuration and key metrics for each model variant under clearly separated runs or groups, consistent with the existing training scripts.
4. WHEN an experiment run encounters an error, THEN the system SHALL still attempt to persist partial logs (e.g., configuration and metrics computed up to failure) to aid debugging, provided that doing so does not corrupt existing artifacts.

### Requirement 6: Fair baselines and trade-off reporting

**User Story:** As a reviewer, I want to see that conclusions about sign-awareness are not artifacts of evaluation or representational mismatch, so that I can trust that improvements are real and not due to an unfair baseline.

#### Acceptance Criteria

1. WHEN Protocol A runs baselines, THEN the system SHALL support at least the following baseline families:
   - `2x_width` non-negative baseline(s) (ReLU SAE and/or non-sign-aware Gated SAE) with width \(2H\) for representational parity on signed axes,
   - a signed-latent baseline that supports negative coefficients (e.g., soft-threshold or signed shrinkage encoder), even if it is not gated.
2. WHEN baselines are evaluated, THEN the system SHALL use the axis-aware evaluation described in Requirement 3, so that a baseline that learns both \(+u_j\) and \(-u_j\) is credited correctly.
3. WHEN summarizing results, THEN the system SHALL produce MSE vs \(L_0\) trade-off curves across a configured list of sparsity coefficients, for each model variant.
4. WHEN sweeping data asymmetry parameters (the "dial"), THEN the system SHALL produce a summary plot and JSON artifact that show how calibration metrics change as a function of:
   - sign probability \(p_+\),
   - and at least one tail-heaviness control (e.g., `exponential_lambda` and/or `lognormal_sigma`).
5. WHEN reporting cross-model comparisons, THEN the system SHALL include model width and an approximate parameter count, so that comparisons can be interpreted under capacity constraints.

6. WHEN using a signed-latent baseline (e.g., Signed Soft-Threshold SAE), THEN the baseline encoder/decoder parameterization SHALL be scale-invariant (e.g., normalize decoder columns in the forward pass and compute encoder projections against the normalized decoder), so that calibration metrics cannot be improved or degraded by trivial rescaling of encoder/decoder weights.

### Requirement 7: Dependency robustness and environment metadata

**User Story:** As a practitioner reproducing results, I want runs to fail fast with clear dependency errors and to record environment metadata, so I can reproduce results and debug issues without guesswork.

#### Acceptance Criteria

1. WHEN the Protocol A runner starts, THEN it SHALL validate that required dependencies are available in compatible versions (at minimum: PyTorch, NumPy, SciPy, Matplotlib; and Einops *if* the codebase uses `einops.einsum`).
2. IF a required API is missing (e.g., `einops.einsum`), THEN the system SHALL raise a clear error message explaining exactly what to install (e.g., “install `requirements.txt`”) rather than failing with an opaque stack trace.
3. WHEN a run begins, THEN the system SHALL write an `environment.json` artifact capturing: Python version, OS, Torch version, CUDA availability, and key library versions (numpy, scipy, matplotlib, einops), to support reproducibility audits.
