# Requirements Document

## Introduction

This document captures the requirements for implementing Protocol B (Directional Deviation Detection; resource constrained) for the Sign-Aware Gated Sparse Autoencoder in the existing `sgnsae` codebase. The goal is to build a reproducible experimental pipeline that:
- Generates synthetic “sensor array” data with directional anomalies (“too high” vs “too low”) matching the Protocol B description in `main.tex`.
- Trains baseline models (standard non-signed variants) and the Sign-Aware variant under a strict latent budget constraint \(H < N\).
- Implements direction-aware anomaly scoring using per-latent thresholds and evaluates Directional Recall @ Fixed FPR separately for “high” and “low” anomalies.
- Produces structured metrics and artifacts to compare models as a function of latent budget.

## Requirements

### Requirement 1: Synthetic dataset generation for directional deviation anomalies

**User Story:** As a research engineer, I want a configurable synthetic dataset generator for a high-dimensional sensor array with directional anomalies, so that I can benchmark whether different SAE variants can detect “too high” and “too low” deviations under a constrained latent budget.

#### Acceptance Criteria

1. WHEN the experiment is run with a specified random seed, THEN the system SHALL generate the same synthetic datasets deterministically.
2. WHEN the experiment is configured with a sensor dimension \(N\), THEN the system SHALL generate “normal” samples \(x \in \mathbb{R}^N\) with \(x \sim \mathcal{N}(0, I)\) componentwise.
3. WHEN the experiment is configured with an anomaly magnitude \(\Delta\) and anomaly specification (affected coordinates, anomaly sign), THEN the system SHALL generate anomalous samples by applying additive offsets of \(+\Delta\) (“high”) or \(-\Delta\) (“low”) on the specified coordinates.
4. WHEN anomalies are generated, THEN the system SHALL produce machine-readable labels that identify:
   - whether the sample is normal/high/low, and
   - which coordinate subset was shifted.
5. IF the user configures the anomaly generator to shift exactly one coordinate per anomaly, THEN the system SHALL support generating balanced per-coordinate, per-direction anomaly test sets (so each channel can be evaluated independently for both “high” and “low” anomalies).
6. IF the experiment is configured with a non-default normal data mode that breaks rotational symmetry (e.g., heteroskedastic per-channel variances), THEN the system SHALL generate “normal” samples according to that mode and SHALL persist the exact mode and parameters in the generator configuration artifact.
7. IF the user requests a diagnostic export of the synthetic datasets, THEN the system SHALL be able to persist a small sample (e.g., Torch tensors) together with the generator configuration for inspection.
8. WHEN \(N\) and/or the configured anomaly test sizes are large, THEN the system SHALL support memory-safe anomaly evaluation by generating anomalies and/or scoring them in batches (streaming), rather than requiring all anomaly samples to be materialized in a single in-memory tensor.

### Requirement 2: Model variants and training loop under a latent budget constraint

**User Story:** As a research engineer, I want a unified training loop that can train multiple SAE variants under an explicit width constraint \(H < N\), so that I can compare directional detection performance at fixed latent budget.

#### Acceptance Criteria

1. WHEN the experiment is configured to use a particular model variant, THEN the system SHALL instantiate that model with width \(H\) and other hyperparameters specified in a configuration object or file.
2. WHEN training is started, THEN the system SHALL train the model using only “normal” training data by default (unsupervised training), and SHALL NOT require anomaly labels during optimization.
3. IF the experiment configuration requests training on a mixture of normal and anomalous data, THEN the system SHALL support this alternative regime and log that it was used.
4. WHEN training completes for a model, THEN the system SHALL persist all relevant training metrics (loss curves, sparsity statistics) and a model checkpoint to a designated output directory, tagged with the model variant and configuration identifier.
5. WHEN multiple model variants are requested in a single run, THEN the system SHALL train and evaluate each variant under the same dataset and latent budget settings, writing results to separate, clearly named subdirectories.

### Requirement 3: Direction-aware anomaly scoring and threshold calibration

**User Story:** As a research scientist, I want separate anomaly detectors for “too high” and “too low” deviations derived from model internals, so that I can measure directional recall at a fixed false positive rate and compare how many channels each model can monitor under the same latent budget.

#### Acceptance Criteria

1. WHEN evaluating a Sign-Aware Gated SAE, THEN the system SHALL compute directional anomaly scores using the Protocol B definitions:
   - \(\text{Score}_{\text{high}}(x) = \sum_i \text{ReLU}(\pi_i(x) - \tau_i^+)\)
   - \(\text{Score}_{\text{low}}(x) = \sum_i \text{ReLU}(-\pi_i(x) - \tau_i^-)\)
   where \(\pi_i(x)\) is the gate pre-activation (or equivalent internal detection signal) and \(\tau_i^+\), \(\tau_i^-\) are calibrated per latent.
2. WHEN calibrating per-latent thresholds, THEN the system SHALL estimate them from a clean (normal-only) validation set using a quantile rule:
   - for Sign-Aware Gated SAE: default 99th percentile of \(\pi_i(x)\) and of \(-\pi_i(x)\), respectively, yielding \(\tau_i^+\) and \(\tau_i^-\).
   - for baseline (non-signed / non-negative) models: default 99th percentile of a one-sided evidence signal (e.g., post-nonlinearity activations), yielding per-latent evidence thresholds used by that baseline’s directional scoring.
3. WHEN evaluating baseline models, THEN the system SHALL implement a comparable pair of directional scores that do NOT grant the baseline access to two-sided evidence from a single latent. Concretely:
   - baseline directional scores SHALL be derived from one-sided evidence signals available to the baseline model (e.g., post-nonlinearity activations for ReLU SAEs; one-sided gate/activation signals for gated baselines),
   - baseline scoring SHALL NOT use the negative tail of raw signed pre-activations from the same latent as a “low” detector,
   - and the system SHALL document in the run artifacts exactly which internal signals and any latent-to-direction assignment rules were used.
4. WHEN choosing detection thresholds for \(\text{Score}_{\text{high}}\) and \(\text{Score}_{\text{low}}\), THEN the system SHALL set per-direction thresholds on normal validation data to achieve a configured fixed false positive rate (default: \(1\%\)).
5. WHEN evaluating on labeled anomaly test sets, THEN the system SHALL compute Directional Recall @ Fixed FPR separately for high and low anomalies, and SHALL report these metrics per anomaly type.
6. IF the experiment is configured to compute per-channel metrics, THEN the system SHALL compute and report per-channel directional recall and an aggregate “both-directions covered” statistic (e.g., number or fraction of channels where both high and low recall exceed a configured target).
7. WHEN the evaluation encounters degenerate calibration conditions (e.g., too few validation samples to estimate a stable quantile), THEN the system SHALL fail with a clear error message or record `null` metrics rather than producing silently incorrect results.

### Requirement 4: Experiment configuration, reproducibility, and CLI interface

**User Story:** As a practitioner running experiments, I want a simple command-line interface and configuration mechanism for Protocol B, so that I can easily reproduce runs and sweep over latent budgets and anomaly magnitudes without modifying code.

#### Acceptance Criteria

1. WHEN the user invokes the Protocol B experiment entry point from the command line with a configuration file path or CLI flags, THEN the system SHALL load all relevant hyperparameters and dataset settings without requiring code changes.
2. WHEN a run is started, THEN the system SHALL log the full configuration (including dataset parameters, anomaly parameters, model hyperparameters, training settings, and random seeds) to the output directory, with any secrets redacted.
3. IF the same configuration (including seed) is used in a subsequent run, THEN the system SHALL reproduce identical synthetic data deterministically and SHALL reproduce evaluation results within numerical tolerances of the underlying frameworks (noting that training may be nondeterministic on some hardware unless explicitly configured for deterministic algorithms).
4. WHEN multiple latent budgets \(H\) are requested in a sweep, THEN the system SHALL run each budget as a separate sub-run and collate results into a single comparison artifact.
5. IF the experiment is configured with multiple random seeds, THEN the system SHALL run each seed as a separate sub-run and collate mean/std (or equivalent) summary statistics into a single comparison artifact.

### Requirement 5: Metrics reporting and comparison artifacts

**User Story:** As a research scientist, I want clear, structured outputs that highlight differences in directional detection performance between models as a function of latent budget, so that I can validate the hypothesized efficiency advantage of the Sign-Aware model.

#### Acceptance Criteria

1. WHEN an experiment run finishes, THEN the system SHALL produce a summary report artifact (e.g., JSON) that includes:
   - Core training statistics (losses, sparsity),
   - Directional recall metrics for “high” and “low” anomalies at fixed FPR,
   - Any per-channel coverage statistics (if enabled),
   - Key configuration details (model type, width \(H\), \(\Delta\), anomaly coordinate settings).
2. WHEN multiple model variants are trained under the same configuration, THEN the system SHALL create a comparison artifact that juxtaposes their key directional detection metrics in a single structure for easy downstream analysis.
3. WHEN an experiment run encounters an error, THEN the system SHALL still attempt to persist partial logs (e.g., configuration and metrics computed up to failure) to aid debugging, provided that doing so does not corrupt existing artifacts.
4. WHEN reporting fixed-FPR results, THEN the system SHALL also report the achieved false positive rate on a held-out normal test set (where available) as a sanity check for calibration drift.
