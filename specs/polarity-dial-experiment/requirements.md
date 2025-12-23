# Requirements Document

## Introduction

This document captures the requirements for implementing Protocol A (the "Polarity Dial" experiment) for the Sign-Aware Gated Sparse Autoencoder in the existing `sgnsae` codebase. The goal is to build a reproducible experimental pipeline that:
- Generates synthetic data with asymmetric, signed feature distributions as described in Protocol A.
- Trains and evaluates baseline models (standard Gated SAE and ReLU SAE) and the Sign-Aware Gated SAE on this data.
- Quantifies calibration quality separately for positive and negative regimes along each learned latent axis (e.g., via the split-scale calibration metrics \(\gamma_+\) and \(\gamma_-\)).
- Produces metrics and artifacts that allow clear comparison between models and verification of the Bi-Jump-ReLU behavior.

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

### Requirement 3: Split-scale calibration and axis alignment evaluation

**User Story:** As a research scientist, I want to quantify how well each model estimates feature magnitudes separately for positive and negative regimes along each latent axis, so that I can verify that the Sign-Aware Gated SAE calibrates both sides simultaneously while baselines do not.

#### Acceptance Criteria

1. WHEN training finishes for a given model, THEN the system SHALL compute a latent–ground-truth alignment between decoder columns and synthetic feature directions (e.g., using cosine similarity and the Hungarian matching procedure).
2. WHEN an alignment is available, THEN the system SHALL compute, for each matched latent–feature pair, the ground-truth scalar coefficients \(c_j(x)\) and the model’s scalar estimates \(\hat{c}_i(x)\) based on reconstructions projected onto the matched direction.
3. WHEN evaluating calibration, THEN the system SHALL estimate separate least-squares scaling factors \(\gamma_+\) and \(\gamma_-\) for the positive and negative halves of the data, respectively, as defined in Protocol A.
4. WHEN calibration metrics are computed, THEN the system SHALL aggregate them into summary statistics (e.g., mean and variance of \(\gamma_+\) and \(\gamma_-\) over matched pairs) and store them in a machine-readable format (such as JSON) for later analysis.
5. IF the user requests diagnostic plots, THEN the system SHALL be able to generate simple visualizations (e.g., histograms or scatter plots) illustrating discrepancies between \(\gamma_+\) and \(\gamma_-\) across models.

### Requirement 4: Experiment configuration, reproducibility, and CLI interface

**User Story:** As a practitioner running experiments, I want a simple command-line interface and configuration mechanism for Protocol A, so that I can easily reproduce runs and sweep over key parameters without modifying code.

#### Acceptance Criteria

1. WHEN the user invokes the Protocol A experiment entry point from the command line with a configuration file path or CLI flags, THEN the system SHALL load all relevant hyperparameters and dataset settings without requiring code changes.
2. WHEN a run is started, THEN the system SHALL log the full configuration (including synthetic data parameters, model hyperparameters, training settings, and random seeds) to the output directory.
3. IF the same configuration (including seed) is used in a subsequent run, THEN the system SHALL reproduce identical synthetic data and training behavior within numerical tolerances of the underlying frameworks.
4. WHEN multiple model variants are requested in a single run (e.g., baseline vs. sign-aware model), THEN the system SHALL train and evaluate each variant under the same dataset and configuration, writing results to separate, clearly named subdirectories.

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

