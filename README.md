# Sign-Aware Sparse Autoencoders

This repository contains research tooling for exploring sign-aware sparse
autoencoders (SAEs).  It now includes end-to-end infrastructure for
Protocol A from the accompanying design document: a reproducible 2-D toy
experiment that analyses how different SAE variants organise correlated and
anti-correlated features.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional (required for figure generation): pip install matplotlib
```

## Running the unit tests

```bash
pytest
```

## Protocol A: Toy 2-D Organisation & Collapse

We provide a runnable script that trains the three SAE variants and produces
metrics, figures, and JSON summaries in the specified output directory.

```bash
python -m sgnsae.scripts.run_protocol_A --config path/to/optional_config.json
```

Running without a configuration file executes the default experimental
settings described in the design document.  The script emits:

* `metrics/protocol_a_summary.csv` – headline MSE, sparsity (L0), collapse
  rates, and decoder/feature alignment summaries for each model.
* `metrics/<model>.json` – detailed decoder/feature angle matrices, sign
  usage statistics, per-feature best-alignment angles, and the list of
  anti-correlated feature pairs captured by a single signed latent.
* `figures/<model>.png` – unit-circle visualisations coloured by ground-truth
  bundles with +/− overlays describing sign usage.
* `history/<model>.json` – per-step training logs containing the reconstruction,
  sparsity, and auxiliary losses.

#### Expected outputs (per model)

Each model's JSON metrics now expose:

* `alignment_angles_deg`: the best alignment angle (degrees) to any decoder,
  up to sign, for every ground-truth feature.
* `alignment_mean_deg` and `alignment_median_deg`: summary statistics that also
  appear in `metrics/protocol_a_summary.csv`.

The collapse detector is orientation-invariant and may optionally require
opposite-signed Pearson correlations when a `corr_threshold` is provided.

### Configuration schema

`run_protocol_A.py` accepts a JSON configuration with the following top-level
keys (all optional, defaulting to the script's built-in configuration):

* `seed`: Global random seed used for dataset generation and training.
* `device`: Torch device string (e.g. `"cpu"` or `"cuda:0"`).
* `output_dir`: Directory to place logs, metrics, and figures.
* `dataset`: Parameters passed to
  :class:`sgnsae.data.toy2d.Toy2DGenerationConfig`.
* `optimizer`: Parameters passed to
  :class:`sgnsae.train.train_sae.OptimizerConfig`.
* `training`: Fields that override the default
  :class:`sgnsae.train.train_sae.TrainingConfig`.
* `models`: A list of model dictionaries matching
  :class:`sgnsae.scripts.run_protocol_A.ProtocolAModelConfig`.

See `sgnsae/scripts/run_protocol_A.py` for the exact defaults and admissible
values.

## Repository structure

```
sgnsae/
  data/                # Synthetic dataset generation utilities
  eval/                # Metrics, evaluation, and visualisation helpers
  models/              # SAE implementations (baseline and gated variants)
  scripts/             # Runnable experiments (Protocol A)
  train/               # Training harness shared across experiments
```

## Contributing

Please ensure that new functionality is covered by unit tests and that `pytest`
passes before submitting changes.
