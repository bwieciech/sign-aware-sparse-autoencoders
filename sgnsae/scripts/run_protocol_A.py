"""Entry point for Protocol A (Toy 2-D Organisation & Collapse)."""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List

import numpy as np
import torch

from sgnsae.data.toy2d import Toy2DGenerationConfig, generate_toy2d_dataset
from sgnsae.eval.toy2d_eval import Toy2DEvaluation, evaluate_toy2d_model
from sgnsae.eval.visualize_toy2d import plot_decoder_arrangement
from sgnsae.models import GatedSAE, SAE
from sgnsae.train.train_sae import (
    OptimizerConfig,
    TrainingConfig,
    detect_device,
    train_model,
)

LOGGER = logging.getLogger("protocol_a")


@dataclass
class ProtocolAModelConfig:
    name: str
    model_type: str
    d_hidden: int
    sparsity_coeff: float
    resample_threshold: int = 2000
    sign_aware: bool = False
    reset_activation_offsets_on_resample: bool = True
    single_activation_offset: bool = False


@dataclass
class ProtocolAConfig:
    seed: int = 0
    device: str | None = None
    output_dir: str = "protocol_a_outputs"
    dataset: Toy2DGenerationConfig = field(
        default_factory=lambda: Toy2DGenerationConfig(
            num_samples=20000,
            num_correlated_bundles=3,
            correlated_bundle_size=3,
            num_anticorrelated_bundles=4,
            anticorrelated_bundle_size=2,
            correlated_on_probability=0.35,
            anticorrelated_on_probability=0.35,
            angular_std=0.1,
            magnitude_range=(0.0, 1.0),
            noise_std=0.0,
            seed=0,
        )
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=5e-4, beta1=0.0, beta2=0.999)
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            steps=250000,
            batch_size=4096,
            sparsity_coeff=2e-3,
            log_every=5000,
            resample_check_interval=1000,
            latent_activity_epsilon=1e-6,
            grad_clip=None,
            shuffle=True,
            seed=0,
        )
    )
    models: List[ProtocolAModelConfig] = field(
        default_factory=lambda: [
            ProtocolAModelConfig(
                name="baseline_relu",
                model_type="sae",
                d_hidden=32,
                sparsity_coeff=5e-3,
                sign_aware=False,
            ),
            ProtocolAModelConfig(
                name="gated",
                model_type="gated",
                d_hidden=32,
                sparsity_coeff=2e-3,
                sign_aware=False,
            ),
            ProtocolAModelConfig(
                name="sg_gated",
                model_type="gated",
                d_hidden=32,
                sparsity_coeff=2e-3,
                sign_aware=True,
                reset_activation_offsets_on_resample=True,
                single_activation_offset=False,
            ),
        ]
    )


def _instantiate_model(config: ProtocolAModelConfig, d_in: int, device: str):
    kwargs = dict(
        n_instances=1,
        d_in=d_in,
        d_hidden=config.d_hidden,
        resample_latents_threshold=config.resample_threshold,
        sign_aware=config.sign_aware,
        device=device,
    )
    if config.model_type == "sae":
        model = SAE(**kwargs)
    elif config.model_type == "gated":
        model = GatedSAE(
            **kwargs,
            reset_activation_offsets_on_resample=config.reset_activation_offsets_on_resample,
            single_activation_offset=config.single_activation_offset,
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    return model


def _evaluation_to_dict(evaluation: Toy2DEvaluation) -> Dict:
    return {
        "model_name": evaluation.model_name,
        "mse": evaluation.mse,
        "l0": evaluation.l0,
        "collapse_rate": evaluation.collapse_rate,
        "collapsed_pairs": evaluation.collapsed_pairs,
        "decoder_angles": evaluation.decoder_angles.tolist(),
        "sign_usage": [asdict(usage) for usage in evaluation.sign_usage],
        "alignment_angles_deg": evaluation.alignment_angles_deg.tolist(),
        "alignment_mean_deg": evaluation.alignment_mean_deg,
        "alignment_median_deg": evaluation.alignment_median_deg,
    }


def _save_training_history(history, path: pathlib.Path):
    serialisable = [asdict(entry) for entry in history]
    path.write_text(json.dumps(serialisable, indent=2))


def _save_evaluation(evaluation: Toy2DEvaluation, path: pathlib.Path):
    path.write_text(json.dumps(_evaluation_to_dict(evaluation), indent=2))


def _save_plot(fig, path: pathlib.Path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required to save protocol figures. Install matplotlib or disable figure generation."
        ) from exc

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_protocol(config: ProtocolAConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = detect_device(config.device)
    LOGGER.info("Using device %s", device)

    output_dir = pathlib.Path(config.output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "history").mkdir(parents=True, exist_ok=True)

    LOGGER.info("Generating synthetic dataset with config: %s", config.dataset.to_json())
    data = generate_toy2d_dataset(config.dataset)
    dataset = data.to_dataset()

    metrics_rows: List[Dict] = []

    for model_config in config.models:
        LOGGER.info("Training model %s", model_config.name)
        model = _instantiate_model(
            model_config,
            d_in=dataset.observations.shape[1],
            device=device,
        )
        model.to(device)

        train_cfg = replace(config.training, optimizer=config.optimizer)
        train_cfg = replace(train_cfg, sparsity_coeff=model_config.sparsity_coeff)

        training_result = train_model(model, dataset, train_cfg)
        history_path = output_dir / "history" / f"{model_config.name}.json"
        _save_training_history(training_result.history, history_path)

        LOGGER.info("Evaluating model %s", model_config.name)
        evaluation = evaluate_toy2d_model(
            training_result.model,
            model_config.name,
            dataset,
            data,
        )
        metrics_rows.append(
            {
                "model_name": model_config.name,
                "mse": evaluation.mse,
                "l0": evaluation.l0,
                "collapse_rate": evaluation.collapse_rate,
                "alignment_mean_deg": evaluation.alignment_mean_deg,
                "alignment_median_deg": evaluation.alignment_median_deg,
            }
        )
        eval_path = output_dir / "metrics" / f"{model_config.name}.json"
        _save_evaluation(evaluation, eval_path)

        decoder_weights = training_result.model.W_dec.detach().cpu()
        if decoder_weights.ndim == 3:
            decoder_vectors = decoder_weights[0].t()
        else:
            decoder_vectors = decoder_weights.t()
        try:
            fig = plot_decoder_arrangement(
                decoder_vectors,
                data,
                evaluation.sign_usage,
                title=f"Decoder arrangement - {model_config.name}",
            )
        except RuntimeError as exc:
            LOGGER.warning("Skipping figure for %s: %s", model_config.name, exc)
        else:
            fig_path = output_dir / "figures" / f"{model_config.name}.png"
            _save_plot(fig, fig_path)

    summary_path = output_dir / "metrics" / "protocol_a_summary.csv"
    headers = [
        "model_name",
        "mse",
        "l0",
        "collapse_rate",
        "alignment_mean_deg",
        "alignment_median_deg",
    ]
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in metrics_rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    config_path = output_dir / "config.json"
    config_payload = asdict(config)
    config_payload["device"] = device
    config_path.write_text(json.dumps(config_payload, indent=2, default=str))
    LOGGER.info("Protocol A complete. Results stored in %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="Optional path to a JSON configuration overriding the defaults.",
    )
    return parser.parse_args()


def load_config(path: pathlib.Path) -> ProtocolAConfig:
    raw = json.loads(path.read_text())
    dataset = Toy2DGenerationConfig(**raw.get("dataset", {}))
    optimizer = OptimizerConfig(**raw.get("optimizer", {}))
    training_dict = raw.get("training", {})
    training = TrainingConfig(
        steps=training_dict.get("steps", 250000),
        batch_size=training_dict.get("batch_size", 4096),
        sparsity_coeff=training_dict.get("sparsity_coeff", 2e-3),
        optimizer=optimizer,
        log_every=training_dict.get("log_every", 5000),
        resample_check_interval=training_dict.get("resample_check_interval", 1000),
        latent_activity_epsilon=training_dict.get("latent_activity_epsilon", 1e-6),
        grad_clip=training_dict.get("grad_clip"),
        shuffle=training_dict.get("shuffle", True),
        seed=training_dict.get("seed", 0),
    )
    models = [ProtocolAModelConfig(**model_dict) for model_dict in raw.get("models", [])]
    return ProtocolAConfig(
        seed=raw.get("seed", 0),
        device=raw.get("device"),
        output_dir=raw.get("output_dir", "protocol_a_outputs"),
        dataset=dataset,
        optimizer=optimizer,
        training=training,
        models=models if models else ProtocolAConfig().models,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        config = ProtocolAConfig()
    run_protocol(config)
