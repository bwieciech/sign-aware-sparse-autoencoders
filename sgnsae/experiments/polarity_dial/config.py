import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]; received {value}.")


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; received {value}.")


def _validate_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative; received {value}.")


def _validate_adam_betas(adam_betas: Tuple[float, float]) -> None:
    if len(adam_betas) != 2:
        raise ValueError(f"adam_betas must contain exactly two values; received {adam_betas}.")
    beta1, beta2 = adam_betas
    for beta_value, beta_name in zip((beta1, beta2), ("beta1", "beta2")):
        if not 0.0 < beta_value < 1.0:
            raise ValueError(f"{beta_name} must be within (0, 1); received {beta_value}.")


@dataclass
class PolarityDialDataConfig:
    seed: int
    num_train: int
    num_val: int
    num_test: int
    input_dim: int
    num_features: int
    activation_prob: float
    positive_sign_prob: float
    lognormal_mu: float
    lognormal_sigma: float
    exponential_lambda: float
    noise_std: float

    def __post_init__(self) -> None:
        for field_name in ("num_train", "num_val", "num_test", "input_dim", "num_features"):
            _validate_positive_int(getattr(self, field_name), field_name)
        _validate_probability(self.activation_prob, "activation_prob")
        _validate_probability(self.positive_sign_prob, "positive_sign_prob")
        if self.lognormal_sigma <= 0:
            raise ValueError(f"lognormal_sigma must be positive; received {self.lognormal_sigma}.")
        if self.exponential_lambda <= 0:
            raise ValueError(f"exponential_lambda must be positive; received {self.exponential_lambda}.")
        _validate_non_negative(self.noise_std, "noise_std")


@dataclass
class PolarityDialModelVariantConfig:
    name: str
    sae_class: str
    sign_aware: bool
    single_activation_offset: bool
    reset_activation_offsets_on_resample: bool
    d_hidden: int
    sparsity_coeff: float
    learning_rate: float
    num_epochs: int
    batch_size: int
    adam_betas: Tuple[float, float]
    symmetric_magnitude: bool = False
    resample_latents_epochs: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model name must be a non-empty string.")
        if self.symmetric_magnitude and not self.sign_aware:
            raise ValueError("symmetric_magnitude requires sign_aware=True.")
        _validate_positive_int(self.d_hidden, "d_hidden")
        _validate_positive_int(self.num_epochs, "num_epochs")
        _validate_positive_int(self.batch_size, "batch_size")
        if self.sparsity_coeff < 0:
            raise ValueError(f"sparsity_coeff must be non-negative; received {self.sparsity_coeff}.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive; received {self.learning_rate}.")
        if self.resample_latents_epochs <= 0:
            raise ValueError(
                f"resample_latents_epochs must be positive; received {self.resample_latents_epochs}."
            )
        _validate_adam_betas(self.adam_betas)


@dataclass
class PolarityDialExperimentConfig:
    output_dir: str
    device: str
    data: PolarityDialDataConfig
    models: List[PolarityDialModelVariantConfig]
    run_id: Optional[str] = None
    log_every_n_steps: int = 25
    log_every_n_epochs: int = 1
    verbose: bool = True
    sample_preview_count: int = 1024
    max_scatter_points_per_regime: int = 5000
    wandb_project: Optional[str] = None
    wandb_api_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must be provided.")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError(f"device must be one of 'cpu', 'cuda', or 'auto'; received {self.device}.")
        if not self.models:
            raise ValueError("At least one model configuration must be provided.")
        if self.log_every_n_steps <= 0:
            raise ValueError(f"log_every_n_steps must be positive; received {self.log_every_n_steps}.")
        if self.log_every_n_epochs <= 0:
            raise ValueError(f"log_every_n_epochs must be positive; received {self.log_every_n_epochs}.")
        _validate_positive_int(self.sample_preview_count, "sample_preview_count")
        _validate_positive_int(self.max_scatter_points_per_regime, "max_scatter_points_per_regime")


def _load_raw_config(path: str) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    with open(path, "r") as f:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ValueError("PyYAML is required to load YAML configs.")
            return yaml.safe_load(f)
        return json.load(f)


def load_experiment_config(path: str) -> PolarityDialExperimentConfig:
    raw_config = _load_raw_config(path)
    if "data" not in raw_config or "models" not in raw_config:
        raise ValueError("Configuration file must include 'data' and 'models' sections.")
    if not isinstance(raw_config["models"], list):
        raise ValueError("The 'models' section must be a list of model configurations.")
    if "output_dir" not in raw_config:
        raise ValueError("Configuration file must include 'output_dir'.")
    data_config = PolarityDialDataConfig(**raw_config["data"])
    model_configs = [
        PolarityDialModelVariantConfig(**model_config) for model_config in raw_config["models"]
    ]
    experiment_config = PolarityDialExperimentConfig(
        output_dir=raw_config["output_dir"],
        device=raw_config.get("device", "auto"),
        data=data_config,
        models=model_configs,
        run_id=raw_config.get("run_id"),
        log_every_n_steps=raw_config.get("log_every_n_steps", 25),
        log_every_n_epochs=raw_config.get("log_every_n_epochs", 1),
        verbose=raw_config.get("verbose", True),
        sample_preview_count=raw_config.get("sample_preview_count", 1024),
        max_scatter_points_per_regime=raw_config.get("max_scatter_points_per_regime", 5000),
        wandb_project=raw_config.get("wandb_project"),
        wandb_api_key=raw_config.get("wandb_api_key"),
    )
    return experiment_config


def save_experiment_config(config: PolarityDialExperimentConfig, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
