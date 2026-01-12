import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; received {value}.")


def _validate_non_negative_float(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative; received {value}.")


def _validate_closed_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]; received {value}.")


def _validate_open_unit_interval(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be within (0, 1); received {value}.")


def _validate_adam_betas(adam_betas: Tuple[float, float]) -> None:
    if len(adam_betas) != 2:
        raise ValueError(
            f"adam_betas must contain exactly two values; received {adam_betas}."
        )
    beta1, beta2 = adam_betas
    for beta_value, beta_name in zip((beta1, beta2), ("beta1", "beta2")):
        if not 0.0 < beta_value < 1.0:
            raise ValueError(
                f"{beta_name} must be within (0, 1); received {beta_value}."
            )


def _calculate_pair_correlation_feasible_interval(
    pair_feature_prob: float,
) -> Tuple[float, float]:
    p = float(pair_feature_prob)
    denominator = p * (1.0 - p)
    if denominator <= 0:
        raise ValueError(
            "pair_feature_prob must be within (0, 1) to define a within-pair correlation;"
            f" received {pair_feature_prob}."
        )

    q_min = max(0.0, 2.0 * p - 1.0)
    q_max = p
    r_min = (q_min - p * p) / denominator
    r_max = (q_max - p * p) / denominator
    r_min = max(-1.0, float(r_min))
    r_max = min(1.0, float(r_max))
    return r_min, r_max


def _validate_pair_correlation(
    pair_feature_prob: float,
    pair_correlation: float,
    name: str,
) -> None:
    r = float(pair_correlation)
    if not -1.0 <= r <= 1.0:
        raise ValueError(f"{name} must be within [-1, 1]; received {pair_correlation}.")

    r_min, r_max = _calculate_pair_correlation_feasible_interval(pair_feature_prob)
    if r < r_min or r > r_max:
        raise ValueError(
            f"{name}={pair_correlation} is infeasible for pair_feature_prob={pair_feature_prob}; "
            f"feasible interval is [{r_min}, {r_max}]."
        )


@dataclass
class Toy2DDataConfig:
    seed: int | List[int]
    num_train: int
    num_val: int
    num_test: int
    num_pairs: int
    direction_mode: Literal["uniform", "random"] = "uniform"
    pair_axis_active_prob: float = 0.2
    active_pairs_per_sample: int | None = None
    pair_feature_prob: float = 0.5
    pair_correlation: Optional[float] = -1.0
    pair_correlation_grid: Optional[List[float]] = None
    magnitude_dist: Literal["lognormal", "constant"] = "lognormal"
    lognormal_mu: float = 0.0
    lognormal_sigma: float = 0.25
    constant_magnitude: float = 1.0
    noise_std: float = 0.01

    def __post_init__(self) -> None:
        if isinstance(self.seed, list):
            if not self.seed:
                raise ValueError(
                    "seed must be a non-empty list when provided as a list."
                )
            for index, seed_value in enumerate(self.seed):
                if not isinstance(seed_value, int) or isinstance(seed_value, bool):
                    raise ValueError(
                        "seed values must be integers; received "
                        f"seed[{index}]={seed_value}."
                    )
        elif not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an integer; received {self.seed}.")

        for field_name in ("num_train", "num_val", "num_test", "num_pairs"):
            _validate_positive_int(getattr(self, field_name), field_name)

        if self.direction_mode not in {"uniform", "random"}:
            raise ValueError(
                "direction_mode must be one of 'uniform' or 'random'; "
                f"received {self.direction_mode}."
            )

        _validate_closed_unit_interval(
            self.pair_axis_active_prob, "pair_axis_active_prob"
        )
        _validate_open_unit_interval(self.pair_feature_prob, "pair_feature_prob")

        if self.pair_correlation_grid is None:
            if self.pair_correlation is None:
                raise ValueError(
                    "pair_correlation must be provided when pair_correlation_grid is not set."
                )
            _validate_pair_correlation(
                self.pair_feature_prob,
                self.pair_correlation,
                "pair_correlation",
            )
        else:
            if not self.pair_correlation_grid:
                raise ValueError(
                    "pair_correlation_grid must be a non-empty list when provided."
                )
            for index, correlation_value in enumerate(self.pair_correlation_grid):
                _validate_pair_correlation(
                    self.pair_feature_prob,
                    correlation_value,
                    f"pair_correlation_grid[{index}]",
                )

        if self.magnitude_dist not in {"lognormal", "constant"}:
            raise ValueError(
                "magnitude_dist must be one of 'lognormal' or 'constant'; "
                f"received {self.magnitude_dist}."
            )

        if self.magnitude_dist == "lognormal":
            if self.lognormal_sigma <= 0:
                raise ValueError(
                    f"lognormal_sigma must be positive; received {self.lognormal_sigma}."
                )
        elif self.constant_magnitude <= 0:
            raise ValueError(
                f"constant_magnitude must be positive; received {self.constant_magnitude}."
            )

        _validate_non_negative_float(self.noise_std, "noise_std")


@dataclass
class Toy2DModelVariantConfig:
    name: str
    sae_class: Literal["SAE", "GatedSAE"]
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
    sparsity_coeff_warmup_epochs: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model name must be a non-empty string.")
        if self.sae_class not in {"SAE", "GatedSAE"}:
            raise ValueError(
                f"sae_class must be one of 'SAE' or 'GatedSAE'; received {self.sae_class}."
            )
        if self.symmetric_magnitude and not self.sign_aware:
            raise ValueError("symmetric_magnitude requires sign_aware=True.")

        _validate_positive_int(self.d_hidden, "d_hidden")
        _validate_positive_int(self.num_epochs, "num_epochs")
        _validate_positive_int(self.batch_size, "batch_size")
        if self.sparsity_coeff < 0:
            raise ValueError(
                f"sparsity_coeff must be non-negative; received {self.sparsity_coeff}."
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive; received {self.learning_rate}."
            )
        if self.resample_latents_epochs <= 0:
            raise ValueError(
                f"resample_latents_epochs must be positive; received {self.resample_latents_epochs}."
            )
        if self.sparsity_coeff_warmup_epochs < 0:
            raise ValueError(
                "sparsity_coeff_warmup_epochs must be >= 0; "
                f"received {self.sparsity_coeff_warmup_epochs}."
            )
        if self.sparsity_coeff_warmup_epochs > self.num_epochs:
            raise ValueError(
                "sparsity_coeff_warmup_epochs must be <= num_epochs; "
                f"received warmup={self.sparsity_coeff_warmup_epochs}, num_epochs={self.num_epochs}."
            )
        _validate_adam_betas(self.adam_betas)


@dataclass
class Toy2DEvaluationConfig:
    min_axis_abs_cos: float = 0.9
    min_activation_corr: float = 0.8
    min_sign_agreement: float = 0.9
    min_regime_samples: int = 100
    hist_num_bins: int = 80

    def __post_init__(self) -> None:
        _validate_closed_unit_interval(self.min_axis_abs_cos, "min_axis_abs_cos")
        _validate_closed_unit_interval(self.min_activation_corr, "min_activation_corr")
        _validate_closed_unit_interval(self.min_sign_agreement, "min_sign_agreement")
        _validate_positive_int(self.min_regime_samples, "min_regime_samples")
        _validate_positive_int(self.hist_num_bins, "hist_num_bins")


@dataclass
class Toy2DExperimentConfig:
    output_dir: str
    device: Literal["cpu", "cuda", "auto"]
    data: Toy2DDataConfig
    models: List[Toy2DModelVariantConfig]
    evaluation: Toy2DEvaluationConfig = field(default_factory=Toy2DEvaluationConfig)
    run_id: Optional[str] = None
    log_every_n_steps: int = 25
    log_every_n_epochs: int = 1
    verbose: bool = True
    sample_preview_count: int = 1024

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must be provided.")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError(
                f"device must be one of 'cpu', 'cuda', or 'auto'; received {self.device}."
            )
        if not self.models:
            raise ValueError("At least one model configuration must be provided.")
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("Model names must be unique.")
        if self.log_every_n_steps <= 0:
            raise ValueError(
                f"log_every_n_steps must be positive; received {self.log_every_n_steps}."
            )
        if self.log_every_n_epochs <= 0:
            raise ValueError(
                f"log_every_n_epochs must be positive; received {self.log_every_n_epochs}."
            )
        _validate_positive_int(self.sample_preview_count, "sample_preview_count")


def _load_raw_config(path: str) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    with open(path, "r") as f:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ValueError("PyYAML is required to load YAML configs.")
            return yaml.safe_load(f)
        return json.load(f)


def load_experiment_config(path: str) -> Toy2DExperimentConfig:
    raw_config = _load_raw_config(path)
    if "data" not in raw_config or "models" not in raw_config:
        raise ValueError(
            "Configuration file must include 'data' and 'models' sections."
        )
    if not isinstance(raw_config["models"], list):
        raise ValueError("The 'models' section must be a list of model configurations.")
    if "output_dir" not in raw_config:
        raise ValueError("Configuration file must include 'output_dir'.")

    data_config = Toy2DDataConfig(**raw_config["data"])
    model_configs = [
        Toy2DModelVariantConfig(**model_config) for model_config in raw_config["models"]
    ]
    evaluation_raw = raw_config.get("evaluation") or {}
    if not isinstance(evaluation_raw, dict):
        raise ValueError("The 'evaluation' section must be an object when provided.")
    evaluation_config = Toy2DEvaluationConfig(**evaluation_raw)

    experiment_config = Toy2DExperimentConfig(
        output_dir=raw_config["output_dir"],
        device=raw_config.get("device", "auto"),
        data=data_config,
        models=model_configs,
        evaluation=evaluation_config,
        run_id=raw_config.get("run_id"),
        log_every_n_steps=raw_config.get("log_every_n_steps", 25),
        log_every_n_epochs=raw_config.get("log_every_n_epochs", 1),
        verbose=raw_config.get("verbose", True),
        sample_preview_count=raw_config.get("sample_preview_count", 1024),
    )
    return experiment_config


def save_experiment_config(config: Toy2DExperimentConfig, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
