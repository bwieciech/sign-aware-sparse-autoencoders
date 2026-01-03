import json
import os
from dataclasses import asdict, dataclass
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


def _validate_open_unit_interval(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be within (0, 1); received {value}.")


def _validate_closed_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]; received {value}.")


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


@dataclass
class DirectionalDeviationDataConfig:
    seed: int
    num_train: int
    num_val: int
    num_test_normal: int
    num_test_high_per_channel: int
    num_test_low_per_channel: int
    num_channels: int
    noise_std: float = 0.0
    normal_mode: Literal[
        "standard_normal",
        "heteroskedastic_normal",
        "iid_non_gaussian",
        "non_gaussian_heteroskedastic",
    ] = "standard_normal"
    channel_stds: Optional[List[float]] = None
    ensure_unique_channel_stds: bool = True
    non_gaussian_dist: Optional[Literal["laplace", "logistic", "student_t"]] = None
    student_t_df: Optional[float] = None
    anomaly_eval_mode: Literal["materialize", "stream"] = "stream"
    train_anomaly_fraction: float = 0.0
    seeds: Optional[List[int]] = None

    def __post_init__(self) -> None:
        for field_name in ("num_train", "num_val", "num_channels"):
            _validate_positive_int(getattr(self, field_name), field_name)
        for field_name in (
            "num_test_normal",
            "num_test_high_per_channel",
            "num_test_low_per_channel",
        ):
            _validate_positive_int(getattr(self, field_name), field_name)

        _validate_non_negative_float(self.noise_std, "noise_std")

        allowed_normal_modes = {
            "standard_normal",
            "heteroskedastic_normal",
            "iid_non_gaussian",
            "non_gaussian_heteroskedastic",
        }
        if self.normal_mode not in allowed_normal_modes:
            raise ValueError(
                "normal_mode must be one of "
                "'standard_normal', 'heteroskedastic_normal', 'iid_non_gaussian', or "
                f"'non_gaussian_heteroskedastic'; received {self.normal_mode}."
            )

        uses_channel_stds = self.normal_mode in {
            "heteroskedastic_normal",
            "non_gaussian_heteroskedastic",
        }
        if uses_channel_stds:
            if self.channel_stds is None:
                raise ValueError(
                    f"channel_stds must be provided when normal_mode='{self.normal_mode}'."
                )
            if len(self.channel_stds) != self.num_channels:
                raise ValueError(
                    f"channel_stds must have length num_channels={self.num_channels}; "
                    f"received {len(self.channel_stds)}."
                )
            for channel_std in self.channel_stds:
                if channel_std <= 0:
                    raise ValueError("All channel_stds entries must be positive.")
            if self.ensure_unique_channel_stds and len(set(self.channel_stds)) != len(
                self.channel_stds
            ):
                raise ValueError(
                    "channel_stds must not contain duplicates when ensure_unique_channel_stds=True."
                )
        elif self.channel_stds is not None:
            raise ValueError(
                f"channel_stds must be omitted when normal_mode='{self.normal_mode}'."
            )

        uses_non_gaussian_dist = self.normal_mode in {
            "iid_non_gaussian",
            "non_gaussian_heteroskedastic",
        }
        if uses_non_gaussian_dist:
            if self.non_gaussian_dist is None:
                raise ValueError(
                    f"non_gaussian_dist must be provided when normal_mode='{self.normal_mode}'."
                )
            if self.non_gaussian_dist not in {"laplace", "logistic", "student_t"}:
                raise ValueError(
                    "non_gaussian_dist must be one of 'laplace', 'logistic', or 'student_t'; "
                    f"received {self.non_gaussian_dist}."
                )
            if self.non_gaussian_dist == "student_t":
                if self.student_t_df is None:
                    raise ValueError(
                        "student_t_df must be provided when non_gaussian_dist='student_t'."
                    )
                if self.student_t_df <= 2:
                    raise ValueError(
                        f"student_t_df must be > 2; received {self.student_t_df}."
                    )
            elif self.student_t_df is not None:
                raise ValueError(
                    "student_t_df must be omitted unless non_gaussian_dist='student_t'."
                )
        else:
            if self.non_gaussian_dist is not None:
                raise ValueError(
                    f"non_gaussian_dist must be omitted when normal_mode='{self.normal_mode}'."
                )
            if self.student_t_df is not None:
                raise ValueError(
                    f"student_t_df must be omitted when normal_mode='{self.normal_mode}'."
                )

        if self.anomaly_eval_mode not in {"materialize", "stream"}:
            raise ValueError(
                "anomaly_eval_mode must be one of 'materialize' or 'stream'; "
                f"received {self.anomaly_eval_mode}."
            )

        if not 0.0 <= self.train_anomaly_fraction <= 1.0:
            raise ValueError(
                "train_anomaly_fraction must be within [0, 1]; "
                f"received {self.train_anomaly_fraction}."
            )

        if self.seeds is not None:
            if not self.seeds:
                raise ValueError("seeds must be a non-empty list when provided.")
            if len(set(self.seeds)) != len(self.seeds):
                raise ValueError("seeds must not contain duplicates.")


@dataclass
class DirectionalDeviationAnomalyConfig:
    delta: float
    shift_mode: Literal["single_channel", "subset", "feature_direction"]
    subset_size: Optional[int] = None
    channels_to_evaluate: Optional[List[int]] = None
    direction_source: Optional[
        Literal["reference_model_decoder", "fixed_random", "per_model_decoder"]
    ] = None
    num_directions: Optional[int] = None
    reference_model_name: str = "sign_aware_gated_sae"
    direction_selection: Optional[Literal["top_purity", "random", "explicit"]] = None
    direction_latent_indices: Optional[List[int]] = None
    normalize_directions: bool = True
    num_test_high_per_direction: Optional[int] = None
    num_test_low_per_direction: Optional[int] = None

    def __post_init__(self) -> None:
        if self.delta <= 0:
            raise ValueError(f"delta must be positive; received {self.delta}.")
        if self.shift_mode not in {"single_channel", "subset", "feature_direction"}:
            raise ValueError(
                "shift_mode must be one of 'single_channel', 'subset', or "
                f"'feature_direction'; received {self.shift_mode}."
            )
        if self.shift_mode == "subset":
            if self.subset_size is None:
                raise ValueError(
                    "subset_size must be provided when shift_mode='subset'."
                )
            _validate_positive_int(self.subset_size, "subset_size")
        elif self.subset_size is not None:
            raise ValueError(
                f"subset_size must be omitted when shift_mode='{self.shift_mode}'."
            )

        if self.channels_to_evaluate is not None and not self.channels_to_evaluate:
            raise ValueError(
                "channels_to_evaluate must be a non-empty list when provided."
            )

        if self.shift_mode != "feature_direction":
            if (
                self.direction_source is not None
                or self.num_directions is not None
                or self.direction_selection is not None
                or self.direction_latent_indices is not None
                or self.num_test_high_per_direction is not None
                or self.num_test_low_per_direction is not None
            ):
                raise ValueError(
                    "Feature-direction fields may only be provided when shift_mode='feature_direction'."
                )
            return

        if self.channels_to_evaluate is not None:
            raise ValueError(
                "channels_to_evaluate must be omitted when shift_mode='feature_direction'."
            )

        if self.direction_source is None:
            self.direction_source = "reference_model_decoder"
        if self.direction_source not in {
            "reference_model_decoder",
            "fixed_random",
            "per_model_decoder",
        }:
            raise ValueError(
                "direction_source must be one of 'reference_model_decoder', 'fixed_random', or "
                f"'per_model_decoder'; received {self.direction_source}."
            )

        if self.num_directions is None:
            raise ValueError(
                "num_directions must be provided when shift_mode='feature_direction'."
            )
        _validate_positive_int(self.num_directions, "num_directions")

        if self.direction_selection is None:
            self.direction_selection = "top_purity"
        if self.direction_selection not in {"top_purity", "random", "explicit"}:
            raise ValueError(
                "direction_selection must be one of 'top_purity', 'random', or 'explicit'; "
                f"received {self.direction_selection}."
            )

        if self.direction_selection == "explicit":
            if self.direction_latent_indices is None:
                raise ValueError(
                    "direction_latent_indices must be provided when direction_selection='explicit'."
                )
            if len(self.direction_latent_indices) != self.num_directions:
                raise ValueError(
                    "direction_latent_indices must have length num_directions="
                    f"{self.num_directions}; received {len(self.direction_latent_indices)}."
                )
            if len(set(self.direction_latent_indices)) != len(
                self.direction_latent_indices
            ):
                raise ValueError(
                    "direction_latent_indices must not contain duplicates."
                )
            for latent_index in self.direction_latent_indices:
                if latent_index < 0:
                    raise ValueError(
                        "direction_latent_indices entries must be non-negative; "
                        f"received {latent_index}."
                    )
        elif self.direction_latent_indices is not None:
            raise ValueError(
                "direction_latent_indices must be omitted unless direction_selection='explicit'."
            )

        if (
            self.direction_source == "reference_model_decoder"
            and not self.reference_model_name
        ):
            raise ValueError(
                "reference_model_name must be a non-empty string when direction_source='reference_model_decoder'."
            )

        if self.num_test_high_per_direction is not None:
            _validate_positive_int(
                self.num_test_high_per_direction, "num_test_high_per_direction"
            )
        if self.num_test_low_per_direction is not None:
            _validate_positive_int(
                self.num_test_low_per_direction, "num_test_low_per_direction"
            )


@dataclass
class DirectionalDeviationThresholdConfig:
    latent_quantile: float = 0.99
    score_fpr: float = 0.01
    min_validation_samples: int = 1000
    deterministic_algorithms: bool = False

    def __post_init__(self) -> None:
        _validate_open_unit_interval(self.latent_quantile, "latent_quantile")
        _validate_open_unit_interval(self.score_fpr, "score_fpr")
        _validate_positive_int(self.min_validation_samples, "min_validation_samples")


@dataclass
class DirectionalDeviationModelVariantConfig:
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
        _validate_adam_betas(self.adam_betas)


@dataclass
class DirectionalDeviationExperimentConfig:
    output_dir: str
    device: Literal["cpu", "cuda", "auto"]
    data: DirectionalDeviationDataConfig
    anomaly: DirectionalDeviationAnomalyConfig
    thresholds: DirectionalDeviationThresholdConfig
    models: List[DirectionalDeviationModelVariantConfig]
    width_sweep: Optional[List[int]] = None
    run_id: Optional[str] = None
    verbose: bool = True
    coverage_recall_target: float = 0.8
    log_every_n_steps: int = 25
    log_every_n_epochs: int = 1
    wandb_project: Optional[str] = None
    wandb_api_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must be provided.")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError(
                f"device must be one of 'cpu', 'cuda', or 'auto'; received {self.device}."
            )
        if not self.models:
            raise ValueError("At least one model configuration must be provided.")
        if self.log_every_n_steps <= 0:
            raise ValueError(
                f"log_every_n_steps must be positive; received {self.log_every_n_steps}."
            )
        if self.log_every_n_epochs <= 0:
            raise ValueError(
                f"log_every_n_epochs must be positive; received {self.log_every_n_epochs}."
            )
        _validate_closed_unit_interval(
            self.coverage_recall_target, "coverage_recall_target"
        )

        if self.data.num_val < self.thresholds.min_validation_samples:
            raise ValueError(
                "num_val must be at least min_validation_samples for stable threshold estimation; "
                f"received num_val={self.data.num_val} and min_validation_samples={self.thresholds.min_validation_samples}."
            )

        if (
            self.data.train_anomaly_fraction > 0.0
            and self.anomaly.shift_mode == "feature_direction"
        ):
            raise ValueError(
                "train_anomaly_fraction must be 0.0 when shift_mode='feature_direction'."
            )

        if self.anomaly.shift_mode == "subset":
            if self.anomaly.subset_size is None:
                raise ValueError(
                    "subset_size must be provided when shift_mode='subset'."
                )
            if self.anomaly.subset_size > self.data.num_channels:
                raise ValueError(
                    f"subset_size={self.anomaly.subset_size} must not exceed num_channels={self.data.num_channels}."
                )

        if self.anomaly.channels_to_evaluate is not None:
            for channel in self.anomaly.channels_to_evaluate:
                if channel < 0 or channel >= self.data.num_channels:
                    raise ValueError(
                        f"channels_to_evaluate must be within [0, {self.data.num_channels - 1}]; received {channel}."
                    )
            if len(set(self.anomaly.channels_to_evaluate)) != len(
                self.anomaly.channels_to_evaluate
            ):
                raise ValueError("channels_to_evaluate must not contain duplicates.")

        if (
            self.anomaly.shift_mode == "feature_direction"
            and self.anomaly.direction_source == "reference_model_decoder"
        ):
            reference_model_name = self.anomaly.reference_model_name
            if not any(model.name == reference_model_name for model in self.models):
                raise ValueError(
                    "reference_model_name must match a model name in config.models when "
                    "direction_source='reference_model_decoder'; received "
                    f"reference_model_name='{reference_model_name}'."
                )

        widths_to_validate = self.width_sweep if self.width_sweep is not None else None
        if widths_to_validate is not None:
            if not widths_to_validate:
                raise ValueError("width_sweep must be a non-empty list when provided.")
            for width in widths_to_validate:
                _validate_positive_int(width, "width_sweep entry")
                if width >= self.data.num_channels:
                    raise ValueError(
                        f"All width_sweep entries must satisfy d_hidden < num_channels={self.data.num_channels}; received {width}."
                    )
        else:
            for model_config in self.models:
                if model_config.d_hidden >= self.data.num_channels:
                    raise ValueError(
                        "Protocol B requires d_hidden < num_channels; received "
                        f"d_hidden={model_config.d_hidden} and num_channels={self.data.num_channels}."
                    )

        if (
            self.anomaly.shift_mode == "feature_direction"
            and self.anomaly.num_directions is not None
        ):
            num_directions = int(self.anomaly.num_directions)
            if self.anomaly.direction_source == "reference_model_decoder":
                if widths_to_validate is not None:
                    if any(int(width) < num_directions for width in widths_to_validate):
                        raise ValueError(
                            "All width_sweep entries must satisfy d_hidden >= num_directions when "
                            "direction_source='reference_model_decoder'; received "
                            f"num_directions={num_directions} and width_sweep={widths_to_validate}."
                        )
                else:
                    reference_model_name = self.anomaly.reference_model_name
                    reference_model_config = next(
                        (
                            model_config
                            for model_config in self.models
                            if model_config.name == reference_model_name
                        ),
                        None,
                    )
                    if reference_model_config is not None and (
                        int(reference_model_config.d_hidden) < num_directions
                    ):
                        raise ValueError(
                            "reference_model_name must refer to a model with d_hidden >= num_directions; "
                            f"received reference_model_name='{reference_model_name}', "
                            f"d_hidden={reference_model_config.d_hidden}, and num_directions={num_directions}."
                        )
                    if (
                        reference_model_config is not None
                        and self.anomaly.direction_latent_indices is not None
                    ):
                        width = int(reference_model_config.d_hidden)
                        for latent_index in self.anomaly.direction_latent_indices:
                            if latent_index >= width:
                                raise ValueError(
                                    "direction_latent_indices entries must be < reference model d_hidden; "
                                    f"received latent_index={latent_index} and d_hidden={width}."
                                )
            elif self.anomaly.direction_source == "per_model_decoder":
                if widths_to_validate is not None:
                    if any(int(width) < num_directions for width in widths_to_validate):
                        raise ValueError(
                            "All width_sweep entries must satisfy d_hidden >= num_directions when "
                            "direction_source='per_model_decoder'; received "
                            f"num_directions={num_directions} and width_sweep={widths_to_validate}."
                        )
                else:
                    for model_config in self.models:
                        if int(model_config.d_hidden) < num_directions:
                            raise ValueError(
                                "All models must satisfy d_hidden >= num_directions when "
                                "direction_source='per_model_decoder'; received model "
                                f"name='{model_config.name}', d_hidden={model_config.d_hidden}, "
                                f"and num_directions={num_directions}."
                            )


def _load_raw_config(path: str) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    with open(path, "r") as f:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ValueError("PyYAML is required to load YAML configs.")
            return yaml.safe_load(f)
        return json.load(f)


def load_experiment_config(path: str) -> DirectionalDeviationExperimentConfig:
    raw_config = _load_raw_config(path)
    required_sections = ("data", "anomaly", "thresholds", "models")
    if any(section not in raw_config for section in required_sections):
        raise ValueError(
            "Configuration file must include 'data', 'anomaly', 'thresholds', and 'models' sections."
        )
    if "output_dir" not in raw_config:
        raise ValueError("Configuration file must include 'output_dir'.")
    if not isinstance(raw_config["models"], list):
        raise ValueError("The 'models' section must be a list of model configurations.")

    data_config = DirectionalDeviationDataConfig(**raw_config["data"])
    anomaly_config = DirectionalDeviationAnomalyConfig(**raw_config["anomaly"])
    threshold_config = DirectionalDeviationThresholdConfig(**raw_config["thresholds"])
    model_configs = [
        DirectionalDeviationModelVariantConfig(**model_config)
        for model_config in raw_config["models"]
    ]
    experiment_config = DirectionalDeviationExperimentConfig(
        output_dir=raw_config["output_dir"],
        device=raw_config.get("device", "auto"),
        data=data_config,
        anomaly=anomaly_config,
        thresholds=threshold_config,
        models=model_configs,
        width_sweep=raw_config.get("width_sweep"),
        run_id=raw_config.get("run_id"),
        verbose=raw_config.get("verbose", True),
        coverage_recall_target=raw_config.get("coverage_recall_target", 0.8),
        log_every_n_steps=raw_config.get("log_every_n_steps", 25),
        log_every_n_epochs=raw_config.get("log_every_n_epochs", 1),
        wandb_project=raw_config.get("wandb_project"),
        wandb_api_key=raw_config.get("wandb_api_key"),
    )
    return experiment_config


def save_experiment_config(
    config: DirectionalDeviationExperimentConfig, path: str
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
