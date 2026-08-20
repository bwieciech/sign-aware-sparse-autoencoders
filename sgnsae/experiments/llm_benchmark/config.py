import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from sgnsae.experiments.device import validate_experiment_device
from sgnsae.experiments.llm_benchmark.model_registry import (
    JUMP_INIT_SAE_CLASS_NAMES,
    KNOWN_SAE_CLASS_NAMES,
    SIGN_AWARE_SAE_CLASS_NAMES,
    THETA_INIT_SAE_CLASS_NAMES,
    TOPK_SAE_CLASS_NAMES,
)
from sgnsae.experiments.llm_benchmark.paths import (
    build_subrun_relative_dir,
    format_sweep_value_for_dir,
    slugify_variant_name,
)
from sgnsae.experiments.llm_benchmark.schema_versions import (
    RESOLVED_CONFIG_SCHEMA_VERSION,
)


Stage = Literal["validate", "cache", "train", "eval", "examples", "plot", "summarize"]
_ALLOWED_STAGES = {
    "validate",
    "cache",
    "train",
    "eval",
    "examples",
    "plot",
    "summarize",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _compute_config_fingerprint(resolved_config: Dict[str, Any]) -> str:
    fingerprint_input = dict(resolved_config)
    for key in (
        "config_fingerprint",
        "rng_fingerprint",
        "output_dir",
        "device",
        "overwrite",
        "resume",
        "stages",
    ):
        fingerprint_input.pop(key, None)
    digest = sha256(_canonical_json_bytes(fingerprint_input)).hexdigest()
    return str(digest)


_RNG_FINGERPRINT_EXCLUDED_TRAINING_KEYS = (
    "multi_sparsity_max_instances",
    "multi_sparsity_enabled",
    "train_steps",
    "log_every_n_steps",
    "checkpoint_model_steps",
    "dump_training_metrics_steps",
)


def _compute_rng_fingerprint(resolved_config: Dict[str, Any]) -> str:
    fingerprint_input = dict(resolved_config)
    for key in (
        "config_fingerprint",
        "rng_fingerprint",
        "output_dir",
        "device",
        "overwrite",
        "resume",
        "stages",
    ):
        fingerprint_input.pop(key, None)

    training = fingerprint_input.get("training")
    if isinstance(training, dict):
        training_fingerprint_input = dict(training)
        for key in _RNG_FINGERPRINT_EXCLUDED_TRAINING_KEYS:
            training_fingerprint_input.pop(key, None)
        fingerprint_input["training"] = training_fingerprint_input

    digest = sha256(_canonical_json_bytes(fingerprint_input)).hexdigest()
    return str(digest)


def _validate_non_empty_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return str(value).strip()


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer; received {value}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive; received {value}.")
    return int(value)


def _validate_non_negative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric; received {value}.")
    value_float = float(value)
    if value_float < 0:
        raise ValueError(f"{name} must be non-negative; received {value_float}.")
    return float(value_float)


class LLMModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    revision: Optional[str] = None
    tokenizer_id: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    torch_dtype: Literal["float32", "float16", "bfloat16", "auto"] = "auto"

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        return _validate_non_empty_string(value, "llm.model_id")

    @field_validator("torch_dtype", mode="before")
    @classmethod
    def _validate_torch_dtype(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(f"llm.torch_dtype must be a string; received {value}.")
        normalized = value.strip().lower()
        allowed_values = {"float32", "float16", "bfloat16", "auto"}
        if normalized not in allowed_values:
            raise ValueError(
                "llm.torch_dtype must be one of "
                "{'float32', 'float16', 'bfloat16', 'auto'}; "
                f"received {value}."
            )
        return normalized


class VisionModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    revision: Optional[str] = None
    image_processor_id: Optional[str] = None
    image_processor_revision: Optional[str] = None
    torch_dtype: Literal["float32", "float16", "bfloat16", "auto"] = "auto"
    image_size: int = 224
    use_auth_token: bool = True

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        return _validate_non_empty_string(value, "vision.model_id")

    @field_validator("image_processor_id")
    @classmethod
    def _validate_image_processor_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _validate_non_empty_string(value, "vision.image_processor_id")

    @field_validator("torch_dtype", mode="before")
    @classmethod
    def _validate_torch_dtype(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(f"vision.torch_dtype must be a string; received {value}.")
        normalized = value.strip().lower()
        allowed_values = {"float32", "float16", "bfloat16", "auto"}
        if normalized not in allowed_values:
            raise ValueError(
                "vision.torch_dtype must be one of "
                "{'float32', 'float16', 'bfloat16', 'auto'}; "
                f"received {value}."
            )
        return normalized

    @field_validator("image_size", mode="before")
    @classmethod
    def _validate_image_size(cls, value: Any) -> int:
        return _validate_positive_int(value, "vision.image_size")


class TextDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["hf", "local_jsonl"]
    dataset_id: Optional[str] = None
    split: Optional[str] = None
    revision: Optional[str] = None
    streaming: bool = True
    text_field: str = "text"
    data_files_glob: Optional[str] = None
    data_files: Optional[List[str]] = None

    @field_validator("text_field")
    @classmethod
    def _validate_text_field(cls, value: str) -> str:
        return _validate_non_empty_string(value, "data.text_field")

    @model_validator(mode="after")
    def _validate_backend_requirements(self) -> "TextDataConfig":
        if self.backend == "hf":
            if self.dataset_id is None or not str(self.dataset_id).strip():
                raise ValueError("data.dataset_id must be provided when backend='hf'.")
            if self.split is None or not str(self.split).strip():
                raise ValueError("data.split must be provided when backend='hf'.")
            if self.data_files_glob is not None or self.data_files is not None:
                raise ValueError(
                    "data_files_glob/data_files must be omitted when backend='hf'."
                )
        else:
            has_glob = self.data_files_glob is not None
            has_files = self.data_files is not None
            if has_glob == has_files:
                raise ValueError(
                    "Exactly one of data.data_files_glob or data.data_files must be provided "
                    "when backend='local_jsonl'."
                )
            if (
                self.dataset_id is not None
                or self.split is not None
                or self.revision is not None
            ):
                raise ValueError(
                    "dataset_id/split/revision must be omitted when backend='local_jsonl'."
                )
            if has_glob:
                if not str(self.data_files_glob).strip():
                    raise ValueError("data.data_files_glob must be a non-empty string.")
            if has_files:
                if not self.data_files:
                    raise ValueError("data.data_files must be a non-empty list.")
                for index, path in enumerate(self.data_files):
                    if not isinstance(path, str) or not path.strip():
                        raise ValueError(
                            f"data.data_files[{index}] must be a non-empty string."
                        )
        return self


class ImageDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["hf", "imagefolder"]
    dataset_id: str
    split: str
    revision: Optional[str] = None
    data_dir: Optional[str] = None
    streaming: bool = True
    image_field: str = "image"
    label_field: Optional[str] = "label"

    @field_validator("dataset_id")
    @classmethod
    def _validate_dataset_id(cls, value: str) -> str:
        return _validate_non_empty_string(value, "data.dataset_id")

    @field_validator("split")
    @classmethod
    def _validate_split(cls, value: str) -> str:
        return _validate_non_empty_string(value, "data.split")

    @field_validator("image_field")
    @classmethod
    def _validate_image_field(cls, value: str) -> str:
        return _validate_non_empty_string(value, "data.image_field")

    @field_validator("label_field")
    @classmethod
    def _validate_label_field(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _validate_non_empty_string(value, "data.label_field")

    @model_validator(mode="after")
    def _validate_backend_requirements(self) -> "ImageDataConfig":
        if self.backend == "hf":
            if self.data_dir is not None:
                raise ValueError(
                    "data.data_dir must be omitted when backend='hf' for image modality."
                )
            return self

        if self.data_dir is None or not str(self.data_dir).strip():
            raise ValueError(
                "data.data_dir must be provided when backend='imagefolder'."
            )
        if not os.path.isdir(self.data_dir):
            raise ValueError(
                f"data.data_dir must exist when backend='imagefolder'; received {self.data_dir}."
            )
        return self


class HookpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hook_name: str
    layer: Optional[int] = None
    layer_selector: Optional[Union[Literal["mid", "late"], float]] = None
    module_path: Optional[str] = None

    @field_validator("hook_name")
    @classmethod
    def _validate_hook_name(cls, value: str) -> str:
        return _validate_non_empty_string(value, "hook.hook_name")

    @field_validator("layer", mode="before")
    @classmethod
    def _validate_layer(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"hook.layer must be an integer; received {value}.")
        if value < 0:
            raise ValueError(f"hook.layer must be >= 0; received {value}.")
        return int(value)

    @field_validator("layer_selector", mode="before")
    @classmethod
    def _validate_layer_selector(cls, value: Any) -> Optional[Union[str, float]]:
        if value is None:
            return None
        if isinstance(value, str):
            selector = value.strip().lower()
            if selector not in {"mid", "late"}:
                raise ValueError(
                    "hook.layer_selector must be 'mid', 'late', or a float in [0, 1]; "
                    f"received {value}."
                )
            return selector
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "hook.layer_selector must be 'mid', 'late', or a float in [0, 1]; "
                f"received {value}."
            )
        value_float = float(value)
        if value_float < 0.0 or value_float > 1.0:
            raise ValueError(
                f"hook.layer_selector must be within [0, 1]; received {value_float}."
            )
        return value_float

    @field_validator("module_path")
    @classmethod
    def _validate_module_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _validate_non_empty_string(value, "hook.module_path")

    @model_validator(mode="after")
    def _validate_layer_spec(self) -> "HookpointConfig":
        has_layer = self.layer is not None
        has_selector = self.layer_selector is not None
        if has_layer == has_selector:
            raise ValueError(
                "Exactly one of hook.layer or hook.layer_selector must be provided."
            )
        return self


class ActivationCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_batch_size: int = Field(
        default=4,
        validation_alias=AliasChoices("llm_batch_size", "batch_size"),
    )
    max_seq_len: int = Field(
        default=128,
        validation_alias=AliasChoices("max_seq_len", "sequence_length"),
    )
    max_tokens: Optional[int] = None
    max_sequences: Optional[int] = None
    vectors_per_shard: int = 262144
    cache_overwrite: bool = False
    split_method: Literal["shard_ranges", "stored_indices"] = "shard_ranges"
    train_fraction: float = 0.9
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    cache_seed: Optional[int] = None
    store_tokens: bool = False
    store_positions: bool = False
    store_sequence_ids: bool = False
    compute_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    storage_dtype: Literal["float32", "float16"] = "float16"
    doc_separator: Literal["eos", "none"] = "eos"
    hf_revision: Optional[str] = None
    text_field: str = "text"

    @field_validator("text_field")
    @classmethod
    def _validate_text_field(cls, value: str) -> str:
        return _validate_non_empty_string(value, "cache.text_field")

    @field_validator("llm_batch_size", mode="before")
    @classmethod
    def _validate_llm_batch_size(cls, value: Any) -> int:
        return _validate_positive_int(value, "cache.llm_batch_size")

    @field_validator("max_seq_len", mode="before")
    @classmethod
    def _validate_max_seq_len(cls, value: Any) -> int:
        return _validate_positive_int(value, "cache.max_seq_len")

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _validate_max_tokens(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"cache.max_tokens must be an integer; received {value}.")
        if value <= 0:
            raise ValueError(
                f"cache.max_tokens must be positive when provided; received {value}."
            )
        return int(value)

    @field_validator("max_sequences", mode="before")
    @classmethod
    def _validate_max_sequences(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"cache.max_sequences must be an integer; received {value}."
            )
        if value <= 0:
            raise ValueError(
                f"cache.max_sequences must be positive when provided; received {value}."
            )
        return int(value)

    @field_validator("vectors_per_shard", mode="before")
    @classmethod
    def _validate_vectors_per_shard(cls, value: Any) -> int:
        return _validate_positive_int(value, "cache.vectors_per_shard")

    @field_validator("cache_seed", mode="before")
    @classmethod
    def _validate_cache_seed(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"cache.cache_seed must be an integer; received {value}.")
        return int(value)

    @field_validator("train_fraction", "val_fraction", "test_fraction", mode="before")
    @classmethod
    def _validate_split_fraction(cls, value: Any, info: Any) -> float:
        field_name = str(info.field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"cache.{field_name} must be numeric; received {value}.")
        value_float = float(value)
        if value_float < 0.0 or value_float > 1.0:
            raise ValueError(
                f"cache.{field_name} must be within [0, 1]; received {value_float}."
            )
        return value_float

    @model_validator(mode="after")
    def _validate_split_fractions_sum(self) -> "ActivationCacheConfig":
        fractions_sum = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(fractions_sum - 1.0) > 1e-6:
            raise ValueError(
                "cache split fractions must sum to 1.0: "
                f"train_fraction + val_fraction + test_fraction = {fractions_sum}."
            )
        return self


class ImageActivationCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_batch_size: int = Field(
        default=4,
        validation_alias=AliasChoices("image_batch_size", "batch_size"),
    )
    image_size: int = 224
    max_images: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("max_images", "max_sequences"),
    )
    vectors_per_shard: int = 262144
    cache_overwrite: bool = False
    split_method: Literal["shard_ranges", "stored_indices"] = "shard_ranges"
    train_fraction: float = 0.9
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    cache_seed: Optional[int] = None
    compute_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    storage_dtype: Literal["float32", "float16"] = "float16"
    token_selection: Literal["patch_only", "cls_only", "all"] = "patch_only"
    representation: Literal["hidden_state"] = "hidden_state"
    hf_revision: Optional[str] = None

    @field_validator("image_batch_size", mode="before")
    @classmethod
    def _validate_image_batch_size(cls, value: Any) -> int:
        return _validate_positive_int(value, "cache.image_batch_size")

    @field_validator("image_size", mode="before")
    @classmethod
    def _validate_image_size(cls, value: Any) -> int:
        return _validate_positive_int(value, "cache.image_size")

    @field_validator("max_images", mode="before")
    @classmethod
    def _validate_max_images(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"cache.max_images must be an integer; received {value}.")
        if value <= 0:
            raise ValueError(
                f"cache.max_images must be positive when provided; received {value}."
            )
        return int(value)

    @field_validator("vectors_per_shard", mode="before")
    @classmethod
    def _validate_vectors_per_shard(cls, value: Any) -> int:
        return _validate_positive_int(value, "cache.vectors_per_shard")

    @field_validator("cache_seed", mode="before")
    @classmethod
    def _validate_cache_seed(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"cache.cache_seed must be an integer; received {value}.")
        return int(value)

    @field_validator("train_fraction", "val_fraction", "test_fraction", mode="before")
    @classmethod
    def _validate_split_fraction(cls, value: Any, info: Any) -> float:
        field_name = str(info.field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"cache.{field_name} must be numeric; received {value}.")
        value_float = float(value)
        if value_float < 0.0 or value_float > 1.0:
            raise ValueError(
                f"cache.{field_name} must be within [0, 1]; received {value_float}."
            )
        return value_float

    @model_validator(mode="after")
    def _validate_split_fractions_sum(self) -> "ImageActivationCacheConfig":
        fractions_sum = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(fractions_sum - 1.0) > 1e-6:
            raise ValueError(
                "cache split fractions must sum to 1.0: "
                f"train_fraction + val_fraction + test_fraction = {fractions_sum}."
            )
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_steps: int = 1000
    batch_size: int = 256
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None
    log_every_n_steps: int = 50
    resample_dead_latents: bool = True
    resample_latents_steps: Optional[int] = None
    checkpoint_model_steps: Optional[int] = None
    dump_training_metrics_steps: Optional[int] = None
    mixed_precision: bool = True
    mixed_precision_dtype: Literal["float16", "bfloat16"] = "float16"
    warmup_steps: int = 0
    threshold_warmup_steps: int = 0
    multi_sparsity_enabled: bool = True
    multi_sparsity_max_instances: Optional[int] = None

    @field_validator("train_steps", mode="before")
    @classmethod
    def _validate_train_steps(cls, value: Any) -> int:
        return _validate_positive_int(value, "training.train_steps")

    @field_validator("batch_size", mode="before")
    @classmethod
    def _validate_batch_size(cls, value: Any) -> int:
        return _validate_positive_int(value, "training.batch_size")

    @field_validator(
        "max_train_samples", "max_val_samples", "max_test_samples", mode="before"
    )
    @classmethod
    def _validate_sample_limit(
        cls,
        value: Any,
        info: Any,
    ) -> Optional[int]:
        if value is None:
            return None
        field_name = str(info.field_name)
        return _validate_positive_int(value, f"training.{field_name}")

    @field_validator("learning_rate", mode="before")
    @classmethod
    def _validate_learning_rate(cls, value: Any) -> float:
        value_float = _validate_non_negative_float(value, "training.learning_rate")
        if value_float <= 0.0:
            raise ValueError(
                f"training.learning_rate must be positive; received {value_float}."
            )
        return float(value_float)

    @field_validator("adam_beta1", "adam_beta2", mode="before")
    @classmethod
    def _validate_adam_beta(cls, value: Any, info: Any) -> float:
        field_name = str(info.field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"training.{field_name} must be numeric; received {value}."
            )
        value_float = float(value)
        if value_float <= 0.0 or value_float >= 1.0:
            raise ValueError(
                f"training.{field_name} must be within (0, 1); received {value_float}."
            )
        return float(value_float)

    @field_validator("weight_decay", mode="before")
    @classmethod
    def _validate_weight_decay(cls, value: Any) -> float:
        return _validate_non_negative_float(value, "training.weight_decay")

    @field_validator("grad_clip_norm", mode="before")
    @classmethod
    def _validate_grad_clip_norm(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        value_float = _validate_non_negative_float(value, "training.grad_clip_norm")
        if value_float <= 0.0:
            raise ValueError(
                f"training.grad_clip_norm must be positive when provided; received {value_float}."
            )
        return float(value_float)

    @field_validator("log_every_n_steps", mode="before")
    @classmethod
    def _validate_log_every_n_steps(cls, value: Any) -> int:
        return _validate_positive_int(value, "training.log_every_n_steps")

    @field_validator("warmup_steps", mode="before")
    @classmethod
    def _validate_warmup_steps(cls, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"training.warmup_steps must be an integer or null; received {value}."
            )
        if value < 0:
            raise ValueError(
                f"training.warmup_steps must be non-negative; received {value}."
            )
        return int(value)

    @field_validator("resample_latents_steps", mode="before")
    @classmethod
    def _validate_resample_latents_steps(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        return _validate_positive_int(value, "training.resample_latents_steps")

    @field_validator("checkpoint_model_steps", mode="before")
    @classmethod
    def _validate_checkpoint_model_steps(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        return _validate_positive_int(value, "training.checkpoint_model_steps")

    @field_validator("dump_training_metrics_steps", mode="before")
    @classmethod
    def _validate_dump_training_metrics_steps(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        return _validate_positive_int(value, "training.dump_training_metrics_steps")

    @field_validator("threshold_warmup_steps", mode="before")
    @classmethod
    def _validate_threshold_warmup_steps(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"training.threshold_warmup_steps must be an integer; received {value}."
            )
        if value < 0:
            raise ValueError(
                "training.threshold_warmup_steps must be non-negative; "
                f"received {value}."
            )
        return int(value)

    @field_validator("multi_sparsity_max_instances", mode="before")
    @classmethod
    def _validate_multi_sparsity_max_instances(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        return _validate_positive_int(value, "training.multi_sparsity_max_instances")


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alive_threshold: float = 1e-6
    alive_min_fraction: float = 1e-4
    min_gamma_samples: int = 32
    gamma_denominator_epsilon: float = 1e-12
    r2_denominator_epsilon: float = 1e-12
    l0_is_nonzero: bool = True

    @field_validator("alive_threshold", mode="before")
    @classmethod
    def _validate_alive_threshold(cls, value: Any) -> float:
        return _validate_non_negative_float(value, "evaluation.alive_threshold")

    @field_validator("alive_min_fraction", mode="before")
    @classmethod
    def _validate_alive_min_fraction(cls, value: Any) -> float:
        value_float = _validate_non_negative_float(
            value, "evaluation.alive_min_fraction"
        )
        if value_float > 1.0:
            raise ValueError(
                "evaluation.alive_min_fraction must be within [0, 1]; "
                f"received {value_float}."
            )
        return float(value_float)

    @field_validator("min_gamma_samples", mode="before")
    @classmethod
    def _validate_min_gamma_samples(cls, value: Any) -> int:
        return _validate_positive_int(value, "evaluation.min_gamma_samples")

    @field_validator(
        "gamma_denominator_epsilon", "r2_denominator_epsilon", mode="before"
    )
    @classmethod
    def _validate_positive_epsilon(cls, value: Any, info: Any) -> float:
        field_name = str(info.field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"evaluation.{field_name} must be numeric; received {value}."
            )
        value_float = float(value)
        if value_float <= 0.0:
            raise ValueError(
                f"evaluation.{field_name} must be positive; received {value_float}."
            )
        return float(value_float)


class LossRecoveredConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_eval_sequences: int = 256
    llm_batch_size: int = 4
    seed: int = 0
    dataset_split: Optional[str] = None
    persist_tokens: bool = True
    denominator_epsilon: float = 1e-12

    @field_validator("max_eval_sequences", mode="before")
    @classmethod
    def _validate_max_eval_sequences(cls, value: Any) -> int:
        return _validate_positive_int(value, "loss_recovered.max_eval_sequences")

    @field_validator("llm_batch_size", mode="before")
    @classmethod
    def _validate_llm_batch_size(cls, value: Any) -> int:
        return _validate_positive_int(value, "loss_recovered.llm_batch_size")

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"loss_recovered.seed must be an integer; received {value}."
            )
        return int(value)

    @field_validator("dataset_split")
    @classmethod
    def _validate_dataset_split(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _validate_non_empty_string(value, "loss_recovered.dataset_split")

    @field_validator("denominator_epsilon", mode="before")
    @classmethod
    def _validate_denominator_epsilon(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "loss_recovered.denominator_epsilon must be numeric; "
                f"received {value}."
            )
        value_float = float(value)
        if value_float <= 0.0:
            raise ValueError(
                "loss_recovered.denominator_epsilon must be positive; "
                f"received {value_float}."
            )
        return value_float

    @model_validator(mode="after")
    def _validate_batch_size_bounds(self) -> "LossRecoveredConfig":
        if int(self.llm_batch_size) > int(self.max_eval_sequences):
            raise ValueError(
                "loss_recovered.llm_batch_size must be <= "
                "loss_recovered.max_eval_sequences."
            )
        return self


class FeatureAblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    num_features: int = 256
    selection: Literal["top_mean_abs", "random_alive"] = "top_mean_abs"
    max_eval_batches: int = 32
    seed: int = 0
    metric: Literal["delta_mse", "delta_lm_loss"] = "delta_mse"
    lm_max_features: int = 32

    @field_validator("num_features", mode="before")
    @classmethod
    def _validate_num_features(cls, value: Any) -> int:
        return _validate_positive_int(value, "feature_ablation.num_features")

    @field_validator("max_eval_batches", mode="before")
    @classmethod
    def _validate_max_eval_batches(cls, value: Any) -> int:
        return _validate_positive_int(value, "feature_ablation.max_eval_batches")

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"feature_ablation.seed must be an integer; received {value}."
            )
        return int(value)

    @field_validator("lm_max_features", mode="before")
    @classmethod
    def _validate_lm_max_features(cls, value: Any) -> int:
        return _validate_positive_int(value, "feature_ablation.lm_max_features")

    @model_validator(mode="after")
    def _validate_lm_feature_budget(self) -> "FeatureAblationConfig":
        if self.metric == "delta_lm_loss" and int(self.lm_max_features) > int(
            self.num_features
        ):
            raise ValueError(
                "feature_ablation.lm_max_features must be <= "
                "feature_ablation.num_features."
            )
        return self


class LatentExamplesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sample_size: int = 512
    max_latents: int = 256
    max_examples_per_latent: int = 8
    activation_threshold: float = 0.0
    latent_selection: Literal[
        "top_by_mean_abs",
        "top_by_max_abs",
        "top_by_freq",
        "explicit",
    ] = "top_by_mean_abs"
    explicit_latents: Optional[List[int]] = None
    target_subruns: Literal["explicit", "matched_l0_anchor", "best_mse"] = (
        "matched_l0_anchor"
    )
    anchor_l0: float = 64.0
    seeds: Optional[List[int]] = None
    output_format: Literal["json"] = "json"
    highlight_markers: Tuple[str, str] = ("|>", "<|")
    image_output_format: Literal["png"] = "png"

    @field_validator(
        "sample_size",
        "max_latents",
        "max_examples_per_latent",
        mode="before",
    )
    @classmethod
    def _validate_positive_integer_fields(cls, value: Any, info: Any) -> int:
        field_name = str(info.field_name)
        return _validate_positive_int(value, f"latent_examples.{field_name}")

    @field_validator("activation_threshold", mode="before")
    @classmethod
    def _validate_activation_threshold(cls, value: Any) -> float:
        return _validate_non_negative_float(
            value, "latent_examples.activation_threshold"
        )

    @field_validator("anchor_l0", mode="before")
    @classmethod
    def _validate_anchor_l0(cls, value: Any) -> float:
        value_float = _validate_non_negative_float(value, "latent_examples.anchor_l0")
        if value_float <= 0.0:
            raise ValueError(
                f"latent_examples.anchor_l0 must be positive; received {value_float}."
            )
        return float(value_float)

    @field_validator("explicit_latents", mode="before")
    @classmethod
    def _validate_explicit_latents(cls, value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(
                "latent_examples.explicit_latents must be a list of integers."
            )
        if not value:
            raise ValueError(
                "latent_examples.explicit_latents must be non-empty when provided."
            )
        normalized: List[int] = []
        for index, entry in enumerate(value):
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise ValueError(
                    f"latent_examples.explicit_latents[{index}] must be an integer; received {entry}."
                )
            if int(entry) < 0:
                raise ValueError(
                    "latent_examples.explicit_latents entries must be >= 0; "
                    f"received {entry}."
                )
            normalized.append(int(entry))
        return normalized

    @field_validator("seeds", mode="before")
    @classmethod
    def _validate_seeds(cls, value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("latent_examples.seeds must be a list of integers.")
        if not value:
            raise ValueError("latent_examples.seeds must be non-empty when provided.")
        normalized: List[int] = []
        for index, entry in enumerate(value):
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise ValueError(
                    f"latent_examples.seeds[{index}] must be an integer; received {entry}."
                )
            normalized.append(int(entry))
        if len(set(normalized)) != len(normalized):
            raise ValueError("latent_examples.seeds must not contain duplicates.")
        return normalized

    @field_validator("highlight_markers", mode="before")
    @classmethod
    def _validate_highlight_markers(cls, value: Any) -> Tuple[str, str]:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                "latent_examples.highlight_markers must contain exactly two strings."
            )
        if not isinstance(value[0], str) or not isinstance(value[1], str):
            raise ValueError(
                "latent_examples.highlight_markers must contain exactly two strings."
            )
        left = _validate_non_empty_string(
            value[0], "latent_examples.highlight_markers[0]"
        )
        right = _validate_non_empty_string(
            value[1], "latent_examples.highlight_markers[1]"
        )
        if left == right:
            raise ValueError(
                "latent_examples.highlight_markers values must be different strings."
            )
        return left, right

    @model_validator(mode="after")
    def _validate_selection_dependencies(self) -> "LatentExamplesConfig":
        if self.latent_selection == "explicit" and self.explicit_latents is None:
            raise ValueError(
                "latent_examples.explicit_latents must be provided when "
                "latent_examples.latent_selection='explicit'."
            )
        if self.latent_selection != "explicit" and self.explicit_latents is not None:
            raise ValueError(
                "latent_examples.explicit_latents can only be provided when "
                "latent_examples.latent_selection='explicit'."
            )
        return self


class ModelVariantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    variant_slug: Optional[str] = None

    sae_class: str
    sign_aware: bool
    d_hidden: int

    sparsity_coeff: Optional[Union[float, List[float]]] = None
    topk_k: Optional[Union[int, List[int]]] = None
    single_activation_offset: bool = False
    reset_activation_offsets_on_resample: bool = True
    symmetric_magnitude: bool = False
    delta_pos_init: Optional[float] = None
    delta_neg_init: Optional[float] = None
    delta_init: Optional[float] = None
    theta_init: Optional[float] = None
    jump_init: Optional[float] = None
    aux_loss_coeff: float = 1.0
    dead_zone_reg_coeff: float = 0.0
    dead_zone_reg_type: Literal["l2"] = "l2"

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _validate_non_empty_string(value, "variants[].name")

    @field_validator("variant_slug")
    @classmethod
    def _validate_variant_slug(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        slug = _validate_non_empty_string(value, "variants[].variant_slug")
        normalized = slugify_variant_name(slug)
        if normalized != slug:
            raise ValueError(
                "variants[].variant_slug must be a filesystem-safe lowercase slug; "
                f"received '{slug}', normalized would be '{normalized}'."
            )
        return slug

    @field_validator("sae_class")
    @classmethod
    def _validate_sae_class(cls, value: str) -> str:
        normalized_value = _validate_non_empty_string(value, "variants[].sae_class")
        if normalized_value not in KNOWN_SAE_CLASS_NAMES:
            raise ValueError(
                f"Unknown variants[].sae_class '{normalized_value}'. "
                f"Allowed values: {sorted(KNOWN_SAE_CLASS_NAMES)}."
            )
        return normalized_value

    @field_validator("d_hidden", mode="before")
    @classmethod
    def _validate_d_hidden(cls, value: Any) -> int:
        return _validate_positive_int(value, "variants[].d_hidden")

    @field_validator("sparsity_coeff", mode="before")
    @classmethod
    def _validate_sparsity_coeff(
        cls, value: Any
    ) -> Optional[Union[float, List[float]]]:
        if value is None:
            return None
        if isinstance(value, list):
            if not value:
                raise ValueError("variants[].sparsity_coeff must be a non-empty list.")
            normalized: List[float] = []
            for index, entry in enumerate(value):
                normalized.append(
                    _validate_non_negative_float(
                        entry, f"variants[].sparsity_coeff[{index}]"
                    )
                )
            return normalized
        return _validate_non_negative_float(value, "variants[].sparsity_coeff")

    @field_validator("topk_k", mode="before")
    @classmethod
    def _validate_topk_k(cls, value: Any) -> Optional[Union[int, List[int]]]:
        if value is None:
            return None
        if isinstance(value, list):
            if not value:
                raise ValueError("variants[].topk_k must be a non-empty list.")
            normalized: List[int] = []
            for index, entry in enumerate(value):
                normalized.append(
                    _validate_positive_int(entry, f"variants[].topk_k[{index}]")
                )
            return normalized
        return _validate_positive_int(value, "variants[].topk_k")

    @model_validator(mode="after")
    def _validate_sweep_controller(self) -> "ModelVariantConfig":
        has_lambda = self.sparsity_coeff is not None
        has_k = self.topk_k is not None
        if has_lambda == has_k:
            raise ValueError(
                "Exactly one of variants[].sparsity_coeff or variants[].topk_k must be provided."
            )
        return self

    @model_validator(mode="after")
    def _validate_sweep_controller_matches_sae_class(self) -> "ModelVariantConfig":
        if self.sae_class in TOPK_SAE_CLASS_NAMES:
            if self.topk_k is None:
                raise ValueError(f"{self.sae_class} requires variants[].topk_k.")
            return self

        if self.topk_k is not None:
            raise ValueError(
                f"{self.sae_class} does not support variants[].topk_k. "
                "Use variants[].sparsity_coeff."
            )
        return self

    @model_validator(mode="after")
    def _validate_topk_k_within_width(self) -> "ModelVariantConfig":
        if self.sae_class in TOPK_SAE_CLASS_NAMES and self.topk_k is not None:
            topk_values = (
                [int(value) for value in self.topk_k]
                if isinstance(self.topk_k, list)
                else [int(self.topk_k)]
            )
            for topk_k in topk_values:
                if int(topk_k) > int(self.d_hidden):
                    raise ValueError(
                        "variants[].topk_k must satisfy topk_k <= d_hidden; "
                        f"received topk_k={topk_k}, d_hidden={self.d_hidden}."
                    )
        return self

    @model_validator(mode="after")
    def _validate_sweep_values_do_not_collide(self) -> "ModelVariantConfig":
        if isinstance(self.sparsity_coeff, list):
            seen_by_formatted_value: Dict[str, float] = {}
            colliding_values_by_formatted_value: Dict[str, List[float]] = {}
            for value in self.sparsity_coeff:
                sparsity_coeff = float(value)
                formatted_value = format_sweep_value_for_dir(sparsity_coeff)
                if formatted_value in seen_by_formatted_value:
                    colliding_values_by_formatted_value.setdefault(
                        formatted_value, [seen_by_formatted_value[formatted_value]]
                    ).append(sparsity_coeff)
                else:
                    seen_by_formatted_value[formatted_value] = sparsity_coeff
            if colliding_values_by_formatted_value:
                colliding_values = ", ".join(
                    f"{formatted_value}: {raw_values}"
                    for formatted_value, raw_values in sorted(
                        colliding_values_by_formatted_value.items()
                    )
                )
                raise ValueError(
                    "variants[].sparsity_coeff contains duplicate values after "
                    f"directory formatting ({colliding_values})."
                )

        if isinstance(self.topk_k, list):
            seen_values: Set[int] = set()
            duplicate_values: Set[int] = set()
            for value in self.topk_k:
                topk_k = int(value)
                if topk_k in seen_values:
                    duplicate_values.add(topk_k)
                else:
                    seen_values.add(topk_k)
            if duplicate_values:
                duplicates = sorted(list(duplicate_values))
                raise ValueError(
                    "variants[].topk_k contains duplicate values: " f"{duplicates}."
                )

        return self

    @field_validator(
        "delta_pos_init",
        "delta_neg_init",
        "delta_init",
        "theta_init",
        "jump_init",
        mode="before",
    )
    @classmethod
    def _validate_optional_non_negative_finite_float(
        cls, value: Any, info: Any
    ) -> Optional[float]:
        if value is None:
            return None
        field_name = str(info.field_name)
        value_float = _validate_non_negative_float(value, f"variants[].{field_name}")
        if not value_float < float("inf"):
            raise ValueError(
                f"variants[].{field_name} must be finite; received {value_float}."
            )
        return float(value_float)

    @field_validator("aux_loss_coeff", "dead_zone_reg_coeff", mode="before")
    @classmethod
    def _validate_non_negative_finite_float(cls, value: Any, info: Any) -> float:
        field_name = str(info.field_name)
        value_float = _validate_non_negative_float(value, f"variants[].{field_name}")
        if not value_float < float("inf"):
            raise ValueError(
                f"variants[].{field_name} must be finite; received {value_float}."
            )
        return float(value_float)

    @field_validator("dead_zone_reg_type", mode="before")
    @classmethod
    def _validate_dead_zone_reg_type(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(
                "variants[].dead_zone_reg_type must be 'l2'; " f"received {value}."
            )
        normalized = value.strip().lower()
        if normalized != "l2":
            raise ValueError(
                "variants[].dead_zone_reg_type must be 'l2'; " f"received {value}."
            )
        return normalized

    @model_validator(mode="after")
    def _validate_variant_options(self) -> "ModelVariantConfig":
        if self.sign_aware and self.sae_class not in SIGN_AWARE_SAE_CLASS_NAMES:
            raise ValueError(
                "variants[].sign_aware=true is only supported for "
                f"{sorted(SIGN_AWARE_SAE_CLASS_NAMES)}; received sae_class='{self.sae_class}'."
            )
        if (
            any(
                value is not None
                for value in (self.delta_init, self.delta_pos_init, self.delta_neg_init)
            )
            and not self.sign_aware
        ):
            raise ValueError(
                "variants[].delta_init, variants[].delta_pos_init, and "
                "variants[].delta_neg_init require sign_aware=true."
            )
        if self.symmetric_magnitude and (
            self.sae_class != "GatedSAE" or not self.sign_aware
        ):
            raise ValueError(
                "variants[].symmetric_magnitude can only be enabled for "
                "sae_class='GatedSAE' with sign_aware=true."
            )
        if self.delta_init is not None and not self.single_activation_offset:
            raise ValueError(
                "variants[].delta_init requires single_activation_offset=true."
            )
        if self.delta_init is not None and (
            self.delta_pos_init is not None or self.delta_neg_init is not None
        ):
            raise ValueError(
                "variants[].delta_init cannot be combined with delta_pos_init/delta_neg_init."
            )
        if (
            self.theta_init is not None
            and self.sae_class not in THETA_INIT_SAE_CLASS_NAMES
        ):
            raise ValueError(
                "variants[].theta_init is only supported for "
                "SignedSoftThresholdSAE and JumpReLUSAE."
            )
        if (
            self.jump_init is not None
            and self.sae_class not in JUMP_INIT_SAE_CLASS_NAMES
        ):
            raise ValueError("variants[].jump_init is only supported for JumpReLUSAE.")
        if self.sae_class == "JumpReLUSAE":
            if self.theta_init is not None and self.theta_init <= 0.0:
                raise ValueError(
                    "variants[].theta_init must be > 0 for sae_class='JumpReLUSAE'."
                )
            if self.jump_init is not None and self.jump_init <= 0.0:
                raise ValueError(
                    "variants[].jump_init must be > 0 for sae_class='JumpReLUSAE'."
                )
        if self.aux_loss_coeff != 1.0 and self.sae_class != "GatedSAE":
            raise ValueError(
                "variants[].aux_loss_coeff can only differ from 1.0 when "
                "sae_class='GatedSAE'."
            )
        if self.dead_zone_reg_coeff > 0.0 and (
            self.sae_class != "GatedSAE" or not self.sign_aware
        ):
            raise ValueError(
                "variants[].dead_zone_reg_coeff > 0 requires "
                "sae_class='GatedSAE' with sign_aware=true."
            )
        return self


class LLMBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str
    device: str = "auto"
    stages: List[Stage] = ["cache", "train", "eval", "plot", "summarize"]
    resume: bool = True
    overwrite: bool = False
    seed: Union[int, List[int]]
    modality: Literal["text", "image"] = "text"

    llm: Optional[LLMModelConfig] = None
    vision: Optional[VisionModelConfig] = None
    data: Union[TextDataConfig, ImageDataConfig]
    hook: HookpointConfig
    cache: Union[ActivationCacheConfig, ImageActivationCacheConfig] = Field(
        default_factory=ActivationCacheConfig
    )
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    variants: List[ModelVariantConfig] = Field(default_factory=list)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    loss_recovered: LossRecoveredConfig = Field(default_factory=LossRecoveredConfig)
    feature_ablation: FeatureAblationConfig = Field(
        default_factory=FeatureAblationConfig
    )
    latent_examples: LatentExamplesConfig = Field(default_factory=LatentExamplesConfig)

    @field_validator("output_dir")
    @classmethod
    def _validate_output_dir(cls, value: str) -> str:
        return _validate_non_empty_string(value, "output_dir")

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, value: Any) -> str:
        return validate_experiment_device(value)

    @field_validator("modality", mode="before")
    @classmethod
    def _normalize_modality(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"modality must be one of ['text', 'image']; received {value}."
            )
        normalized = value.strip().lower()
        if normalized not in {"text", "image"}:
            raise ValueError(
                f"modality must be one of ['text', 'image']; received {value}."
            )
        return normalized

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: Any) -> Union[int, List[int]]:
        if isinstance(value, bool):
            raise ValueError(
                f"seed must be an integer or list of integers; received {value}."
            )
        if isinstance(value, list):
            if not value:
                raise ValueError(
                    "seed must be a non-empty list when provided as a list."
                )
            normalized: List[int] = []
            for index, entry in enumerate(value):
                if isinstance(entry, bool) or not isinstance(entry, int):
                    raise ValueError(
                        f"seed[{index}] must be an integer; received {entry}."
                    )
                normalized.append(int(entry))
            if len(set(normalized)) != len(normalized):
                raise ValueError("seed must not contain duplicates.")
            return normalized
        if not isinstance(value, int):
            raise ValueError(f"seed must be an integer; received {value}.")
        return int(value)

    @field_validator("stages", mode="before")
    @classmethod
    def _normalize_stages(cls, value: Any) -> Any:
        if isinstance(value, list):
            normalized: List[str] = []
            for entry in value:
                if not isinstance(entry, str):
                    raise ValueError("stages must be a list of strings.")
                normalized.append(entry.strip().lower())
            return normalized
        return value

    @field_validator("data", mode="before")
    @classmethod
    def _validate_data_by_modality(cls, value: Any, info: Any) -> Any:
        modality_raw = info.data.get("modality", "text")
        modality = str(modality_raw).strip().lower()
        if modality == "image":
            if isinstance(value, ImageDataConfig):
                return value
            return ImageDataConfig.model_validate(value)
        if isinstance(value, TextDataConfig):
            return value
        return TextDataConfig.model_validate(value)

    @field_validator("cache", mode="before")
    @classmethod
    def _validate_cache_by_modality(cls, value: Any, info: Any) -> Any:
        modality_raw = info.data.get("modality", "text")
        modality = str(modality_raw).strip().lower()
        if modality == "image":
            if isinstance(value, ImageActivationCacheConfig):
                return value
            return ImageActivationCacheConfig.model_validate(value)
        if isinstance(value, ActivationCacheConfig):
            return value
        return ActivationCacheConfig.model_validate(value)

    @model_validator(mode="after")
    def _validate_stages(self) -> "LLMBenchmarkConfig":
        if "validate" in self.stages and len(self.stages) != 1:
            raise ValueError(
                "If 'validate' is present in stages, it must be the only stage."
            )
        return self

    @model_validator(mode="after")
    def _validate_variants_unique(self) -> "LLMBenchmarkConfig":
        if not self.variants:
            return self
        if len({variant.name for variant in self.variants}) != len(self.variants):
            raise ValueError("Variant names must be unique.")
        return self

    @model_validator(mode="after")
    def _validate_modality_requirements(self) -> "LLMBenchmarkConfig":
        if self.modality == "text":
            if self.llm is None:
                raise ValueError("llm must be provided when modality='text'.")
            if self.vision is not None:
                raise ValueError("vision must be omitted when modality='text'.")
            if not isinstance(self.data, TextDataConfig):
                raise ValueError("data must use TextDataConfig when modality='text'.")
            if not isinstance(self.cache, ActivationCacheConfig):
                raise ValueError(
                    "cache must use ActivationCacheConfig when modality='text'."
                )
            return self

        if self.vision is None:
            raise ValueError("vision must be provided when modality='image'.")
        if self.llm is not None:
            raise ValueError("llm must be omitted when modality='image'.")
        if not isinstance(self.data, ImageDataConfig):
            raise ValueError("data must use ImageDataConfig when modality='image'.")
        if not isinstance(self.cache, ImageActivationCacheConfig):
            raise ValueError(
                "cache must use ImageActivationCacheConfig when modality='image'."
            )
        if self.loss_recovered.enabled:
            raise ValueError(
                "loss_recovered.enabled=true is not supported when modality='image'."
            )
        if self.hook.hook_name != "hidden_state":
            raise ValueError(
                "When modality='image', hook.hook_name must be 'hidden_state'."
            )
        if self.hook.module_path is not None:
            raise ValueError("When modality='image', hook.module_path must be omitted.")
        return self

    @model_validator(mode="after")
    def _validate_text_field_consistency(self) -> "LLMBenchmarkConfig":
        if self.modality != "text":
            return self
        if not isinstance(self.data, TextDataConfig) or not isinstance(
            self.cache, ActivationCacheConfig
        ):
            raise ValueError(
                "text modality requires TextDataConfig and ActivationCacheConfig."
            )
        data_has_explicit_text_field = "text_field" in self.data.model_fields_set
        cache_has_explicit_text_field = "text_field" in self.cache.model_fields_set

        if data_has_explicit_text_field and cache_has_explicit_text_field:
            if self.data.text_field != self.cache.text_field:
                raise ValueError(
                    f"data.text_field ({self.data.text_field}) must match "
                    f"cache.text_field ({self.cache.text_field})."
                )
            return self

        if data_has_explicit_text_field and not cache_has_explicit_text_field:
            self.cache.text_field = self.data.text_field
            return self

        if cache_has_explicit_text_field and not data_has_explicit_text_field:
            self.data.text_field = self.cache.text_field
            return self

        if self.data.text_field != self.cache.text_field:
            self.cache.text_field = self.data.text_field

        return self

    @model_validator(mode="after")
    def _validate_image_size_consistency(self) -> "LLMBenchmarkConfig":
        if self.modality != "image":
            return self
        if not isinstance(self.vision, VisionModelConfig) or not isinstance(
            self.cache, ImageActivationCacheConfig
        ):
            raise ValueError(
                "image modality requires VisionModelConfig and ImageActivationCacheConfig."
            )

        vision_has_explicit_image_size = "image_size" in self.vision.model_fields_set
        cache_has_explicit_image_size = "image_size" in self.cache.model_fields_set

        if vision_has_explicit_image_size and cache_has_explicit_image_size:
            if int(self.vision.image_size) != int(self.cache.image_size):
                raise ValueError(
                    f"vision.image_size ({self.vision.image_size}) must match "
                    f"cache.image_size ({self.cache.image_size}) when both are set."
                )
            return self

        if vision_has_explicit_image_size and not cache_has_explicit_image_size:
            self.cache.image_size = int(self.vision.image_size)
            return self

        if cache_has_explicit_image_size and not vision_has_explicit_image_size:
            self.vision.image_size = int(self.cache.image_size)
            return self

        if int(self.vision.image_size) != int(self.cache.image_size):
            self.cache.image_size = int(self.vision.image_size)
        return self

    @model_validator(mode="after")
    def _validate_feature_ablation_dependencies(self) -> "LLMBenchmarkConfig":
        if (
            self.feature_ablation.metric == "delta_lm_loss"
            and not self.loss_recovered.enabled
        ):
            raise ValueError(
                "feature_ablation.metric='delta_lm_loss' requires "
                "loss_recovered.enabled=true."
            )
        return self


def _load_raw_config(path: str) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    with open(path, "r") as f:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ValueError("PyYAML is required to load YAML configs.")
            return yaml.safe_load(f)
        return json.load(f)


def _resolve_local_path(path: str, config_dir: str) -> str:
    expanded_path = os.path.expanduser(path)
    if os.path.isabs(expanded_path):
        return os.path.abspath(expanded_path)
    return os.path.abspath(os.path.join(config_dir, expanded_path))


def _normalize_local_data_paths(
    raw_config: Dict[str, Any],
    config_path: str,
) -> Dict[str, Any]:
    normalized_config = dict(raw_config)
    data = normalized_config.get("data")
    if not isinstance(data, dict):
        return normalized_config

    normalized_data = dict(data)
    backend_raw = normalized_data.get("backend")
    backend = (
        str(backend_raw).strip().lower()
        if isinstance(backend_raw, str)
        else backend_raw
    )
    config_dir = os.path.dirname(os.path.abspath(config_path))

    if backend == "imagefolder":
        data_dir = normalized_data.get("data_dir")
        if isinstance(data_dir, str) and data_dir.strip():
            normalized_data["data_dir"] = _resolve_local_path(
                path=data_dir,
                config_dir=config_dir,
            )

    if backend == "local_jsonl":
        data_files = normalized_data.get("data_files")
        if isinstance(data_files, list):
            resolved_files: List[Any] = []
            for entry in data_files:
                if isinstance(entry, str) and entry.strip():
                    resolved_files.append(
                        _resolve_local_path(path=entry, config_dir=config_dir)
                    )
                else:
                    resolved_files.append(entry)
            normalized_data["data_files"] = resolved_files

        data_files_glob = normalized_data.get("data_files_glob")
        if isinstance(data_files_glob, str) and data_files_glob.strip():
            normalized_data["data_files_glob"] = _resolve_local_path(
                path=data_files_glob,
                config_dir=config_dir,
            )

    normalized_config["data"] = normalized_data
    return normalized_config


def load_benchmark_config(path: str) -> LLMBenchmarkConfig:
    raw_config = _load_raw_config(path)
    if not isinstance(raw_config, dict):
        raise ValueError("Benchmark configuration must be a JSON/YAML object.")
    raw_config = _normalize_local_data_paths(raw_config=raw_config, config_path=path)
    return LLMBenchmarkConfig.model_validate(raw_config)


def resolve_benchmark_config(
    config: LLMBenchmarkConfig,
    stages_override: Optional[List[str]] = None,
    validate_only: bool = False,
) -> Dict[str, Any]:
    if validate_only and stages_override is not None:
        raise ValueError("validate_only and stages_override cannot both be provided.")

    effective_stages: List[str]
    if validate_only:
        effective_stages = ["validate"]
    elif stages_override is not None:
        effective_stages = [stage.strip().lower() for stage in stages_override]
    else:
        effective_stages = [str(stage) for stage in config.stages]

    if "validate" in effective_stages and len(effective_stages) != 1:
        raise ValueError(
            "If 'validate' is present in stages, it must be the only stage."
        )
    unknown_stages = [
        stage for stage in effective_stages if stage not in _ALLOWED_STAGES
    ]
    if unknown_stages:
        raise ValueError(
            "Unknown stage(s): "
            f"{unknown_stages}. Allowed stages: {sorted(_ALLOWED_STAGES)}."
        )

    if isinstance(config.seed, list):
        seeds = sorted([int(seed) for seed in config.seed])
    else:
        seeds = [int(config.seed)]

    seen_slugs: Dict[str, str] = {}
    resolved_variants: List[Dict[str, Any]] = []
    expanded_variants: List[Tuple[str, str, str, List[Union[int, float]]]] = []
    for variant in config.variants:
        derived_slug = variant.variant_slug or slugify_variant_name(variant.name)
        existing_name = seen_slugs.get(derived_slug)
        if existing_name is not None:
            raise ValueError(
                "variants produce a colliding variant_slug. Colliding slug "
                f"'{derived_slug}' comes from '{existing_name}' and '{variant.name}'."
            )
        seen_slugs[derived_slug] = variant.name

        sweep_param: str
        sweep_values: List[Union[int, float]]
        if variant.sparsity_coeff is not None:
            sweep_param = "lambda"
            if isinstance(variant.sparsity_coeff, list):
                sweep_values = [float(value) for value in variant.sparsity_coeff]
            else:
                sweep_values = [float(variant.sparsity_coeff)]
        else:
            sweep_param = "k"
            if isinstance(variant.topk_k, list):
                sweep_values = [int(value) for value in variant.topk_k]
            else:
                sweep_values = [int(variant.topk_k)]

        resolved_variant = variant.model_dump()
        resolved_variant["variant_slug"] = derived_slug
        resolved_variants.append(resolved_variant)
        expanded_variants.append(
            (variant.name, derived_slug, sweep_param, sweep_values)
        )

    subruns: List[Dict[str, Any]] = []
    for variant_name, variant_slug, sweep_param, sweep_values in expanded_variants:
        for seed in seeds:
            for sweep_value in sweep_values:
                subruns.append(
                    {
                        "variant_name": variant_name,
                        "variant_slug": variant_slug,
                        "seed": int(seed),
                        "sweep_param": sweep_param,
                        "sweep_value": sweep_value,
                        "relative_subrun_dir": build_subrun_relative_dir(
                            variant_slug=variant_slug,
                            seed=int(seed),
                            sweep_param=sweep_param,
                            sweep_value=sweep_value,
                        ),
                    }
                )

    resolved_config: Dict[str, Any] = {
        "schema_version": RESOLVED_CONFIG_SCHEMA_VERSION,
        "config_fingerprint": None,
        "rng_fingerprint": None,
        "output_dir": config.output_dir,
        "device": config.device,
        "stages": effective_stages,
        "resume": bool(config.resume),
        "overwrite": bool(config.overwrite),
        "seeds": seeds,
        "modality": config.modality,
        "llm": config.llm.model_dump() if config.llm is not None else None,
        "vision": config.vision.model_dump() if config.vision is not None else None,
        "data": config.data.model_dump(),
        "hook": config.hook.model_dump(),
        "cache": config.cache.model_dump(),
        "training": config.training.model_dump(),
        "variants": resolved_variants,
        "evaluation": config.evaluation.model_dump(),
        "loss_recovered": config.loss_recovered.model_dump(),
        "feature_ablation": config.feature_ablation.model_dump(),
        "latent_examples": config.latent_examples.model_dump(),
        "subruns": subruns,
    }
    resolved_config["config_fingerprint"] = _compute_config_fingerprint(resolved_config)
    resolved_config["rng_fingerprint"] = _compute_rng_fingerprint(resolved_config)
    return resolved_config
