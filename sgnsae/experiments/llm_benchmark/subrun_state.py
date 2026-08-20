import json
import os
from typing import Any, Dict, List, Literal, Optional, Set

from sgnsae.experiments.llm_benchmark.schema_versions import (
    TRAINING_METRICS_SCHEMA_VERSION,
)


TrainingState = Literal["completed", "legacy_unmigrated", "incomplete"]

LEGACY_SUCCESS_REQUIRED_KEYS: Set[str] = {
    "schema_version",
    "variant_name",
    "variant_slug",
    "seed",
    "sweep_param",
    "sweep_value",
    "d_in",
    "d_hidden",
    "train_steps",
    "batch_size",
    "learning_rate",
    "adam_betas",
    "warmup_steps",
    "mixed_precision",
    "mixed_precision_dtype",
    "threshold_warmup_steps",
    "resample_dead_latents",
    "resample_latents_threshold",
    "loss_knobs",
    "history",
}


def training_metrics_path(subrun_dir: str) -> str:
    return os.path.join(subrun_dir, "training_metrics.json")


def load_training_metrics_if_present(subrun_dir: str) -> Optional[Dict[str, Any]]:
    path = training_metrics_path(subrun_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as file:
            payload = json.load(file)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_completed_payload(payload: Dict[str, Any]) -> bool:
    return (
        payload.get("status") == "completed"
        and not bool(payload.get("terminated_early", False))
        and payload.get("failure") in (None, {})
    )


def _is_non_bool_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_non_bool_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_completed_train_steps(subrun_dir: str) -> Optional[int]:
    metrics = load_training_metrics_if_present(subrun_dir)
    if metrics is None:
        return None
    if not is_completed_payload(metrics):
        return None
    train_steps = metrics.get("train_steps")
    if not _is_non_bool_int(train_steps) or int(train_steps) <= 0:
        return None
    return int(train_steps)


# Retained as public API for runner continuation detection (task 6)
# and external tooling. _load_continuation_state reads history inline
# to avoid a redundant disk read.
def load_completed_training_history(
    subrun_dir: str,
) -> Optional[List[Dict[str, Any]]]:
    metrics = load_training_metrics_if_present(subrun_dir)
    if metrics is None:
        return None
    if not is_completed_payload(metrics):
        return None
    history = metrics.get("history")
    if not isinstance(history, list):
        return None
    return list(history)


def is_legacy_success_training_metrics_payload(payload: Dict[str, Any]) -> bool:
    if "status" in payload:
        return False
    if bool(payload.get("terminated_early", False)):
        return False
    if payload.get("failure") not in (None, {}):
        return False
    if not LEGACY_SUCCESS_REQUIRED_KEYS.issubset(payload.keys()):
        return False
    if payload.get("schema_version") != TRAINING_METRICS_SCHEMA_VERSION:
        return False
    if not _is_non_empty_str(payload.get("variant_name")):
        return False
    if not _is_non_empty_str(payload.get("variant_slug")):
        return False
    if not _is_non_bool_int(payload.get("seed")) or int(payload["seed"]) < 0:
        return False
    if not _is_non_empty_str(payload.get("sweep_param")):
        return False
    if not _is_non_bool_number(payload.get("sweep_value")):
        return False
    for key in ("d_in", "d_hidden", "train_steps", "batch_size"):
        value = payload.get(key)
        if not _is_non_bool_int(value) or int(value) <= 0:
            return False
    if not _is_non_bool_number(payload.get("learning_rate")):
        return False
    if float(payload["learning_rate"]) <= 0.0:
        return False
    for key in (
        "warmup_steps",
        "threshold_warmup_steps",
        "resample_latents_threshold",
    ):
        value = payload.get(key)
        if not _is_non_bool_int(value) or int(value) < 0:
            return False
    if payload.get("mixed_precision") not in {True, False}:
        return False
    if not _is_non_empty_str(payload.get("mixed_precision_dtype")):
        return False
    if payload.get("resample_dead_latents") not in {True, False}:
        return False
    if not isinstance(payload.get("loss_knobs"), dict):
        return False
    if not isinstance(payload.get("history"), list):
        return False
    adam_betas = payload.get("adam_betas")
    if not (
        isinstance(adam_betas, list)
        and len(adam_betas) == 2
        and all(_is_non_bool_number(value) for value in adam_betas)
    ):
        return False
    if not all(0.0 <= float(value) < 1.0 for value in adam_betas):
        return False
    return True


def subrun_training_state(subrun_dir: str) -> TrainingState:
    metrics = load_training_metrics_if_present(subrun_dir)
    if metrics is None:
        return "incomplete"
    if is_completed_payload(metrics):
        return "completed"
    if is_legacy_success_training_metrics_payload(metrics):
        return "legacy_unmigrated"
    return "incomplete"


def subrun_training_completed(subrun_dir: str) -> bool:
    return subrun_training_state(subrun_dir) == "completed"


def legacy_migration_error(output_dir: str, subrun_dir: str) -> ValueError:
    return ValueError(
        "Encountered legacy pre-status training_metrics.json in "
        f"subrun_dir={subrun_dir}. "
        "Run `python scripts/migrate_llm_training_status.py "
        f"{output_dir}` to mark trusted historical LLM benchmark outputs as "
        "completed, or rerun with overwrite=true to regenerate them."
    )
