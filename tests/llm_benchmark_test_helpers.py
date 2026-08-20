from typing import Any, Dict

from sgnsae.experiments.llm_benchmark.schema_versions import (
    TRAINING_METRICS_SCHEMA_VERSION,
)


def legacy_success_training_metrics_payload() -> Dict[str, Any]:
    return {
        "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
        "variant_name": "baseline_relu",
        "variant_slug": "baseline_relu",
        "seed": 0,
        "sweep_param": "lambda",
        "sweep_value": 1e-3,
        "d_in": 768,
        "d_hidden": 4096,
        "train_steps": 1000,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "adam_betas": [0.9, 0.999],
        "warmup_steps": 0,
        "mixed_precision": False,
        "mixed_precision_dtype": "bfloat16",
        "threshold_warmup_steps": 0,
        "resample_dead_latents": False,
        "resample_latents_threshold": 0,
        "loss_knobs": {},
        "history": [],
    }
