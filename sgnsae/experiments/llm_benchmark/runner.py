import json
import os
import re
import shutil
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from tqdm import tqdm

from sgnsae.experiments.llm_benchmark.config import (
    LLMBenchmarkConfig,
    load_benchmark_config,
    resolve_benchmark_config,
)
from sgnsae.experiments.llm_benchmark.cache import run_cache_stage
from sgnsae.experiments.llm_benchmark.environment import persist_environment_metadata
from sgnsae.experiments.llm_benchmark.examples import (
    build_and_persist_latent_examples_index,
    build_and_persist_paired_duplicate_case_studies,
    generate_latent_examples,
    select_target_subruns_for_examples,
    validate_examples_cache_metadata,
)
from sgnsae.experiments.llm_benchmark.loss_recovered import (
    ensure_run_level_loss_recovered_artifacts,
)
from sgnsae.experiments.llm_benchmark.matched_points import (
    compute_and_persist_matched_points,
    matched_points_path,
)
from sgnsae.experiments.llm_benchmark.metrics import (
    build_and_persist_comparison_summary,
    eval_artifact_paths,
    run_evaluation_for_subrun,
)
from sgnsae.experiments.llm_benchmark.plotting import (
    ensure_summary_has_matched_l0_anchors,
    generate_plots_for_run,
    plot_stage_artifact_paths,
)
from sgnsae.experiments.llm_benchmark.schema_versions import RUN_STATUS_SCHEMA_VERSION
from sgnsae.experiments.llm_benchmark.subrun_state import (
    legacy_migration_error,
    load_completed_train_steps,
    load_training_metrics_if_present as load_subrun_training_metrics_if_present,
    subrun_training_state,
    subrun_training_completed,
    training_metrics_path,
)
from sgnsae.experiments.llm_benchmark.training import (
    load_activation_manifest,
    resolve_manifest_path,
    run_training_for_lambda_chunk,
    run_training_for_subrun,
    train_artifact_paths,
)


_STAGE_UNIVERSE = [
    "validate",
    "cache",
    "train",
    "eval",
    "examples",
    "plot",
    "summarize",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _delete_run_level_downstream_artifacts(output_dir: str) -> None:
    paths = [
        os.path.join(output_dir, "comparison_summary.json"),
        os.path.join(output_dir, "results_summary.json"),
        os.path.join(output_dir, "latent_examples_index.json"),
        os.path.join(output_dir, "paired_duplicate_case_studies.json"),
        *plot_stage_artifact_paths(output_dir=output_dir).values(),
    ]
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)


def _load_existing_fingerprint(output_dir: str) -> Optional[str]:
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        existing = _read_json(config_path)
    except Exception:
        return None
    fingerprint = existing.get("config_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        return None
    return str(fingerprint)


def _load_existing_rng_fingerprint(output_dir: str) -> Optional[str]:
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        existing = _read_json(config_path)
    except Exception:
        return None
    rng_fingerprint = existing.get("rng_fingerprint")
    if not isinstance(rng_fingerprint, str) or not rng_fingerprint.strip():
        return None
    return str(rng_fingerprint)


def _load_existing_run_status(output_dir: str) -> Optional[Dict[str, Any]]:
    run_status_path = os.path.join(output_dir, "run_status.json")
    if not os.path.isfile(run_status_path):
        return None
    try:
        raw = _read_json(run_status_path)
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _load_training_metrics_if_present(subrun_dir: str) -> Optional[Dict[str, Any]]:
    return load_subrun_training_metrics_if_present(subrun_dir)


def _training_completed_successfully(subrun_dir: str) -> bool:
    artifacts = train_artifact_paths(subrun_dir)
    if not (
        os.path.isfile(artifacts["model"])
        and os.path.isfile(training_metrics_path(subrun_dir))
    ):
        return False
    return subrun_training_completed(subrun_dir)


def _assert_no_legacy_subruns(
    *, output_dir: str, subruns: List[Dict[str, Any]]
) -> None:
    for subrun in subruns:
        subrun_dir = _output_subdir_path(
            output_dir,
            str(subrun["relative_subrun_dir"]),
        )
        if subrun_training_state(subrun_dir) == "legacy_unmigrated":
            raise legacy_migration_error(
                output_dir=output_dir,
                subrun_dir=subrun_dir,
            )


def _build_stage_statuses_for_validate_only() -> Dict[str, Dict[str, Any]]:
    stages: Dict[str, Dict[str, Any]] = {}
    for stage in _STAGE_UNIVERSE:
        stages[stage] = {
            "status": "done" if stage == "validate" else "not_requested",
        }
    return stages


def _build_stage_statuses_for_execution(
    requested_stages: List[str],
) -> Dict[str, Dict[str, Any]]:
    requested_set = set(requested_stages)
    stages: Dict[str, Dict[str, Any]] = {}
    for stage in _STAGE_UNIVERSE:
        if stage not in requested_set:
            stages[stage] = {"status": "not_requested"}
        else:
            stages[stage] = {"status": "pending"}
    return stages


def _persist_run_status(
    output_dir: str,
    config_fingerprint: str,
    requested_stages: List[str],
    resume_check: Dict[str, Any],
    stage_statuses: Dict[str, Dict[str, Any]],
    resolve_timestamps: Tuple[str, str],
) -> str:
    existing = _load_existing_run_status(output_dir)
    created_at = (
        str(existing.get("created_at"))
        if existing is not None and isinstance(existing.get("created_at"), str)
        else _utc_now_iso()
    )
    updated_at = _utc_now_iso()
    resolve_started_at, resolve_ended_at = resolve_timestamps
    payload: Dict[str, Any] = {
        "schema_version": RUN_STATUS_SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint,
        "requested_stages": requested_stages,
        "created_at": created_at,
        "updated_at": updated_at,
        "resume_check": resume_check,
        "internal_steps": {
            "resolve": {
                "status": "done",
                "started_at": resolve_started_at,
                "ended_at": resolve_ended_at,
            }
        },
        "stages": stage_statuses,
    }
    run_status_path = os.path.join(output_dir, "run_status.json")
    _write_json(run_status_path, payload)
    return run_status_path


def _effective_stages(
    config: LLMBenchmarkConfig,
    stages_override: Optional[List[str]],
    validate_only: bool,
) -> List[str]:
    if validate_only and stages_override is not None:
        raise ValueError("validate_only and stages_override cannot both be provided.")
    if validate_only:
        return ["validate"]
    if stages_override is None:
        return [str(stage) for stage in config.stages]
    normalized = [str(stage).strip().lower() for stage in stages_override]
    unknown = [stage for stage in normalized if stage not in _STAGE_UNIVERSE]
    if unknown:
        raise ValueError(
            f"Unknown stage(s): {unknown}. Allowed stages: {_STAGE_UNIVERSE}."
        )
    return normalized


def _output_subdir_path(output_dir: str, relative_dir: str) -> str:
    normalized = relative_dir.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        raise ValueError("Resolved config contains an empty relative_subrun_dir.")
    for part in parts:
        if part in {".", ".."}:
            raise ValueError(
                f"Resolved config contains an invalid relative_subrun_dir segment: '{part}'."
            )

    base_directory = os.path.abspath(output_dir)
    candidate_path = os.path.abspath(os.path.join(base_directory, *parts))
    try:
        common_path = os.path.commonpath([base_directory, candidate_path])
    except ValueError as exc:
        raise ValueError(
            f"Resolved relative_subrun_dir escapes output_dir: {relative_dir}"
        ) from exc
    if common_path != base_directory:
        raise ValueError(
            f"Resolved relative_subrun_dir escapes output_dir: {relative_dir}"
        )
    return candidate_path


def _are_artifacts_present(paths: Dict[str, str]) -> bool:
    return all(os.path.isfile(path) for path in paths.values())


def _resolve_effective_loss_knobs(variant: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "aux_loss_coeff": float(variant.get("aux_loss_coeff", 1.0)),
        "dead_zone_reg_coeff": float(variant.get("dead_zone_reg_coeff", 0.0)),
        "dead_zone_reg_type": str(variant.get("dead_zone_reg_type", "l2")),
    }


def _ensure_model_config_for_reused_subrun(
    *,
    output_dir: str,
    subrun_dir: str,
    variant: Dict[str, Any],
    subrun: Dict[str, Any],
    config: LLMBenchmarkConfig,
) -> None:
    artifacts = train_artifact_paths(subrun_dir)
    if os.path.isfile(artifacts["model_config"]):
        return

    manifest_path = os.path.join(output_dir, "activations", "activations_manifest.json")
    manifest_cache_fingerprint: Optional[str] = None
    try:
        resolved_manifest_path = resolve_manifest_path(output_dir=output_dir)
        manifest = load_activation_manifest(manifest_path=resolved_manifest_path)
        manifest_path = resolved_manifest_path
        manifest_cache_fingerprint = manifest.get("cache_fingerprint")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        manifest_cache_fingerprint = None

    resample_latents_threshold = 0
    if os.path.isfile(artifacts["training_metrics"]):
        try:
            training_metrics = _read_json(artifacts["training_metrics"])
            if isinstance(training_metrics, dict):
                raw_threshold = training_metrics.get("resample_latents_threshold")
                if isinstance(raw_threshold, int) and not isinstance(
                    raw_threshold, bool
                ):
                    resample_latents_threshold = int(raw_threshold)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            resample_latents_threshold = 0

    persisted_subrun = {
        key: value for key, value in subrun.items() if key != "sweep_index"
    }
    model_config_payload = {
        "variant": dict(variant),
        "subrun": persisted_subrun,
        "training": config.training.model_dump(),
        "manifest_path": manifest_path,
        "manifest_cache_fingerprint": manifest_cache_fingerprint,
        "resample_latents_threshold": int(resample_latents_threshold),
        "loss_knobs": _resolve_effective_loss_knobs(variant=variant),
    }
    _write_json(artifacts["model_config"], model_config_payload)


def _classify_completed_subrun(
    completed_steps: Optional[int],
    new_train_steps: int,
    subrun_dir: str,
) -> Literal["continue", "reuse"]:
    if completed_steps is not None and new_train_steps > completed_steps:
        return "continue"
    if completed_steps is not None and new_train_steps < completed_steps:
        raise ValueError(
            f"Cannot reduce train_steps from "
            f"{completed_steps} to {new_train_steps} "
            f"for subrun {subrun_dir}. Increase "
            f"train_steps, or remove resume=true and set "
            f"overwrite=true to retrain from scratch."
        )
    return "reuse"


def _version_existing_config(output_dir: str) -> Optional[str]:
    # Not atomic across concurrent processes: two runs could compute
    # the same version number. Acceptable for single-process research
    # pipelines; use os.link + os.unlink to harden if needed.
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    max_version = 0
    for filename in os.listdir(output_dir):
        match = re.match(r"^config-v(\d+)-deprecated\.json$", filename)
        if match:
            max_version = max(max_version, int(match.group(1)))
    version = max_version + 1
    dest = os.path.join(output_dir, f"config-v{version}-deprecated.json")
    os.rename(config_path, dest)
    return dest


def _check_resume_no_prior_run(
    subrun_dir: str,
    pre_existing_subrun_dirs: Set[str],
) -> None:
    metrics_path = training_metrics_path(subrun_dir)
    metrics = _load_training_metrics_if_present(subrun_dir)
    if os.path.isfile(metrics_path) and metrics is None:
        raise ValueError(
            f"training_metrics.json in {subrun_dir} exists "
            f"but could not be parsed. The file may be corrupt."
        )
    if metrics is not None and metrics.get("status") == "completed":
        warnings.warn(
            f"resume=true: training_metrics.json in {subrun_dir} "
            f"shows status='completed' but model.pt is missing. "
            f"The subrun will be retrained from scratch.",
            stacklevel=2,
        )
    if metrics is None and subrun_dir in pre_existing_subrun_dirs:
        raise ValueError(
            f"resume=true but no prior training run exists "
            f"for subrun {subrun_dir}. Remove resume=true "
            f"to start a fresh run."
        )


def _check_no_overwrite_completed_run(subrun_dir: str) -> None:
    if _training_completed_successfully(subrun_dir):
        raise ValueError(
            f"A completed training run already exists in "
            f"{subrun_dir}. Set resume=true to continue/reuse, "
            f"or overwrite=true to retrain from scratch."
        )


def _run_train_stage(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    output_dir: str,
    config_path: str = "",
    pre_existing_subrun_dirs: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if pre_existing_subrun_dirs is None:
        pre_existing_subrun_dirs = set()
    if bool(config.resume) and not bool(config.overwrite):
        _assert_no_legacy_subruns(
            output_dir=output_dir,
            subruns=list(resolved_config.get("subruns", [])),
        )

    variants_by_slug: Dict[str, Dict[str, Any]] = {}
    for variant in resolved_config.get("variants", []):
        variant_slug = str(variant["variant_slug"])
        variants_by_slug[variant_slug] = dict(variant)

    grouped_subruns: Dict[Tuple[str, int, str], List[Tuple[Dict[str, Any], int]]] = {}
    sweep_indices_by_group: Dict[Tuple[str, int, str], int] = {}
    for subrun in resolved_config.get("subruns", []):
        variant_slug = str(subrun["variant_slug"])
        seed = int(subrun["seed"])
        sweep_param = str(subrun["sweep_param"])
        group_key = (variant_slug, seed, sweep_param)
        sweep_index = int(sweep_indices_by_group.get(group_key, 0))
        sweep_indices_by_group[group_key] = sweep_index + 1
        grouped_subruns.setdefault(group_key, []).append((dict(subrun), sweep_index))

    trained_subruns = 0
    reused_subruns = 0
    downstream_artifacts_invalidated = False

    def _invalidate_run_level_downstream_artifacts() -> None:
        nonlocal downstream_artifacts_invalidated
        if downstream_artifacts_invalidated:
            return
        _delete_run_level_downstream_artifacts(output_dir=output_dir)
        downstream_artifacts_invalidated = True

    for (variant_slug, seed, sweep_param), grouped_entries in grouped_subruns.items():
        if variant_slug not in variants_by_slug:
            raise ValueError(
                "Resolved config subrun references unknown variant_slug: "
                f"{variant_slug}."
            )
        variant = variants_by_slug[variant_slug]
        sparsity_coeff = variant.get("sparsity_coeff")
        topk_k = variant.get("topk_k")
        is_multi_sparsity_group = bool(config.training.multi_sparsity_enabled) and (
            (
                sweep_param == "lambda"
                and isinstance(sparsity_coeff, list)
                and len(sparsity_coeff) > 1
            )
            or (sweep_param == "k" and isinstance(topk_k, list) and len(topk_k) > 1)
        )

        if is_multi_sparsity_group:
            remaining_subruns: List[Dict[str, Any]] = []
            continuation_subruns: List[Dict[str, Any]] = []
            continuation_from_steps_value: Optional[int] = None
            for subrun, sweep_index in grouped_entries:
                subrun_dir = _output_subdir_path(
                    output_dir, str(subrun["relative_subrun_dir"])
                )
                if bool(config.resume) and not bool(config.overwrite):
                    if _training_completed_successfully(subrun_dir):
                        completed_steps = load_completed_train_steps(subrun_dir)
                        new_train_steps = int(config.training.train_steps)
                        action = _classify_completed_subrun(
                            completed_steps,
                            new_train_steps,
                            subrun_dir,
                        )
                        if action == "continue":
                            if (
                                continuation_from_steps_value is not None
                                and continuation_from_steps_value != completed_steps
                            ):
                                raise ValueError(
                                    "Mixed continuation points "
                                    "within sweep group "
                                    f"(variant_slug="
                                    f"{variant_slug}, "
                                    f"seed={seed}, "
                                    f"sweep_param="
                                    f"{sweep_param}): "
                                    "expected "
                                    f"{continuation_from_steps_value}"
                                    f", found {completed_steps} "
                                    f"in {subrun_dir}."
                                )
                            continuation_from_steps_value = completed_steps
                            subrun_with_index = dict(subrun)
                            subrun_with_index["sweep_index"] = int(sweep_index)
                            continuation_subruns.append(subrun_with_index)
                            continue
                        _ensure_model_config_for_reused_subrun(
                            output_dir=output_dir,
                            subrun_dir=subrun_dir,
                            variant=variant,
                            subrun=subrun,
                            config=config,
                        )
                        tqdm.write(
                            f"Skipping {subrun_dir}. Reason: "
                            f"completed at {completed_steps} steps "
                            f"(matches train_steps={new_train_steps})."
                        )
                        reused_subruns += 1
                        continue
                    else:
                        _check_resume_no_prior_run(
                            subrun_dir=subrun_dir,
                            pre_existing_subrun_dirs=pre_existing_subrun_dirs,
                        )
                elif not bool(config.resume) and not bool(config.overwrite):
                    _check_no_overwrite_completed_run(subrun_dir)
                subrun_with_index = dict(subrun)
                subrun_with_index["sweep_index"] = int(sweep_index)
                remaining_subruns.append(subrun_with_index)

            if remaining_subruns and continuation_subruns:
                raise ValueError(
                    "Cannot mix fresh and continuation-eligible "
                    "subruns within the same sweep group "
                    f"(variant_slug={variant_slug}, seed={seed}"
                    f", sweep_param={sweep_param})."
                )

            if not remaining_subruns and not continuation_subruns:
                continue

            subruns_to_dispatch = (
                continuation_subruns if continuation_subruns else remaining_subruns
            )
            dispatch_continuation = (
                continuation_from_steps_value if continuation_subruns else None
            )

            _invalidate_run_level_downstream_artifacts()
            max_instances = config.training.multi_sparsity_max_instances
            if max_instances is None:
                subrun_chunks = [subruns_to_dispatch]
            else:
                subrun_chunks = [
                    subruns_to_dispatch[start : start + int(max_instances)]
                    for start in range(
                        0,
                        len(subruns_to_dispatch),
                        int(max_instances),
                    )
                ]

            summary = {
                "config_path": config_path,
                "output_dir": config.output_dir,
                "device": config.device,
                "training": {
                    "train_steps": config.training.train_steps,
                    "batch_size": config.training.batch_size,
                    "learning_rate": config.training.learning_rate,
                },
            }
            tqdm.write(
                f"Training variant_slug={variant_slug}, "
                f"seed={seed}, sweep_param={sweep_param} "
                f"({len(subruns_to_dispatch)} subruns in "
                f"{len(subrun_chunks)} batch(es))\n"
                f"Config: {summary}"
            )
            for chunk_idx, chunk_subruns in enumerate(subrun_chunks):
                sweep_values = [s.get("sweep_value") for s in chunk_subruns]
                tqdm.write(
                    f"Starting batch {chunk_idx + 1}/"
                    f"{len(subrun_chunks)}. "
                    f"{sweep_param} values: {sweep_values}"
                )
                _ = run_training_for_lambda_chunk(
                    config=config,
                    resolved_config=resolved_config,
                    variant=variant,
                    seed=int(seed),
                    chunk_subruns=chunk_subruns,
                    output_dir=output_dir,
                    overwrite=bool(config.overwrite),
                    continuation_from_steps=dispatch_continuation,
                )
                trained_subruns += len(chunk_subruns)
            continue

        for subrun, _ in grouped_entries:
            subrun_dir = _output_subdir_path(
                output_dir, str(subrun["relative_subrun_dir"])
            )
            if bool(config.resume) and not bool(config.overwrite):
                if _training_completed_successfully(subrun_dir):
                    completed_steps = load_completed_train_steps(subrun_dir)
                    new_train_steps = int(config.training.train_steps)
                    action = _classify_completed_subrun(
                        completed_steps,
                        new_train_steps,
                        subrun_dir,
                    )
                    if action == "continue":
                        _invalidate_run_level_downstream_artifacts()
                        _ = run_training_for_subrun(
                            config=config,
                            resolved_config=resolved_config,
                            subrun=subrun,
                            output_dir=output_dir,
                            overwrite=False,
                            continuation_from_steps=completed_steps,
                        )
                        trained_subruns += 1
                        continue
                    _ensure_model_config_for_reused_subrun(
                        output_dir=output_dir,
                        subrun_dir=subrun_dir,
                        variant=variant,
                        subrun=subrun,
                        config=config,
                    )
                    tqdm.write(
                        f"Skipping {subrun_dir}. Reason: "
                        f"completed at {completed_steps} steps "
                        f"(matches train_steps={new_train_steps})."
                    )
                    reused_subruns += 1
                    continue
                else:
                    _check_resume_no_prior_run(
                        subrun_dir=subrun_dir,
                        pre_existing_subrun_dirs=pre_existing_subrun_dirs,
                    )
            elif not bool(config.resume) and not bool(config.overwrite):
                _check_no_overwrite_completed_run(subrun_dir)
            _invalidate_run_level_downstream_artifacts()
            _ = run_training_for_subrun(
                config=config,
                resolved_config=resolved_config,
                subrun=subrun,
                output_dir=output_dir,
                overwrite=bool(config.overwrite),
            )
            trained_subruns += 1

    return {
        "trained_subruns": trained_subruns,
        "reused_subruns": reused_subruns,
        "message": (
            f"Train stage completed. trained_subruns={trained_subruns}, "
            f"reused_subruns={reused_subruns}."
        ),
    }


def _run_eval_stage(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    if bool(config.loss_recovered.enabled) and config.modality != "text":
        raise ValueError(
            "loss_recovered.enabled=true is only supported for modality='text'."
        )
    if (
        bool(config.feature_ablation.enabled)
        and config.feature_ablation.metric == "delta_lm_loss"
        and config.modality != "text"
    ):
        raise ValueError(
            "feature_ablation.metric='delta_lm_loss' is only supported for "
            "modality='text'."
        )
    _assert_no_legacy_subruns(
        output_dir=output_dir,
        subruns=list(resolved_config.get("subruns", [])),
    )
    if bool(config.loss_recovered.enabled):
        _ = ensure_run_level_loss_recovered_artifacts(
            config=config,
            output_dir=output_dir,
            overwrite=bool(config.overwrite),
        )

    evaluated_subruns = 0
    reused_subruns = 0
    skipped_failed_subruns = 0
    downstream_artifacts_invalidated = False

    def _invalidate_run_level_downstream_artifacts() -> None:
        nonlocal downstream_artifacts_invalidated
        if downstream_artifacts_invalidated:
            return
        _delete_run_level_downstream_artifacts(output_dir=output_dir)
        downstream_artifacts_invalidated = True

    subruns = list(resolved_config.get("subruns", []))
    for subrun in tqdm(subruns, desc="Evaluating subruns", unit="subrun"):
        subrun_dir = _output_subdir_path(output_dir, str(subrun["relative_subrun_dir"]))
        state = subrun_training_state(subrun_dir)
        if state != "completed":
            _invalidate_run_level_downstream_artifacts()
            artifacts = eval_artifact_paths(subrun_dir, config=config)
            for path in artifacts.values():
                if os.path.isfile(path):
                    os.remove(path)
            skipped_failed_subruns += 1
            tqdm.write(
                f"Skipping {subrun_dir}. Reason: " f"training state is '{state}'."
            )
            continue
        artifacts = eval_artifact_paths(subrun_dir, config=config)
        if (
            bool(config.resume)
            and not bool(config.overwrite)
            and _are_artifacts_present(artifacts)
        ):
            train_artifacts = train_artifact_paths(subrun_dir)
            if not (
                os.path.isfile(train_artifacts["model"])
                and os.path.isfile(train_artifacts["model_config"])
            ):
                raise ValueError(
                    "Cannot reuse evaluation artifacts without a trained checkpoint. "
                    "Missing model.pt or model_config.json in subrun_dir="
                    f"{subrun_dir}."
                )
            reused_subruns += 1
            tqdm.write(
                f"Skipping {subrun_dir}. Reason: "
                f"evaluation artifacts already present."
            )
            continue
        _invalidate_run_level_downstream_artifacts()
        _ = run_evaluation_for_subrun(
            config=config,
            resolved_config=resolved_config,
            subrun=subrun,
            output_dir=output_dir,
            overwrite=bool(config.overwrite),
        )
        evaluated_subruns += 1

    comparison_summary = build_and_persist_comparison_summary(
        resolved_config=resolved_config,
        output_dir=output_dir,
    )
    return {
        "evaluated_subruns": evaluated_subruns,
        "reused_subruns": reused_subruns,
        "skipped_failed_subruns": skipped_failed_subruns,
        "comparison_subruns": len(comparison_summary.get("subruns", [])),
        "message": (
            f"Eval stage completed. evaluated_subruns={evaluated_subruns}, "
            f"reused_subruns={reused_subruns}, "
            f"skipped_failed_subruns={skipped_failed_subruns}."
        ),
    }


def _run_examples_stage(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    if not bool(config.latent_examples.enabled):
        return {
            "generated_subruns": 0,
            "reused_subruns": 0,
            "selected_subruns": 0,
            "eligible_subruns": 0,
            "skipped_failed_subruns": 0,
            "index_path": None,
            "message": (
                "Examples stage skipped because latent_examples.enabled=false."
            ),
        }

    selected_subruns = select_target_subruns_for_examples(
        config=config,
        resolved_config=resolved_config,
        output_dir=output_dir,
    )
    _assert_no_legacy_subruns(
        output_dir=output_dir,
        subruns=selected_subruns,
    )
    eligible_subruns: List[Dict[str, Any]] = []
    skipped_failed_subruns = 0
    for subrun in selected_subruns:
        subrun_dir = _output_subdir_path(output_dir, str(subrun["relative_subrun_dir"]))
        state = subrun_training_state(subrun_dir)
        if state == "completed":
            eligible_subruns.append(subrun)
            continue
        latent_examples_path = os.path.join(subrun_dir, "latent_examples.json")
        assets_dir = os.path.join(subrun_dir, "latent_examples_assets")
        if os.path.isfile(latent_examples_path):
            os.remove(latent_examples_path)
        if os.path.isdir(assets_dir):
            shutil.rmtree(assets_dir)
        skipped_failed_subruns += 1
    if str(config.modality) == "text" and eligible_subruns:
        validate_examples_cache_metadata(config=config, output_dir=output_dir)
    generated_subruns = 0
    reused_subruns = 0
    index_entries: List[Dict[str, Any]] = []
    for subrun in eligible_subruns:
        result = generate_latent_examples(
            config=config,
            resolved_config=resolved_config,
            subrun=subrun,
            output_dir=output_dir,
            overwrite=bool(config.overwrite),
        )
        relative_subrun_dir = str(subrun["relative_subrun_dir"]).replace("\\", "/")
        relative_examples_path = (
            os.path.join(relative_subrun_dir, "latent_examples.json")
            .replace("\\", "/")
            .strip("/")
        )
        index_entries.append(
            {
                "variant_slug": str(subrun["variant_slug"]),
                "seed": int(subrun["seed"]),
                "sweep_param": str(subrun["sweep_param"]),
                "sweep_value": subrun["sweep_value"],
                "latent_examples_path": relative_examples_path,
            }
        )
        if bool(result.get("reused", False)):
            reused_subruns += 1
        else:
            generated_subruns += 1

    index_result = build_and_persist_latent_examples_index(
        output_dir=output_dir,
        subrun_entries=index_entries,
    )
    paired_case_study_path = None
    paired_case_study_count = 0
    if str(config.modality) == "text":
        paired_case_study_result = build_and_persist_paired_duplicate_case_studies(
            config=config,
            resolved_config=resolved_config,
            selected_subruns=eligible_subruns,
            output_dir=output_dir,
        )
        paired_case_study_path = paired_case_study_result.get("case_studies_path")
        paired_case_study_count = int(
            paired_case_study_result.get("num_case_studies", 0)
        )
    return {
        "generated_subruns": generated_subruns,
        "reused_subruns": reused_subruns,
        "selected_subruns": len(selected_subruns),
        "eligible_subruns": len(eligible_subruns),
        "skipped_failed_subruns": skipped_failed_subruns,
        "index_path": index_result.get("index_path"),
        "paired_duplicate_case_studies_path": paired_case_study_path,
        "paired_duplicate_case_studies_count": paired_case_study_count,
        "message": (
            "Examples stage completed. "
            f"generated_subruns={generated_subruns}, "
            f"reused_subruns={reused_subruns}, "
            f"selected_subruns={len(selected_subruns)}, "
            f"eligible_subruns={len(eligible_subruns)}, "
            f"skipped_failed_subruns={skipped_failed_subruns}."
        ),
    }


def _delete_plot_stage_artifacts_if_present(paths: Dict[str, str]) -> None:
    for path in paths.values():
        if os.path.isfile(path):
            os.remove(path)


def _delete_plot_files_if_present(paths: Dict[str, str]) -> None:
    _delete_plot_stage_artifacts_if_present(paths)


def _run_plot_stage(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    _assert_no_legacy_subruns(
        output_dir=output_dir,
        subruns=list(resolved_config.get("subruns", [])),
    )
    comparison_summary = build_and_persist_comparison_summary(
        resolved_config=resolved_config,
        output_dir=output_dir,
    )
    _ = ensure_summary_has_matched_l0_anchors(
        output_dir=output_dir,
        summary=comparison_summary,
    )
    plot_paths = plot_stage_artifact_paths(output_dir=output_dir)
    if bool(config.overwrite):
        _delete_plot_stage_artifacts_if_present(plot_paths)
    if (
        bool(config.resume)
        and not bool(config.overwrite)
        and _are_artifacts_present(plot_paths)
    ):
        return {
            "reused": True,
            "plots_written": 0,
            "message": "Plot stage reused existing plot artifacts.",
        }
    plot_result = generate_plots_for_run(output_dir=output_dir)
    return {
        "reused": False,
        "plots_written": int(plot_result.get("plots_written", 0)),
        "message": "Plot stage completed.",
    }


def _select_best_metric_entry(
    aggregates: List[Dict[str, Any]],
    metric_name: str,
    maximize: bool,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in aggregates:
        variant_slug = entry.get("variant_slug")
        if not isinstance(variant_slug, str):
            continue
        metric_value = entry.get(metric_name)
        if metric_value is None:
            continue
        grouped.setdefault(variant_slug, []).append(entry)

    output: List[Dict[str, Any]] = []
    for variant_slug, entries in grouped.items():
        if maximize:
            best = max(entries, key=lambda item: float(item[metric_name]))
        else:
            best = min(entries, key=lambda item: float(item[metric_name]))
        output.append(
            {
                "variant_slug": variant_slug,
                "sweep_param": best.get("sweep_param"),
                "sweep_value": best.get("sweep_value"),
                metric_name: best.get(metric_name),
                "num_seeds": best.get("num_seeds"),
            }
        )
    output.sort(key=lambda item: str(item["variant_slug"]))
    return output


def _run_summarize_stage(
    config: LLMBenchmarkConfig,
    resolved_config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    del config
    comparison_summary = build_and_persist_comparison_summary(
        resolved_config=resolved_config,
        output_dir=output_dir,
    )
    comparison_summary = ensure_summary_has_matched_l0_anchors(
        output_dir=output_dir,
        summary=comparison_summary,
    )
    matched_points_payload = compute_and_persist_matched_points(
        summary=comparison_summary,
        output_dir=output_dir,
    )
    aggregates = comparison_summary.get("aggregates", [])
    if not isinstance(aggregates, list):
        aggregates = []
    matched_l0 = comparison_summary.get("matched_l0_anchors")
    results_summary = {
        "schema_version": 1,
        "metadata": comparison_summary.get("metadata", {}),
        "best_mse_by_variant": _select_best_metric_entry(
            aggregates=aggregates,
            metric_name="mse_mean",
            maximize=False,
        ),
        "best_loss_recovered_by_variant": _select_best_metric_entry(
            aggregates=aggregates,
            metric_name="loss_recovered_mean",
            maximize=True,
        ),
        "matched_l0_anchors": matched_l0,
        "matched_points": matched_points_payload,
        "matched_points_path": matched_points_path(output_dir=output_dir),
        "num_subruns": len(comparison_summary.get("subruns", [])),
    }
    results_summary_path = os.path.join(output_dir, "results_summary.json")
    _write_json(results_summary_path, results_summary)
    return {
        "results_summary_path": results_summary_path,
        "message": "Summarize stage completed.",
    }


def run_llm_sae_benchmark(
    config_path: str,
    stages_override: Optional[List[str]] = None,
    validate_only: bool = False,
) -> str:
    resolve_started_at = _utc_now_iso()
    config = load_benchmark_config(config_path)
    if bool(config.resume) and bool(config.overwrite):
        raise ValueError(
            "resume=true and overwrite=true are mutually exclusive. "
            "Set one or the other, not both."
        )
    requested_stages = _effective_stages(
        config=config,
        stages_override=stages_override,
        validate_only=validate_only,
    )
    if "validate" in requested_stages and len(requested_stages) != 1:
        raise ValueError(
            "If 'validate' is present in stages, it must be the only stage."
        )

    resolved_config = resolve_benchmark_config(
        config=config,
        stages_override=requested_stages,
        validate_only=False,
    )
    resolve_ended_at = _utc_now_iso()
    output_dir = str(resolved_config["output_dir"])
    config_fingerprint = str(resolved_config["config_fingerprint"])
    rng_fingerprint = str(resolved_config.get("rng_fingerprint", ""))

    existing_fingerprint = _load_existing_fingerprint(output_dir)
    existing_rng_fingerprint = _load_existing_rng_fingerprint(output_dir)

    if existing_fingerprint is None:
        resume_check: Dict[str, Any] = {"status": "not_applicable"}
    else:
        resume_check = {
            "status": "passed",
            "fingerprints_match": existing_fingerprint == config_fingerprint,
            "existing_fingerprint": existing_fingerprint,
            "new_fingerprint": config_fingerprint,
            "existing_rng_fingerprint": existing_rng_fingerprint,
            "new_rng_fingerprint": (
                rng_fingerprint if rng_fingerprint.strip() else None
            ),
        }

    os.makedirs(output_dir, exist_ok=True)

    if bool(config.resume):
        _version_existing_config(output_dir)
    _write_json(os.path.join(output_dir, "config.json"), resolved_config)
    persist_environment_metadata(output_dir)

    if requested_stages == ["validate"]:
        stage_statuses = _build_stage_statuses_for_validate_only()
        _persist_run_status(
            output_dir=output_dir,
            config_fingerprint=config_fingerprint,
            requested_stages=["validate"],
            resume_check=resume_check,
            stage_statuses=stage_statuses,
            resolve_timestamps=(resolve_started_at, resolve_ended_at),
        )
        return output_dir

    stages_requiring_variants = {"train", "eval", "examples", "plot", "summarize"}
    if not config.variants and stages_requiring_variants & set(requested_stages):
        raise ValueError(
            "At least one variant must be provided for stages: "
            f"{sorted(stages_requiring_variants & set(requested_stages))}."
        )

    pre_existing_subrun_dirs: Set[str] = set()
    for subrun in resolved_config.get("subruns", []):
        relative_dir = subrun.get("relative_subrun_dir")
        if not isinstance(relative_dir, str) or not relative_dir:
            # The makedirs loop immediately after will raise.
            continue
        full_path = _output_subdir_path(output_dir, relative_dir)
        if os.path.isdir(full_path):
            pre_existing_subrun_dirs.add(full_path)

    for subrun in resolved_config.get("subruns", []):
        relative_dir = subrun.get("relative_subrun_dir")
        if not isinstance(relative_dir, str) or not relative_dir:
            raise ValueError("Resolved config contains an invalid relative_subrun_dir.")
        os.makedirs(_output_subdir_path(output_dir, relative_dir), exist_ok=True)

    stage_statuses = _build_stage_statuses_for_execution(requested_stages)
    stage_to_index = {
        stage_name: index for index, stage_name in enumerate(requested_stages)
    }

    def _mark_blocked_stages(stage_name: str) -> None:
        stage_index = stage_to_index[stage_name]
        for blocked_stage in requested_stages[stage_index + 1 :]:
            stage_statuses[blocked_stage] = {
                "status": "pending",
                "message": "Not run (blocked by earlier failure).",
            }

    for stage_name in requested_stages:
        if stage_name == "cache":
            try:
                cache_result = run_cache_stage(
                    config=config,
                    output_dir=output_dir,
                    runner_overwrite=bool(config.overwrite),
                )
            except Exception as exc:
                stage_statuses[stage_name] = {
                    "status": "failed",
                    "message": str(exc),
                }
                _mark_blocked_stages(stage_name)
                _persist_run_status(
                    output_dir=output_dir,
                    config_fingerprint=config_fingerprint,
                    requested_stages=requested_stages,
                    resume_check=resume_check,
                    stage_statuses=stage_statuses,
                    resolve_timestamps=(resolve_started_at, resolve_ended_at),
                )
                raise

            stage_statuses[stage_name] = {
                "status": "done",
                "message": str(cache_result.get("message", "Cache stage completed.")),
            }
            continue

        if stage_name == "train":
            try:
                train_result = _run_train_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=output_dir,
                    config_path=config_path,
                    pre_existing_subrun_dirs=pre_existing_subrun_dirs,
                )
            except Exception as exc:
                stage_statuses[stage_name] = {
                    "status": "failed",
                    "message": str(exc),
                }
                _mark_blocked_stages(stage_name)
                _persist_run_status(
                    output_dir=output_dir,
                    config_fingerprint=config_fingerprint,
                    requested_stages=requested_stages,
                    resume_check=resume_check,
                    stage_statuses=stage_statuses,
                    resolve_timestamps=(resolve_started_at, resolve_ended_at),
                )
                raise

            stage_statuses[stage_name] = {
                "status": "done",
                "message": str(train_result.get("message", "Train stage completed.")),
                "trained_subruns": int(train_result.get("trained_subruns", 0)),
                "reused_subruns": int(train_result.get("reused_subruns", 0)),
            }
            continue

        if stage_name == "eval":
            try:
                eval_result = _run_eval_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=output_dir,
                )
            except Exception as exc:
                stage_statuses[stage_name] = {
                    "status": "failed",
                    "message": str(exc),
                }
                _mark_blocked_stages(stage_name)
                _persist_run_status(
                    output_dir=output_dir,
                    config_fingerprint=config_fingerprint,
                    requested_stages=requested_stages,
                    resume_check=resume_check,
                    stage_statuses=stage_statuses,
                    resolve_timestamps=(resolve_started_at, resolve_ended_at),
                )
                raise

            stage_statuses[stage_name] = {
                "status": "done",
                "message": str(eval_result.get("message", "Eval stage completed.")),
                "evaluated_subruns": int(eval_result.get("evaluated_subruns", 0)),
                "reused_subruns": int(eval_result.get("reused_subruns", 0)),
                "skipped_failed_subruns": int(
                    eval_result.get("skipped_failed_subruns", 0)
                ),
                "comparison_subruns": int(eval_result.get("comparison_subruns", 0)),
            }
            continue

        if stage_name == "examples":
            try:
                examples_result = _run_examples_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=output_dir,
                )
            except Exception as exc:
                stage_statuses[stage_name] = {
                    "status": "failed",
                    "message": str(exc),
                }
                _mark_blocked_stages(stage_name)
                _persist_run_status(
                    output_dir=output_dir,
                    config_fingerprint=config_fingerprint,
                    requested_stages=requested_stages,
                    resume_check=resume_check,
                    stage_statuses=stage_statuses,
                    resolve_timestamps=(resolve_started_at, resolve_ended_at),
                )
                raise

            stage_statuses[stage_name] = {
                "status": "done",
                "message": str(
                    examples_result.get("message", "Examples stage completed.")
                ),
                "generated_subruns": int(examples_result.get("generated_subruns", 0)),
                "reused_subruns": int(examples_result.get("reused_subruns", 0)),
                "selected_subruns": int(examples_result.get("selected_subruns", 0)),
                "eligible_subruns": int(examples_result.get("eligible_subruns", 0)),
                "skipped_failed_subruns": int(
                    examples_result.get("skipped_failed_subruns", 0)
                ),
                "index_path": examples_result.get("index_path"),
                "paired_duplicate_case_studies_path": examples_result.get(
                    "paired_duplicate_case_studies_path"
                ),
                "paired_duplicate_case_studies_count": int(
                    examples_result.get("paired_duplicate_case_studies_count", 0)
                ),
            }
            continue

        if stage_name == "plot":
            try:
                plot_result = _run_plot_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=output_dir,
                )
            except Exception as exc:
                stage_statuses[stage_name] = {
                    "status": "failed",
                    "message": str(exc),
                }
                _mark_blocked_stages(stage_name)
                _persist_run_status(
                    output_dir=output_dir,
                    config_fingerprint=config_fingerprint,
                    requested_stages=requested_stages,
                    resume_check=resume_check,
                    stage_statuses=stage_statuses,
                    resolve_timestamps=(resolve_started_at, resolve_ended_at),
                )
                raise

            stage_statuses[stage_name] = {
                "status": "done",
                "message": str(plot_result.get("message", "Plot stage completed.")),
                "reused": bool(plot_result.get("reused", False)),
                "plots_written": int(plot_result.get("plots_written", 0)),
            }
            continue

        if stage_name == "summarize":
            try:
                summarize_result = _run_summarize_stage(
                    config=config,
                    resolved_config=resolved_config,
                    output_dir=output_dir,
                )
            except Exception as exc:
                stage_statuses[stage_name] = {
                    "status": "failed",
                    "message": str(exc),
                }
                _mark_blocked_stages(stage_name)
                _persist_run_status(
                    output_dir=output_dir,
                    config_fingerprint=config_fingerprint,
                    requested_stages=requested_stages,
                    resume_check=resume_check,
                    stage_statuses=stage_statuses,
                    resolve_timestamps=(resolve_started_at, resolve_ended_at),
                )
                raise

            stage_statuses[stage_name] = {
                "status": "done",
                "message": str(
                    summarize_result.get("message", "Summarize stage completed.")
                ),
                "results_summary_path": summarize_result.get("results_summary_path"),
            }
            continue

        stage_statuses[stage_name] = {
            "status": "not_implemented",
            "message": "Stage not implemented yet.",
        }
        _mark_blocked_stages(stage_name)
        _persist_run_status(
            output_dir=output_dir,
            config_fingerprint=config_fingerprint,
            requested_stages=requested_stages,
            resume_check=resume_check,
            stage_statuses=stage_statuses,
            resolve_timestamps=(resolve_started_at, resolve_ended_at),
        )
        raise NotImplementedError(
            "Stage "
            f"'{stage_name}' is not implemented yet. Implemented stages: "
            "['validate', 'cache', 'train', 'eval', "
            "'examples', 'plot', 'summarize']."
        )

    _persist_run_status(
        output_dir=output_dir,
        config_fingerprint=config_fingerprint,
        requested_stages=requested_stages,
        resume_check=resume_check,
        stage_statuses=stage_statuses,
        resolve_timestamps=(resolve_started_at, resolve_ended_at),
    )
    return output_dir
