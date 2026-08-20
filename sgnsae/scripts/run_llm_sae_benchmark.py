import argparse
import importlib
import json
import sys
from types import ModuleType
from typing import List, Optional

INSTALL_HINT = ".venv/bin/pip install -r requirements.txt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        nargs="?",
        type=str,
        help="Path to LLM SAE benchmark configuration (JSON or YAML).",
    )
    parser.add_argument(
        "--config-path",
        dest="config_path_flag",
        default=None,
        type=str,
        help="Path to LLM SAE benchmark configuration (JSON or YAML).",
    )
    parser.add_argument(
        "--stages",
        default=None,
        type=str,
        help="Comma-separated stage list (validate,cache,train,eval,examples,plot,summarize).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config, write run-level artifacts, and exit successfully.",
    )
    return parser.parse_args()


def _import_or_raise(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency '{module_name}' required for the LLM SAE benchmark. "
            f"Install dependencies with: {INSTALL_HINT}"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import '{module_name}' due to an ImportError. "
            f"Install dependencies with: {INSTALL_HINT}"
        ) from exc


def _parse_stage_list(stages_csv: str) -> List[str]:
    entries = [entry.strip() for entry in stages_csv.split(",")]
    entries = [entry for entry in entries if entry]
    if not entries:
        raise ValueError("--stages must contain at least one stage.")
    return entries


def _resolve_config_path(positional: Optional[str], flag: Optional[str]) -> str:
    if positional and flag:
        raise ValueError(
            "Provide config path either positionally or via --config-path, not both."
        )
    path = flag or positional
    if path is None:
        raise ValueError(
            "Missing config path. Provide it positionally or via --config-path."
        )
    return str(path)


def main() -> None:
    args = parse_arguments()
    try:
        config_path = _resolve_config_path(args.config_path, args.config_path_flag)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None

    try:
        _import_or_raise("torch")
        _import_or_raise("transformers")
        _import_or_raise("pydantic")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None

    stages_override = None
    if args.validate_only and args.stages is not None:
        print("--validate-only and --stages cannot both be provided.", file=sys.stderr)
        raise SystemExit(2)
    if args.stages is not None:
        try:
            stages_override = _parse_stage_list(args.stages)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from None

    from sgnsae.experiments.llm_benchmark import (
        load_benchmark_config,
        resolve_benchmark_config,
        run_llm_sae_benchmark,
    )

    config = load_benchmark_config(config_path)
    resolved_preview = resolve_benchmark_config(
        config=config,
        stages_override=stages_override,
        validate_only=args.validate_only,
    )
    print("Loaded LLM SAE benchmark config:")
    json.dump(resolved_preview, sys.stdout, indent=2, default=str)
    print("\nStarting run...")

    try:
        output_dir = run_llm_sae_benchmark(
            config_path=config_path,
            stages_override=stages_override,
            validate_only=args.validate_only,
        )
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Run complete. Artifacts written to: {output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
