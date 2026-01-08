import argparse
import importlib
import json
import sys
from dataclasses import asdict
from types import ModuleType

INSTALL_HINT = ".venv/bin/pip install -r requirements.txt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
        help="Path to Protocol C experiment configuration (JSON or YAML).",
    )
    return parser.parse_args()


def _import_or_raise(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Missing dependency '{module_name}' required for the Toy Model Geometry "
            f"Validation experiment. Install dependencies with: {INSTALL_HINT}"
        ) from exc


def validate_toy_model_geometry_validation_dependencies() -> None:
    _import_or_raise("torch")
    _import_or_raise("numpy")
    matplotlib = _import_or_raise("matplotlib")
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    _import_or_raise("matplotlib.pyplot")
    _import_or_raise("scipy")

    einops = _import_or_raise("einops")
    if not hasattr(einops, "einsum"):
        raise RuntimeError(
            "Toy Model Geometry Validation requires 'einops.einsum', but it is "
            "unavailable in this einops version. Install/upgrade dependencies with: "
            f"{INSTALL_HINT}"
        )


def main() -> None:
    args = parse_arguments()
    try:
        validate_toy_model_geometry_validation_dependencies()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None

    from sgnsae.experiments.toy_model_geometry_validation import (
        load_experiment_config,
        run_toy_model_geometry_validation_experiment,
    )

    config = load_experiment_config(args.config_path)
    print("Loaded Toy Model Geometry Validation config:")
    json.dump(asdict(config), sys.stdout, indent=2, default=str)
    print("\nStarting run...")
    run_dir = run_toy_model_geometry_validation_experiment(args.config_path)
    print(f"Run complete. Artifacts written to: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
