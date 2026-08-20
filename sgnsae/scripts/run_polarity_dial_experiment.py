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
        help="Path to Protocol A experiment configuration (JSON or YAML).",
    )
    return parser.parse_args()


def _import_or_raise(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Missing dependency '{module_name}' required for the Polarity Dial "
            f"experiment. Install dependencies with: {INSTALL_HINT}"
        ) from exc


def validate_polarity_dial_dependencies() -> None:
    _import_or_raise("torch")
    _import_or_raise("numpy")
    matplotlib = _import_or_raise("matplotlib")
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    _import_or_raise("matplotlib.pyplot")

    _import_or_raise("scipy")
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Polarity Dial requires 'scipy.optimize.linear_sum_assignment'. Install "
            f"dependencies with: {INSTALL_HINT}"
        ) from exc

    einops = _import_or_raise("einops")
    if not hasattr(einops, "einsum"):
        raise RuntimeError(
            "Polarity Dial requires 'einops.einsum', but it is unavailable in this "
            f"einops version. Install/upgrade dependencies with: {INSTALL_HINT}"
        )


def main() -> None:
    args = parse_arguments()
    try:
        validate_polarity_dial_dependencies()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None

    from sgnsae.experiments.polarity_dial import (
        load_experiment_config,
        run_polarity_dial_experiment,
    )

    config = load_experiment_config(args.config_path)
    print("Loaded Polarity Dial config:")
    json.dump(asdict(config), sys.stdout, indent=2, default=str)
    print("\nStarting run...")
    run_output = run_polarity_dial_experiment(args.config_path)
    if isinstance(run_output, list):
        for run_dir in run_output:
            print(f"Run complete. Artifacts written to: {run_dir}")
    else:
        print(f"Run complete. Artifacts written to: {run_output}")


if __name__ == "__main__":  # pragma: no cover
    main()
