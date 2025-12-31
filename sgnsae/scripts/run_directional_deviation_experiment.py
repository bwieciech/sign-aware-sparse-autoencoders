import argparse
import json
import sys
from dataclasses import asdict

from sgnsae.experiments.directional_deviation import (
    load_experiment_config,
    run_directional_deviation_experiment,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
        help="Path to Protocol B experiment configuration (JSON or YAML).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_experiment_config(args.config_path)
    print("Loaded Directional Deviation config:")
    json.dump(asdict(config), sys.stdout, indent=2, default=str)
    print("\nStarting run...")
    run_dir = run_directional_deviation_experiment(args.config_path)
    print(f"Run complete. Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()

