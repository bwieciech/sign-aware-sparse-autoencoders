import argparse
import json
import sys
from dataclasses import asdict

from sgnsae.experiments.polarity_dial import (
    load_experiment_config,
    run_polarity_dial_experiment,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
        help="Path to Protocol A experiment configuration (JSON or YAML).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_experiment_config(args.config_path)
    print("Loaded Polarity Dial config:")
    json.dump(asdict(config), sys.stdout, indent=2, default=str)
    print("\nStarting run...")
    run_dir = run_polarity_dial_experiment(args.config_path)
    print(f"Run complete. Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
