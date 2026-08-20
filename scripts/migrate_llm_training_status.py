import argparse
import json
import os
import sys
from typing import Any, Dict


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from sgnsae.experiments.llm_benchmark.subrun_state import (
    is_legacy_success_training_metrics_payload,
)


def migrate_training_metrics_file(path: str) -> bool:
    try:
        with open(path, "r") as file:
            payload = json.load(file)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if not is_legacy_success_training_metrics_payload(payload):
        return False

    payload["status"] = "completed"
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)
    return True


def walk(root: str) -> int:
    updated = 0
    for current_root, _, files in os.walk(root):
        if "training_metrics.json" not in files:
            continue
        path = os.path.join(current_root, "training_metrics.json")
        if migrate_training_metrics_file(path):
            updated += 1
    return updated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    arguments = _parse_args()
    print(walk(arguments.root))
