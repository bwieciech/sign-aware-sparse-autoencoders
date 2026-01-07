import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional

import torch


def _package_version(package_name: str) -> Optional[str]:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def collect_environment_metadata() -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = str(torch.cuda.get_device_name(0))
        except Exception:
            gpu_name = None

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "gpu_name": gpu_name,
        },
        "dependencies": {
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "matplotlib": _package_version("matplotlib"),
            "einops": _package_version("einops"),
        },
    }


def persist_environment_metadata(run_dir: str) -> str:
    environment_path = os.path.join(run_dir, "environment.json")
    metadata = collect_environment_metadata()
    with open(environment_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return environment_path
