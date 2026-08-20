import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Tuple

import torch

from sgnsae.experiments.llm_benchmark.schema_versions import ENVIRONMENT_SCHEMA_VERSION


_PIP_FREEZE_CACHE: Optional[Tuple[Optional[List[str]], Optional[str]]] = None


def _package_version(package_name: str) -> Optional[str]:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _collect_pip_freeze() -> Tuple[Optional[List[str]], Optional[str]]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return None, f"pip freeze failed: {exc}"

    if result.returncode != 0:
        return None, f"pip freeze exited with code {result.returncode}: {result.stderr}"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines, None


def _collect_pip_freeze_cached() -> Tuple[Optional[List[str]], Optional[str]]:
    global _PIP_FREEZE_CACHE
    if _PIP_FREEZE_CACHE is not None:
        return _PIP_FREEZE_CACHE
    _PIP_FREEZE_CACHE = _collect_pip_freeze()
    return _PIP_FREEZE_CACHE


def clear_pip_freeze_cache() -> None:
    global _PIP_FREEZE_CACHE
    _PIP_FREEZE_CACHE = None


def collect_environment_metadata() -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = str(torch.cuda.get_device_name(0))
        except Exception:
            gpu_name = None

    pip_freeze, pip_freeze_error = _collect_pip_freeze_cached()

    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
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
            "transformers": _package_version("transformers"),
            "pydantic": _package_version("pydantic"),
        },
        "pip_freeze": pip_freeze,
        "pip_freeze_error": pip_freeze_error,
    }


def persist_environment_metadata(output_dir: str) -> str:
    environment_path = os.path.join(output_dir, "environment.json")
    metadata = collect_environment_metadata()
    with open(environment_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return environment_path
