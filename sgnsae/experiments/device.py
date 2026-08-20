import re
from typing import Any, Optional

import torch


_CUDA_DEVICE_PATTERN = re.compile(r"^cuda:(\d+)$")


def _extract_cuda_index(device: str) -> Optional[int]:
    match = _CUDA_DEVICE_PATTERN.fullmatch(device)
    if match is None:
        return None
    return int(match.group(1))


def validate_experiment_device(device: Any) -> str:
    if not isinstance(device, str):
        raise ValueError(f"device must be a string; received {device}.")
    if device in {"cpu", "cuda", "auto"}:
        return device
    if _extract_cuda_index(device) is not None:
        return device
    raise ValueError(
        "device must be one of 'cpu', 'cuda', 'auto', or 'cuda:<index>'; "
        f"received {device}."
    )


def resolve_experiment_device(requested_device: str) -> str:
    device = validate_experiment_device(requested_device)
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise ValueError(f"device '{device}' requires CUDA, but CUDA is not available.")
    if device == "cuda":
        return "cuda"
    cuda_index = _extract_cuda_index(device)
    assert cuda_index is not None
    device_count = int(torch.cuda.device_count())
    if cuda_index >= device_count:
        raise ValueError(
            f"device '{device}' requested CUDA index {cuda_index}, but only "
            f"{device_count} CUDA device(s) are available."
        )
    return device
