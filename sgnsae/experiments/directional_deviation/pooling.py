from typing import Literal, Optional

import torch
from torch import Tensor


PoolingMode = Literal["sum", "max", "topk"]


def pool_exceedances(
    exceedances: Tensor,
    pooling: PoolingMode,
    topk_k: Optional[int] = None,
) -> Tensor:
    if exceedances.ndim != 2:
        raise ValueError(
            f"Expected exceedances.ndim == 2, got exceedances.ndim={exceedances.ndim}."
        )
    if exceedances.shape[1] == 0:
        return torch.zeros(
            exceedances.shape[0],
            dtype=exceedances.dtype,
            device=exceedances.device,
        )

    if pooling == "sum":
        return exceedances.sum(dim=1)
    if pooling == "max":
        return exceedances.max(dim=1).values
    if pooling == "topk":
        if topk_k is None:
            raise ValueError("topk_k must be provided when pooling='topk'.")
        if topk_k <= 0:
            raise ValueError(f"topk_k must be positive; received {topk_k}.")
        k_effective = min(int(topk_k), int(exceedances.shape[1]))
        return exceedances.topk(k_effective, dim=1).values.sum(dim=1)

    raise ValueError(f"Unsupported pooling mode: {pooling}.")
