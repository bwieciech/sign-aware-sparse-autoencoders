from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor


def validate_and_prepare_topk_k(
    topk_k: Union[int, List[int]],
    n_instances: int,
    d_hidden: int,
    device: str,
) -> Tuple[int, Optional[List[int]], Optional[Tensor]]:
    if isinstance(topk_k, list):
        if not topk_k:
            raise ValueError("topk_k list must be non-empty.")
        if len(topk_k) != n_instances:
            raise ValueError(
                f"topk_k list length must match n_instances; "
                f"received len(topk_k)={len(topk_k)}, n_instances={n_instances}."
            )
        for k in topk_k:
            if not isinstance(k, int) or isinstance(k, bool):
                raise ValueError(
                    f"Each topk_k value must be an int; "
                    f"received {k!r} (type {type(k).__name__})."
                )
            if k <= 0 or k > d_hidden:
                raise ValueError(
                    f"Each topk_k value must satisfy 0 < k <= d_hidden; "
                    f"received k={k}, d_hidden={d_hidden}."
                )
        return (
            topk_k[0],
            list(topk_k),
            torch.tensor(topk_k, dtype=torch.int64, device=device),
        )
    if not isinstance(topk_k, int) or isinstance(topk_k, bool):
        raise ValueError(
            f"topk_k must be an int or List[int]; "
            f"received {topk_k!r} (type {type(topk_k).__name__})."
        )
    if topk_k <= 0 or topk_k > d_hidden:
        raise ValueError(
            f"topk_k must satisfy 0 < topk_k <= d_hidden; "
            f"received topk_k={topk_k}, d_hidden={d_hidden}."
        )
    return int(topk_k), None, None


def deterministic_topk_mask_per_instance(
    scores: Tensor,
    topk_k_per_instance: Tensor,
) -> Tensor:
    if topk_k_per_instance.shape[0] != scores.shape[1]:
        raise ValueError(
            f"topk_k_per_instance length ({topk_k_per_instance.shape[0]}) "
            f"must match scores n_instances dim ({scores.shape[1]})."
        )
    masks = []
    for i in range(scores.shape[1]):
        instance_scores = scores[:, i : i + 1, :]
        instance_mask = deterministic_topk_mask(
            instance_scores, int(topk_k_per_instance[i].item())
        )
        masks.append(instance_mask)
    return torch.cat(masks, dim=1)


def deterministic_topk_mask(scores: Tensor, topk_k: int) -> Tensor:
    scores = torch.nan_to_num(
        scores,
        nan=-float("inf"),
        posinf=float("inf"),
        neginf=-float("inf"),
    )
    kth_values = torch.topk(scores, k=topk_k, dim=-1).values[..., -1:].contiguous()

    strictly_greater = scores > kth_values
    equal_to_kth = scores == kth_values
    strictly_greater_count = strictly_greater.sum(dim=-1, keepdim=True)
    required_from_equal = topk_k - strictly_greater_count

    equal_rank = torch.cumsum(equal_to_kth.to(torch.int32), dim=-1)
    equal_pick = equal_to_kth & (equal_rank <= required_from_equal)
    return strictly_greater | equal_pick
