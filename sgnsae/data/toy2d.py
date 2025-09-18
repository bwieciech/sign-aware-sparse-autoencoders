"""Utilities for generating the 2-D synthetic dataset used in Protocol A.

The module implements the correlated and anti-correlated feature bundles
introduced in ``Toy Models of Superposition`` and adapts them to the
sign-aware sparse autoencoder setting.  Each dataset is fully specified by
:class:`Toy2DGenerationConfig` and deterministic with respect to the
provided seed so that experiments can be reproduced bit-for-bit.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


BUNDLE_TYPE_CORRELATED = "correlated"
BUNDLE_TYPE_ANTICORRELATED = "anticorrelated"


@dataclass(frozen=True)
class Toy2DBundle:
    """Metadata describing a feature bundle.

    Attributes
    ----------
    name:
        Human readable identifier (e.g. ``"corr_0"``).
    indices:
        Indices of features that belong to the bundle.
    bundle_type:
        Either :data:`BUNDLE_TYPE_CORRELATED` or
        :data:`BUNDLE_TYPE_ANTICORRELATED`.
    base_angle:
        Base angle on the unit circle (in radians) used to sample the
        feature directions for the bundle.  The ``anticorrelated`` bundles
        use this angle for the "positive" direction and add :math:`\\pi` for
        the negative directions so that the ground-truth features live on
        opposite sides of the circle.
    """

    name: str
    indices: Tuple[int, ...]
    bundle_type: str
    base_angle: float


@dataclass(frozen=True)
class Toy2DGenerationConfig:
    """Configuration controlling the synthetic dataset construction."""

    num_samples: int
    num_correlated_bundles: int
    correlated_bundle_size: int
    num_anticorrelated_bundles: int
    anticorrelated_bundle_size: int
    correlated_on_probability: float
    anticorrelated_on_probability: float
    angular_std: float = 0.12
    magnitude_range: Tuple[float, float] = (0.0, 1.0)
    noise_std: float = 0.0
    seed: int = 0

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, indent=2)


@dataclass
class Toy2DData:
    """Container returned by :func:`generate_toy2d_dataset`.

    The container stores both the tensors required for model training as
    well as metadata that is useful for evaluation and visualization.
    """

    observations: torch.Tensor
    activations: torch.Tensor
    features: torch.Tensor
    bundles: List[Toy2DBundle]
    config: Toy2DGenerationConfig
    feature_to_bundle: List[int]

    def to_dataset(self) -> "Toy2DTensorDataset":
        """Create a :class:`torch.utils.data.Dataset` view over observations."""

        return Toy2DTensorDataset(self.observations, self.activations)

    @property
    def num_features(self) -> int:
        return self.features.shape[0]

    def anticorrelated_pairs(self) -> List[Tuple[int, int]]:
        """Return a list of feature index pairs belonging to anti bundles."""

        pairs: List[Tuple[int, int]] = []
        for bundle in self.bundles:
            if bundle.bundle_type != BUNDLE_TYPE_ANTICORRELATED:
                continue
            indices = list(bundle.indices)
            if len(indices) < 2:
                continue
            # We treat anti-correlated bundles as composed of ordered pairs.
            # This keeps the logic explicit and allows for bundle sizes > 2.
            for i in range(len(indices) - 1):
                for j in range(i + 1, len(indices)):
                    pairs.append((indices[i], indices[j]))
        return pairs

    def correlated_groups(self) -> List[Tuple[int, ...]]:
        groups: List[Tuple[int, ...]] = []
        for bundle in self.bundles:
            if bundle.bundle_type == BUNDLE_TYPE_CORRELATED:
                groups.append(bundle.indices)
        return groups


class Toy2DTensorDataset(Dataset):
    """Simple tensor-backed dataset exposing observations and activations."""

    def __init__(self, observations: torch.Tensor, activations: torch.Tensor):
        if observations.ndim != 2 or observations.shape[1] != 2:
            raise ValueError("Toy2DTensorDataset expects observations of shape (N, 2)")
        if activations.ndim != 2:
            raise ValueError("Activations must be a rank-2 tensor")
        if observations.shape[0] != activations.shape[0]:
            raise ValueError("Observations and activations must have the same batch dimension")
        self._observations = observations
        self._activations = activations

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._observations.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "observation": self._observations[idx],
            "activation": self._activations[idx],
        }

    @property
    def observations(self) -> torch.Tensor:
        return self._observations

    @property
    def activations(self) -> torch.Tensor:
        return self._activations


def _sample_angles(
    base_angle: float,
    count: int,
    std: float,
    rng: np.random.Generator,
    *,
    mirrored: bool = False,
) -> List[float]:
    """Sample ``count`` angles around ``base_angle`` with optional mirroring."""

    offsets = rng.normal(loc=0.0, scale=std, size=count)
    angles = []
    for i in range(count):
        angle = base_angle + offsets[i]
        if mirrored and i % 2 == 1:
            angle = angle + math.pi
        angles.append(float(angle % (2 * math.pi)))
    return angles


def _angles_to_unit_vectors(angles: Sequence[float]) -> torch.Tensor:
    vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return torch.tensor(vectors, dtype=torch.float32)


def generate_toy2d_dataset(config: Toy2DGenerationConfig) -> Toy2DData:
    """Generate a toy 2-D dataset containing correlated/anticorrelated bundles.

    The function follows the specification in the Protocol A design document.
    Correlated bundles share a Bernoulli gate so that the features within the
    bundle co-activate, whereas anti-correlated bundles sample at most one
    active feature at a time.  Magnitudes are drawn from a uniform
    distribution in ``config.magnitude_range`` whenever a feature is active.
    """

    if not (0.0 <= config.correlated_on_probability <= 1.0):
        raise ValueError("correlated_on_probability must lie in [0, 1]")
    if not (0.0 <= config.anticorrelated_on_probability <= 1.0):
        raise ValueError("anticorrelated_on_probability must lie in [0, 1]")

    rng = np.random.default_rng(config.seed)

    bundles: List[Toy2DBundle] = []
    angles: List[float] = []
    feature_to_bundle: List[int] = []

    # Correlated bundles -------------------------------------------------
    for bundle_id in range(config.num_correlated_bundles):
        base_angle = rng.uniform(0.0, 2 * math.pi)
        feature_indices = list(
            range(
                len(angles),
                len(angles) + config.correlated_bundle_size,
            )
        )
        bundles.append(
            Toy2DBundle(
                name=f"corr_{bundle_id}",
                indices=tuple(feature_indices),
                bundle_type=BUNDLE_TYPE_CORRELATED,
                base_angle=base_angle,
            )
        )
        angles.extend(
            _sample_angles(base_angle, config.correlated_bundle_size, config.angular_std, rng)
        )
        feature_to_bundle.extend([len(bundles) - 1] * config.correlated_bundle_size)

    # Anti-correlated bundles -------------------------------------------
    for bundle_id in range(config.num_anticorrelated_bundles):
        base_angle = rng.uniform(0.0, 2 * math.pi)
        feature_indices = list(
            range(
                len(angles),
                len(angles) + config.anticorrelated_bundle_size,
            )
        )
        bundles.append(
            Toy2DBundle(
                name=f"anti_{bundle_id}",
                indices=tuple(feature_indices),
                bundle_type=BUNDLE_TYPE_ANTICORRELATED,
                base_angle=base_angle,
            )
        )
        angles.extend(
            _sample_angles(
                base_angle,
                config.anticorrelated_bundle_size,
                config.angular_std,
                rng,
                mirrored=True,
            )
        )
        feature_to_bundle.extend([len(bundles) - 1] * config.anticorrelated_bundle_size)

    features = _angles_to_unit_vectors(angles)
    num_features = features.shape[0]

    low, high = config.magnitude_range
    if high <= low:
        raise ValueError("magnitude_range must satisfy low < high")

    activations = torch.zeros(config.num_samples, num_features, dtype=torch.float32)

    for sample_idx in range(config.num_samples):
        # Correlated bundles: activate all features together
        for bundle in bundles:
            if bundle.bundle_type == BUNDLE_TYPE_CORRELATED:
                if rng.uniform() < config.correlated_on_probability:
                    magnitudes = rng.uniform(low, high, size=len(bundle.indices))
                    activations[sample_idx, list(bundle.indices)] = torch.tensor(
                        magnitudes, dtype=torch.float32
                    )
                continue

            # Anti-correlated bundles: choose at most one feature
            if rng.uniform() < config.anticorrelated_on_probability:
                feature_idx = int(rng.choice(bundle.indices))
                magnitude = float(rng.uniform(low, high))
                activations[sample_idx, feature_idx] = magnitude

    observations = activations @ features
    if config.noise_std > 0:
        noise = torch.randn_like(observations) * config.noise_std
        observations = observations + noise

    return Toy2DData(
        observations=observations,
        activations=activations,
        features=features,
        bundles=bundles,
        config=config,
        feature_to_bundle=feature_to_bundle,
    )


def batched_iterator(
    dataset: Toy2DTensorDataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: Optional[int] = None,
) -> Iterable[Dict[str, torch.Tensor]]:
    """Create a deterministic iterator over batches of the dataset.

    PyTorch's :class:`DataLoader` introduces multi-threaded behaviour and a
    fair amount of boilerplate for such a small dataset, so we provide a
    minimal iterator that keeps reproducibility simple.
    """

    num_samples = len(dataset)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    position = 0

    while True:
        if shuffle:
            rng.shuffle(indices)
        while position < num_samples:
            batch_indices = indices[position : position + batch_size]
            position += batch_size
            batch = {
                "observation": dataset.observations[batch_indices],
                "activation": dataset.activations[batch_indices],
            }
            yield batch
        position = 0


__all__ = [
    "BUNDLE_TYPE_ANTICORRELATED",
    "BUNDLE_TYPE_CORRELATED",
    "Toy2DBundle",
    "Toy2DData",
    "Toy2DTensorDataset",
    "Toy2DGenerationConfig",
    "batched_iterator",
    "generate_toy2d_dataset",
]
