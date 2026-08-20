from typing import FrozenSet

from sgnsae.models import (
    AbsTopKSAE,
    AbsTopKGatedMagnitudeSAE,
    GatedSAE,
    JumpReLUSAE,
    SAE,
    SignedSoftThresholdSAE,
    TopKSAE,
)


KNOWN_SAE_CLASS_NAMES: FrozenSet[str] = frozenset(
    {
        SAE.__name__,
        GatedSAE.__name__,
        SignedSoftThresholdSAE.__name__,
        JumpReLUSAE.__name__,
        TopKSAE.__name__,
        AbsTopKSAE.__name__,
        AbsTopKGatedMagnitudeSAE.__name__,
    }
)
TOPK_SAE_CLASS_NAMES: FrozenSet[str] = frozenset(
    {TopKSAE.__name__, AbsTopKSAE.__name__, AbsTopKGatedMagnitudeSAE.__name__}
)
LAMBDA_SWEEP_SAE_CLASS_NAMES: FrozenSet[str] = frozenset(
    KNOWN_SAE_CLASS_NAMES - TOPK_SAE_CLASS_NAMES
)
SIGN_AWARE_SAE_CLASS_NAMES: FrozenSet[str] = frozenset(
    {SAE.__name__, GatedSAE.__name__, AbsTopKGatedMagnitudeSAE.__name__}
)
THETA_INIT_SAE_CLASS_NAMES: FrozenSet[str] = frozenset(
    {SignedSoftThresholdSAE.__name__, JumpReLUSAE.__name__}
)
JUMP_INIT_SAE_CLASS_NAMES: FrozenSet[str] = frozenset({JumpReLUSAE.__name__})
