"""Model architecture definitions."""

from models.architectures.losses import (
    EMDLoss,
    FocalLoss,
    LossConfig,
    MultiHorizonLoss,
    SoftCrossEntropyLoss,
    compute_cdf,
    create_soft_labels,
    earth_movers_distance,
    get_loss_function,
)

__all__ = [
    "LossConfig",
    "EMDLoss",
    "SoftCrossEntropyLoss",
    "FocalLoss",
    "MultiHorizonLoss",
    "compute_cdf",
    "earth_movers_distance",
    "create_soft_labels",
    "get_loss_function",
]
