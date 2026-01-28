"""Loss functions for probabilistic price prediction.

This module implements ordinal-aware loss functions for multi-horizon
bucket classification:
- Earth Mover's Distance (EMD) loss - penalizes predictions far from true bucket
- Soft-label cross-entropy - uses Gaussian-smoothed targets for ordinal awareness
- Focal loss variant - addresses class imbalance in extreme buckets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LossConfig:
    """Configuration for loss functions.

    Attributes:
        num_buckets: Number of output buckets.
        loss_type: Type of loss function ("emd", "soft_ce", "focal").
        label_smoothing_sigma: Sigma for Gaussian label smoothing (soft_ce only).
        focal_gamma: Focusing parameter for focal loss (focal only).
        emd_p: Power for EMD distance (1 for L1, 2 for L2).
        reduction: Reduction method ("mean", "sum", "none").
    """

    num_buckets: int = 101
    loss_type: Literal["emd", "soft_ce", "focal"] = "emd"
    label_smoothing_sigma: float = 1.0
    focal_gamma: float = 2.0
    emd_p: int = 1
    reduction: Literal["mean", "sum", "none"] = "mean"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_buckets < 3:
            raise ValueError("num_buckets must be at least 3")
        if self.loss_type not in ("emd", "soft_ce", "focal"):
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        if self.label_smoothing_sigma <= 0:
            raise ValueError("label_smoothing_sigma must be positive")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if self.emd_p not in (1, 2):
            raise ValueError("emd_p must be 1 or 2")
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction: {self.reduction}")


def compute_cdf(probs: Tensor) -> Tensor:
    """Compute cumulative distribution function from probabilities.

    Args:
        probs: Probability distribution of shape (..., num_buckets).

    Returns:
        CDF of same shape, where cdf[..., k] = sum(probs[..., :k+1]).
    """
    return torch.cumsum(probs, dim=-1)


def earth_movers_distance(
    pred_probs: Tensor,
    target_probs: Tensor,
    p: int = 1,
) -> Tensor:
    """Compute Earth Mover's Distance between probability distributions.

    EMD (also called Wasserstein distance) measures the minimum "work" needed
    to transform one distribution into another. For ordinal buckets, this
    penalizes predictions far from the true bucket more than nearby errors.

    Args:
        pred_probs: Predicted probabilities of shape (batch, num_buckets).
        target_probs: Target probabilities of shape (batch, num_buckets).
        p: Power for distance (1 for L1-EMD, 2 for L2-EMD).

    Returns:
        EMD values of shape (batch,).
    """
    # EMD between 1D distributions equals L_p distance between CDFs
    pred_cdf = compute_cdf(pred_probs)
    target_cdf = compute_cdf(target_probs)

    if p == 1:
        # L1 EMD: sum of absolute CDF differences
        emd = torch.sum(torch.abs(pred_cdf - target_cdf), dim=-1)
    else:
        # L2 EMD: sqrt of sum of squared CDF differences
        emd = torch.sqrt(torch.sum((pred_cdf - target_cdf) ** 2, dim=-1))

    return emd


class EMDLoss(nn.Module):
    """Earth Mover's Distance loss for ordinal bucket classification.

    This loss penalizes predictions far from the true bucket more than
    nearby predictions, making it well-suited for ordinal targets like
    price buckets where bucket 25 is "closer" to bucket 26 than to bucket 75.

    Attributes:
        config: Loss configuration.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        """Initialize EMD loss.

        Args:
            config: Loss configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or LossConfig(loss_type="emd")

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute EMD loss.

        Args:
            logits: Raw model outputs of shape (batch, num_buckets).
            targets: Target bucket indices of shape (batch,) as integers,
                or target probabilities of shape (batch, num_buckets).

        Returns:
            Loss value (scalar if reduction="mean" or "sum", else (batch,)).
        """
        # Convert logits to probabilities
        pred_probs = F.softmax(logits, dim=-1)

        # Handle integer targets by converting to one-hot
        if targets.dim() == 1:
            target_probs = F.one_hot(
                targets.long(), num_classes=self.config.num_buckets
            ).float()
        else:
            target_probs = targets

        # Compute EMD
        emd = earth_movers_distance(pred_probs, target_probs, p=self.config.emd_p)

        # Apply reduction
        if self.config.reduction == "mean":
            return emd.mean()
        elif self.config.reduction == "sum":
            return emd.sum()
        else:
            return emd


def create_soft_labels(
    targets: Tensor,
    num_buckets: int,
    sigma: float = 1.0,
) -> Tensor:
    """Create soft labels with Gaussian smoothing around target bucket.

    This converts hard bucket indices to soft probability distributions
    where nearby buckets receive partial probability mass.

    Args:
        targets: Target bucket indices of shape (batch,).
        num_buckets: Total number of buckets.
        sigma: Standard deviation of Gaussian in bucket units.

    Returns:
        Soft labels of shape (batch, num_buckets).
    """
    device = targets.device

    # Create bucket indices: shape (num_buckets,)
    bucket_indices = torch.arange(num_buckets, device=device, dtype=torch.float32)

    # Expand for broadcasting: targets (batch, 1), indices (1, num_buckets)
    targets_expanded = targets.float().unsqueeze(-1)  # (batch, 1)
    indices_expanded = bucket_indices.unsqueeze(0)  # (1, num_buckets)

    # Compute distances and Gaussian weights
    distances = indices_expanded - targets_expanded  # (batch, num_buckets)
    weights = torch.exp(-0.5 * (distances / sigma) ** 2)

    # Normalize to sum to 1
    soft_labels = weights / weights.sum(dim=-1, keepdim=True)

    return soft_labels


class SoftCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with Gaussian-smoothed soft labels.

    Instead of one-hot targets, this uses soft probability distributions
    centered on the true bucket, providing ordinal awareness where nearby
    buckets receive partial credit.

    Attributes:
        config: Loss configuration.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        """Initialize soft cross-entropy loss.

        Args:
            config: Loss configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or LossConfig(loss_type="soft_ce")

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute soft cross-entropy loss.

        Args:
            logits: Raw model outputs of shape (batch, num_buckets).
            targets: Target bucket indices of shape (batch,) as integers,
                or target probabilities of shape (batch, num_buckets).

        Returns:
            Loss value (scalar if reduction="mean" or "sum", else (batch,)).
        """
        # Get soft target distribution
        if targets.dim() == 1:
            soft_targets = create_soft_labels(
                targets,
                self.config.num_buckets,
                self.config.label_smoothing_sigma,
            )
        else:
            soft_targets = targets

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Cross-entropy with soft targets: -sum(target * log_prob)
        loss = -torch.sum(soft_targets * log_probs, dim=-1)

        # Apply reduction
        if self.config.reduction == "mean":
            return loss.mean()
        elif self.config.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in bucket classification.

    Focal loss down-weights well-classified examples, focusing training
    on hard examples. This is useful when extreme buckets are rare.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.

    Attributes:
        config: Loss configuration.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        """Initialize focal loss.

        Args:
            config: Loss configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or LossConfig(loss_type="focal")

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs of shape (batch, num_buckets).
            targets: Target bucket indices of shape (batch,) as integers.

        Returns:
            Loss value (scalar if reduction="mean" or "sum", else (batch,)).
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probability of true class
        if targets.dim() == 1:
            # Gather probabilities at target indices
            targets_long = targets.long()
            p_t = probs.gather(dim=-1, index=targets_long.unsqueeze(-1)).squeeze(-1)
        else:
            # Weighted probability with soft targets
            p_t = (probs * targets).sum(dim=-1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.config.focal_gamma

        # Cross-entropy term: -log(p_t)
        ce_loss = -torch.log(p_t.clamp(min=1e-8))

        # Focal loss
        loss = focal_weight * ce_loss

        # Apply reduction
        if self.config.reduction == "mean":
            return loss.mean()
        elif self.config.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MultiHorizonLoss(nn.Module):
    """Combined loss for multi-horizon price prediction.

    Computes loss for each prediction horizon and combines them,
    optionally with different weights per horizon.

    Attributes:
        config: Loss configuration.
        horizon_weights: Optional weights for each horizon.
        loss_fn: Base loss function for each horizon.
    """

    horizon_weights: Tensor  # Buffer type annotation
    loss_fn: EMDLoss | SoftCrossEntropyLoss | FocalLoss

    def __init__(
        self,
        config: LossConfig | None = None,
        horizon_weights: list[float] | None = None,
        num_horizons: int = 8,
    ) -> None:
        """Initialize multi-horizon loss.

        Args:
            config: Loss configuration.
            horizon_weights: Weights for each horizon loss. If None, equal weights.
            num_horizons: Number of prediction horizons.
        """
        super().__init__()
        self.config = config or LossConfig()
        self.num_horizons = num_horizons

        # Set up horizon weights
        if horizon_weights is None:
            self.register_buffer(
                "horizon_weights",
                torch.ones(num_horizons) / num_horizons,
            )
        else:
            if len(horizon_weights) != num_horizons:
                raise ValueError(
                    f"Expected {num_horizons} weights, got {len(horizon_weights)}"
                )
            weights = torch.tensor(horizon_weights, dtype=torch.float32)
            # Normalize to sum to 1
            self.register_buffer("horizon_weights", weights / weights.sum())

        # Create base loss function with no reduction (we'll reduce ourselves)
        per_sample_config = LossConfig(
            num_buckets=self.config.num_buckets,
            loss_type=self.config.loss_type,
            label_smoothing_sigma=self.config.label_smoothing_sigma,
            focal_gamma=self.config.focal_gamma,
            emd_p=self.config.emd_p,
            reduction="none",
        )

        if self.config.loss_type == "emd":
            self.loss_fn = EMDLoss(per_sample_config)
        elif self.config.loss_type == "soft_ce":
            self.loss_fn = SoftCrossEntropyLoss(per_sample_config)
        else:
            self.loss_fn = FocalLoss(per_sample_config)

    def forward(
        self,
        logits: dict[int, Tensor] | list[Tensor],
        targets: dict[int, Tensor] | list[Tensor],
    ) -> Tensor:
        """Compute combined multi-horizon loss.

        Args:
            logits: Per-horizon logits, either as dict {horizon_idx: tensor}
                or list of tensors. Each tensor has shape (batch, num_buckets).
            targets: Per-horizon targets, same format as logits.
                Each tensor has shape (batch,) for indices or (batch, num_buckets).

        Returns:
            Combined loss value (scalar).
        """
        # Convert to list if dict
        if isinstance(logits, dict):
            logits_list = [logits[i] for i in range(self.num_horizons)]
        else:
            logits_list = list(logits)

        if isinstance(targets, dict):
            targets_list = [targets[i] for i in range(self.num_horizons)]
        else:
            targets_list = list(targets)

        if len(logits_list) != self.num_horizons:
            raise ValueError(
                f"Expected {self.num_horizons} horizons, got {len(logits_list)}"
            )

        # Compute loss for each horizon
        horizon_losses = []
        for horizon_logits, horizon_targets in zip(
            logits_list, targets_list, strict=True
        ):
            loss = self.loss_fn(horizon_logits, horizon_targets)
            horizon_losses.append(loss.mean())  # Mean over batch

        # Stack and weight
        losses = torch.stack(horizon_losses)
        weighted_loss = (losses * self.horizon_weights).sum()

        return weighted_loss


def get_loss_function(config: LossConfig) -> nn.Module:
    """Factory function to create loss function from config.

    Args:
        config: Loss configuration.

    Returns:
        Configured loss function module.
    """
    if config.loss_type == "emd":
        return EMDLoss(config)
    elif config.loss_type == "soft_ce":
        return SoftCrossEntropyLoss(config)
    elif config.loss_type == "focal":
        return FocalLoss(config)
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")
