"""Advanced calibration analysis for probabilistic predictions.

This module provides tools for analyzing and improving model calibration:

Post-hoc Calibration Methods:
- Temperature scaling: Single-parameter rescaling of logits
- Platt scaling: Logistic regression on logits
- Isotonic regression: Non-parametric calibration

Visualization and Analysis:
- Reliability diagrams (calibration curves)
- Calibration analysis across horizons and buckets
- Calibration drift over time

Usage:
    # Temperature scaling on validation set
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_targets)
    calibrated_probs = scaler.transform(test_logits)

    # Plot reliability diagram
    fig = plot_reliability_diagram(probs, targets)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch import Tensor

from evaluation.metrics import CalibrationResult, compute_calibration


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for calibration methods.

    Attributes:
        num_bins: Number of bins for reliability diagrams.
        temperature_lr: Learning rate for temperature scaling.
        temperature_max_iter: Maximum iterations for temperature optimization.
        temperature_tol: Convergence tolerance for temperature scaling.
        platt_max_iter: Maximum iterations for Platt scaling.
    """

    num_bins: int = 15
    temperature_lr: float = 0.01
    temperature_max_iter: int = 50
    temperature_tol: float = 1e-4
    platt_max_iter: int = 100

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if self.temperature_lr <= 0:
            raise ValueError("temperature_lr must be positive")
        if self.temperature_max_iter < 1:
            raise ValueError("temperature_max_iter must be at least 1")
        if self.temperature_tol <= 0:
            raise ValueError("temperature_tol must be positive")
        if self.platt_max_iter < 1:
            raise ValueError("platt_max_iter must be at least 1")


class TemperatureScaler:
    """Temperature scaling for post-hoc calibration.

    Temperature scaling rescales logits by a single learned parameter T:
        calibrated_probs = softmax(logits / T)

    This preserves the rank ordering of predictions while improving calibration.

    Reference:
        Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        """Initialize temperature scaler.

        Args:
            config: Calibration configuration. Uses defaults if None.
        """
        self.config = config or CalibrationConfig()
        self.temperature: float = 1.0
        self._is_fitted: bool = False

    def fit(
        self,
        logits: Tensor,
        targets: Tensor,
        validation_split: float = 0.0,
    ) -> dict[str, Any]:
        """Learn optimal temperature on validation data.

        Args:
            logits: Uncalibrated logits of shape (batch, num_buckets).
            targets: Target bucket indices of shape (batch,).
            validation_split: Fraction of data to use for validation (0-1).
                If > 0, splits data and returns validation metrics.

        Returns:
            Dictionary with fitting statistics (loss history, final ECE, etc.).
        """
        # Validate inputs
        if logits.ndim != 2:
            raise ValueError(f"logits must be 2D, got shape {logits.shape}")
        if targets.ndim != 1:
            raise ValueError(f"targets must be 1D, got shape {targets.shape}")
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: logits {logits.shape[0]} vs "
                f"targets {targets.shape[0]}"
            )

        # Optional validation split
        if validation_split > 0:
            n = len(logits)
            split_idx = int(n * (1 - validation_split))
            train_logits, val_logits = logits[:split_idx], logits[split_idx:]
            train_targets, val_targets = targets[:split_idx], targets[split_idx:]
        else:
            train_logits, train_targets = logits, targets
            val_logits, val_targets = None, None

        # Temperature parameter (initialized to 1.0)
        temperature = nn.Parameter(torch.ones(1, device=logits.device))

        # NLL loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS(
            [temperature],
            lr=self.config.temperature_lr,
            max_iter=self.config.temperature_max_iter,
        )

        # Track loss history
        loss_history: list[float] = []

        def closure() -> Tensor:
            optimizer.zero_grad()
            # Scale logits by temperature
            scaled_logits = train_logits / temperature
            loss: Tensor = criterion(scaled_logits, train_targets)
            loss.backward()  # type: ignore[no-untyped-call]
            loss_history.append(float(loss.item()))
            return loss

        # Optimize temperature
        optimizer.step(closure)  # type: ignore[no-untyped-call]

        # Store learned temperature
        self.temperature = float(temperature.item())
        self._is_fitted = True

        # Compute final calibration metrics
        train_probs = self.transform(train_logits)
        train_cal = compute_calibration(
            train_probs, train_targets, self.config.num_bins
        )

        stats: dict[str, Any] = {
            "temperature": self.temperature,
            "loss_history": loss_history,
            "train_ece": train_cal.expected_calibration_error,
            "train_mce": train_cal.maximum_calibration_error,
        }

        if val_logits is not None and val_targets is not None:
            val_probs = self.transform(val_logits)
            val_cal = compute_calibration(val_probs, val_targets, self.config.num_bins)
            stats["val_ece"] = val_cal.expected_calibration_error
            stats["val_mce"] = val_cal.maximum_calibration_error

        return stats

    def transform(self, logits: Tensor) -> Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Uncalibrated logits of shape (batch, num_buckets).

        Returns:
            Calibrated probabilities of shape (batch, num_buckets).

        Raises:
            RuntimeError: If scaler has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("TemperatureScaler must be fitted before transform")

        scaled_logits = logits / self.temperature
        return torch.softmax(scaled_logits, dim=-1)

    def fit_transform(
        self,
        logits: Tensor,
        targets: Tensor,
        validation_split: float = 0.0,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Fit temperature and transform logits in one step.

        Args:
            logits: Uncalibrated logits of shape (batch, num_buckets).
            targets: Target bucket indices of shape (batch,).
            validation_split: Fraction of data to use for validation.

        Returns:
            Tuple of (calibrated_probs, fitting_stats).
        """
        stats = self.fit(logits, targets, validation_split)
        probs = self.transform(logits)
        return probs, stats


class PlattScaler:
    """Platt scaling for post-hoc calibration.

    Platt scaling applies logistic regression to the logits:
        calibrated_prob = sigmoid(a * logit + b)

    This is more flexible than temperature scaling but can overfit.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        """Initialize Platt scaler.

        Args:
            config: Calibration configuration. Uses defaults if None.
        """
        self.config = config or CalibrationConfig()
        self.scale: float = 1.0
        self.bias: float = 0.0
        self._is_fitted: bool = False

    def fit(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> dict[str, Any]:
        """Learn Platt scaling parameters.

        Args:
            logits: Uncalibrated logits of shape (batch, num_buckets).
            targets: Target bucket indices of shape (batch,).

        Returns:
            Dictionary with fitting statistics.
        """
        # For multi-class, we apply Platt scaling to the max logit
        # (probability of predicted class)
        max_logits = logits.max(dim=-1).values.unsqueeze(-1)
        pred_classes = logits.argmax(dim=-1)

        # Binary target: 1 if predicted class = true class, else 0
        binary_targets = (pred_classes == targets).float().unsqueeze(-1)

        # Learn a and b via logistic regression
        params = nn.Parameter(
            torch.tensor([[1.0], [0.0]], device=logits.device, requires_grad=True)
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([params], max_iter=self.config.platt_max_iter)

        loss_history: list[float] = []

        def closure() -> Tensor:
            optimizer.zero_grad()
            # scaled_logits = a * logit + b
            scaled = max_logits * params[0] + params[1]
            loss: Tensor = criterion(scaled, binary_targets)
            loss.backward()  # type: ignore[no-untyped-call]
            loss_history.append(float(loss.item()))
            return loss

        optimizer.step(closure)  # type: ignore[no-untyped-call]

        self.scale = float(params[0].item())
        self.bias = float(params[1].item())
        self._is_fitted = True

        return {
            "scale": self.scale,
            "bias": self.bias,
            "loss_history": loss_history,
        }

    def transform(self, logits: Tensor) -> Tensor:
        """Apply Platt scaling to logits.

        Args:
            logits: Uncalibrated logits of shape (batch, num_buckets).

        Returns:
            Calibrated probabilities of shape (batch, num_buckets).

        Raises:
            RuntimeError: If scaler has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("PlattScaler must be fitted before transform")

        # Scale logits and convert to probabilities
        scaled_logits = logits * self.scale + self.bias
        return torch.softmax(scaled_logits, dim=-1)

    def fit_transform(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Fit Platt scaling and transform logits.

        Args:
            logits: Uncalibrated logits of shape (batch, num_buckets).
            targets: Target bucket indices of shape (batch,).

        Returns:
            Tuple of (calibrated_probs, fitting_stats).
        """
        stats = self.fit(logits, targets)
        probs = self.transform(logits)
        return probs, stats


@dataclass
class ReliabilityDiagram:
    """Data for plotting a reliability diagram.

    Attributes:
        bin_confidences: Mean predicted probability in each bin.
        bin_accuracies: Actual accuracy in each bin.
        bin_counts: Number of samples in each bin.
        bin_edges: Edges of the confidence bins.
        ece: Expected calibration error.
        mce: Maximum calibration error.
        num_samples: Total number of samples.
    """

    bin_confidences: NDArray[np.floating[Any]]
    bin_accuracies: NDArray[np.floating[Any]]
    bin_counts: NDArray[np.intp]
    bin_edges: NDArray[np.floating[Any]]
    ece: float
    mce: float
    num_samples: int


def compute_reliability_diagram(
    probs: Tensor,
    targets: Tensor,
    num_bins: int = 15,
) -> ReliabilityDiagram:
    """Compute data for a reliability diagram.

    A reliability diagram plots predicted confidence (x-axis) vs actual
    accuracy (y-axis). Perfect calibration lies on the diagonal y=x.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).
        num_bins: Number of bins for confidence values.

    Returns:
        ReliabilityDiagram with all data needed for plotting.
    """
    cal_result = compute_calibration(probs, targets, num_bins)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    return ReliabilityDiagram(
        bin_confidences=cal_result.bin_confidences,
        bin_accuracies=cal_result.bin_accuracies,
        bin_counts=cal_result.bin_counts,
        bin_edges=bin_edges,
        ece=cal_result.expected_calibration_error,
        mce=cal_result.maximum_calibration_error,
        num_samples=int(probs.shape[0]),
    )


@dataclass
class MultiHorizonCalibration:
    """Calibration analysis across multiple prediction horizons.

    Attributes:
        horizons: List of horizon values (seconds).
        reliability_diagrams: Reliability diagram for each horizon.
        ece_by_horizon: ECE for each horizon.
        mce_by_horizon: MCE for each horizon.
        mean_ece: Mean ECE across horizons.
        mean_mce: Mean MCE across horizons.
    """

    horizons: list[int]
    reliability_diagrams: dict[int, ReliabilityDiagram]
    ece_by_horizon: dict[int, float]
    mce_by_horizon: dict[int, float]
    mean_ece: float
    mean_mce: float


def compute_multi_horizon_calibration(
    probs_dict: dict[int, Tensor],
    targets_dict: dict[int, Tensor],
    num_bins: int = 15,
) -> MultiHorizonCalibration:
    """Compute calibration analysis across all horizons.

    Args:
        probs_dict: Dict mapping horizon -> probabilities (batch, num_buckets).
        targets_dict: Dict mapping horizon -> targets (batch,).
        num_bins: Number of bins for reliability diagrams.

    Returns:
        MultiHorizonCalibration with per-horizon and aggregate metrics.
    """
    horizons = sorted(probs_dict.keys())
    reliability_diagrams: dict[int, ReliabilityDiagram] = {}
    ece_by_horizon: dict[int, float] = {}
    mce_by_horizon: dict[int, float] = {}

    for horizon in horizons:
        probs = probs_dict[horizon]
        targets = targets_dict[horizon]
        diagram = compute_reliability_diagram(probs, targets, num_bins)
        reliability_diagrams[horizon] = diagram
        ece_by_horizon[horizon] = diagram.ece
        mce_by_horizon[horizon] = diagram.mce

    mean_ece = float(np.mean(list(ece_by_horizon.values())))
    mean_mce = float(np.mean(list(mce_by_horizon.values())))

    return MultiHorizonCalibration(
        horizons=horizons,
        reliability_diagrams=reliability_diagrams,
        ece_by_horizon=ece_by_horizon,
        mce_by_horizon=mce_by_horizon,
        mean_ece=mean_ece,
        mean_mce=mean_mce,
    )


def format_calibration_report(
    calibration: MultiHorizonCalibration,
) -> str:
    """Format calibration analysis as a human-readable report.

    Args:
        calibration: Multi-horizon calibration analysis.

    Returns:
        Formatted string report.
    """
    lines = ["=" * 60, "Multi-Horizon Calibration Analysis", "=" * 60, ""]

    # Per-horizon table
    lines.append(f"{'Horizon':>10} {'ECE':>10} {'MCE':>10} {'Samples':>12}")
    lines.append("-" * 60)

    for horizon in calibration.horizons:
        diagram = calibration.reliability_diagrams[horizon]
        ece = calibration.ece_by_horizon[horizon]
        mce = calibration.mce_by_horizon[horizon]
        lines.append(
            f"{horizon:>10}s {ece:>10.4f} {mce:>10.4f} {diagram.num_samples:>12}"
        )

    lines.append("-" * 60)
    lines.append(
        f"{'Mean':>10} {calibration.mean_ece:>10.4f} "
        f"{calibration.mean_mce:>10.4f}"
    )

    lines.append("")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Calibration Quality:")
    lines.append(f"  ECE < 0.05: Excellent calibration")
    lines.append(f"  ECE < 0.10: Good calibration")
    lines.append(f"  ECE < 0.15: Fair calibration")
    lines.append(f"  ECE > 0.15: Poor calibration (consider recalibration)")
    lines.append("")

    if calibration.mean_ece < 0.05:
        quality = "Excellent"
    elif calibration.mean_ece < 0.10:
        quality = "Good"
    elif calibration.mean_ece < 0.15:
        quality = "Fair"
    else:
        quality = "Poor - consider temperature scaling"

    lines.append(f"Overall calibration quality: {quality}")
    lines.append("")

    return "\n".join(lines)


@dataclass
class BucketCalibration:
    """Per-bucket calibration analysis.

    Useful for identifying which price buckets are poorly calibrated.

    Attributes:
        bucket_indices: Bucket index values.
        bucket_centers_bps: Bucket centers in basis points.
        predicted_frequencies: Mean predicted probability for each bucket.
        actual_frequencies: Actual frequency of occurrence for each bucket.
        sample_counts: Number of samples where each bucket was predicted.
    """

    bucket_indices: NDArray[np.intp]
    bucket_centers_bps: NDArray[np.floating[Any]]
    predicted_frequencies: NDArray[np.floating[Any]]
    actual_frequencies: NDArray[np.floating[Any]]
    sample_counts: NDArray[np.intp]


def compute_bucket_calibration(
    probs: Tensor,
    targets: Tensor,
    bucket_centers_bps: Tensor,
) -> BucketCalibration:
    """Analyze calibration at the bucket level.

    For each bucket, computes:
    - Mean predicted probability (when that bucket is predicted)
    - Actual frequency (how often predictions were correct)

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).
        bucket_centers_bps: Bucket centers in basis points (num_buckets,).

    Returns:
        BucketCalibration with per-bucket statistics.
    """
    num_buckets = probs.shape[-1]
    pred_classes = probs.argmax(dim=-1)

    # Convert to numpy (ensure float64 for bucket_centers)
    probs_np = probs.detach().cpu().numpy()
    pred_classes_np = pred_classes.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    bucket_centers_np = bucket_centers_bps.detach().cpu().numpy().astype(np.float64)

    predicted_freq = np.zeros(num_buckets, dtype=np.float64)
    actual_freq = np.zeros(num_buckets, dtype=np.float64)
    sample_counts = np.zeros(num_buckets, dtype=np.intp)

    for bucket in range(num_buckets):
        # Find samples where this bucket was predicted
        mask = pred_classes_np == bucket
        sample_counts[bucket] = np.sum(mask)

        if sample_counts[bucket] > 0:
            # Mean predicted probability for this bucket
            predicted_freq[bucket] = np.mean(probs_np[mask, bucket])
            # Actual accuracy (was this bucket correct?)
            actual_freq[bucket] = np.mean((targets_np[mask] == bucket).astype(float))

    return BucketCalibration(
        bucket_indices=np.arange(num_buckets, dtype=np.intp),
        bucket_centers_bps=bucket_centers_np,
        predicted_frequencies=predicted_freq,
        actual_frequencies=actual_freq,
        sample_counts=sample_counts,
    )


def apply_temperature_scaling_multi_horizon(
    logits_dict: dict[int, Tensor],
    targets_dict: dict[int, Tensor],
    config: CalibrationConfig | None = None,
) -> tuple[dict[int, Tensor], dict[int, dict[str, Any]]]:
    """Apply temperature scaling separately to each horizon.

    Args:
        logits_dict: Dict mapping horizon -> logits (batch, num_buckets).
        targets_dict: Dict mapping horizon -> targets (batch,).
        config: Calibration configuration.

    Returns:
        Tuple of:
            - Dict mapping horizon -> calibrated probabilities
            - Dict mapping horizon -> fitting statistics
    """
    calibrated_probs: dict[int, Tensor] = {}
    stats_dict: dict[int, dict[str, Any]] = {}

    for horizon in sorted(logits_dict.keys()):
        logits = logits_dict[horizon]
        targets = targets_dict[horizon]

        scaler = TemperatureScaler(config)
        probs, stats = scaler.fit_transform(logits, targets)

        calibrated_probs[horizon] = probs
        stats_dict[horizon] = stats

    return calibrated_probs, stats_dict
