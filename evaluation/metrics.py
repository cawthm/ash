"""Evaluation metrics for probabilistic price prediction.

This module implements metrics for evaluating multi-horizon probability
distribution predictions:

Probabilistic Quality Metrics:
- Log-likelihood: Primary metric measuring prediction quality
- Brier score: Decomposable score for probability calibration
- Expected Calibration Error (ECE): Calibration measure

Trading-Relevant Metrics:
- Directional accuracy: Whether predicted mean has correct sign
- Sharpness: Concentration of predicted distribution
- Profit/loss simulation: Backtesting on held-out data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for metrics computation.

    Attributes:
        num_buckets: Number of price buckets.
        bucket_centers_bps: Basis point values at bucket centers.
        num_calibration_bins: Number of bins for calibration analysis.
        eps: Small value for numerical stability.
    """

    num_buckets: int = 101
    bucket_centers_bps: tuple[float, ...] | None = None
    num_calibration_bins: int = 10
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_buckets < 3:
            raise ValueError("num_buckets must be at least 3")
        if (
            self.bucket_centers_bps is not None
            and len(self.bucket_centers_bps) != self.num_buckets
        ):
            raise ValueError(
                f"bucket_centers_bps length ({len(self.bucket_centers_bps)}) "
                f"must match num_buckets ({self.num_buckets})"
            )
        if self.num_calibration_bins < 2:
            raise ValueError("num_calibration_bins must be at least 2")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def get_bucket_centers(self) -> NDArray[np.floating[Any]]:
        """Get bucket centers in basis points.

        Returns:
            Array of bucket centers. Uses configured values if provided,
            otherwise defaults to uniform spacing from -50 to +50 bps.
        """
        if self.bucket_centers_bps is not None:
            return np.array(self.bucket_centers_bps, dtype=np.float64)
        # Default: uniform spacing from -50 to +50 bps
        return np.linspace(-50.0, 50.0, self.num_buckets, dtype=np.float64)


def log_likelihood(
    probs: Tensor,
    targets: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Compute log-likelihood of targets under predicted distributions.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).
        eps: Small value for numerical stability.

    Returns:
        Log-likelihood values of shape (batch,).
    """
    # Gather probabilities at target indices
    target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # Clamp for numerical stability
    return torch.log(target_probs.clamp(min=eps))


def negative_log_likelihood(
    probs: Tensor,
    targets: Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tensor:
    """Compute negative log-likelihood (NLL) loss.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).
        eps: Small value for numerical stability.
        reduction: Reduction method ("mean", "sum", "none").

    Returns:
        NLL value (scalar if reduction is "mean" or "sum").
    """
    ll = log_likelihood(probs, targets, eps)
    nll = -ll

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll


def brier_score(
    probs: Tensor,
    targets: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute Brier score for probability predictions.

    The Brier score measures the mean squared difference between
    predicted probabilities and the true outcome (one-hot encoded).
    Lower is better, with 0 being perfect.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).
        reduction: Reduction method ("mean", "sum", "none").

    Returns:
        Brier score (scalar if reduction is "mean" or "sum").
    """
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(-1, targets.unsqueeze(-1), 1.0)

    # Mean squared error per sample
    mse = torch.sum((probs - one_hot) ** 2, dim=-1)

    if reduction == "mean":
        return mse.mean()
    elif reduction == "sum":
        return mse.sum()
    else:
        return mse


def brier_score_per_bucket(
    probs: Tensor,
    targets: Tensor,
) -> Tensor:
    """Compute Brier score decomposed per bucket.

    Useful for identifying which buckets are poorly calibrated.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).

    Returns:
        Per-bucket Brier score of shape (num_buckets,).
    """
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(-1, targets.unsqueeze(-1), 1.0)

    # Squared error per bucket, averaged over samples
    squared_errors = (probs - one_hot) ** 2
    return squared_errors.mean(dim=0)


@dataclass
class CalibrationResult:
    """Results from calibration analysis.

    Attributes:
        bin_confidences: Mean predicted probability in each bin.
        bin_accuracies: Actual accuracy in each bin.
        bin_counts: Number of samples in each bin.
        expected_calibration_error: Overall ECE value.
        maximum_calibration_error: Maximum calibration error across bins.
    """

    bin_confidences: NDArray[np.floating[Any]]
    bin_accuracies: NDArray[np.floating[Any]]
    bin_counts: NDArray[np.intp]
    expected_calibration_error: float
    maximum_calibration_error: float


def compute_calibration(
    probs: Tensor,
    targets: Tensor,
    num_bins: int = 10,
) -> CalibrationResult:
    """Compute calibration metrics and reliability diagram data.

    For perfectly calibrated predictions, when the model predicts
    probability p, the event should occur with frequency p.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).
        num_bins: Number of bins for the reliability diagram.

    Returns:
        CalibrationResult with bin statistics and ECE/MCE.
    """
    # Get the probability of the true class for each sample
    target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Convert to numpy for binning
    confs = target_probs.detach().cpu().numpy()

    # Compute correctness (1 if predicted class = true class, else 0)
    pred_classes = probs.argmax(dim=-1)
    correct = (pred_classes == targets).float().detach().cpu().numpy()

    # Bin by confidence (probability of predicted class, not true class)
    pred_probs = probs.max(dim=-1).values.detach().cpu().numpy()

    # Create bins from 0 to 1
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(pred_probs, bin_edges[1:-1])

    # Compute per-bin statistics
    bin_confidences = np.zeros(num_bins, dtype=np.float64)
    bin_accuracies = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.intp)

    for i in range(num_bins):
        mask = bin_indices == i
        bin_counts[i] = np.sum(mask)
        if bin_counts[i] > 0:
            bin_confidences[i] = np.mean(pred_probs[mask])
            bin_accuracies[i] = np.mean(correct[mask])

    # Compute ECE (weighted average of |confidence - accuracy|)
    total_samples = len(confs)
    ece = 0.0
    mce = 0.0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            gap = abs(bin_confidences[i] - bin_accuracies[i])
            ece += (bin_counts[i] / total_samples) * gap
            mce = max(mce, gap)

    return CalibrationResult(
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        expected_calibration_error=float(ece),
        maximum_calibration_error=float(mce),
    )


def expected_calibration_error(
    probs: Tensor,
    targets: Tensor,
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the average gap between confidence and accuracy
    across probability bins.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).
        num_bins: Number of bins for calibration computation.

    Returns:
        ECE value (lower is better).
    """
    result = compute_calibration(probs, targets, num_bins)
    return result.expected_calibration_error


def predicted_mean_bps(
    probs: Tensor,
    bucket_centers_bps: Tensor,
) -> Tensor:
    """Compute predicted mean in basis points.

    The predicted mean is the expectation of the price change
    under the predicted distribution.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        bucket_centers_bps: Bucket centers in basis points (num_buckets,).

    Returns:
        Predicted mean in bps of shape (batch,).
    """
    # Weighted sum: E[X] = sum(p_i * x_i)
    return torch.sum(probs * bucket_centers_bps.unsqueeze(0), dim=-1)


def predicted_std_bps(
    probs: Tensor,
    bucket_centers_bps: Tensor,
) -> Tensor:
    """Compute predicted standard deviation in basis points.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        bucket_centers_bps: Bucket centers in basis points (num_buckets,).

    Returns:
        Predicted std in bps of shape (batch,).
    """
    mean = predicted_mean_bps(probs, bucket_centers_bps)
    # E[X^2] = sum(p_i * x_i^2)
    squared = bucket_centers_bps.unsqueeze(0) ** 2
    e_x2 = torch.sum(probs * squared, dim=-1)
    # Var[X] = E[X^2] - E[X]^2
    variance = e_x2 - mean**2
    # Clamp to handle numerical issues
    return torch.sqrt(variance.clamp(min=1e-8))


def directional_accuracy(
    probs: Tensor,
    targets: Tensor,
    bucket_centers_bps: Tensor,
    threshold_bps: float = 0.0,
) -> Tensor:
    """Compute directional accuracy (sign of predicted mean vs actual).

    Measures whether the model correctly predicts the direction
    of price movement (up vs down).

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).
        bucket_centers_bps: Bucket centers in basis points (num_buckets,).
        threshold_bps: Ignore predictions within this threshold of zero.

    Returns:
        Directional accuracy (scalar between 0 and 1).
    """
    pred_mean = predicted_mean_bps(probs, bucket_centers_bps)
    actual_bps = bucket_centers_bps[targets]

    # Compute signs (positive or negative)
    pred_sign = torch.sign(pred_mean)
    actual_sign = torch.sign(actual_bps)

    # Mask out cases where predicted or actual is too close to zero
    mask = (torch.abs(pred_mean) > threshold_bps) & (
        torch.abs(actual_bps) > threshold_bps
    )

    if mask.sum() == 0:
        return torch.tensor(0.5)  # No valid samples

    correct = (pred_sign == actual_sign).float()
    return correct[mask].mean()


def sharpness(
    probs: Tensor,
    bucket_centers_bps: Tensor,
) -> Tensor:
    """Compute sharpness (concentration) of predictions.

    Lower standard deviation means sharper (more confident) predictions.
    Returns the average standard deviation across predictions.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        bucket_centers_bps: Bucket centers in basis points (num_buckets,).

    Returns:
        Mean predicted standard deviation in bps (scalar).
    """
    stds = predicted_std_bps(probs, bucket_centers_bps)
    return stds.mean()


def entropy(probs: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute entropy of probability distributions.

    Higher entropy means more uncertain predictions.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        eps: Small value for numerical stability.

    Returns:
        Entropy values of shape (batch,).
    """
    # H = -sum(p * log(p))
    log_probs = torch.log(probs.clamp(min=eps))
    return -torch.sum(probs * log_probs, dim=-1)


def mean_entropy(probs: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute mean entropy across predictions.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        eps: Small value for numerical stability.

    Returns:
        Mean entropy (scalar).
    """
    return entropy(probs, eps).mean()


@dataclass
class PnLResult:
    """Results from profit/loss simulation.

    Attributes:
        total_pnl_bps: Total P&L in basis points.
        mean_pnl_bps: Mean P&L per trade in basis points.
        win_rate: Fraction of trades with positive P&L.
        sharpe_ratio: Sharpe ratio of returns.
        num_trades: Number of trades taken.
        pnl_series: P&L for each trade.
    """

    total_pnl_bps: float
    mean_pnl_bps: float
    win_rate: float
    sharpe_ratio: float
    num_trades: int
    pnl_series: NDArray[np.floating[Any]]


def simulate_pnl(
    probs: Tensor,
    targets: Tensor,
    bucket_centers_bps: Tensor,
    confidence_threshold: float = 0.0,
    direction_threshold_bps: float = 0.0,
) -> PnLResult:
    """Simulate profit/loss for a simple directional strategy.

    Strategy: If predicted mean exceeds threshold, take a position
    in that direction. P&L is the actual move times position sign.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).
        bucket_centers_bps: Bucket centers in basis points (num_buckets,).
        confidence_threshold: Only trade if max probability exceeds this.
        direction_threshold_bps: Only trade if |predicted_mean| exceeds this.

    Returns:
        PnLResult with simulation statistics.
    """
    pred_mean = predicted_mean_bps(probs, bucket_centers_bps)
    actual_bps = bucket_centers_bps[targets]
    max_prob = probs.max(dim=-1).values

    # Determine which samples to trade
    trade_mask = (max_prob > confidence_threshold) & (
        torch.abs(pred_mean) > direction_threshold_bps
    )

    if trade_mask.sum() == 0:
        return PnLResult(
            total_pnl_bps=0.0,
            mean_pnl_bps=0.0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            num_trades=0,
            pnl_series=np.array([], dtype=np.float64),
        )

    # Compute P&L for each trade
    positions = torch.sign(pred_mean)  # Long if positive, short if negative
    pnl = positions * actual_bps  # P&L per trade

    # Filter to traded samples
    traded_pnl = pnl[trade_mask].detach().cpu().numpy()
    num_trades = len(traded_pnl)

    total_pnl = float(traded_pnl.sum())
    mean_pnl = float(traded_pnl.mean())
    win_rate = float((traded_pnl > 0).sum() / num_trades) if num_trades > 0 else 0.0

    # Compute Sharpe ratio (assuming daily returns, annualize)
    std_pnl = float(traded_pnl.std()) if num_trades > 1 else 0.0
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    return PnLResult(
        total_pnl_bps=total_pnl,
        mean_pnl_bps=mean_pnl,
        win_rate=win_rate,
        sharpe_ratio=sharpe,
        num_trades=num_trades,
        pnl_series=traded_pnl,
    )


@dataclass
class HorizonMetrics:
    """Metrics for a single prediction horizon.

    Attributes:
        horizon: The prediction horizon in seconds.
        nll: Negative log-likelihood.
        brier_score: Brier score.
        ece: Expected calibration error.
        directional_accuracy: Directional accuracy.
        sharpness: Prediction sharpness (std).
        mean_entropy: Mean entropy of predictions.
    """

    horizon: int
    nll: float
    brier_score: float
    ece: float
    directional_accuracy: float
    sharpness: float
    mean_entropy: float


@dataclass
class MultiHorizonMetrics:
    """Metrics aggregated across all prediction horizons.

    Attributes:
        horizons: List of HorizonMetrics for each horizon.
        mean_nll: Mean NLL across horizons.
        mean_brier: Mean Brier score across horizons.
        mean_ece: Mean ECE across horizons.
        mean_directional_accuracy: Mean directional accuracy.
    """

    horizons: list[HorizonMetrics]
    mean_nll: float
    mean_brier: float
    mean_ece: float
    mean_directional_accuracy: float


def compute_horizon_metrics(
    probs: Tensor,
    targets: Tensor,
    horizon: int,
    bucket_centers_bps: Tensor,
    num_calibration_bins: int = 10,
) -> HorizonMetrics:
    """Compute all metrics for a single horizon.

    Args:
        probs: Predicted probabilities of shape (batch, num_buckets).
        targets: Target bucket indices of shape (batch,).
        horizon: The prediction horizon in seconds.
        bucket_centers_bps: Bucket centers in basis points.
        num_calibration_bins: Number of bins for ECE.

    Returns:
        HorizonMetrics with all computed metrics.
    """
    nll = float(negative_log_likelihood(probs, targets, reduction="mean"))
    bs = float(brier_score(probs, targets, reduction="mean"))
    ece = expected_calibration_error(probs, targets, num_calibration_bins)
    da = float(directional_accuracy(probs, targets, bucket_centers_bps))
    sharp = float(sharpness(probs, bucket_centers_bps))
    ent = float(mean_entropy(probs))

    return HorizonMetrics(
        horizon=horizon,
        nll=nll,
        brier_score=bs,
        ece=ece,
        directional_accuracy=da,
        sharpness=sharp,
        mean_entropy=ent,
    )


def compute_multi_horizon_metrics(
    probs_dict: dict[int, Tensor],
    targets_dict: dict[int, Tensor],
    bucket_centers_bps: Tensor,
    num_calibration_bins: int = 10,
) -> MultiHorizonMetrics:
    """Compute metrics across all prediction horizons.

    Args:
        probs_dict: Dict mapping horizon -> probabilities (batch, num_buckets).
        targets_dict: Dict mapping horizon -> targets (batch,).
        bucket_centers_bps: Bucket centers in basis points.
        num_calibration_bins: Number of bins for ECE.

    Returns:
        MultiHorizonMetrics with per-horizon and aggregate metrics.
    """
    horizon_metrics: list[HorizonMetrics] = []

    for horizon in sorted(probs_dict.keys()):
        probs = probs_dict[horizon]
        targets = targets_dict[horizon]
        metrics = compute_horizon_metrics(
            probs, targets, horizon, bucket_centers_bps, num_calibration_bins
        )
        horizon_metrics.append(metrics)

    # Compute means
    mean_nll = np.mean([m.nll for m in horizon_metrics])
    mean_brier = np.mean([m.brier_score for m in horizon_metrics])
    mean_ece = np.mean([m.ece for m in horizon_metrics])
    mean_da = np.mean([m.directional_accuracy for m in horizon_metrics])

    return MultiHorizonMetrics(
        horizons=horizon_metrics,
        mean_nll=float(mean_nll),
        mean_brier=float(mean_brier),
        mean_ece=float(mean_ece),
        mean_directional_accuracy=float(mean_da),
    )


def format_metrics_report(metrics: MultiHorizonMetrics) -> str:
    """Format metrics as a human-readable report.

    Args:
        metrics: Multi-horizon metrics to format.

    Returns:
        Formatted string report.
    """
    lines = ["=" * 60, "Multi-Horizon Evaluation Metrics", "=" * 60, ""]

    # Per-horizon table
    lines.append(
        f"{'Horizon':>10} {'NLL':>8} {'Brier':>8} {'ECE':>8} "
        f"{'Dir Acc':>8} {'Sharpness':>10}"
    )
    lines.append("-" * 60)

    for h in metrics.horizons:
        lines.append(
            f"{h.horizon:>10} {h.nll:>8.4f} {h.brier_score:>8.4f} "
            f"{h.ece:>8.4f} {h.directional_accuracy:>8.4f} {h.sharpness:>10.2f}"
        )

    lines.append("-" * 60)
    lines.append(
        f"{'Mean':>10} {metrics.mean_nll:>8.4f} {metrics.mean_brier:>8.4f} "
        f"{metrics.mean_ece:>8.4f} {metrics.mean_directional_accuracy:>8.4f}"
    )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
