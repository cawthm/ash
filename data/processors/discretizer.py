"""Discretizer for mapping continuous values to discrete buckets.

This module implements log-return based discretization for price predictions.
The buckets are symmetric around 0%, using basis points (1 bp = 0.01%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BucketConfig:
    """Configuration for discretization buckets.

    Attributes:
        num_buckets: Total number of buckets (must be odd for symmetric around 0).
        min_bps: Minimum value in basis points (e.g., -50 for -0.5%).
        max_bps: Maximum value in basis points (e.g., +50 for +0.5%).
    """

    num_buckets: int = 101
    min_bps: float = -50.0
    max_bps: float = 50.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_buckets < 3:
            raise ValueError("num_buckets must be at least 3")
        if self.num_buckets % 2 == 0:
            raise ValueError("num_buckets must be odd for symmetric buckets around 0")
        if self.min_bps >= self.max_bps:
            raise ValueError("min_bps must be less than max_bps")
        if self.min_bps > 0 or self.max_bps < 0:
            raise ValueError("Bucket range must include 0")


class LogReturnDiscretizer:
    """Discretizes log returns into buckets.

    Uses log-return (percentage change) buckets centered on 0%, symmetric
    around price changes. Values outside the bucket range are clipped to
    the edge buckets.

    Attributes:
        config: Bucket configuration.
        boundaries: Array of bucket boundaries in basis points.
        centers: Array of bucket centers in basis points.
    """

    def __init__(self, config: BucketConfig | None = None) -> None:
        """Initialize discretizer with configuration.

        Args:
            config: Bucket configuration. Uses defaults if None.
        """
        self.config = config or BucketConfig()
        self.boundaries = self._compute_boundaries()
        self.centers = self._compute_centers()

    def _compute_boundaries(self) -> NDArray[np.floating[Any]]:
        """Compute bucket boundaries.

        Returns:
            Array of shape (num_buckets + 1,) with bucket boundaries.
        """
        # Create evenly spaced boundaries from min to max
        return np.linspace(
            self.config.min_bps,
            self.config.max_bps,
            self.config.num_buckets + 1,
            dtype=np.float64,
        )

    def _compute_centers(self) -> NDArray[np.floating[Any]]:
        """Compute bucket centers.

        Returns:
            Array of shape (num_buckets,) with bucket centers.
        """
        # Center of each bucket is average of its boundaries
        return (self.boundaries[:-1] + self.boundaries[1:]) / 2.0

    def price_to_log_return_bps(
        self,
        current_price: NDArray[np.floating[Any]] | float,
        reference_price: NDArray[np.floating[Any]] | float,
    ) -> NDArray[np.floating[Any]]:
        """Convert prices to log returns in basis points.

        Args:
            current_price: Current price(s).
            reference_price: Reference price(s) for computing return.

        Returns:
            Log return(s) in basis points.
        """
        current = np.asarray(current_price, dtype=np.float64)
        reference = np.asarray(reference_price, dtype=np.float64)

        if np.any(reference <= 0):
            raise ValueError("Reference price must be positive")
        if np.any(current <= 0):
            raise ValueError("Current price must be positive")

        # Log return in basis points (1 bp = 0.01% = 0.0001)
        log_return = np.log(current / reference)
        return log_return * 10000.0  # Convert to basis points

    def discretize(
        self,
        values_bps: NDArray[np.floating[Any]] | float,
    ) -> NDArray[np.intp]:
        """Map values in basis points to bucket indices.

        Values below min_bps map to bucket 0.
        Values above max_bps map to bucket (num_buckets - 1).

        Args:
            values_bps: Value(s) in basis points to discretize.

        Returns:
            Bucket index/indices (0 to num_buckets - 1).
        """
        values = np.asarray(values_bps, dtype=np.float64)

        # np.searchsorted returns index where value would be inserted
        # We subtract 1 because boundaries[i] is left edge of bucket i
        # Values exactly on boundary go to higher bucket, except rightmost
        indices = np.searchsorted(self.boundaries[1:], values, side="left")

        # Clip to valid range [0, num_buckets - 1]
        return np.clip(indices, 0, self.config.num_buckets - 1)

    def discretize_prices(
        self,
        current_prices: NDArray[np.floating[Any]] | float,
        reference_prices: NDArray[np.floating[Any]] | float,
    ) -> NDArray[np.intp]:
        """Discretize price changes directly.

        Convenience method that combines price-to-return and discretization.

        Args:
            current_prices: Current price(s).
            reference_prices: Reference price(s).

        Returns:
            Bucket index/indices.
        """
        bps = self.price_to_log_return_bps(current_prices, reference_prices)
        return self.discretize(bps)

    def bucket_to_bps(
        self,
        bucket_indices: NDArray[np.intp] | int,
    ) -> NDArray[np.floating[Any]]:
        """Convert bucket indices back to basis point values (bucket centers).

        Args:
            bucket_indices: Bucket index/indices.

        Returns:
            Basis point value(s) at bucket center(s).
        """
        indices = np.asarray(bucket_indices)
        if np.any(indices < 0) or np.any(indices >= self.config.num_buckets):
            raise ValueError(
                f"Bucket indices must be in [0, {self.config.num_buckets - 1}]"
            )
        result: NDArray[np.floating[Any]] = self.centers[indices]
        return result

    def get_zero_bucket(self) -> int:
        """Get the index of the bucket containing 0.

        Returns:
            Index of the center bucket (containing 0 return).
        """
        return self.config.num_buckets // 2

    def create_soft_labels(
        self,
        true_bucket: int,
        sigma: float = 1.0,
    ) -> NDArray[np.floating[Any]]:
        """Create soft labels with Gaussian smoothing around true bucket.

        Useful for ordinal-aware training where nearby buckets should
        receive partial credit.

        Args:
            true_bucket: The true bucket index.
            sigma: Standard deviation of Gaussian in bucket units.

        Returns:
            Array of shape (num_buckets,) with soft label probabilities.
        """
        if true_bucket < 0 or true_bucket >= self.config.num_buckets:
            raise ValueError(
                f"true_bucket must be in [0, {self.config.num_buckets - 1}]"
            )

        bucket_indices = np.arange(self.config.num_buckets, dtype=np.float64)
        # Gaussian centered on true_bucket
        distances = bucket_indices - true_bucket
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        # Normalize to sum to 1
        result: NDArray[np.floating[Any]] = weights / weights.sum()
        return result
