"""Discretizer for mapping continuous values to discrete buckets.

This module implements discretization for:
- Log-return based price predictions (symmetric around 0%, using basis points)
- Volatility discretization (implied and realized volatility)
- Volume discretization (relative to historical average)
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


@dataclass(frozen=True)
class VolatilityConfig:
    """Configuration for volatility discretization buckets.

    Attributes:
        num_buckets: Total number of buckets.
        min_vol: Minimum volatility percentage (e.g., 5.0 for 5%).
        max_vol: Maximum volatility percentage (e.g., 150.0 for 150%).
    """

    num_buckets: int = 101
    min_vol: float = 5.0
    max_vol: float = 150.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_buckets < 3:
            raise ValueError("num_buckets must be at least 3")
        if self.min_vol < 0:
            raise ValueError("min_vol must be non-negative")
        if self.min_vol >= self.max_vol:
            raise ValueError("min_vol must be less than max_vol")


class VolatilityDiscretizer:
    """Discretizes volatility values (IV or RV) into buckets.

    Volatility is expressed as annualized percentage (e.g., 20.0 for 20%).
    Values outside the bucket range are clipped to the edge buckets.

    Attributes:
        config: Volatility bucket configuration.
        boundaries: Array of bucket boundaries in volatility percentage.
        centers: Array of bucket centers in volatility percentage.
    """

    def __init__(self, config: VolatilityConfig | None = None) -> None:
        """Initialize discretizer with configuration.

        Args:
            config: Volatility bucket configuration. Uses defaults if None.
        """
        self.config = config or VolatilityConfig()
        self.boundaries = self._compute_boundaries()
        self.centers = self._compute_centers()

    def _compute_boundaries(self) -> NDArray[np.floating[Any]]:
        """Compute bucket boundaries.

        Returns:
            Array of shape (num_buckets + 1,) with bucket boundaries.
        """
        return np.linspace(
            self.config.min_vol,
            self.config.max_vol,
            self.config.num_buckets + 1,
            dtype=np.float64,
        )

    def _compute_centers(self) -> NDArray[np.floating[Any]]:
        """Compute bucket centers.

        Returns:
            Array of shape (num_buckets,) with bucket centers.
        """
        return (self.boundaries[:-1] + self.boundaries[1:]) / 2.0

    def discretize(
        self,
        volatility: NDArray[np.floating[Any]] | float,
    ) -> NDArray[np.intp]:
        """Map volatility values to bucket indices.

        Values below min_vol map to bucket 0.
        Values above max_vol map to bucket (num_buckets - 1).

        Args:
            volatility: Volatility value(s) as percentage (e.g., 20.0 for 20%).

        Returns:
            Bucket index/indices (0 to num_buckets - 1).
        """
        values = np.asarray(volatility, dtype=np.float64)
        indices = np.searchsorted(self.boundaries[1:], values, side="left")
        return np.clip(indices, 0, self.config.num_buckets - 1)

    def bucket_to_volatility(
        self,
        bucket_indices: NDArray[np.intp] | int,
    ) -> NDArray[np.floating[Any]]:
        """Convert bucket indices back to volatility values (bucket centers).

        Args:
            bucket_indices: Bucket index/indices.

        Returns:
            Volatility value(s) at bucket center(s) as percentage.
        """
        indices = np.asarray(bucket_indices)
        if np.any(indices < 0) or np.any(indices >= self.config.num_buckets):
            raise ValueError(
                f"Bucket indices must be in [0, {self.config.num_buckets - 1}]"
            )
        result: NDArray[np.floating[Any]] = self.centers[indices]
        return result

    def create_soft_labels(
        self,
        true_bucket: int,
        sigma: float = 1.0,
    ) -> NDArray[np.floating[Any]]:
        """Create soft labels with Gaussian smoothing around true bucket.

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
        distances = bucket_indices - true_bucket
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        result: NDArray[np.floating[Any]] = weights / weights.sum()
        return result


@dataclass(frozen=True)
class VolumeConfig:
    """Configuration for volume discretization buckets.

    Volume is expressed as a ratio relative to historical average
    (e.g., 1.0 = 100% of average, 2.0 = 200% of average).

    Attributes:
        num_buckets: Total number of buckets.
        min_ratio: Minimum volume ratio (e.g., 0.1 for 10% of average).
        max_ratio: Maximum volume ratio (e.g., 5.0 for 500% of average).
        log_scale: If True, use logarithmic spacing for buckets.
    """

    num_buckets: int = 101
    min_ratio: float = 0.1
    max_ratio: float = 5.0
    log_scale: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_buckets < 3:
            raise ValueError("num_buckets must be at least 3")
        if self.min_ratio <= 0:
            raise ValueError("min_ratio must be positive")
        if self.min_ratio >= self.max_ratio:
            raise ValueError("min_ratio must be less than max_ratio")


class VolumeDiscretizer:
    """Discretizes relative volume into buckets.

    Volume is expressed as a ratio relative to historical average.
    By default, uses logarithmic spacing which is natural for ratios
    (e.g., 0.5x and 2x are equidistant from 1x in log space).

    Values outside the bucket range are clipped to the edge buckets.

    Attributes:
        config: Volume bucket configuration.
        boundaries: Array of bucket boundaries as volume ratios.
        centers: Array of bucket centers as volume ratios.
    """

    def __init__(self, config: VolumeConfig | None = None) -> None:
        """Initialize discretizer with configuration.

        Args:
            config: Volume bucket configuration. Uses defaults if None.
        """
        self.config = config or VolumeConfig()
        self.boundaries = self._compute_boundaries()
        self.centers = self._compute_centers()

    def _compute_boundaries(self) -> NDArray[np.floating[Any]]:
        """Compute bucket boundaries.

        Returns:
            Array of shape (num_buckets + 1,) with bucket boundaries.
        """
        if self.config.log_scale:
            # Logarithmic spacing
            return np.geomspace(
                self.config.min_ratio,
                self.config.max_ratio,
                self.config.num_buckets + 1,
                dtype=np.float64,
            )
        else:
            # Linear spacing
            return np.linspace(
                self.config.min_ratio,
                self.config.max_ratio,
                self.config.num_buckets + 1,
                dtype=np.float64,
            )

    def _compute_centers(self) -> NDArray[np.floating[Any]]:
        """Compute bucket centers.

        For log-scale, uses geometric mean; for linear, uses arithmetic mean.

        Returns:
            Array of shape (num_buckets,) with bucket centers.
        """
        if self.config.log_scale:
            # Geometric mean for log-scale
            return np.sqrt(self.boundaries[:-1] * self.boundaries[1:])
        else:
            # Arithmetic mean for linear scale
            return (self.boundaries[:-1] + self.boundaries[1:]) / 2.0

    def discretize(
        self,
        volume_ratio: NDArray[np.floating[Any]] | float,
    ) -> NDArray[np.intp]:
        """Map volume ratio values to bucket indices.

        Values below min_ratio map to bucket 0.
        Values above max_ratio map to bucket (num_buckets - 1).

        Args:
            volume_ratio: Volume ratio(s) (e.g., 1.5 for 150% of average).

        Returns:
            Bucket index/indices (0 to num_buckets - 1).
        """
        values = np.asarray(volume_ratio, dtype=np.float64)
        indices = np.searchsorted(self.boundaries[1:], values, side="left")
        return np.clip(indices, 0, self.config.num_buckets - 1)

    def get_average_bucket(self) -> int:
        """Get the bucket index that contains the average (ratio = 1.0).

        Returns:
            Bucket index containing ratio 1.0, or closest bucket if 1.0
            is outside the configured range.
        """
        if self.config.min_ratio > 1.0:
            return 0
        if self.config.max_ratio < 1.0:
            return self.config.num_buckets - 1
        return int(self.discretize(1.0))

    def bucket_to_ratio(
        self,
        bucket_indices: NDArray[np.intp] | int,
    ) -> NDArray[np.floating[Any]]:
        """Convert bucket indices back to volume ratio values (bucket centers).

        Args:
            bucket_indices: Bucket index/indices.

        Returns:
            Volume ratio value(s) at bucket center(s).
        """
        indices = np.asarray(bucket_indices)
        if np.any(indices < 0) or np.any(indices >= self.config.num_buckets):
            raise ValueError(
                f"Bucket indices must be in [0, {self.config.num_buckets - 1}]"
            )
        result: NDArray[np.floating[Any]] = self.centers[indices]
        return result

    def create_soft_labels(
        self,
        true_bucket: int,
        sigma: float = 1.0,
    ) -> NDArray[np.floating[Any]]:
        """Create soft labels with Gaussian smoothing around true bucket.

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
        distances = bucket_indices - true_bucket
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        result: NDArray[np.floating[Any]] = weights / weights.sum()
        return result
