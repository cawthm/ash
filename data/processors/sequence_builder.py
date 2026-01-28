"""Sequence builder for temporal alignment of multi-source data.

This module constructs fixed-length input sequences for the model by:
- Aligning data from multiple sources (stock trades, options) to a common timeline
- Handling asynchronous data arrivals with forward-fill semantics
- Managing overnight gaps and market hours

The sequence builder produces regularly-sampled time series ready for
feature computation and model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class OvernightStrategy(Enum):
    """Strategy for handling overnight gaps in data.

    Attributes:
        RESET: Reset all state at market open (features start fresh).
        MASK: Mark overnight periods as invalid (NaN).
        INTERPOLATE: Interpolate across overnight gap (not recommended).
    """

    RESET = "reset"
    MASK = "mask"
    INTERPOLATE = "interpolate"


@dataclass(frozen=True)
class SequenceConfig:
    """Configuration for sequence construction.

    Attributes:
        lookback_seconds: Total lookback window length in seconds.
        sample_interval: Sampling interval in seconds (time between samples).
        overnight_strategy: How to handle overnight gaps.
        market_open_hour: Market open hour (0-23) in local time.
        market_close_hour: Market close hour (0-23) in local time.
    """

    lookback_seconds: int = 300
    sample_interval: int = 1
    overnight_strategy: OvernightStrategy = OvernightStrategy.RESET
    market_open_hour: int = 9
    market_close_hour: int = 16

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.lookback_seconds < 1:
            raise ValueError("lookback_seconds must be at least 1")
        if self.sample_interval < 1:
            raise ValueError("sample_interval must be at least 1")
        if self.lookback_seconds < self.sample_interval:
            raise ValueError(
                "lookback_seconds must be >= sample_interval"
            )
        if not 0 <= self.market_open_hour < 24:
            raise ValueError("market_open_hour must be in range [0, 24)")
        if not 0 <= self.market_close_hour <= 24:
            raise ValueError("market_close_hour must be in range [0, 24]")

    @property
    def sequence_length(self) -> int:
        """Number of samples in a complete sequence."""
        return self.lookback_seconds // self.sample_interval


@dataclass
class AlignedSequence:
    """Result of sequence alignment containing aligned time series data.

    Attributes:
        timestamps: 1D array of aligned timestamps (Unix seconds).
        prices: 1D array of prices at each timestamp.
        volumes: 1D array of volumes at each timestamp.
        valid_mask: 1D boolean array indicating valid data points.
        options_data: Optional dict of options arrays aligned to timestamps.
    """

    timestamps: NDArray[np.floating[Any]]
    prices: NDArray[np.floating[Any]]
    volumes: NDArray[np.floating[Any]]
    valid_mask: NDArray[np.bool_]
    options_data: dict[str, NDArray[np.floating[Any]]] = field(default_factory=dict)


class SequenceBuilder:
    """Builds aligned sequences from multi-source asynchronous data.

    The sequence builder takes raw tick data and options snapshots,
    aligns them to a regular time grid, and produces fixed-length
    sequences suitable for model input.

    Data alignment uses forward-fill semantics: each grid point contains
    the most recent observation at or before that time.

    Attributes:
        config: Sequence configuration.
    """

    def __init__(self, config: SequenceConfig | None = None) -> None:
        """Initialize sequence builder with configuration.

        Args:
            config: Sequence configuration. Uses defaults if None.
        """
        self.config = config or SequenceConfig()

    def create_time_grid(
        self,
        end_time: float,
    ) -> NDArray[np.floating[Any]]:
        """Create a regular time grid ending at the specified time.

        Args:
            end_time: End timestamp (Unix seconds) for the grid.

        Returns:
            1D array of timestamps at regular intervals.
        """
        n = self.config.sequence_length
        interval = float(self.config.sample_interval)

        # Grid ends at end_time and goes back lookback_seconds
        grid = np.arange(n, dtype=np.float64) * interval
        grid = end_time - (n - 1) * interval + grid

        return grid

    def forward_fill_to_grid(
        self,
        timestamps: NDArray[np.floating[Any]],
        values: NDArray[np.floating[Any]],
        grid: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.bool_]]:
        """Align values to a time grid using forward-fill.

        For each grid point, finds the most recent observation at or before
        that time and uses its value. Grid points before any observation
        are marked as invalid.

        Args:
            timestamps: 1D array of observation timestamps (must be sorted).
            values: 1D array of observation values.
            grid: 1D array of target timestamps.

        Returns:
            Tuple of (aligned_values, valid_mask).
            - aligned_values: Values aligned to grid (NaN where invalid).
            - valid_mask: Boolean mask indicating valid grid points.
        """
        timestamps = np.asarray(timestamps, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        grid = np.asarray(grid, dtype=np.float64)

        if timestamps.ndim != 1 or values.ndim != 1:
            raise ValueError("timestamps and values must be 1-dimensional")
        if len(timestamps) != len(values):
            raise ValueError("timestamps and values must have same length")
        if grid.ndim != 1:
            raise ValueError("grid must be 1-dimensional")

        n_grid = len(grid)
        result = np.full(n_grid, np.nan, dtype=np.float64)
        valid = np.zeros(n_grid, dtype=np.bool_)

        if len(timestamps) == 0:
            return result, valid

        # Use searchsorted to find insertion points
        # For each grid point, find index of first timestamp > grid point
        # Then use the previous index (last timestamp <= grid point)
        indices = np.searchsorted(timestamps, grid, side="right") - 1

        # Valid if there's at least one observation before this grid point
        valid = indices >= 0
        valid_indices = np.where(valid)[0]

        if len(valid_indices) > 0:
            result[valid_indices] = values[indices[valid_indices]]

        return result, valid

    def forward_fill_2d_to_grid(
        self,
        timestamps: NDArray[np.floating[Any]],
        values: NDArray[np.floating[Any]],
        grid: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.bool_]]:
        """Align 2D values to a time grid using forward-fill.

        Same as forward_fill_to_grid but for 2D value arrays where each
        row corresponds to a timestamp.

        Args:
            timestamps: 1D array of observation timestamps (must be sorted).
            values: 2D array of shape (n_timestamps, n_features).
            grid: 1D array of target timestamps.

        Returns:
            Tuple of (aligned_values, valid_mask).
            - aligned_values: 2D array of shape (n_grid, n_features).
            - valid_mask: Boolean mask indicating valid grid points.
        """
        timestamps = np.asarray(timestamps, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        grid = np.asarray(grid, dtype=np.float64)

        if timestamps.ndim != 1:
            raise ValueError("timestamps must be 1-dimensional")
        if values.ndim != 2:
            raise ValueError("values must be 2-dimensional")
        if len(timestamps) != len(values):
            raise ValueError("timestamps and values must have same length")
        if grid.ndim != 1:
            raise ValueError("grid must be 1-dimensional")

        n_grid = len(grid)
        n_features = values.shape[1]
        result = np.full((n_grid, n_features), np.nan, dtype=np.float64)
        valid = np.zeros(n_grid, dtype=np.bool_)

        if len(timestamps) == 0:
            return result, valid

        indices = np.searchsorted(timestamps, grid, side="right") - 1
        valid = indices >= 0
        valid_indices = np.where(valid)[0]

        if len(valid_indices) > 0:
            result[valid_indices] = values[indices[valid_indices]]

        return result, valid

    def detect_overnight_gaps(
        self,
        timestamps: NDArray[np.floating[Any]],
    ) -> NDArray[np.bool_]:
        """Detect overnight gaps in timestamp sequence.

        An overnight gap is detected when the time difference between
        consecutive samples exceeds a threshold (default: 4 hours).

        Args:
            timestamps: 1D array of timestamps.

        Returns:
            Boolean array marking positions after overnight gaps.
        """
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if len(timestamps) < 2:
            return np.zeros(len(timestamps), dtype=np.bool_)

        # Gap threshold: 4 hours in seconds
        gap_threshold = 4 * 3600

        diffs = np.diff(timestamps)
        gap_mask = np.zeros(len(timestamps), dtype=np.bool_)
        gap_mask[1:] = diffs > gap_threshold

        return gap_mask

    def apply_overnight_strategy(
        self,
        data: NDArray[np.floating[Any]],
        gap_mask: NDArray[np.bool_],
    ) -> NDArray[np.floating[Any]]:
        """Apply overnight gap handling strategy to data.

        Args:
            data: 1D or 2D array of data values.
            gap_mask: Boolean mask indicating positions after overnight gaps.

        Returns:
            Data with overnight strategy applied.
        """
        data = np.asarray(data, dtype=np.float64).copy()

        if self.config.overnight_strategy == OvernightStrategy.RESET:
            # Mark all positions before the gap as invalid
            # This resets feature calculations at each market open
            if np.any(gap_mask):
                gap_indices = np.where(gap_mask)[0]
                for gap_idx in gap_indices:
                    data[:gap_idx] = np.nan
        elif self.config.overnight_strategy == OvernightStrategy.MASK:
            # Just mark the gap positions as invalid
            if data.ndim == 1:
                data[gap_mask] = np.nan
            else:
                data[gap_mask, :] = np.nan
        # INTERPOLATE: do nothing, let values propagate through
        return data

    def align_trade_data(
        self,
        trade_timestamps: NDArray[np.floating[Any]],
        trade_prices: NDArray[np.floating[Any]],
        trade_volumes: NDArray[np.floating[Any]],
        end_time: float,
    ) -> AlignedSequence:
        """Align trade data to a regular time grid.

        Args:
            trade_timestamps: 1D array of trade timestamps (Unix seconds).
            trade_prices: 1D array of trade prices.
            trade_volumes: 1D array of trade volumes.
            end_time: End timestamp for the sequence.

        Returns:
            AlignedSequence with prices and volumes aligned to time grid.
        """
        trade_timestamps = np.asarray(trade_timestamps, dtype=np.float64)
        trade_prices = np.asarray(trade_prices, dtype=np.float64)
        trade_volumes = np.asarray(trade_volumes, dtype=np.float64)

        if not (len(trade_timestamps) == len(trade_prices) == len(trade_volumes)):
            raise ValueError(
                "trade_timestamps, trade_prices, and trade_volumes must have same length"
            )

        # Create time grid
        grid = self.create_time_grid(end_time)

        # Align prices (forward-fill)
        aligned_prices, price_valid = self.forward_fill_to_grid(
            trade_timestamps, trade_prices, grid
        )

        # For volumes, we aggregate within each interval rather than forward-fill
        # This gives total volume traded in each time bucket
        aligned_volumes = self._aggregate_volumes_to_grid(
            trade_timestamps, trade_volumes, grid
        )

        # Detect overnight gaps
        gap_mask = self.detect_overnight_gaps(grid)

        # Apply overnight strategy
        aligned_prices = self.apply_overnight_strategy(aligned_prices, gap_mask)
        aligned_volumes = self.apply_overnight_strategy(aligned_volumes, gap_mask)

        # Update valid mask based on overnight strategy
        if self.config.overnight_strategy in (
            OvernightStrategy.RESET,
            OvernightStrategy.MASK,
        ):
            valid_mask = price_valid & ~np.isnan(aligned_prices)
        else:
            valid_mask = price_valid

        return AlignedSequence(
            timestamps=grid,
            prices=aligned_prices,
            volumes=aligned_volumes,
            valid_mask=valid_mask,
        )

    def _aggregate_volumes_to_grid(
        self,
        timestamps: NDArray[np.floating[Any]],
        volumes: NDArray[np.floating[Any]],
        grid: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Aggregate volumes into time buckets.

        For each grid interval [t, t+interval), sum all volumes with
        timestamps in that range.

        Args:
            timestamps: 1D array of trade timestamps.
            volumes: 1D array of trade volumes.
            grid: 1D array of grid timestamps.

        Returns:
            1D array of aggregated volumes for each grid bucket.
        """
        n_grid = len(grid)
        result = np.zeros(n_grid, dtype=np.float64)

        if len(timestamps) == 0:
            return result

        interval = float(self.config.sample_interval)

        # For each grid point, find trades in [grid[i], grid[i]+interval)
        for i in range(n_grid):
            bucket_start = grid[i]
            bucket_end = bucket_start + interval
            mask = (timestamps >= bucket_start) & (timestamps < bucket_end)
            result[i] = np.sum(volumes[mask])

        return result

    def align_options_data(
        self,
        options_timestamps: NDArray[np.floating[Any]],
        options_values: dict[str, NDArray[np.floating[Any]]],
        grid: NDArray[np.floating[Any]],
    ) -> dict[str, NDArray[np.floating[Any]]]:
        """Align options data to a time grid.

        Options data is typically snapshot-based (less frequent than trades),
        so forward-fill is used to propagate the last known snapshot.

        Args:
            options_timestamps: 1D array of snapshot timestamps.
            options_values: Dict mapping field names to value arrays.
                Each value array can be 1D or 2D.
            grid: Target time grid.

        Returns:
            Dict with aligned values for each field.
        """
        aligned: dict[str, NDArray[np.floating[Any]]] = {}

        for field_name, values in options_values.items():
            values = np.asarray(values, dtype=np.float64)

            if values.ndim == 1:
                aligned_vals, _ = self.forward_fill_to_grid(
                    options_timestamps, values, grid
                )
            elif values.ndim == 2:
                aligned_vals, _ = self.forward_fill_2d_to_grid(
                    options_timestamps, values, grid
                )
            else:
                raise ValueError(f"Values for {field_name} must be 1D or 2D")

            aligned[field_name] = aligned_vals

        return aligned

    def build_sequence(
        self,
        trade_timestamps: NDArray[np.floating[Any]],
        trade_prices: NDArray[np.floating[Any]],
        trade_volumes: NDArray[np.floating[Any]],
        end_time: float,
        options_timestamps: NDArray[np.floating[Any]] | None = None,
        options_values: dict[str, NDArray[np.floating[Any]]] | None = None,
    ) -> AlignedSequence:
        """Build a complete aligned sequence from multi-source data.

        This is the main entry point for sequence construction. It:
        1. Creates a regular time grid ending at end_time
        2. Aligns trade data using forward-fill for prices, aggregation for volumes
        3. Aligns options data using forward-fill
        4. Applies overnight gap handling

        Args:
            trade_timestamps: Trade timestamps (Unix seconds, sorted).
            trade_prices: Trade prices.
            trade_volumes: Trade volumes.
            end_time: End timestamp for the sequence.
            options_timestamps: Optional options snapshot timestamps.
            options_values: Optional dict of options data arrays.

        Returns:
            AlignedSequence containing all aligned data.
        """
        # First align trade data
        sequence = self.align_trade_data(
            trade_timestamps, trade_prices, trade_volumes, end_time
        )

        # Then align options data if provided
        if options_timestamps is not None and options_values is not None:
            aligned_options = self.align_options_data(
                options_timestamps, options_values, sequence.timestamps
            )
            sequence = AlignedSequence(
                timestamps=sequence.timestamps,
                prices=sequence.prices,
                volumes=sequence.volumes,
                valid_mask=sequence.valid_mask,
                options_data=aligned_options,
            )

        return sequence

    def get_valid_range(
        self,
        sequence: AlignedSequence,
        min_valid_fraction: float = 0.8,
    ) -> tuple[int, int] | None:
        """Find the longest contiguous valid range in a sequence.

        Args:
            sequence: The aligned sequence.
            min_valid_fraction: Minimum fraction of valid data required.

        Returns:
            Tuple of (start_idx, end_idx) for the valid range, or None if
            no range meets the minimum valid fraction requirement.
        """
        valid = sequence.valid_mask
        n = len(valid)

        if n == 0:
            return None

        # Find contiguous valid regions
        # Pad with False to detect boundaries
        padded = np.concatenate([[False], valid, [False]])
        diffs = np.diff(padded.astype(np.int8))

        # starts: indices where valid regions begin
        # ends: indices where valid regions end
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        if len(starts) == 0:
            return None

        # Find longest region
        lengths = ends - starts
        best_idx = np.argmax(lengths)
        best_start = starts[best_idx]
        best_end = ends[best_idx]
        best_length = lengths[best_idx]

        # Check minimum valid fraction
        if best_length / n < min_valid_fraction:
            return None

        return int(best_start), int(best_end)
