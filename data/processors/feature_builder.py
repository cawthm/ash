"""Feature builder for computing model input features.

This module implements feature computation for:
- Price history features (returns at multiple lookback windows, VWAP)
- Volatility features (realized volatility, IV/RV spread)
- Order flow features (trade direction imbalance, size distribution, arrival rate)
- Options features (IV surface, Greeks, put/call ratios, term structure)

Features are computed from raw price/volume data and produce fixed-length
feature vectors suitable for model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PriceFeatureConfig:
    """Configuration for price history features.

    Attributes:
        return_windows: Lookback windows (in seconds) for computing returns.
        include_vwap: Whether to include VWAP deviation feature.
        sample_interval: Sampling interval in seconds (for aligning windows).
    """

    return_windows: tuple[int, ...] = (1, 5, 10, 30, 60, 120, 300)
    include_vwap: bool = True
    sample_interval: int = 1


@dataclass(frozen=True)
class VolatilityFeatureConfig:
    """Configuration for volatility features.

    Attributes:
        rv_windows: Windows (in seconds) for realized volatility calculation.
        include_iv_rv_spread: Whether to include IV/RV spread feature.
        include_vol_of_vol: Whether to include volatility of volatility.
        annualization_factor: Factor to annualize volatility (sqrt of trading seconds/year).
    """

    rv_windows: tuple[int, ...] = (60, 300, 600)
    include_iv_rv_spread: bool = True
    include_vol_of_vol: bool = True
    annualization_factor: float = field(default_factory=lambda: np.sqrt(252 * 6.5 * 3600))


class PriceFeatureBuilder:
    """Computes price history features from raw price and volume data.

    Features computed:
    - Log returns at multiple lookback windows
    - VWAP deviation (optional)

    All features are computed from aligned time series data where each row
    represents a fixed time interval.

    Attributes:
        config: Price feature configuration.
    """

    def __init__(self, config: PriceFeatureConfig | None = None) -> None:
        """Initialize feature builder with configuration.

        Args:
            config: Price feature configuration. Uses defaults if None.
        """
        self.config = config or PriceFeatureConfig()

    def compute_log_returns(
        self,
        prices: NDArray[np.floating[Any]],
        window: int,
    ) -> NDArray[np.floating[Any]]:
        """Compute log returns over a lookback window.

        Args:
            prices: 1D array of prices, ordered from oldest to newest.
            window: Lookback window in number of samples.

        Returns:
            Log returns for each position where the window is valid.
            Positions where window extends before data start are NaN.
        """
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim != 1:
            raise ValueError("prices must be 1-dimensional")
        if window < 1:
            raise ValueError("window must be at least 1")

        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)

        if window >= n:
            return result

        # Current prices and lagged prices
        current = prices[window:]
        lagged = prices[:-window]

        # Log return = ln(current / lagged)
        # Handle zero/negative prices by leaving as NaN
        valid_mask = (current > 0) & (lagged > 0)
        valid_indices = np.where(valid_mask)[0] + window
        result[valid_indices] = np.log(current[valid_mask] / lagged[valid_mask])

        return result

    def compute_returns_multi_window(
        self,
        prices: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Compute log returns at all configured lookback windows.

        Args:
            prices: 1D array of prices, ordered from oldest to newest.

        Returns:
            2D array of shape (len(prices), num_windows) with log returns.
            Each column corresponds to a window from config.return_windows.
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)
        num_windows = len(self.config.return_windows)

        result = np.full((n, num_windows), np.nan, dtype=np.float64)

        for i, window in enumerate(self.config.return_windows):
            # Convert window from seconds to samples
            window_samples = window // self.config.sample_interval
            if window_samples >= 1:
                result[:, i] = self.compute_log_returns(prices, window_samples)

        return result

    def compute_vwap(
        self,
        prices: NDArray[np.floating[Any]],
        volumes: NDArray[np.floating[Any]],
        window: int,
    ) -> NDArray[np.floating[Any]]:
        """Compute Volume Weighted Average Price over a rolling window.

        Args:
            prices: 1D array of prices.
            volumes: 1D array of volumes, same length as prices.
            window: Rolling window size in samples.

        Returns:
            VWAP values. Positions where window extends before data start are NaN.
        """
        prices = np.asarray(prices, dtype=np.float64)
        volumes = np.asarray(volumes, dtype=np.float64)

        if prices.ndim != 1 or volumes.ndim != 1:
            raise ValueError("prices and volumes must be 1-dimensional")
        if len(prices) != len(volumes):
            raise ValueError("prices and volumes must have same length")
        if window < 1:
            raise ValueError("window must be at least 1")

        n = len(prices)
        result = np.full(n, np.nan, dtype=np.float64)

        if window > n:
            return result

        # Price * Volume for each sample
        pv = prices * volumes

        # Rolling sums using cumsum for efficiency
        cumsum_pv = np.cumsum(pv)
        cumsum_v = np.cumsum(volumes)

        # For positions >= window, compute rolling sum
        # rolling_sum[i] = cumsum[i] - cumsum[i-window]
        result[window - 1] = cumsum_pv[window - 1] / max(cumsum_v[window - 1], 1e-10)

        if window < n:
            rolling_pv = cumsum_pv[window:] - cumsum_pv[:-window]
            rolling_v = cumsum_v[window:] - cumsum_v[:-window]
            # Avoid division by zero
            valid_v = np.maximum(rolling_v, 1e-10)
            result[window:] = rolling_pv / valid_v

        return result

    def compute_vwap_deviation(
        self,
        prices: NDArray[np.floating[Any]],
        volumes: NDArray[np.floating[Any]],
        window: int,
    ) -> NDArray[np.floating[Any]]:
        """Compute price deviation from VWAP as a log ratio.

        Args:
            prices: 1D array of prices.
            volumes: 1D array of volumes.
            window: Rolling window for VWAP calculation.

        Returns:
            Log ratio of current price to VWAP: ln(price / vwap).
            Positions where VWAP is invalid are NaN.
        """
        vwap = self.compute_vwap(prices, volumes, window)

        result = np.full(len(prices), np.nan, dtype=np.float64)
        valid_mask = (vwap > 0) & (np.asarray(prices) > 0) & ~np.isnan(vwap)
        result[valid_mask] = np.log(np.asarray(prices)[valid_mask] / vwap[valid_mask])

        return result

    def compute_features(
        self,
        prices: NDArray[np.floating[Any]],
        volumes: NDArray[np.floating[Any]] | None = None,
        vwap_window: int = 60,
    ) -> NDArray[np.floating[Any]]:
        """Compute all price history features.

        Args:
            prices: 1D array of prices.
            volumes: 1D array of volumes (required if include_vwap is True).
            vwap_window: Window size for VWAP calculation in samples.

        Returns:
            2D array of shape (len(prices), num_features) with all features.
            Feature order: [return_windows..., vwap_deviation (if enabled)]
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)

        # Returns at multiple windows
        returns = self.compute_returns_multi_window(prices)
        features_list: list[NDArray[np.floating[Any]]] = [returns]

        # VWAP deviation
        if self.config.include_vwap:
            if volumes is None:
                raise ValueError("volumes required when include_vwap is True")
            vwap_dev = self.compute_vwap_deviation(prices, volumes, vwap_window)
            features_list.append(vwap_dev.reshape(n, 1))

        return np.hstack(features_list)

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names.

        Returns:
            List of feature names corresponding to columns in compute_features output.
        """
        names = [f"log_return_{w}s" for w in self.config.return_windows]
        if self.config.include_vwap:
            names.append("vwap_deviation")
        return names

    @property
    def num_features(self) -> int:
        """Total number of features produced by compute_features."""
        count = len(self.config.return_windows)
        if self.config.include_vwap:
            count += 1
        return count


class VolatilityFeatureBuilder:
    """Computes volatility features from price data.

    Features computed:
    - Realized volatility at multiple windows (annualized)
    - Volatility of volatility (optional)

    Note: IV/RV spread requires implied volatility data which must be
    provided separately.

    Attributes:
        config: Volatility feature configuration.
    """

    def __init__(self, config: VolatilityFeatureConfig | None = None) -> None:
        """Initialize feature builder with configuration.

        Args:
            config: Volatility feature configuration. Uses defaults if None.
        """
        self.config = config or VolatilityFeatureConfig()

    def compute_realized_volatility(
        self,
        log_returns: NDArray[np.floating[Any]],
        window: int,
    ) -> NDArray[np.floating[Any]]:
        """Compute realized volatility over a rolling window.

        Uses standard deviation of log returns, annualized.

        Args:
            log_returns: 1D array of log returns (not percentage).
            window: Rolling window size in samples.

        Returns:
            Annualized realized volatility as percentage (e.g., 20.0 for 20%).
            Positions with insufficient data are NaN.
        """
        log_returns = np.asarray(log_returns, dtype=np.float64)
        if log_returns.ndim != 1:
            raise ValueError("log_returns must be 1-dimensional")
        if window < 2:
            raise ValueError("window must be at least 2 for volatility calculation")

        n = len(log_returns)
        result = np.full(n, np.nan, dtype=np.float64)

        if window > n:
            return result

        # Rolling standard deviation using cumsum approach for efficiency
        # Var(X) = E[X^2] - E[X]^2
        cumsum = np.cumsum(log_returns)
        cumsum_sq = np.cumsum(log_returns**2)

        # First valid position
        mean_first = cumsum[window - 1] / window
        mean_sq_first = cumsum_sq[window - 1] / window
        var_first = mean_sq_first - mean_first**2
        result[window - 1] = np.sqrt(max(var_first, 0)) * self.config.annualization_factor

        # Remaining positions
        if window < n:
            rolling_sum = cumsum[window:] - cumsum[:-window]
            rolling_sum_sq = cumsum_sq[window:] - cumsum_sq[:-window]
            rolling_mean = rolling_sum / window
            rolling_mean_sq = rolling_sum_sq / window
            rolling_var = rolling_mean_sq - rolling_mean**2
            # Handle numerical issues that could give small negative variance
            rolling_var = np.maximum(rolling_var, 0)
            result[window:] = np.sqrt(rolling_var) * self.config.annualization_factor

        # Convert to percentage
        return result * 100.0

    def compute_rv_multi_window(
        self,
        log_returns: NDArray[np.floating[Any]],
        sample_interval: int = 1,
    ) -> NDArray[np.floating[Any]]:
        """Compute realized volatility at all configured windows.

        Args:
            log_returns: 1D array of log returns.
            sample_interval: Sampling interval in seconds.

        Returns:
            2D array of shape (len(log_returns), num_windows) with RV values.
        """
        log_returns = np.asarray(log_returns, dtype=np.float64)
        n = len(log_returns)
        num_windows = len(self.config.rv_windows)

        result = np.full((n, num_windows), np.nan, dtype=np.float64)

        for i, window_seconds in enumerate(self.config.rv_windows):
            window_samples = window_seconds // sample_interval
            if window_samples >= 2:
                result[:, i] = self.compute_realized_volatility(log_returns, window_samples)

        return result

    def compute_volatility_of_volatility(
        self,
        realized_volatility: NDArray[np.floating[Any]],
        window: int,
    ) -> NDArray[np.floating[Any]]:
        """Compute volatility of volatility (standard deviation of RV).

        Args:
            realized_volatility: 1D array of realized volatility values.
            window: Rolling window for computing vol-of-vol.

        Returns:
            Standard deviation of RV over the window.
        """
        rv = np.asarray(realized_volatility, dtype=np.float64)
        if rv.ndim != 1:
            raise ValueError("realized_volatility must be 1-dimensional")
        if window < 2:
            raise ValueError("window must be at least 2")

        n = len(rv)
        result = np.full(n, np.nan, dtype=np.float64)

        if window > n:
            return result

        # Use same rolling variance approach, but on RV values
        # Need to handle NaN values in RV
        cumsum = np.nancumsum(rv)
        cumsum_sq = np.nancumsum(rv**2)
        count = np.nancumsum(~np.isnan(rv))

        for i in range(window - 1, n):
            start_idx = i - window + 1
            if start_idx == 0:
                valid_count = count[i]
                if valid_count >= 2:
                    mean = cumsum[i] / valid_count
                    mean_sq = cumsum_sq[i] / valid_count
                    var = mean_sq - mean**2
                    result[i] = np.sqrt(max(var, 0))
            else:
                valid_count = count[i] - count[start_idx - 1]
                if valid_count >= 2:
                    rolling_sum = cumsum[i] - cumsum[start_idx - 1]
                    rolling_sum_sq = cumsum_sq[i] - cumsum_sq[start_idx - 1]
                    mean = rolling_sum / valid_count
                    mean_sq = rolling_sum_sq / valid_count
                    var = mean_sq - mean**2
                    result[i] = np.sqrt(max(var, 0))

        return result

    def compute_iv_rv_spread(
        self,
        implied_volatility: NDArray[np.floating[Any]],
        realized_volatility: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Compute IV - RV spread.

        Args:
            implied_volatility: IV values (same scale as RV, e.g., percentage).
            realized_volatility: RV values.

        Returns:
            IV - RV spread. NaN where either input is NaN.
        """
        iv = np.asarray(implied_volatility, dtype=np.float64)
        rv = np.asarray(realized_volatility, dtype=np.float64)

        if iv.shape != rv.shape:
            raise ValueError("implied_volatility and realized_volatility must have same shape")

        return iv - rv

    def compute_features(
        self,
        log_returns: NDArray[np.floating[Any]],
        implied_volatility: NDArray[np.floating[Any]] | None = None,
        sample_interval: int = 1,
        vol_of_vol_window: int = 60,
    ) -> NDArray[np.floating[Any]]:
        """Compute all volatility features.

        Args:
            log_returns: 1D array of log returns.
            implied_volatility: Optional IV data for IV/RV spread.
            sample_interval: Sampling interval in seconds.
            vol_of_vol_window: Window for volatility of volatility calculation.

        Returns:
            2D array of shape (len(log_returns), num_features).
        """
        log_returns = np.asarray(log_returns, dtype=np.float64)
        n = len(log_returns)

        features_list: list[NDArray[np.floating[Any]]] = []

        # Realized volatility at multiple windows
        rv_features = self.compute_rv_multi_window(log_returns, sample_interval)
        features_list.append(rv_features)

        # IV/RV spread (using shortest RV window for spread)
        if self.config.include_iv_rv_spread:
            if implied_volatility is not None:
                spread = self.compute_iv_rv_spread(
                    implied_volatility, rv_features[:, 0]
                )
            else:
                # If no IV provided, fill with NaN
                spread = np.full(n, np.nan, dtype=np.float64)
            features_list.append(spread.reshape(n, 1))

        # Volatility of volatility (using shortest RV window)
        if self.config.include_vol_of_vol:
            vol_of_vol = self.compute_volatility_of_volatility(
                rv_features[:, 0], vol_of_vol_window
            )
            features_list.append(vol_of_vol.reshape(n, 1))

        return np.hstack(features_list)

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names.

        Returns:
            List of feature names corresponding to columns in compute_features output.
        """
        names = [f"rv_{w}s" for w in self.config.rv_windows]
        if self.config.include_iv_rv_spread:
            names.append("iv_rv_spread")
        if self.config.include_vol_of_vol:
            names.append("vol_of_vol")
        return names

    @property
    def num_features(self) -> int:
        """Total number of features produced by compute_features."""
        count = len(self.config.rv_windows)
        if self.config.include_iv_rv_spread:
            count += 1
        if self.config.include_vol_of_vol:
            count += 1
        return count


@dataclass(frozen=True)
class OrderFlowFeatureConfig:
    """Configuration for order flow features.

    Attributes:
        imbalance_window: Window (in samples) for trade direction imbalance.
        include_size_distribution: Whether to include trade size distribution features.
        include_arrival_rate: Whether to include trade arrival rate.
        size_quantiles: Quantiles for trade size distribution (as fractions).
        arrival_rate_window: Window (in samples) for arrival rate calculation.
    """

    imbalance_window: int = 30
    include_size_distribution: bool = True
    include_arrival_rate: bool = True
    size_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    arrival_rate_window: int = 60


class OrderFlowFeatureBuilder:
    """Computes order flow features from trade data.

    Features computed:
    - Trade direction imbalance (net buy/sell pressure)
    - Trade size distribution quantiles (optional)
    - Arrival rate (trades per second, optional)

    Trade direction is inferred using the tick rule (comparing price
    to previous price).

    Attributes:
        config: Order flow feature configuration.
    """

    def __init__(self, config: OrderFlowFeatureConfig | None = None) -> None:
        """Initialize feature builder with configuration.

        Args:
            config: Order flow feature configuration. Uses defaults if None.
        """
        self.config = config or OrderFlowFeatureConfig()

    def infer_trade_direction(
        self,
        prices: NDArray[np.floating[Any]],
    ) -> NDArray[np.signedinteger[Any]]:
        """Infer trade direction using the tick rule.

        The tick rule classifies trades as:
        - Buy (+1): if price > previous price (uptick)
        - Sell (-1): if price < previous price (downtick)
        - Unknown (0): if price == previous price (zero tick)

        For zero ticks, we propagate the last known direction.

        Args:
            prices: 1D array of trade prices.

        Returns:
            Array of trade directions: +1 (buy), -1 (sell), 0 (unknown).
        """
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim != 1:
            raise ValueError("prices must be 1-dimensional")

        n = len(prices)
        if n == 0:
            return np.array([], dtype=np.int64)

        direction = np.zeros(n, dtype=np.int64)

        # Compare each price to previous
        if n > 1:
            diff = prices[1:] - prices[:-1]
            direction[1:] = np.sign(diff).astype(np.int64)

            # Propagate last known direction for zero ticks
            last_dir = 0
            for i in range(n):
                if direction[i] != 0:
                    last_dir = direction[i]
                else:
                    direction[i] = last_dir

        return direction

    def compute_trade_imbalance(
        self,
        directions: NDArray[np.signedinteger[Any]],
        sizes: NDArray[np.floating[Any]],
        window: int,
    ) -> NDArray[np.floating[Any]]:
        """Compute volume-weighted trade direction imbalance.

        Imbalance = (buy_volume - sell_volume) / total_volume

        Args:
            directions: Trade directions (+1 buy, -1 sell).
            sizes: Trade sizes (volumes).
            window: Rolling window size.

        Returns:
            Imbalance values in range [-1, 1]. NaN where window is invalid.
        """
        directions = np.asarray(directions, dtype=np.int64)
        sizes = np.asarray(sizes, dtype=np.float64)

        if directions.shape != sizes.shape:
            raise ValueError("directions and sizes must have same shape")
        if directions.ndim != 1:
            raise ValueError("directions must be 1-dimensional")
        if window < 1:
            raise ValueError("window must be at least 1")

        n = len(directions)
        result = np.full(n, np.nan, dtype=np.float64)

        if window > n:
            return result

        # Signed volume: positive for buys, negative for sells
        signed_volume = directions.astype(np.float64) * sizes

        # Rolling sums using cumsum
        cumsum_signed = np.cumsum(signed_volume)
        cumsum_total = np.cumsum(np.abs(sizes))

        # First valid position
        if cumsum_total[window - 1] > 0:
            result[window - 1] = cumsum_signed[window - 1] / cumsum_total[window - 1]

        # Remaining positions
        if window < n:
            rolling_signed = cumsum_signed[window:] - cumsum_signed[:-window]
            rolling_total = cumsum_total[window:] - cumsum_total[:-window]
            valid_mask = rolling_total > 0
            result[window:][valid_mask] = rolling_signed[valid_mask] / rolling_total[valid_mask]

        return result

    def compute_size_quantiles(
        self,
        sizes: NDArray[np.floating[Any]],
        window: int,
    ) -> NDArray[np.floating[Any]]:
        """Compute rolling quantiles of trade sizes.

        Args:
            sizes: 1D array of trade sizes.
            window: Rolling window size.

        Returns:
            2D array of shape (len(sizes), num_quantiles) with quantile values.
            NaN where window is invalid.
        """
        sizes = np.asarray(sizes, dtype=np.float64)
        if sizes.ndim != 1:
            raise ValueError("sizes must be 1-dimensional")
        if window < 1:
            raise ValueError("window must be at least 1")

        n = len(sizes)
        num_quantiles = len(self.config.size_quantiles)
        result = np.full((n, num_quantiles), np.nan, dtype=np.float64)

        if window > n:
            return result

        # Use stride_tricks for efficient rolling window
        # But for simplicity and clarity, use explicit loop
        # (can be optimized later if needed)
        for i in range(window - 1, n):
            window_data = sizes[i - window + 1 : i + 1]
            for j, q in enumerate(self.config.size_quantiles):
                result[i, j] = np.quantile(window_data, q)

        return result

    def compute_arrival_rate(
        self,
        timestamps: NDArray[np.floating[Any]],
        window: int,
    ) -> NDArray[np.floating[Any]]:
        """Compute trade arrival rate (trades per second).

        Args:
            timestamps: 1D array of trade timestamps (in seconds).
            window: Number of trades to use for rate calculation.

        Returns:
            Arrival rate in trades per second. NaN where window is invalid.
        """
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be 1-dimensional")
        if window < 2:
            raise ValueError("window must be at least 2 for rate calculation")

        n = len(timestamps)
        result = np.full(n, np.nan, dtype=np.float64)

        if window > n:
            return result

        # Rate = (window - 1) trades / time_elapsed
        for i in range(window - 1, n):
            time_elapsed = timestamps[i] - timestamps[i - window + 1]
            if time_elapsed > 0:
                result[i] = (window - 1) / time_elapsed

        return result

    def compute_features(
        self,
        prices: NDArray[np.floating[Any]],
        sizes: NDArray[np.floating[Any]],
        timestamps: NDArray[np.floating[Any]] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Compute all order flow features.

        Args:
            prices: 1D array of trade prices.
            sizes: 1D array of trade sizes.
            timestamps: 1D array of timestamps (required if include_arrival_rate).

        Returns:
            2D array of shape (len(prices), num_features) with all features.
            Feature order: [imbalance, size_quantiles..., arrival_rate]
        """
        prices = np.asarray(prices, dtype=np.float64)
        sizes = np.asarray(sizes, dtype=np.float64)

        if prices.shape != sizes.shape:
            raise ValueError("prices and sizes must have same shape")

        n = len(prices)
        features_list: list[NDArray[np.floating[Any]]] = []

        # Trade direction imbalance
        directions = self.infer_trade_direction(prices)
        imbalance = self.compute_trade_imbalance(
            directions, sizes, self.config.imbalance_window
        )
        features_list.append(imbalance.reshape(n, 1))

        # Size distribution quantiles
        if self.config.include_size_distribution:
            quantiles = self.compute_size_quantiles(
                sizes, self.config.imbalance_window
            )
            features_list.append(quantiles)

        # Arrival rate
        if self.config.include_arrival_rate:
            if timestamps is None:
                raise ValueError("timestamps required when include_arrival_rate is True")
            arrival = self.compute_arrival_rate(
                timestamps, self.config.arrival_rate_window
            )
            features_list.append(arrival.reshape(n, 1))

        return np.hstack(features_list)

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names.

        Returns:
            List of feature names corresponding to columns in compute_features output.
        """
        names = ["trade_imbalance"]
        if self.config.include_size_distribution:
            for q in self.config.size_quantiles:
                names.append(f"size_q{int(q * 100)}")
        if self.config.include_arrival_rate:
            names.append("arrival_rate")
        return names

    @property
    def num_features(self) -> int:
        """Total number of features produced by compute_features."""
        count = 1  # imbalance
        if self.config.include_size_distribution:
            count += len(self.config.size_quantiles)
        if self.config.include_arrival_rate:
            count += 1
        return count


@dataclass(frozen=True)
class OptionsFeatureConfig:
    """Configuration for options-derived features.

    Attributes:
        include_iv_surface: Whether to include IV surface features (ATM IV by expiry).
        include_greeks: Whether to include aggregated Greeks features.
        include_put_call_ratio: Whether to include put/call volume/OI ratios.
        include_term_structure: Whether to include IV term structure slope.
        iv_moneyness_buckets: Moneyness levels for IV surface (as ratios to spot).
        expiry_buckets_days: Days to expiration buckets for term structure.
    """

    include_iv_surface: bool = True
    include_greeks: bool = True
    include_put_call_ratio: bool = True
    include_term_structure: bool = True
    iv_moneyness_buckets: tuple[float, ...] = (0.95, 0.975, 1.0, 1.025, 1.05)
    expiry_buckets_days: tuple[int, ...] = (7, 30, 60, 90)


class OptionsFeatureBuilder:
    """Computes options-derived features from options chain data.

    Features computed:
    - IV surface: ATM implied volatility across expiration buckets
    - Greeks: Aggregated delta, gamma, vega, theta exposures
    - Put/call ratios: Volume and open interest ratios
    - Term structure slope: IV term structure gradient

    Options data is expected as structured arrays with fields for:
    strike, expiration, option_type (call/put), iv, delta, gamma, vega, theta,
    volume, open_interest.

    Attributes:
        config: Options feature configuration.
    """

    def __init__(self, config: OptionsFeatureConfig | None = None) -> None:
        """Initialize feature builder with configuration.

        Args:
            config: Options feature configuration. Uses defaults if None.
        """
        self.config = config or OptionsFeatureConfig()

    def compute_moneyness(
        self,
        strikes: NDArray[np.floating[Any]],
        spot_price: float,
    ) -> NDArray[np.floating[Any]]:
        """Compute moneyness ratio for each strike.

        Args:
            strikes: Array of strike prices.
            spot_price: Current underlying spot price.

        Returns:
            Moneyness ratios (strike / spot).
        """
        if spot_price <= 0:
            raise ValueError("spot_price must be positive")
        strikes = np.asarray(strikes, dtype=np.float64)
        return strikes / spot_price

    def compute_days_to_expiry(
        self,
        expirations: NDArray[np.floating[Any]],
        current_time: float,
    ) -> NDArray[np.floating[Any]]:
        """Compute days to expiration.

        Args:
            expirations: Array of expiration timestamps (Unix seconds or days).
            current_time: Current timestamp (same units as expirations).

        Returns:
            Days to expiration for each option.
        """
        expirations = np.asarray(expirations, dtype=np.float64)
        # Assuming timestamps are in seconds, convert to days
        dte = (expirations - current_time) / (24 * 3600)
        return np.maximum(dte, 0)  # Can't have negative DTE

    def compute_atm_iv_by_expiry(
        self,
        strikes: NDArray[np.floating[Any]],
        expirations: NDArray[np.floating[Any]],
        ivs: NDArray[np.floating[Any]],
        spot_price: float,
        current_time: float,
    ) -> NDArray[np.floating[Any]]:
        """Compute ATM implied volatility for each expiry bucket.

        Selects options closest to ATM (moneyness ~1.0) and averages IV
        within each expiry bucket.

        Args:
            strikes: Array of strike prices.
            expirations: Array of expiration timestamps.
            ivs: Array of implied volatilities (as decimals, e.g., 0.25 for 25%).
            spot_price: Current underlying spot price.
            current_time: Current timestamp.

        Returns:
            Array of ATM IVs for each expiry bucket in config.expiry_buckets_days.
            NaN for buckets with no valid options.
        """
        strikes = np.asarray(strikes, dtype=np.float64)
        ivs = np.asarray(ivs, dtype=np.float64)

        moneyness = self.compute_moneyness(strikes, spot_price)
        dte = self.compute_days_to_expiry(expirations, current_time)

        result = np.full(len(self.config.expiry_buckets_days), np.nan, dtype=np.float64)

        # ATM is within 2% of spot
        atm_mask = np.abs(moneyness - 1.0) < 0.02

        for i, bucket_days in enumerate(self.config.expiry_buckets_days):
            # Options within ±7 days of bucket target
            bucket_margin = min(bucket_days * 0.3, 7)  # 30% or 7 days max
            dte_mask = np.abs(dte - bucket_days) <= bucket_margin
            combined_mask = atm_mask & dte_mask & ~np.isnan(ivs)

            if np.any(combined_mask):
                # Weight by how close to ATM
                weights = 1.0 / (np.abs(moneyness[combined_mask] - 1.0) + 0.001)
                result[i] = np.average(ivs[combined_mask], weights=weights)

        return result

    def compute_iv_surface_features(
        self,
        strikes: NDArray[np.floating[Any]],
        expirations: NDArray[np.floating[Any]],
        ivs: NDArray[np.floating[Any]],
        spot_price: float,
        current_time: float,
    ) -> NDArray[np.floating[Any]]:
        """Compute IV surface features across moneyness and expiry.

        Args:
            strikes: Array of strike prices.
            expirations: Array of expiration timestamps.
            ivs: Array of implied volatilities.
            spot_price: Current underlying spot price.
            current_time: Current timestamp.

        Returns:
            Flattened array of IV values: [expiry1_m1, expiry1_m2, ..., expiry2_m1, ...]
            Shape: (num_expiry_buckets * num_moneyness_buckets,)
        """
        strikes = np.asarray(strikes, dtype=np.float64)
        ivs = np.asarray(ivs, dtype=np.float64)

        moneyness = self.compute_moneyness(strikes, spot_price)
        dte = self.compute_days_to_expiry(expirations, current_time)

        num_expiry = len(self.config.expiry_buckets_days)
        num_moneyness = len(self.config.iv_moneyness_buckets)
        result = np.full(num_expiry * num_moneyness, np.nan, dtype=np.float64)

        for i, bucket_days in enumerate(self.config.expiry_buckets_days):
            bucket_margin = min(bucket_days * 0.3, 7)
            dte_mask = np.abs(dte - bucket_days) <= bucket_margin

            for j, target_moneyness in enumerate(self.config.iv_moneyness_buckets):
                # Moneyness within 1.5% of target
                moneyness_mask = np.abs(moneyness - target_moneyness) < 0.015
                combined_mask = dte_mask & moneyness_mask & ~np.isnan(ivs)

                if np.any(combined_mask):
                    # Weight by proximity to target moneyness
                    weights = 1.0 / (
                        np.abs(moneyness[combined_mask] - target_moneyness) + 0.001
                    )
                    result[i * num_moneyness + j] = np.average(
                        ivs[combined_mask], weights=weights
                    )

        return result

    def compute_aggregated_greeks(
        self,
        deltas: NDArray[np.floating[Any]],
        gammas: NDArray[np.floating[Any]],
        vegas: NDArray[np.floating[Any]],
        thetas: NDArray[np.floating[Any]],
        open_interests: NDArray[np.floating[Any]],
        option_types: NDArray[np.int8],
    ) -> NDArray[np.floating[Any]]:
        """Compute aggregated Greeks weighted by open interest.

        Greeks are aggregated as OI-weighted averages, providing a measure
        of market-wide exposure.

        Args:
            deltas: Array of option deltas.
            gammas: Array of option gammas.
            vegas: Array of option vegas.
            thetas: Array of option thetas.
            open_interests: Array of open interest values.
            option_types: Array of option types (1 for call, -1 for put).

        Returns:
            Array of [net_delta, total_gamma, total_vega, total_theta].
        """
        deltas = np.asarray(deltas, dtype=np.float64)
        gammas = np.asarray(gammas, dtype=np.float64)
        vegas = np.asarray(vegas, dtype=np.float64)
        thetas = np.asarray(thetas, dtype=np.float64)
        open_interests = np.asarray(open_interests, dtype=np.float64)
        option_types = np.asarray(option_types, dtype=np.int8)

        result = np.zeros(4, dtype=np.float64)
        total_oi = np.sum(open_interests)

        if total_oi <= 0:
            return np.full(4, np.nan, dtype=np.float64)

        # Net delta (calls positive, puts negative relative contribution)
        # Delta already accounts for direction, so just weight by OI
        result[0] = np.sum(deltas * open_interests) / total_oi

        # Total gamma (absolute, both calls and puts contribute positively)
        result[1] = np.sum(np.abs(gammas) * open_interests) / total_oi

        # Total vega (positive for both calls and puts)
        result[2] = np.sum(vegas * open_interests) / total_oi

        # Total theta (typically negative, time decay)
        result[3] = np.sum(thetas * open_interests) / total_oi

        return result

    def compute_put_call_ratios(
        self,
        option_types: NDArray[np.int8],
        volumes: NDArray[np.floating[Any]],
        open_interests: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Compute put/call ratios for volume and open interest.

        Args:
            option_types: Array of option types (1 for call, -1 for put).
            volumes: Array of trading volumes.
            open_interests: Array of open interest values.

        Returns:
            Array of [volume_pc_ratio, oi_pc_ratio].
            Ratio is put/call, so >1 means more put activity.
        """
        option_types = np.asarray(option_types, dtype=np.int8)
        volumes = np.asarray(volumes, dtype=np.float64)
        open_interests = np.asarray(open_interests, dtype=np.float64)

        call_mask = option_types == 1
        put_mask = option_types == -1

        result = np.full(2, np.nan, dtype=np.float64)

        # Volume ratio
        call_volume = np.sum(volumes[call_mask])
        put_volume = np.sum(volumes[put_mask])
        if call_volume > 0:
            result[0] = put_volume / call_volume

        # OI ratio
        call_oi = np.sum(open_interests[call_mask])
        put_oi = np.sum(open_interests[put_mask])
        if call_oi > 0:
            result[1] = put_oi / call_oi

        return result

    def compute_term_structure_slope(
        self,
        atm_ivs_by_expiry: NDArray[np.floating[Any]],
    ) -> float:
        """Compute IV term structure slope.

        Fits a linear regression of IV vs days-to-expiry and returns
        the slope coefficient.

        Args:
            atm_ivs_by_expiry: ATM IVs for each expiry bucket.

        Returns:
            Slope of IV term structure (IV change per day).
            Positive means contango, negative means backwardation.
        """
        expiry_days = np.array(self.config.expiry_buckets_days, dtype=np.float64)
        ivs = np.asarray(atm_ivs_by_expiry, dtype=np.float64)

        # Filter out NaN values
        valid_mask = ~np.isnan(ivs)
        if np.sum(valid_mask) < 2:
            return np.nan

        valid_days = expiry_days[valid_mask]
        valid_ivs = ivs[valid_mask]

        # Simple linear regression: slope = cov(x,y) / var(x)
        mean_days = np.mean(valid_days)
        mean_ivs = np.mean(valid_ivs)

        numerator = np.sum((valid_days - mean_days) * (valid_ivs - mean_ivs))
        denominator = np.sum((valid_days - mean_days) ** 2)

        if denominator < 1e-10:
            return np.nan

        return float(numerator / denominator)

    def compute_features(
        self,
        strikes: NDArray[np.floating[Any]],
        expirations: NDArray[np.floating[Any]],
        ivs: NDArray[np.floating[Any]],
        spot_price: float,
        current_time: float,
        option_types: NDArray[np.int8] | None = None,
        deltas: NDArray[np.floating[Any]] | None = None,
        gammas: NDArray[np.floating[Any]] | None = None,
        vegas: NDArray[np.floating[Any]] | None = None,
        thetas: NDArray[np.floating[Any]] | None = None,
        volumes: NDArray[np.floating[Any]] | None = None,
        open_interests: NDArray[np.floating[Any]] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Compute all options features.

        Args:
            strikes: Array of strike prices.
            expirations: Array of expiration timestamps.
            ivs: Array of implied volatilities.
            spot_price: Current underlying spot price.
            current_time: Current timestamp.
            option_types: Option types (1=call, -1=put). Required for Greeks/ratios.
            deltas: Option deltas. Required if include_greeks.
            gammas: Option gammas. Required if include_greeks.
            vegas: Option vegas. Required if include_greeks.
            thetas: Option thetas. Required if include_greeks.
            volumes: Trading volumes. Required if include_put_call_ratio.
            open_interests: Open interest. Required if include_greeks or ratios.

        Returns:
            1D array of all computed features.
        """
        features_list: list[NDArray[np.floating[Any]]] = []

        # IV surface features
        if self.config.include_iv_surface:
            iv_surface = self.compute_iv_surface_features(
                strikes, expirations, ivs, spot_price, current_time
            )
            features_list.append(iv_surface)

        # Greeks
        if self.config.include_greeks:
            if any(
                x is None
                for x in [option_types, deltas, gammas, vegas, thetas, open_interests]
            ):
                raise ValueError(
                    "Greeks data required when include_greeks is True: "
                    "option_types, deltas, gammas, vegas, thetas, open_interests"
                )
            # Type checker needs help here since we checked above
            assert option_types is not None
            assert deltas is not None
            assert gammas is not None
            assert vegas is not None
            assert thetas is not None
            assert open_interests is not None
            greeks = self.compute_aggregated_greeks(
                deltas, gammas, vegas, thetas, open_interests, option_types
            )
            features_list.append(greeks)

        # Put/call ratios
        if self.config.include_put_call_ratio:
            if option_types is None or volumes is None or open_interests is None:
                raise ValueError(
                    "option_types, volumes, and open_interests required "
                    "when include_put_call_ratio is True"
                )
            ratios = self.compute_put_call_ratios(option_types, volumes, open_interests)
            features_list.append(ratios)

        # Term structure slope
        if self.config.include_term_structure:
            atm_ivs = self.compute_atm_iv_by_expiry(
                strikes, expirations, ivs, spot_price, current_time
            )
            slope = self.compute_term_structure_slope(atm_ivs)
            features_list.append(np.array([slope], dtype=np.float64))

        if not features_list:
            return np.array([], dtype=np.float64)

        return np.concatenate(features_list)

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names.

        Returns:
            List of feature names corresponding to columns in compute_features output.
        """
        names: list[str] = []

        if self.config.include_iv_surface:
            for expiry in self.config.expiry_buckets_days:
                for moneyness in self.config.iv_moneyness_buckets:
                    names.append(f"iv_{expiry}d_m{int(moneyness * 100)}")

        if self.config.include_greeks:
            names.extend(["net_delta", "total_gamma", "total_vega", "total_theta"])

        if self.config.include_put_call_ratio:
            names.extend(["volume_pc_ratio", "oi_pc_ratio"])

        if self.config.include_term_structure:
            names.append("iv_term_slope")

        return names

    @property
    def num_features(self) -> int:
        """Total number of features produced by compute_features."""
        count = 0
        if self.config.include_iv_surface:
            count += len(self.config.expiry_buckets_days) * len(
                self.config.iv_moneyness_buckets
            )
        if self.config.include_greeks:
            count += 4  # delta, gamma, vega, theta
        if self.config.include_put_call_ratio:
            count += 2  # volume ratio, OI ratio
        if self.config.include_term_structure:
            count += 1  # slope
        return count
