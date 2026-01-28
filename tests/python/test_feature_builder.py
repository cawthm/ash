"""Tests for feature_builder.py."""

from __future__ import annotations

import numpy as np
import pytest

from data.processors.feature_builder import (
    PriceFeatureBuilder,
    PriceFeatureConfig,
    VolatilityFeatureBuilder,
    VolatilityFeatureConfig,
)


class TestPriceFeatureConfig:
    """Tests for PriceFeatureConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config has expected values."""
        config = PriceFeatureConfig()
        assert config.return_windows == (1, 5, 10, 30, 60, 120, 300)
        assert config.include_vwap is True
        assert config.sample_interval == 1

    def test_custom_config(self) -> None:
        """Custom config accepts valid values."""
        config = PriceFeatureConfig(
            return_windows=(5, 10, 20),
            include_vwap=False,
            sample_interval=5,
        )
        assert config.return_windows == (5, 10, 20)
        assert config.include_vwap is False
        assert config.sample_interval == 5


class TestPriceFeatureBuilder:
    """Tests for PriceFeatureBuilder class."""

    def test_default_initialization(self) -> None:
        """Builder initializes with default config."""
        builder = PriceFeatureBuilder()
        assert builder.config.return_windows == (1, 5, 10, 30, 60, 120, 300)

    def test_compute_log_returns_single_window(self) -> None:
        """Log returns are computed correctly for a single window."""
        builder = PriceFeatureBuilder()
        # Prices: 100, 101, 102 (1% gain each)
        prices = np.array([100.0, 101.0, 102.0])
        returns = builder.compute_log_returns(prices, window=1)

        # First element is NaN (no prior price)
        assert np.isnan(returns[0])
        # Second element: ln(101/100)
        assert np.isclose(returns[1], np.log(101 / 100))
        # Third element: ln(102/101)
        assert np.isclose(returns[2], np.log(102 / 101))

    def test_compute_log_returns_larger_window(self) -> None:
        """Log returns with window > 1."""
        builder = PriceFeatureBuilder()
        prices = np.array([100.0, 101.0, 103.0, 106.0])
        returns = builder.compute_log_returns(prices, window=2)

        # First two elements are NaN
        assert np.isnan(returns[0])
        assert np.isnan(returns[1])
        # Third element: ln(103/100)
        assert np.isclose(returns[2], np.log(103 / 100))
        # Fourth element: ln(106/101)
        assert np.isclose(returns[3], np.log(106 / 101))

    def test_compute_log_returns_window_equals_length(self) -> None:
        """Window equal to data length gives all NaN except last."""
        builder = PriceFeatureBuilder()
        prices = np.array([100.0, 105.0, 110.0])
        returns = builder.compute_log_returns(prices, window=3)

        # All elements should be NaN when window >= len(prices)
        assert np.all(np.isnan(returns))

    def test_compute_log_returns_rejects_2d_input(self) -> None:
        """2D input raises ValueError."""
        builder = PriceFeatureBuilder()
        prices = np.array([[100.0, 101.0], [102.0, 103.0]])
        with pytest.raises(ValueError, match="1-dimensional"):
            builder.compute_log_returns(prices, window=1)

    def test_compute_log_returns_rejects_zero_window(self) -> None:
        """Window of 0 raises ValueError."""
        builder = PriceFeatureBuilder()
        prices = np.array([100.0, 101.0])
        with pytest.raises(ValueError, match="at least 1"):
            builder.compute_log_returns(prices, window=0)

    def test_compute_log_returns_handles_zero_price(self) -> None:
        """Zero prices result in NaN returns."""
        builder = PriceFeatureBuilder()
        prices = np.array([100.0, 0.0, 102.0])
        returns = builder.compute_log_returns(prices, window=1)

        assert np.isnan(returns[0])  # No prior
        assert np.isnan(returns[1])  # Current is 0
        assert np.isnan(returns[2])  # Prior is 0

    def test_compute_returns_multi_window(self) -> None:
        """Multi-window returns computes all windows."""
        config = PriceFeatureConfig(return_windows=(1, 2), sample_interval=1)
        builder = PriceFeatureBuilder(config)
        prices = np.array([100.0, 101.0, 103.0, 106.0])
        returns = builder.compute_returns_multi_window(prices)

        assert returns.shape == (4, 2)
        # Window 1 returns
        assert np.isnan(returns[0, 0])
        assert np.isclose(returns[1, 0], np.log(101 / 100))
        # Window 2 returns
        assert np.isnan(returns[0, 1])
        assert np.isnan(returns[1, 1])
        assert np.isclose(returns[2, 1], np.log(103 / 100))

    def test_compute_vwap(self) -> None:
        """VWAP is computed correctly."""
        builder = PriceFeatureBuilder()
        prices = np.array([10.0, 11.0, 12.0, 13.0])
        volumes = np.array([100.0, 200.0, 150.0, 250.0])

        vwap = builder.compute_vwap(prices, volumes, window=2)

        # First element is NaN
        assert np.isnan(vwap[0])
        # Second element: (10*100 + 11*200) / (100 + 200) = 3200/300 = 10.666...
        expected_1 = (10 * 100 + 11 * 200) / (100 + 200)
        assert np.isclose(vwap[1], expected_1)
        # Third element: (11*200 + 12*150) / (200 + 150) = 4000/350
        expected_2 = (11 * 200 + 12 * 150) / (200 + 150)
        assert np.isclose(vwap[2], expected_2)

    def test_compute_vwap_mismatched_lengths(self) -> None:
        """Mismatched price/volume lengths raises ValueError."""
        builder = PriceFeatureBuilder()
        prices = np.array([10.0, 11.0, 12.0])
        volumes = np.array([100.0, 200.0])
        with pytest.raises(ValueError, match="same length"):
            builder.compute_vwap(prices, volumes, window=2)

    def test_compute_vwap_deviation(self) -> None:
        """VWAP deviation is log ratio of price to VWAP."""
        builder = PriceFeatureBuilder()
        prices = np.array([10.0, 11.0, 12.0, 11.5])
        volumes = np.array([100.0, 100.0, 100.0, 100.0])

        deviation = builder.compute_vwap_deviation(prices, volumes, window=2)

        # First element is NaN (no VWAP)
        assert np.isnan(deviation[0])
        # For equal volumes, VWAP is arithmetic mean of prices
        # At index 1: VWAP = (10+11)/2 = 10.5, deviation = ln(11/10.5)
        vwap_1 = (10 + 11) / 2
        assert np.isclose(deviation[1], np.log(11 / vwap_1))

    def test_compute_features(self) -> None:
        """compute_features returns correct shape and includes all features."""
        config = PriceFeatureConfig(return_windows=(1, 2), include_vwap=True)
        builder = PriceFeatureBuilder(config)

        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        volumes = np.array([1000.0, 1100.0, 900.0, 1200.0, 1000.0])

        features = builder.compute_features(prices, volumes, vwap_window=2)

        # 2 return windows + 1 VWAP deviation = 3 features
        assert features.shape == (5, 3)

    def test_compute_features_no_vwap(self) -> None:
        """compute_features without VWAP has correct shape."""
        config = PriceFeatureConfig(return_windows=(1, 2, 5), include_vwap=False)
        builder = PriceFeatureBuilder(config)

        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        features = builder.compute_features(prices)

        # 3 return windows, no VWAP
        assert features.shape == (6, 3)

    def test_compute_features_requires_volumes_for_vwap(self) -> None:
        """compute_features with include_vwap requires volumes."""
        config = PriceFeatureConfig(include_vwap=True)
        builder = PriceFeatureBuilder(config)
        prices = np.array([100.0, 101.0, 102.0])

        with pytest.raises(ValueError, match="volumes required"):
            builder.compute_features(prices)

    def test_get_feature_names(self) -> None:
        """Feature names match feature order."""
        config = PriceFeatureConfig(return_windows=(1, 5, 10), include_vwap=True)
        builder = PriceFeatureBuilder(config)

        names = builder.get_feature_names()
        assert names == ["log_return_1s", "log_return_5s", "log_return_10s", "vwap_deviation"]

    def test_get_feature_names_no_vwap(self) -> None:
        """Feature names without VWAP."""
        config = PriceFeatureConfig(return_windows=(1, 5), include_vwap=False)
        builder = PriceFeatureBuilder(config)

        names = builder.get_feature_names()
        assert names == ["log_return_1s", "log_return_5s"]

    def test_num_features(self) -> None:
        """num_features property returns correct count."""
        config = PriceFeatureConfig(return_windows=(1, 5, 10, 30), include_vwap=True)
        builder = PriceFeatureBuilder(config)
        assert builder.num_features == 5  # 4 returns + 1 VWAP

        config_no_vwap = PriceFeatureConfig(return_windows=(1, 5), include_vwap=False)
        builder_no_vwap = PriceFeatureBuilder(config_no_vwap)
        assert builder_no_vwap.num_features == 2


class TestVolatilityFeatureConfig:
    """Tests for VolatilityFeatureConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config has expected values."""
        config = VolatilityFeatureConfig()
        assert config.rv_windows == (60, 300, 600)
        assert config.include_iv_rv_spread is True
        assert config.include_vol_of_vol is True
        assert config.annualization_factor > 0

    def test_custom_config(self) -> None:
        """Custom config accepts valid values."""
        config = VolatilityFeatureConfig(
            rv_windows=(30, 60),
            include_iv_rv_spread=False,
            include_vol_of_vol=False,
            annualization_factor=100.0,
        )
        assert config.rv_windows == (30, 60)
        assert config.include_iv_rv_spread is False
        assert config.include_vol_of_vol is False
        assert config.annualization_factor == 100.0


class TestVolatilityFeatureBuilder:
    """Tests for VolatilityFeatureBuilder class."""

    def test_default_initialization(self) -> None:
        """Builder initializes with default config."""
        builder = VolatilityFeatureBuilder()
        assert builder.config.rv_windows == (60, 300, 600)

    def test_compute_realized_volatility(self) -> None:
        """RV is computed correctly."""
        # Use simple annualization factor for testing
        config = VolatilityFeatureConfig(annualization_factor=1.0)
        builder = VolatilityFeatureBuilder(config)

        # Known returns with std = 0.01
        log_returns = np.array([0.01, -0.01, 0.01, -0.01, 0.01])
        rv = builder.compute_realized_volatility(log_returns, window=4)

        # First 3 elements are NaN
        assert np.all(np.isnan(rv[:3]))
        # At index 3: std of [0.01, -0.01, 0.01, -0.01] = 0.01
        # As percentage: 1.0%
        assert np.isclose(rv[3], 1.0, atol=0.01)

    def test_compute_realized_volatility_constant_returns(self) -> None:
        """Constant returns give zero volatility."""
        config = VolatilityFeatureConfig(annualization_factor=1.0)
        builder = VolatilityFeatureBuilder(config)

        log_returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        rv = builder.compute_realized_volatility(log_returns, window=3)

        # After warm-up, volatility should be ~0
        assert np.isclose(rv[2], 0.0, atol=1e-10)
        assert np.isclose(rv[3], 0.0, atol=1e-10)

    def test_compute_realized_volatility_rejects_small_window(self) -> None:
        """Window < 2 raises ValueError."""
        builder = VolatilityFeatureBuilder()
        log_returns = np.array([0.01, 0.02, 0.03])
        with pytest.raises(ValueError, match="at least 2"):
            builder.compute_realized_volatility(log_returns, window=1)

    def test_compute_realized_volatility_rejects_2d_input(self) -> None:
        """2D input raises ValueError."""
        builder = VolatilityFeatureBuilder()
        log_returns = np.array([[0.01, 0.02], [0.03, 0.04]])
        with pytest.raises(ValueError, match="1-dimensional"):
            builder.compute_realized_volatility(log_returns, window=2)

    def test_compute_rv_multi_window(self) -> None:
        """Multi-window RV computes all windows."""
        config = VolatilityFeatureConfig(rv_windows=(2, 4), annualization_factor=1.0)
        builder = VolatilityFeatureBuilder(config)

        # 10 data points
        log_returns = np.random.randn(10) * 0.01
        rv = builder.compute_rv_multi_window(log_returns, sample_interval=1)

        assert rv.shape == (10, 2)
        # First window (2): should have data from index 1
        assert np.isnan(rv[0, 0])
        assert not np.isnan(rv[1, 0])
        # Second window (4): should have data from index 3
        assert np.isnan(rv[2, 1])
        assert not np.isnan(rv[3, 1])

    def test_compute_volatility_of_volatility(self) -> None:
        """Vol-of-vol is computed correctly."""
        builder = VolatilityFeatureBuilder()

        # RV values with some variation
        rv = np.array([20.0, 22.0, 19.0, 23.0, 21.0, 24.0])
        vol_of_vol = builder.compute_volatility_of_volatility(rv, window=3)

        # First 2 elements are NaN
        assert np.all(np.isnan(vol_of_vol[:2]))
        # At index 2: std of [20, 22, 19]
        expected_std = np.std([20, 22, 19], ddof=0)
        assert np.isclose(vol_of_vol[2], expected_std, atol=0.01)

    def test_compute_volatility_of_volatility_with_nan(self) -> None:
        """Vol-of-vol handles NaN in input."""
        builder = VolatilityFeatureBuilder()

        rv = np.array([np.nan, 22.0, 19.0, 23.0, 21.0])
        vol_of_vol = builder.compute_volatility_of_volatility(rv, window=3)

        # Should still compute where enough valid data exists
        assert np.isnan(vol_of_vol[0])
        assert np.isnan(vol_of_vol[1])
        # Index 2 has only 2 valid values in window, should compute
        assert not np.isnan(vol_of_vol[3])  # Window [22, 19, 23] all valid

    def test_compute_iv_rv_spread(self) -> None:
        """IV/RV spread is simple difference."""
        builder = VolatilityFeatureBuilder()

        iv = np.array([25.0, 26.0, 24.0])
        rv = np.array([20.0, 22.0, 23.0])

        spread = builder.compute_iv_rv_spread(iv, rv)

        assert np.allclose(spread, [5.0, 4.0, 1.0])

    def test_compute_iv_rv_spread_mismatched_shapes(self) -> None:
        """Mismatched IV/RV shapes raises ValueError."""
        builder = VolatilityFeatureBuilder()
        iv = np.array([25.0, 26.0])
        rv = np.array([20.0, 22.0, 23.0])

        with pytest.raises(ValueError, match="same shape"):
            builder.compute_iv_rv_spread(iv, rv)

    def test_compute_features(self) -> None:
        """compute_features returns correct shape."""
        config = VolatilityFeatureConfig(rv_windows=(5, 10))
        builder = VolatilityFeatureBuilder(config)

        log_returns = np.random.randn(100) * 0.01
        iv = np.full(100, 25.0)

        features = builder.compute_features(
            log_returns, implied_volatility=iv, sample_interval=1
        )

        # 2 RV windows + IV/RV spread + vol-of-vol = 4
        assert features.shape == (100, 4)

    def test_compute_features_no_iv(self) -> None:
        """compute_features without IV fills spread with NaN."""
        config = VolatilityFeatureConfig(rv_windows=(5, 10))
        builder = VolatilityFeatureBuilder(config)

        log_returns = np.random.randn(100) * 0.01
        features = builder.compute_features(log_returns, sample_interval=1)

        # IV/RV spread column should be all NaN
        assert features.shape == (100, 4)
        iv_rv_col = 2  # After 2 RV windows
        assert np.all(np.isnan(features[:, iv_rv_col]))

    def test_compute_features_minimal(self) -> None:
        """compute_features with minimal options."""
        config = VolatilityFeatureConfig(
            rv_windows=(5,),
            include_iv_rv_spread=False,
            include_vol_of_vol=False,
        )
        builder = VolatilityFeatureBuilder(config)

        log_returns = np.random.randn(20) * 0.01
        features = builder.compute_features(log_returns)

        # Only 1 RV window
        assert features.shape == (20, 1)

    def test_get_feature_names(self) -> None:
        """Feature names match feature order."""
        config = VolatilityFeatureConfig(rv_windows=(60, 300))
        builder = VolatilityFeatureBuilder(config)

        names = builder.get_feature_names()
        assert names == ["rv_60s", "rv_300s", "iv_rv_spread", "vol_of_vol"]

    def test_get_feature_names_minimal(self) -> None:
        """Feature names with minimal options."""
        config = VolatilityFeatureConfig(
            rv_windows=(30,),
            include_iv_rv_spread=False,
            include_vol_of_vol=False,
        )
        builder = VolatilityFeatureBuilder(config)

        names = builder.get_feature_names()
        assert names == ["rv_30s"]

    def test_num_features(self) -> None:
        """num_features property returns correct count."""
        config = VolatilityFeatureConfig(rv_windows=(60, 300, 600))
        builder = VolatilityFeatureBuilder(config)
        assert builder.num_features == 5  # 3 RV + spread + vol-of-vol

        config_minimal = VolatilityFeatureConfig(
            rv_windows=(60,),
            include_iv_rv_spread=False,
            include_vol_of_vol=False,
        )
        builder_minimal = VolatilityFeatureBuilder(config_minimal)
        assert builder_minimal.num_features == 1


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_empty_array(self) -> None:
        """Empty arrays are handled gracefully."""
        price_builder = PriceFeatureBuilder()
        vol_builder = VolatilityFeatureBuilder()

        empty = np.array([])
        assert price_builder.compute_log_returns(empty, window=1).shape == (0,)
        assert price_builder.compute_vwap(empty, empty, window=1).shape == (0,)
        assert vol_builder.compute_realized_volatility(empty, window=2).shape == (0,)

    def test_single_element(self) -> None:
        """Single element arrays produce NaN (insufficient data)."""
        price_builder = PriceFeatureBuilder()

        single = np.array([100.0])
        returns = price_builder.compute_log_returns(single, window=1)
        assert len(returns) == 1
        assert np.isnan(returns[0])

    def test_large_array_performance(self) -> None:
        """Large arrays complete without issues."""
        price_builder = PriceFeatureBuilder()
        vol_builder = VolatilityFeatureBuilder()

        # 1 million data points
        n = 1_000_000
        prices = 100 + np.cumsum(np.random.randn(n) * 0.01)
        volumes = np.abs(np.random.randn(n)) * 1000

        # Should complete quickly
        returns = price_builder.compute_log_returns(prices, window=100)
        assert returns.shape == (n,)

        vwap = price_builder.compute_vwap(prices, volumes, window=100)
        assert vwap.shape == (n,)

        rv = vol_builder.compute_realized_volatility(returns[~np.isnan(returns)], window=100)
        assert len(rv) > 0

    def test_numerical_stability_small_values(self) -> None:
        """Small price values don't cause numerical issues."""
        builder = PriceFeatureBuilder()

        # Very small prices (e.g., penny stock)
        prices = np.array([0.01, 0.0101, 0.0102, 0.0103])
        returns = builder.compute_log_returns(prices, window=1)

        # Should still be valid
        assert not np.all(np.isnan(returns[1:]))
        assert np.all(np.isfinite(returns[1:]))

    def test_numerical_stability_large_values(self) -> None:
        """Large price values don't cause numerical issues."""
        builder = PriceFeatureBuilder()

        # Large prices (e.g., BRK.A)
        prices = np.array([500000.0, 500100.0, 500200.0, 500300.0])
        returns = builder.compute_log_returns(prices, window=1)

        assert not np.all(np.isnan(returns[1:]))
        assert np.all(np.isfinite(returns[1:]))
