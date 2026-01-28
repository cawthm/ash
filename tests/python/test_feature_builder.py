"""Tests for feature_builder.py."""

from __future__ import annotations

import numpy as np
import pytest

from data.processors.feature_builder import (
    OptionsFeatureBuilder,
    OptionsFeatureConfig,
    OrderFlowFeatureBuilder,
    OrderFlowFeatureConfig,
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


class TestOrderFlowFeatureConfig:
    """Tests for OrderFlowFeatureConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config has expected values."""
        config = OrderFlowFeatureConfig()
        assert config.imbalance_window == 30
        assert config.include_size_distribution is True
        assert config.include_arrival_rate is True
        assert config.size_quantiles == (0.25, 0.5, 0.75)
        assert config.arrival_rate_window == 60

    def test_custom_config(self) -> None:
        """Custom config accepts valid values."""
        config = OrderFlowFeatureConfig(
            imbalance_window=60,
            include_size_distribution=False,
            include_arrival_rate=False,
            size_quantiles=(0.1, 0.5, 0.9),
            arrival_rate_window=30,
        )
        assert config.imbalance_window == 60
        assert config.include_size_distribution is False
        assert config.include_arrival_rate is False
        assert config.size_quantiles == (0.1, 0.5, 0.9)
        assert config.arrival_rate_window == 30


class TestOrderFlowFeatureBuilder:
    """Tests for OrderFlowFeatureBuilder class."""

    def test_default_initialization(self) -> None:
        """Builder initializes with default config."""
        builder = OrderFlowFeatureBuilder()
        assert builder.config.imbalance_window == 30

    def test_infer_trade_direction_upticks(self) -> None:
        """Upticks are classified as buys (+1)."""
        builder = OrderFlowFeatureBuilder()
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        directions = builder.infer_trade_direction(prices)

        assert directions[0] == 0  # First has no prior
        assert directions[1] == 1  # Uptick
        assert directions[2] == 1  # Uptick
        assert directions[3] == 1  # Uptick

    def test_infer_trade_direction_downticks(self) -> None:
        """Downticks are classified as sells (-1)."""
        builder = OrderFlowFeatureBuilder()
        prices = np.array([103.0, 102.0, 101.0, 100.0])
        directions = builder.infer_trade_direction(prices)

        assert directions[0] == 0  # First has no prior
        assert directions[1] == -1  # Downtick
        assert directions[2] == -1  # Downtick
        assert directions[3] == -1  # Downtick

    def test_infer_trade_direction_zero_ticks(self) -> None:
        """Zero ticks propagate last known direction."""
        builder = OrderFlowFeatureBuilder()
        prices = np.array([100.0, 101.0, 101.0, 101.0, 100.0])
        directions = builder.infer_trade_direction(prices)

        assert directions[0] == 0  # First
        assert directions[1] == 1  # Uptick
        assert directions[2] == 1  # Zero tick, propagates +1
        assert directions[3] == 1  # Zero tick, propagates +1
        assert directions[4] == -1  # Downtick

    def test_infer_trade_direction_empty(self) -> None:
        """Empty array returns empty result."""
        builder = OrderFlowFeatureBuilder()
        directions = builder.infer_trade_direction(np.array([]))
        assert len(directions) == 0

    def test_infer_trade_direction_single(self) -> None:
        """Single element returns zero direction."""
        builder = OrderFlowFeatureBuilder()
        directions = builder.infer_trade_direction(np.array([100.0]))
        assert len(directions) == 1
        assert directions[0] == 0

    def test_infer_trade_direction_rejects_2d(self) -> None:
        """2D input raises ValueError."""
        builder = OrderFlowFeatureBuilder()
        with pytest.raises(ValueError, match="1-dimensional"):
            builder.infer_trade_direction(np.array([[100.0, 101.0]]))

    def test_compute_trade_imbalance_all_buys(self) -> None:
        """All buys gives imbalance of +1."""
        builder = OrderFlowFeatureBuilder()
        directions = np.array([1, 1, 1, 1, 1])
        sizes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        imbalance = builder.compute_trade_imbalance(directions, sizes, window=3)

        assert np.isnan(imbalance[0])
        assert np.isnan(imbalance[1])
        assert np.isclose(imbalance[2], 1.0)
        assert np.isclose(imbalance[3], 1.0)
        assert np.isclose(imbalance[4], 1.0)

    def test_compute_trade_imbalance_all_sells(self) -> None:
        """All sells gives imbalance of -1."""
        builder = OrderFlowFeatureBuilder()
        directions = np.array([-1, -1, -1, -1, -1])
        sizes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        imbalance = builder.compute_trade_imbalance(directions, sizes, window=3)

        assert np.isclose(imbalance[2], -1.0)
        assert np.isclose(imbalance[4], -1.0)

    def test_compute_trade_imbalance_balanced(self) -> None:
        """Balanced buys/sells gives imbalance of 0."""
        builder = OrderFlowFeatureBuilder()
        directions = np.array([1, -1, 1, -1])
        sizes = np.array([100.0, 100.0, 100.0, 100.0])
        imbalance = builder.compute_trade_imbalance(directions, sizes, window=2)

        # Window of 2: [1, -1] -> (100 - 100) / 200 = 0
        assert np.isclose(imbalance[1], 0.0)
        assert np.isclose(imbalance[2], 0.0)
        assert np.isclose(imbalance[3], 0.0)

    def test_compute_trade_imbalance_volume_weighted(self) -> None:
        """Imbalance is volume-weighted."""
        builder = OrderFlowFeatureBuilder()
        directions = np.array([1, -1, 1])
        sizes = np.array([300.0, 100.0, 100.0])  # Big buy, small sell, small buy
        imbalance = builder.compute_trade_imbalance(directions, sizes, window=3)

        # (300 - 100 + 100) / (300 + 100 + 100) = 300/500 = 0.6
        expected = (300 - 100 + 100) / 500
        assert np.isclose(imbalance[2], expected)

    def test_compute_trade_imbalance_mismatched_shapes(self) -> None:
        """Mismatched shapes raises ValueError."""
        builder = OrderFlowFeatureBuilder()
        with pytest.raises(ValueError, match="same shape"):
            builder.compute_trade_imbalance(
                np.array([1, -1, 1]),
                np.array([100.0, 100.0]),
                window=2,
            )

    def test_compute_trade_imbalance_invalid_window(self) -> None:
        """Window < 1 raises ValueError."""
        builder = OrderFlowFeatureBuilder()
        with pytest.raises(ValueError, match="at least 1"):
            builder.compute_trade_imbalance(
                np.array([1, -1]),
                np.array([100.0, 100.0]),
                window=0,
            )

    def test_compute_size_quantiles(self) -> None:
        """Size quantiles are computed correctly."""
        config = OrderFlowFeatureConfig(size_quantiles=(0.25, 0.5, 0.75))
        builder = OrderFlowFeatureBuilder(config)
        sizes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        quantiles = builder.compute_size_quantiles(sizes, window=4)

        # First 3 elements are NaN
        assert np.all(np.isnan(quantiles[:3, :]))
        # At index 3: window = [10, 20, 30, 40]
        expected_q25 = np.quantile([10, 20, 30, 40], 0.25)
        expected_q50 = np.quantile([10, 20, 30, 40], 0.5)
        expected_q75 = np.quantile([10, 20, 30, 40], 0.75)
        assert np.isclose(quantiles[3, 0], expected_q25)
        assert np.isclose(quantiles[3, 1], expected_q50)
        assert np.isclose(quantiles[3, 2], expected_q75)

    def test_compute_size_quantiles_rejects_2d(self) -> None:
        """2D input raises ValueError."""
        builder = OrderFlowFeatureBuilder()
        with pytest.raises(ValueError, match="1-dimensional"):
            builder.compute_size_quantiles(np.array([[10.0, 20.0]]), window=2)

    def test_compute_arrival_rate(self) -> None:
        """Arrival rate is computed correctly."""
        builder = OrderFlowFeatureBuilder()
        # 5 trades over 4 seconds = 1 trade/second
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        rate = builder.compute_arrival_rate(timestamps, window=5)

        # At index 4: 4 trades in 4 seconds = 1 trade/sec
        assert np.isnan(rate[0])
        assert np.isnan(rate[3])
        assert np.isclose(rate[4], 1.0)

    def test_compute_arrival_rate_variable_timing(self) -> None:
        """Arrival rate handles variable timing."""
        builder = OrderFlowFeatureBuilder()
        # Trades at 0, 0.5, 1.0, 2.0 (accelerating, then slowing)
        timestamps = np.array([0.0, 0.5, 1.0, 2.0])
        rate = builder.compute_arrival_rate(timestamps, window=3)

        # At index 2: 2 trades in 1 second = 2 trades/sec
        assert np.isclose(rate[2], 2.0)
        # At index 3: 2 trades in 1.5 seconds = 1.33 trades/sec
        assert np.isclose(rate[3], 2 / 1.5)

    def test_compute_arrival_rate_rejects_small_window(self) -> None:
        """Window < 2 raises ValueError."""
        builder = OrderFlowFeatureBuilder()
        with pytest.raises(ValueError, match="at least 2"):
            builder.compute_arrival_rate(np.array([0.0, 1.0]), window=1)

    def test_compute_features(self) -> None:
        """compute_features returns correct shape."""
        config = OrderFlowFeatureConfig(
            imbalance_window=3,
            include_size_distribution=True,
            include_arrival_rate=True,
            size_quantiles=(0.5,),
            arrival_rate_window=3,
        )
        builder = OrderFlowFeatureBuilder(config)

        prices = np.array([100.0, 101.0, 100.5, 101.5, 102.0])
        sizes = np.array([100.0, 150.0, 200.0, 100.0, 150.0])
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        features = builder.compute_features(prices, sizes, timestamps)

        # 1 imbalance + 1 quantile + 1 arrival rate = 3
        assert features.shape == (5, 3)

    def test_compute_features_minimal(self) -> None:
        """compute_features with minimal options."""
        config = OrderFlowFeatureConfig(
            imbalance_window=2,
            include_size_distribution=False,
            include_arrival_rate=False,
        )
        builder = OrderFlowFeatureBuilder(config)

        prices = np.array([100.0, 101.0, 102.0])
        sizes = np.array([100.0, 100.0, 100.0])

        features = builder.compute_features(prices, sizes)

        # Only imbalance
        assert features.shape == (3, 1)

    def test_compute_features_requires_timestamps(self) -> None:
        """compute_features with arrival_rate requires timestamps."""
        config = OrderFlowFeatureConfig(include_arrival_rate=True)
        builder = OrderFlowFeatureBuilder(config)

        prices = np.array([100.0, 101.0, 102.0])
        sizes = np.array([100.0, 100.0, 100.0])

        with pytest.raises(ValueError, match="timestamps required"):
            builder.compute_features(prices, sizes)

    def test_compute_features_mismatched_shapes(self) -> None:
        """Mismatched prices/sizes raises ValueError."""
        builder = OrderFlowFeatureBuilder()
        with pytest.raises(ValueError, match="same shape"):
            builder.compute_features(
                np.array([100.0, 101.0]),
                np.array([100.0]),
            )

    def test_get_feature_names(self) -> None:
        """Feature names match feature order."""
        config = OrderFlowFeatureConfig(
            include_size_distribution=True,
            include_arrival_rate=True,
            size_quantiles=(0.25, 0.5, 0.75),
        )
        builder = OrderFlowFeatureBuilder(config)

        names = builder.get_feature_names()
        assert names == [
            "trade_imbalance",
            "size_q25",
            "size_q50",
            "size_q75",
            "arrival_rate",
        ]

    def test_get_feature_names_minimal(self) -> None:
        """Feature names with minimal options."""
        config = OrderFlowFeatureConfig(
            include_size_distribution=False,
            include_arrival_rate=False,
        )
        builder = OrderFlowFeatureBuilder(config)

        names = builder.get_feature_names()
        assert names == ["trade_imbalance"]

    def test_num_features(self) -> None:
        """num_features property returns correct count."""
        config = OrderFlowFeatureConfig(
            include_size_distribution=True,
            include_arrival_rate=True,
            size_quantiles=(0.25, 0.5, 0.75),
        )
        builder = OrderFlowFeatureBuilder(config)
        assert builder.num_features == 5  # 1 + 3 + 1

        config_minimal = OrderFlowFeatureConfig(
            include_size_distribution=False,
            include_arrival_rate=False,
        )
        builder_minimal = OrderFlowFeatureBuilder(config_minimal)
        assert builder_minimal.num_features == 1


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_empty_array(self) -> None:
        """Empty arrays are handled gracefully."""
        price_builder = PriceFeatureBuilder()
        vol_builder = VolatilityFeatureBuilder()
        order_builder = OrderFlowFeatureBuilder()

        empty = np.array([])
        assert price_builder.compute_log_returns(empty, window=1).shape == (0,)
        assert price_builder.compute_vwap(empty, empty, window=1).shape == (0,)
        assert vol_builder.compute_realized_volatility(empty, window=2).shape == (0,)
        assert order_builder.infer_trade_direction(empty).shape == (0,)

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


class TestOptionsFeatureConfig:
    """Tests for OptionsFeatureConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config has expected values."""
        config = OptionsFeatureConfig()
        assert config.include_iv_surface is True
        assert config.include_greeks is True
        assert config.include_put_call_ratio is True
        assert config.include_term_structure is True
        assert config.iv_moneyness_buckets == (0.95, 0.975, 1.0, 1.025, 1.05)
        assert config.expiry_buckets_days == (7, 30, 60, 90)

    def test_custom_config(self) -> None:
        """Custom config accepts valid values."""
        config = OptionsFeatureConfig(
            include_iv_surface=False,
            include_greeks=False,
            include_put_call_ratio=True,
            include_term_structure=False,
            iv_moneyness_buckets=(0.9, 1.0, 1.1),
            expiry_buckets_days=(14, 28, 56),
        )
        assert config.include_iv_surface is False
        assert config.include_greeks is False
        assert config.include_put_call_ratio is True
        assert config.include_term_structure is False
        assert config.iv_moneyness_buckets == (0.9, 1.0, 1.1)
        assert config.expiry_buckets_days == (14, 28, 56)


class TestOptionsFeatureBuilder:
    """Tests for OptionsFeatureBuilder class."""

    def test_default_initialization(self) -> None:
        """Builder initializes with default config."""
        builder = OptionsFeatureBuilder()
        assert builder.config.include_iv_surface is True
        assert builder.config.expiry_buckets_days == (7, 30, 60, 90)

    def test_compute_moneyness(self) -> None:
        """Moneyness is computed correctly."""
        builder = OptionsFeatureBuilder()
        strikes = np.array([95.0, 100.0, 105.0])
        moneyness = builder.compute_moneyness(strikes, spot_price=100.0)

        assert np.allclose(moneyness, [0.95, 1.0, 1.05])

    def test_compute_moneyness_zero_spot(self) -> None:
        """Zero spot price raises ValueError."""
        builder = OptionsFeatureBuilder()
        with pytest.raises(ValueError, match="spot_price must be positive"):
            builder.compute_moneyness(np.array([100.0]), spot_price=0.0)

    def test_compute_moneyness_negative_spot(self) -> None:
        """Negative spot price raises ValueError."""
        builder = OptionsFeatureBuilder()
        with pytest.raises(ValueError, match="spot_price must be positive"):
            builder.compute_moneyness(np.array([100.0]), spot_price=-10.0)

    def test_compute_days_to_expiry(self) -> None:
        """Days to expiry computed correctly."""
        builder = OptionsFeatureBuilder()
        current = 1000000.0  # Unix seconds
        day_seconds = 24 * 3600
        expirations = np.array([
            current + 7 * day_seconds,
            current + 30 * day_seconds,
            current + 60 * day_seconds,
        ])
        dte = builder.compute_days_to_expiry(expirations, current)

        assert np.allclose(dte, [7.0, 30.0, 60.0])

    def test_compute_days_to_expiry_past_expiration(self) -> None:
        """Past expirations return 0 DTE (clamped)."""
        builder = OptionsFeatureBuilder()
        current = 1000000.0
        expirations = np.array([current - 86400])  # 1 day ago
        dte = builder.compute_days_to_expiry(expirations, current)

        assert dte[0] == 0.0

    def test_compute_atm_iv_by_expiry(self) -> None:
        """ATM IV by expiry bucket computed correctly."""
        config = OptionsFeatureConfig(expiry_buckets_days=(7, 30))
        builder = OptionsFeatureBuilder(config)

        spot = 100.0
        current = 1000000.0
        day_seconds = 24 * 3600

        # Create options at various strikes and expiries
        strikes = np.array([99.0, 100.0, 101.0, 99.0, 100.0, 101.0])
        expirations = np.array([
            current + 7 * day_seconds,
            current + 7 * day_seconds,
            current + 7 * day_seconds,
            current + 30 * day_seconds,
            current + 30 * day_seconds,
            current + 30 * day_seconds,
        ])
        ivs = np.array([0.22, 0.20, 0.21, 0.27, 0.25, 0.26])

        atm_ivs = builder.compute_atm_iv_by_expiry(
            strikes, expirations, ivs, spot, current
        )

        assert len(atm_ivs) == 2
        # 7-day bucket: ATM strike 100 has IV 0.20
        # weighted average will favor the 100 strike
        assert 0.19 < atm_ivs[0] < 0.23
        # 30-day bucket: ATM strike 100 has IV 0.25
        assert 0.24 < atm_ivs[1] < 0.28

    def test_compute_atm_iv_by_expiry_no_matching_options(self) -> None:
        """Returns NaN when no options match a bucket."""
        config = OptionsFeatureConfig(expiry_buckets_days=(7, 90))
        builder = OptionsFeatureBuilder(config)

        spot = 100.0
        current = 1000000.0
        day_seconds = 24 * 3600

        # Only have 30-day options
        strikes = np.array([100.0])
        expirations = np.array([current + 30 * day_seconds])
        ivs = np.array([0.25])

        atm_ivs = builder.compute_atm_iv_by_expiry(
            strikes, expirations, ivs, spot, current
        )

        # 7-day and 90-day buckets have no options
        assert np.isnan(atm_ivs[0])
        assert np.isnan(atm_ivs[1])

    def test_compute_iv_surface_features(self) -> None:
        """IV surface features computed for moneyness x expiry grid."""
        config = OptionsFeatureConfig(
            iv_moneyness_buckets=(0.95, 1.0, 1.05),
            expiry_buckets_days=(7, 30),
        )
        builder = OptionsFeatureBuilder(config)

        spot = 100.0
        current = 1000000.0
        day_seconds = 24 * 3600

        # Create a grid of options
        strikes = np.array([95.0, 100.0, 105.0, 95.0, 100.0, 105.0])
        expirations = np.array([
            current + 7 * day_seconds,
            current + 7 * day_seconds,
            current + 7 * day_seconds,
            current + 30 * day_seconds,
            current + 30 * day_seconds,
            current + 30 * day_seconds,
        ])
        # IV smile: higher IV for OTM
        ivs = np.array([0.25, 0.20, 0.24, 0.30, 0.25, 0.29])

        surface = builder.compute_iv_surface_features(
            strikes, expirations, ivs, spot, current
        )

        # 2 expiries x 3 moneyness = 6 values
        assert len(surface) == 6
        # Check that values are reasonable (not all NaN)
        assert not np.all(np.isnan(surface))

    def test_compute_aggregated_greeks(self) -> None:
        """Aggregated Greeks computed correctly."""
        builder = OptionsFeatureBuilder()

        deltas = np.array([0.5, 0.3, -0.4, -0.6])  # Call, call, put, put
        gammas = np.array([0.05, 0.03, 0.04, 0.06])
        vegas = np.array([10.0, 8.0, 9.0, 11.0])
        thetas = np.array([-0.5, -0.4, -0.3, -0.6])
        open_interests = np.array([1000.0, 500.0, 800.0, 700.0])
        option_types = np.array([1, 1, -1, -1], dtype=np.int8)

        greeks = builder.compute_aggregated_greeks(
            deltas, gammas, vegas, thetas, open_interests, option_types
        )

        assert len(greeks) == 4
        # Net delta: OI-weighted average
        total_oi = 3000.0
        expected_delta = (0.5 * 1000 + 0.3 * 500 - 0.4 * 800 - 0.6 * 700) / total_oi
        assert np.isclose(greeks[0], expected_delta)

    def test_compute_aggregated_greeks_zero_oi(self) -> None:
        """Returns NaN when total OI is zero."""
        builder = OptionsFeatureBuilder()

        greeks = builder.compute_aggregated_greeks(
            np.array([0.5]),
            np.array([0.05]),
            np.array([10.0]),
            np.array([-0.5]),
            np.array([0.0]),  # Zero OI
            np.array([1], dtype=np.int8),
        )

        assert np.all(np.isnan(greeks))

    def test_compute_put_call_ratios(self) -> None:
        """Put/call ratios computed correctly."""
        builder = OptionsFeatureBuilder()

        option_types = np.array([1, 1, -1, -1], dtype=np.int8)
        volumes = np.array([1000.0, 500.0, 800.0, 1200.0])
        open_interests = np.array([5000.0, 3000.0, 4000.0, 6000.0])

        ratios = builder.compute_put_call_ratios(option_types, volumes, open_interests)

        # Volume ratio: put_vol / call_vol = 2000 / 1500 = 1.333
        assert np.isclose(ratios[0], 2000 / 1500)
        # OI ratio: put_oi / call_oi = 10000 / 8000 = 1.25
        assert np.isclose(ratios[1], 10000 / 8000)

    def test_compute_put_call_ratios_no_calls(self) -> None:
        """Returns NaN when no call volume/OI."""
        builder = OptionsFeatureBuilder()

        option_types = np.array([-1, -1], dtype=np.int8)
        volumes = np.array([1000.0, 500.0])
        open_interests = np.array([5000.0, 3000.0])

        ratios = builder.compute_put_call_ratios(option_types, volumes, open_interests)

        assert np.isnan(ratios[0])  # No call volume
        assert np.isnan(ratios[1])  # No call OI

    def test_compute_term_structure_slope_contango(self) -> None:
        """Term structure slope positive for contango (IV increases with DTE)."""
        config = OptionsFeatureConfig(expiry_buckets_days=(7, 30, 60, 90))
        builder = OptionsFeatureBuilder(config)

        # IV increases with DTE (contango)
        atm_ivs = np.array([0.20, 0.22, 0.24, 0.26])

        slope = builder.compute_term_structure_slope(atm_ivs)

        # Positive slope
        assert slope > 0

    def test_compute_term_structure_slope_backwardation(self) -> None:
        """Term structure slope negative for backwardation (IV decreases with DTE)."""
        config = OptionsFeatureConfig(expiry_buckets_days=(7, 30, 60, 90))
        builder = OptionsFeatureBuilder(config)

        # IV decreases with DTE (backwardation)
        atm_ivs = np.array([0.30, 0.26, 0.24, 0.22])

        slope = builder.compute_term_structure_slope(atm_ivs)

        # Negative slope
        assert slope < 0

    def test_compute_term_structure_slope_flat(self) -> None:
        """Term structure slope near zero for flat curve."""
        config = OptionsFeatureConfig(expiry_buckets_days=(7, 30, 60, 90))
        builder = OptionsFeatureBuilder(config)

        # Flat IV term structure
        atm_ivs = np.array([0.25, 0.25, 0.25, 0.25])

        slope = builder.compute_term_structure_slope(atm_ivs)

        assert np.isclose(slope, 0.0, atol=1e-10)

    def test_compute_term_structure_slope_insufficient_data(self) -> None:
        """Returns NaN with fewer than 2 valid data points."""
        config = OptionsFeatureConfig(expiry_buckets_days=(7, 30, 60, 90))
        builder = OptionsFeatureBuilder(config)

        # Only one valid IV
        atm_ivs = np.array([0.25, np.nan, np.nan, np.nan])

        slope = builder.compute_term_structure_slope(atm_ivs)

        assert np.isnan(slope)

    def test_compute_features_all_enabled(self) -> None:
        """compute_features returns all features when enabled."""
        config = OptionsFeatureConfig(
            iv_moneyness_buckets=(0.95, 1.0, 1.05),
            expiry_buckets_days=(7, 30),
        )
        builder = OptionsFeatureBuilder(config)

        spot = 100.0
        current = 1000000.0
        day_seconds = 24 * 3600

        strikes = np.array([95.0, 100.0, 105.0, 95.0, 100.0, 105.0])
        expirations = np.array([
            current + 7 * day_seconds,
            current + 7 * day_seconds,
            current + 7 * day_seconds,
            current + 30 * day_seconds,
            current + 30 * day_seconds,
            current + 30 * day_seconds,
        ])
        ivs = np.array([0.25, 0.20, 0.24, 0.30, 0.25, 0.29])
        option_types = np.array([1, 1, 1, -1, -1, -1], dtype=np.int8)
        deltas = np.array([0.3, 0.5, 0.7, -0.7, -0.5, -0.3])
        gammas = np.array([0.05, 0.04, 0.03, 0.03, 0.04, 0.05])
        vegas = np.array([10.0, 12.0, 10.0, 10.0, 12.0, 10.0])
        thetas = np.array([-0.5, -0.4, -0.3, -0.3, -0.4, -0.5])
        volumes = np.array([100.0, 200.0, 150.0, 180.0, 250.0, 120.0])
        open_interests = np.array([1000.0, 2000.0, 1500.0, 1800.0, 2500.0, 1200.0])

        features = builder.compute_features(
            strikes,
            expirations,
            ivs,
            spot,
            current,
            option_types=option_types,
            deltas=deltas,
            gammas=gammas,
            vegas=vegas,
            thetas=thetas,
            volumes=volumes,
            open_interests=open_interests,
        )

        # IV surface: 2 expiries x 3 moneyness = 6
        # Greeks: 4
        # Put/call ratios: 2
        # Term structure: 1
        # Total: 13
        expected_features = 6 + 4 + 2 + 1
        assert len(features) == expected_features
        assert features.shape == (expected_features,)

    def test_compute_features_iv_surface_only(self) -> None:
        """compute_features with only IV surface enabled."""
        config = OptionsFeatureConfig(
            include_iv_surface=True,
            include_greeks=False,
            include_put_call_ratio=False,
            include_term_structure=False,
            iv_moneyness_buckets=(1.0,),
            expiry_buckets_days=(30,),
        )
        builder = OptionsFeatureBuilder(config)

        spot = 100.0
        current = 1000000.0
        day_seconds = 24 * 3600

        strikes = np.array([100.0])
        expirations = np.array([current + 30 * day_seconds])
        ivs = np.array([0.25])

        features = builder.compute_features(
            strikes, expirations, ivs, spot, current
        )

        # Only 1 IV surface point
        assert len(features) == 1

    def test_compute_features_requires_greeks_data(self) -> None:
        """compute_features raises error when Greeks data missing."""
        config = OptionsFeatureConfig(include_greeks=True)
        builder = OptionsFeatureBuilder(config)

        with pytest.raises(ValueError, match="Greeks data required"):
            builder.compute_features(
                np.array([100.0]),
                np.array([1000000.0]),
                np.array([0.25]),
                100.0,
                1000000.0,
            )

    def test_compute_features_requires_ratio_data(self) -> None:
        """compute_features raises error when ratio data missing."""
        config = OptionsFeatureConfig(
            include_greeks=False,
            include_put_call_ratio=True,
        )
        builder = OptionsFeatureBuilder(config)

        with pytest.raises(ValueError, match="option_types, volumes, and open_interests"):
            builder.compute_features(
                np.array([100.0]),
                np.array([1000000.0]),
                np.array([0.25]),
                100.0,
                1000000.0,
            )

    def test_compute_features_empty_when_all_disabled(self) -> None:
        """compute_features returns empty array when all features disabled."""
        config = OptionsFeatureConfig(
            include_iv_surface=False,
            include_greeks=False,
            include_put_call_ratio=False,
            include_term_structure=False,
        )
        builder = OptionsFeatureBuilder(config)

        features = builder.compute_features(
            np.array([100.0]),
            np.array([1000000.0]),
            np.array([0.25]),
            100.0,
            1000000.0,
        )

        assert len(features) == 0

    def test_get_feature_names(self) -> None:
        """Feature names match expected format."""
        config = OptionsFeatureConfig(
            iv_moneyness_buckets=(0.95, 1.0, 1.05),
            expiry_buckets_days=(7, 30),
        )
        builder = OptionsFeatureBuilder(config)

        names = builder.get_feature_names()

        # IV surface: 6 names (2 expiry x 3 moneyness)
        # Greeks: 4
        # Ratios: 2
        # Term structure: 1
        assert len(names) == 13

        # Check IV surface naming format
        assert "iv_7d_m95" in names
        assert "iv_30d_m100" in names
        assert "iv_7d_m105" in names

        # Check Greeks names
        assert "net_delta" in names
        assert "total_gamma" in names
        assert "total_vega" in names
        assert "total_theta" in names

        # Check ratio names
        assert "volume_pc_ratio" in names
        assert "oi_pc_ratio" in names

        # Check term structure name
        assert "iv_term_slope" in names

    def test_get_feature_names_minimal(self) -> None:
        """Feature names with minimal config."""
        config = OptionsFeatureConfig(
            include_iv_surface=False,
            include_greeks=False,
            include_put_call_ratio=True,
            include_term_structure=False,
        )
        builder = OptionsFeatureBuilder(config)

        names = builder.get_feature_names()

        assert names == ["volume_pc_ratio", "oi_pc_ratio"]

    def test_num_features(self) -> None:
        """num_features property returns correct count."""
        config = OptionsFeatureConfig(
            iv_moneyness_buckets=(0.95, 1.0, 1.05),
            expiry_buckets_days=(7, 30),
        )
        builder = OptionsFeatureBuilder(config)

        # 6 IV surface + 4 Greeks + 2 ratios + 1 term structure = 13
        assert builder.num_features == 13

    def test_num_features_default(self) -> None:
        """num_features with default config."""
        builder = OptionsFeatureBuilder()

        # 4 expiry x 5 moneyness = 20 IV surface
        # + 4 Greeks + 2 ratios + 1 term structure = 27
        assert builder.num_features == 27

    def test_num_features_minimal(self) -> None:
        """num_features with minimal config."""
        config = OptionsFeatureConfig(
            include_iv_surface=False,
            include_greeks=False,
            include_put_call_ratio=False,
            include_term_structure=True,
        )
        builder = OptionsFeatureBuilder(config)

        assert builder.num_features == 1
