"""Tests for sequence_builder module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from data.processors.sequence_builder import (
    AlignedSequence,
    OvernightStrategy,
    SequenceBuilder,
    SequenceConfig,
)


class TestSequenceConfig:
    """Tests for SequenceConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SequenceConfig()
        assert config.lookback_seconds == 300
        assert config.sample_interval == 1
        assert config.overnight_strategy == OvernightStrategy.RESET
        assert config.market_open_hour == 9
        assert config.market_close_hour == 16

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SequenceConfig(
            lookback_seconds=600,
            sample_interval=5,
            overnight_strategy=OvernightStrategy.MASK,
            market_open_hour=8,
            market_close_hour=17,
        )
        assert config.lookback_seconds == 600
        assert config.sample_interval == 5
        assert config.overnight_strategy == OvernightStrategy.MASK

    def test_sequence_length_property(self) -> None:
        """Test sequence_length computation."""
        config = SequenceConfig(lookback_seconds=300, sample_interval=1)
        assert config.sequence_length == 300

        config2 = SequenceConfig(lookback_seconds=300, sample_interval=5)
        assert config2.sequence_length == 60

    def test_invalid_lookback_seconds(self) -> None:
        """Test validation of lookback_seconds."""
        with pytest.raises(ValueError, match="lookback_seconds must be at least 1"):
            SequenceConfig(lookback_seconds=0)

    def test_invalid_sample_interval(self) -> None:
        """Test validation of sample_interval."""
        with pytest.raises(ValueError, match="sample_interval must be at least 1"):
            SequenceConfig(sample_interval=0)

    def test_lookback_less_than_interval(self) -> None:
        """Test that lookback must be >= sample_interval."""
        with pytest.raises(ValueError, match="lookback_seconds must be >= sample_interval"):
            SequenceConfig(lookback_seconds=1, sample_interval=5)

    def test_invalid_market_hours(self) -> None:
        """Test validation of market hours."""
        with pytest.raises(ValueError, match="market_open_hour"):
            SequenceConfig(market_open_hour=-1)

        with pytest.raises(ValueError, match="market_open_hour"):
            SequenceConfig(market_open_hour=24)

        with pytest.raises(ValueError, match="market_close_hour"):
            SequenceConfig(market_close_hour=-1)

        with pytest.raises(ValueError, match="market_close_hour"):
            SequenceConfig(market_close_hour=25)


class TestSequenceBuilderInit:
    """Tests for SequenceBuilder initialization."""

    def test_default_config(self) -> None:
        """Test initialization with default config."""
        builder = SequenceBuilder()
        assert builder.config.lookback_seconds == 300

    def test_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = SequenceConfig(lookback_seconds=60, sample_interval=2)
        builder = SequenceBuilder(config)
        assert builder.config.lookback_seconds == 60
        assert builder.config.sample_interval == 2


class TestCreateTimeGrid:
    """Tests for time grid creation."""

    def test_basic_grid(self) -> None:
        """Test basic time grid creation."""
        config = SequenceConfig(lookback_seconds=10, sample_interval=1)
        builder = SequenceBuilder(config)

        end_time = 1000.0
        grid = builder.create_time_grid(end_time)

        assert len(grid) == 10
        assert grid[-1] == 1000.0
        assert grid[0] == 991.0  # 1000 - 9
        np.testing.assert_allclose(np.diff(grid), 1.0)

    def test_grid_with_interval(self) -> None:
        """Test grid with non-unit sample interval."""
        config = SequenceConfig(lookback_seconds=20, sample_interval=5)
        builder = SequenceBuilder(config)

        end_time = 100.0
        grid = builder.create_time_grid(end_time)

        assert len(grid) == 4  # 20 / 5
        assert grid[-1] == 100.0
        assert grid[0] == 85.0  # 100 - 15
        np.testing.assert_allclose(np.diff(grid), 5.0)


class TestForwardFillToGrid:
    """Tests for forward-fill alignment."""

    def test_basic_forward_fill(self) -> None:
        """Test basic forward-fill alignment."""
        builder = SequenceBuilder()

        timestamps = np.array([0.0, 2.0, 5.0])
        values = np.array([10.0, 20.0, 30.0])
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        result, valid = builder.forward_fill_to_grid(timestamps, values, grid)

        # At t=0: value 10 (exact match)
        # At t=1: value 10 (forward-fill from t=0)
        # At t=2: value 20 (exact match)
        # At t=3,4: value 20 (forward-fill from t=2)
        # At t=5: value 30 (exact match)
        expected = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
        np.testing.assert_allclose(result, expected)
        assert all(valid)

    def test_grid_before_first_observation(self) -> None:
        """Test grid points before first observation are invalid."""
        builder = SequenceBuilder()

        timestamps = np.array([5.0, 10.0])
        values = np.array([100.0, 200.0])
        grid = np.array([0.0, 2.0, 5.0, 7.0, 10.0])

        result, valid = builder.forward_fill_to_grid(timestamps, values, grid)

        # t=0,2 are before first obs, should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert not valid[0]
        assert not valid[1]

        # t=5,7,10 are valid
        assert result[2] == 100.0
        assert result[3] == 100.0
        assert result[4] == 200.0
        assert valid[2]
        assert valid[3]
        assert valid[4]

    def test_empty_observations(self) -> None:
        """Test with no observations."""
        builder = SequenceBuilder()

        timestamps = np.array([])
        values = np.array([])
        grid = np.array([0.0, 1.0, 2.0])

        result, valid = builder.forward_fill_to_grid(timestamps, values, grid)

        assert all(np.isnan(result))
        assert not any(valid)

    def test_empty_grid(self) -> None:
        """Test with empty grid."""
        builder = SequenceBuilder()

        timestamps = np.array([1.0, 2.0])
        values = np.array([10.0, 20.0])
        grid = np.array([])

        result, valid = builder.forward_fill_to_grid(timestamps, values, grid)

        assert len(result) == 0
        assert len(valid) == 0

    def test_invalid_input_dimensions(self) -> None:
        """Test error handling for invalid input dimensions."""
        builder = SequenceBuilder()

        with pytest.raises(ValueError, match="1-dimensional"):
            builder.forward_fill_to_grid(
                np.array([[1.0, 2.0]]),  # 2D
                np.array([1.0, 2.0]),
                np.array([1.0]),
            )

        with pytest.raises(ValueError, match="same length"):
            builder.forward_fill_to_grid(
                np.array([1.0, 2.0]),
                np.array([1.0]),  # different length
                np.array([1.0]),
            )


class TestForwardFill2DToGrid:
    """Tests for 2D forward-fill alignment."""

    def test_basic_2d_forward_fill(self) -> None:
        """Test basic 2D forward-fill alignment."""
        builder = SequenceBuilder()

        timestamps = np.array([0.0, 5.0])
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        grid = np.array([0.0, 2.0, 5.0, 7.0])

        result, valid = builder.forward_fill_2d_to_grid(timestamps, values, grid)

        assert result.shape == (4, 3)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[1], [1.0, 2.0, 3.0])  # forward-fill
        np.testing.assert_allclose(result[2], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(result[3], [4.0, 5.0, 6.0])  # forward-fill

    def test_2d_invalid_dimensions(self) -> None:
        """Test error handling for invalid 2D input dimensions."""
        builder = SequenceBuilder()

        with pytest.raises(ValueError, match="2-dimensional"):
            builder.forward_fill_2d_to_grid(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),  # 1D, should be 2D
                np.array([1.0]),
            )


class TestDetectOvernightGaps:
    """Tests for overnight gap detection."""

    def test_no_gaps(self) -> None:
        """Test when there are no overnight gaps."""
        builder = SequenceBuilder()

        timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        gap_mask = builder.detect_overnight_gaps(timestamps)

        assert not any(gap_mask)

    def test_single_overnight_gap(self) -> None:
        """Test detection of a single overnight gap."""
        builder = SequenceBuilder()

        # Gap of 5 hours (18000 seconds) exceeds 4-hour threshold
        timestamps = np.array([0.0, 1.0, 2.0, 18002.0, 18003.0])
        gap_mask = builder.detect_overnight_gaps(timestamps)

        expected = np.array([False, False, False, True, False])
        np.testing.assert_array_equal(gap_mask, expected)

    def test_multiple_gaps(self) -> None:
        """Test detection of multiple overnight gaps."""
        builder = SequenceBuilder()

        timestamps = np.array([0.0, 1.0, 20000.0, 20001.0, 40000.0])
        gap_mask = builder.detect_overnight_gaps(timestamps)

        expected = np.array([False, False, True, False, True])
        np.testing.assert_array_equal(gap_mask, expected)

    def test_empty_timestamps(self) -> None:
        """Test with empty timestamps."""
        builder = SequenceBuilder()

        gap_mask = builder.detect_overnight_gaps(np.array([]))
        assert len(gap_mask) == 0

    def test_single_timestamp(self) -> None:
        """Test with single timestamp."""
        builder = SequenceBuilder()

        gap_mask = builder.detect_overnight_gaps(np.array([100.0]))
        assert len(gap_mask) == 1
        assert not gap_mask[0]


class TestApplyOvernightStrategy:
    """Tests for overnight gap strategy application."""

    def test_reset_strategy(self) -> None:
        """Test RESET strategy invalidates data before gaps."""
        config = SequenceConfig(overnight_strategy=OvernightStrategy.RESET)
        builder = SequenceBuilder(config)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gap_mask = np.array([False, False, False, True, False])

        result = builder.apply_overnight_strategy(data, gap_mask)

        # All positions before gap (indices 0,1,2) should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert result[3] == 4.0
        assert result[4] == 5.0

    def test_mask_strategy(self) -> None:
        """Test MASK strategy only invalidates gap positions."""
        config = SequenceConfig(overnight_strategy=OvernightStrategy.MASK)
        builder = SequenceBuilder(config)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gap_mask = np.array([False, False, False, True, False])

        result = builder.apply_overnight_strategy(data, gap_mask)

        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] == 3.0
        assert np.isnan(result[3])  # Only gap position is NaN
        assert result[4] == 5.0

    def test_interpolate_strategy(self) -> None:
        """Test INTERPOLATE strategy leaves data unchanged."""
        config = SequenceConfig(overnight_strategy=OvernightStrategy.INTERPOLATE)
        builder = SequenceBuilder(config)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gap_mask = np.array([False, False, False, True, False])

        result = builder.apply_overnight_strategy(data, gap_mask)

        np.testing.assert_allclose(result, data)

    def test_2d_mask_strategy(self) -> None:
        """Test MASK strategy with 2D data."""
        config = SequenceConfig(overnight_strategy=OvernightStrategy.MASK)
        builder = SequenceBuilder(config)

        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        gap_mask = np.array([False, True, False])

        result = builder.apply_overnight_strategy(data, gap_mask)

        np.testing.assert_allclose(result[0], [1.0, 2.0])
        assert all(np.isnan(result[1]))
        np.testing.assert_allclose(result[2], [5.0, 6.0])


class TestAggregateVolumes:
    """Tests for volume aggregation to grid."""

    def test_basic_aggregation(self) -> None:
        """Test basic volume aggregation."""
        config = SequenceConfig(lookback_seconds=10, sample_interval=2)
        builder = SequenceBuilder(config)

        # Trades at various times
        timestamps = np.array([0.0, 0.5, 1.0, 2.5, 3.0, 4.0])
        volumes = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        # Grid at [0, 2, 4, 6, 8]
        grid = np.array([0.0, 2.0, 4.0, 6.0, 8.0])

        result = builder._aggregate_volumes_to_grid(timestamps, volumes, grid)

        # [0, 2): trades at 0.0, 0.5, 1.0 -> 10 + 20 + 30 = 60
        # [2, 4): trades at 2.5, 3.0 -> 40 + 50 = 90
        # [4, 6): trade at 4.0 -> 60
        # [6, 8): no trades -> 0
        # [8, 10): no trades -> 0
        expected = np.array([60.0, 90.0, 60.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_empty_volumes(self) -> None:
        """Test with no trades."""
        builder = SequenceBuilder()

        timestamps = np.array([])
        volumes = np.array([])
        grid = np.array([0.0, 1.0, 2.0])

        result = builder._aggregate_volumes_to_grid(timestamps, volumes, grid)

        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


class TestAlignTradeData:
    """Tests for trade data alignment."""

    def test_basic_alignment(self) -> None:
        """Test basic trade data alignment."""
        config = SequenceConfig(lookback_seconds=5, sample_interval=1)
        builder = SequenceBuilder(config)

        # Trades
        trade_ts = np.array([95.0, 97.0, 99.0])
        trade_prices = np.array([100.0, 101.0, 102.0])
        trade_volumes = np.array([1000.0, 2000.0, 3000.0])

        result = builder.align_trade_data(trade_ts, trade_prices, trade_volumes, 100.0)

        # Grid should be [96, 97, 98, 99, 100]
        assert len(result.timestamps) == 5
        assert result.timestamps[-1] == 100.0

        # Prices: forward-fill
        # t=96: from t=95 -> 100.0
        # t=97: from t=97 -> 101.0
        # t=98: from t=97 -> 101.0
        # t=99: from t=99 -> 102.0
        # t=100: from t=99 -> 102.0
        expected_prices = np.array([100.0, 101.0, 101.0, 102.0, 102.0])
        np.testing.assert_allclose(result.prices, expected_prices)

    def test_mismatched_lengths(self) -> None:
        """Test error handling for mismatched array lengths."""
        builder = SequenceBuilder()

        with pytest.raises(ValueError, match="same length"):
            builder.align_trade_data(
                np.array([1.0, 2.0]),
                np.array([100.0]),  # wrong length
                np.array([1000.0, 2000.0]),
                10.0,
            )


class TestAlignOptionsData:
    """Tests for options data alignment."""

    def test_1d_options_alignment(self) -> None:
        """Test alignment of 1D options data."""
        builder = SequenceBuilder()

        options_ts = np.array([0.0, 5.0])
        options_values = {"iv": np.array([0.25, 0.30])}
        grid = np.array([0.0, 2.0, 5.0, 7.0])

        result = builder.align_options_data(options_ts, options_values, grid)

        assert "iv" in result
        expected_iv = np.array([0.25, 0.25, 0.30, 0.30])
        np.testing.assert_allclose(result["iv"], expected_iv)

    def test_2d_options_alignment(self) -> None:
        """Test alignment of 2D options data."""
        builder = SequenceBuilder()

        options_ts = np.array([0.0, 5.0])
        options_values = {
            "greeks": np.array([[0.5, 0.1], [0.6, 0.2]])  # delta, gamma
        }
        grid = np.array([0.0, 2.0, 5.0])

        result = builder.align_options_data(options_ts, options_values, grid)

        assert result["greeks"].shape == (3, 2)
        np.testing.assert_allclose(result["greeks"][0], [0.5, 0.1])
        np.testing.assert_allclose(result["greeks"][1], [0.5, 0.1])
        np.testing.assert_allclose(result["greeks"][2], [0.6, 0.2])

    def test_invalid_dimensions(self) -> None:
        """Test error for 3D options data."""
        builder = SequenceBuilder()

        options_ts = np.array([0.0])
        # Intentionally pass 3D array to test error handling
        options_values: dict[str, Any] = {"bad": np.ones((1, 2, 3))}
        grid = np.array([0.0])

        with pytest.raises(ValueError, match="1D or 2D"):
            builder.align_options_data(options_ts, options_values, grid)


class TestBuildSequence:
    """Tests for full sequence building."""

    def test_build_with_trades_only(self) -> None:
        """Test sequence building with only trade data."""
        config = SequenceConfig(lookback_seconds=5, sample_interval=1)
        builder = SequenceBuilder(config)

        trade_ts = np.array([96.0, 98.0, 100.0])
        trade_prices = np.array([100.0, 101.0, 102.0])
        trade_volumes = np.array([1000.0, 2000.0, 3000.0])

        result = builder.build_sequence(
            trade_ts, trade_prices, trade_volumes, end_time=100.0
        )

        assert isinstance(result, AlignedSequence)
        assert len(result.timestamps) == 5
        assert len(result.prices) == 5
        assert len(result.volumes) == 5
        assert len(result.valid_mask) == 5
        assert result.options_data == {}

    def test_build_with_options(self) -> None:
        """Test sequence building with trade and options data."""
        config = SequenceConfig(lookback_seconds=5, sample_interval=1)
        builder = SequenceBuilder(config)

        trade_ts = np.array([96.0, 98.0, 100.0])
        trade_prices = np.array([100.0, 101.0, 102.0])
        trade_volumes = np.array([1000.0, 2000.0, 3000.0])

        options_ts = np.array([95.0, 99.0])
        options_values = {"atm_iv": np.array([0.25, 0.28])}

        result = builder.build_sequence(
            trade_ts,
            trade_prices,
            trade_volumes,
            end_time=100.0,
            options_timestamps=options_ts,
            options_values=options_values,
        )

        assert "atm_iv" in result.options_data
        assert len(result.options_data["atm_iv"]) == 5


class TestGetValidRange:
    """Tests for valid range detection."""

    def test_all_valid(self) -> None:
        """Test when all data is valid."""
        builder = SequenceBuilder()

        sequence = AlignedSequence(
            timestamps=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            prices=np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
            volumes=np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0]),
            valid_mask=np.array([True, True, True, True, True]),
        )

        result = builder.get_valid_range(sequence)

        assert result == (0, 5)

    def test_partial_valid(self) -> None:
        """Test with partially valid data."""
        builder = SequenceBuilder()

        sequence = AlignedSequence(
            timestamps=np.arange(10.0),
            prices=np.ones(10),
            volumes=np.ones(10),
            valid_mask=np.array(
                [False, False, True, True, True, True, True, False, False, False]
            ),
        )

        # 5/10 = 50% valid, so use a lower threshold
        result = builder.get_valid_range(sequence, min_valid_fraction=0.4)

        assert result == (2, 7)

    def test_insufficient_valid_fraction(self) -> None:
        """Test when valid fraction is below threshold."""
        builder = SequenceBuilder()

        sequence = AlignedSequence(
            timestamps=np.arange(10.0),
            prices=np.ones(10),
            volumes=np.ones(10),
            valid_mask=np.array(
                [False, False, True, True, False, False, False, False, False, False]
            ),
        )

        # Only 2/10 = 20% valid, below 80% threshold
        result = builder.get_valid_range(sequence, min_valid_fraction=0.8)
        assert result is None

        # But passes with lower threshold
        result = builder.get_valid_range(sequence, min_valid_fraction=0.1)
        assert result == (2, 4)

    def test_empty_sequence(self) -> None:
        """Test with empty sequence."""
        builder = SequenceBuilder()

        sequence = AlignedSequence(
            timestamps=np.array([]),
            prices=np.array([]),
            volumes=np.array([]),
            valid_mask=np.array([], dtype=np.bool_),
        )

        result = builder.get_valid_range(sequence)
        assert result is None

    def test_no_valid_data(self) -> None:
        """Test when no data is valid."""
        builder = SequenceBuilder()

        sequence = AlignedSequence(
            timestamps=np.arange(5.0),
            prices=np.ones(5),
            volumes=np.ones(5),
            valid_mask=np.array([False, False, False, False, False]),
        )

        result = builder.get_valid_range(sequence)
        assert result is None

    def test_multiple_valid_regions(self) -> None:
        """Test with multiple disjoint valid regions."""
        builder = SequenceBuilder()

        sequence = AlignedSequence(
            timestamps=np.arange(10.0),
            prices=np.ones(10),
            volumes=np.ones(10),
            valid_mask=np.array(
                [True, True, False, False, True, True, True, True, False, False]
            ),
        )

        # Should return the longest region (indices 4-7, length 4)
        result = builder.get_valid_range(sequence, min_valid_fraction=0.3)
        assert result == (4, 8)


class TestAlignedSequence:
    """Tests for AlignedSequence dataclass."""

    def test_creation(self) -> None:
        """Test basic AlignedSequence creation."""
        seq = AlignedSequence(
            timestamps=np.array([0.0, 1.0]),
            prices=np.array([100.0, 101.0]),
            volumes=np.array([1000.0, 2000.0]),
            valid_mask=np.array([True, True]),
        )

        assert len(seq.timestamps) == 2
        assert len(seq.prices) == 2
        assert len(seq.volumes) == 2
        assert len(seq.valid_mask) == 2
        assert seq.options_data == {}

    def test_with_options_data(self) -> None:
        """Test AlignedSequence with options data."""
        seq = AlignedSequence(
            timestamps=np.array([0.0, 1.0]),
            prices=np.array([100.0, 101.0]),
            volumes=np.array([1000.0, 2000.0]),
            valid_mask=np.array([True, True]),
            options_data={"iv": np.array([0.25, 0.26])},
        )

        assert "iv" in seq.options_data
        np.testing.assert_allclose(seq.options_data["iv"], [0.25, 0.26])


class TestOvernightStrategy:
    """Tests for OvernightStrategy enum."""

    def test_enum_values(self) -> None:
        """Test enum string values."""
        assert OvernightStrategy.RESET.value == "reset"
        assert OvernightStrategy.MASK.value == "mask"
        assert OvernightStrategy.INTERPOLATE.value == "interpolate"

    def test_from_string(self) -> None:
        """Test creating enum from string."""
        assert OvernightStrategy("reset") == OvernightStrategy.RESET
        assert OvernightStrategy("mask") == OvernightStrategy.MASK
        assert OvernightStrategy("interpolate") == OvernightStrategy.INTERPOLATE
