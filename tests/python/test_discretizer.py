"""Tests for the discretizer module."""

import numpy as np
import pytest

from data.processors.discretizer import (
    BucketConfig,
    LogReturnDiscretizer,
    VolatilityConfig,
    VolatilityDiscretizer,
    VolumeConfig,
    VolumeDiscretizer,
)


class TestBucketConfig:
    """Tests for BucketConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BucketConfig()
        assert config.num_buckets == 101
        assert config.min_bps == -50.0
        assert config.max_bps == 50.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BucketConfig(num_buckets=51, min_bps=-25.0, max_bps=25.0)
        assert config.num_buckets == 51
        assert config.min_bps == -25.0
        assert config.max_bps == 25.0

    def test_num_buckets_too_small(self) -> None:
        """Test that num_buckets must be at least 3."""
        with pytest.raises(ValueError, match="num_buckets must be at least 3"):
            BucketConfig(num_buckets=2)

    def test_num_buckets_must_be_odd(self) -> None:
        """Test that num_buckets must be odd for symmetry."""
        with pytest.raises(ValueError, match="num_buckets must be odd"):
            BucketConfig(num_buckets=100)

    def test_min_must_be_less_than_max(self) -> None:
        """Test that min_bps must be less than max_bps."""
        with pytest.raises(ValueError, match="min_bps must be less than max_bps"):
            BucketConfig(min_bps=50.0, max_bps=-50.0)

    def test_range_must_include_zero(self) -> None:
        """Test that bucket range must include 0."""
        with pytest.raises(ValueError, match="Bucket range must include 0"):
            BucketConfig(min_bps=10.0, max_bps=50.0)


class TestLogReturnDiscretizer:
    """Tests for LogReturnDiscretizer class."""

    def test_default_initialization(self) -> None:
        """Test initialization with default config."""
        discretizer = LogReturnDiscretizer()
        assert discretizer.config.num_buckets == 101
        assert len(discretizer.boundaries) == 102  # num_buckets + 1
        assert len(discretizer.centers) == 101

    def test_boundaries_are_evenly_spaced(self) -> None:
        """Test that bucket boundaries are evenly spaced."""
        discretizer = LogReturnDiscretizer()
        diffs = np.diff(discretizer.boundaries)
        # All differences should be equal (within floating point tolerance)
        assert np.allclose(diffs, diffs[0])

    def test_centers_are_midpoints(self) -> None:
        """Test that centers are midpoints of boundaries."""
        discretizer = LogReturnDiscretizer()
        expected_centers = (
            discretizer.boundaries[:-1] + discretizer.boundaries[1:]
        ) / 2.0
        assert np.allclose(discretizer.centers, expected_centers)

    def test_zero_bucket_is_center(self) -> None:
        """Test that zero bucket is the center bucket."""
        discretizer = LogReturnDiscretizer()
        zero_bucket = discretizer.get_zero_bucket()
        assert zero_bucket == 50  # Middle of 0-100
        # Center of zero bucket should be 0
        assert np.isclose(discretizer.centers[zero_bucket], 0.0)

    def test_price_to_log_return_bps_single_value(self) -> None:
        """Test price to log return conversion for a single value."""
        discretizer = LogReturnDiscretizer()
        # 1% increase: log(1.01) * 10000 ≈ 99.5 bps
        bps = discretizer.price_to_log_return_bps(101.0, 100.0)
        expected = np.log(1.01) * 10000.0
        assert np.isclose(bps, expected)

    def test_price_to_log_return_bps_array(self) -> None:
        """Test price to log return conversion for arrays."""
        discretizer = LogReturnDiscretizer()
        current = np.array([101.0, 102.0, 100.0])
        reference = np.array([100.0, 100.0, 100.0])
        bps = discretizer.price_to_log_return_bps(current, reference)
        expected = np.log(current / reference) * 10000.0
        assert np.allclose(bps, expected)

    def test_price_to_log_return_rejects_zero_reference(self) -> None:
        """Test that zero reference price raises error."""
        discretizer = LogReturnDiscretizer()
        with pytest.raises(ValueError, match="Reference price must be positive"):
            discretizer.price_to_log_return_bps(100.0, 0.0)

    def test_price_to_log_return_rejects_negative_price(self) -> None:
        """Test that negative price raises error."""
        discretizer = LogReturnDiscretizer()
        with pytest.raises(ValueError, match="Current price must be positive"):
            discretizer.price_to_log_return_bps(-100.0, 100.0)

    def test_discretize_zero_returns_center_bucket(self) -> None:
        """Test that 0 bps returns the center bucket."""
        discretizer = LogReturnDiscretizer()
        bucket = discretizer.discretize(0.0)
        assert bucket == discretizer.get_zero_bucket()

    def test_discretize_clips_below_min(self) -> None:
        """Test that values below min_bps map to bucket 0."""
        discretizer = LogReturnDiscretizer()
        bucket = discretizer.discretize(-100.0)  # Below -50 bps
        assert bucket == 0

    def test_discretize_clips_above_max(self) -> None:
        """Test that values above max_bps map to last bucket."""
        discretizer = LogReturnDiscretizer()
        bucket = discretizer.discretize(100.0)  # Above +50 bps
        assert bucket == 100

    def test_discretize_array(self) -> None:
        """Test discretization of array input."""
        discretizer = LogReturnDiscretizer()
        values = np.array([-60.0, -25.0, 0.0, 25.0, 60.0])
        buckets = discretizer.discretize(values)
        assert buckets[0] == 0  # Clipped to min
        assert buckets[2] == 50  # Center
        assert buckets[4] == 100  # Clipped to max
        assert 0 < buckets[1] < 50  # Negative range
        assert 50 < buckets[3] < 100  # Positive range

    def test_discretize_prices_integration(self) -> None:
        """Test discretize_prices convenience method."""
        discretizer = LogReturnDiscretizer()
        # Small price increase should give bucket above center
        bucket = discretizer.discretize_prices(100.1, 100.0)
        assert bucket > discretizer.get_zero_bucket()

        # Small price decrease should give bucket below center
        bucket = discretizer.discretize_prices(99.9, 100.0)
        assert bucket < discretizer.get_zero_bucket()

    def test_bucket_to_bps_roundtrip(self) -> None:
        """Test that bucket centers match bucket_to_bps output."""
        discretizer = LogReturnDiscretizer()
        for i in range(discretizer.config.num_buckets):
            bps = discretizer.bucket_to_bps(i)
            assert np.isclose(bps, discretizer.centers[i])

    def test_bucket_to_bps_array(self) -> None:
        """Test bucket_to_bps with array input."""
        discretizer = LogReturnDiscretizer()
        indices = np.array([0, 50, 100])
        bps = discretizer.bucket_to_bps(indices)
        assert len(bps) == 3
        assert np.isclose(bps[1], 0.0)  # Center bucket

    def test_bucket_to_bps_rejects_invalid_index(self) -> None:
        """Test that invalid bucket indices raise error."""
        discretizer = LogReturnDiscretizer()
        with pytest.raises(ValueError, match="Bucket indices must be in"):
            discretizer.bucket_to_bps(-1)
        with pytest.raises(ValueError, match="Bucket indices must be in"):
            discretizer.bucket_to_bps(101)

    def test_create_soft_labels_sum_to_one(self) -> None:
        """Test that soft labels sum to 1."""
        discretizer = LogReturnDiscretizer()
        labels = discretizer.create_soft_labels(50)
        assert np.isclose(labels.sum(), 1.0)

    def test_create_soft_labels_peak_at_true_bucket(self) -> None:
        """Test that soft labels peak at true bucket."""
        discretizer = LogReturnDiscretizer()
        true_bucket = 30
        labels = discretizer.create_soft_labels(true_bucket)
        assert labels[true_bucket] == labels.max()

    def test_create_soft_labels_symmetric(self) -> None:
        """Test that soft labels are symmetric around true bucket."""
        discretizer = LogReturnDiscretizer()
        true_bucket = 50  # Center to avoid edge effects
        labels = discretizer.create_soft_labels(true_bucket)
        # Labels should be symmetric
        for offset in range(1, 40):
            assert np.isclose(labels[true_bucket - offset], labels[true_bucket + offset])

    def test_create_soft_labels_sigma_affects_spread(self) -> None:
        """Test that larger sigma creates wider spread."""
        discretizer = LogReturnDiscretizer()
        narrow = discretizer.create_soft_labels(50, sigma=0.5)
        wide = discretizer.create_soft_labels(50, sigma=2.0)
        # Narrow labels should be more concentrated at center
        assert narrow[50] > wide[50]

    def test_create_soft_labels_rejects_invalid_bucket(self) -> None:
        """Test that invalid true_bucket raises error."""
        discretizer = LogReturnDiscretizer()
        with pytest.raises(ValueError, match="true_bucket must be in"):
            discretizer.create_soft_labels(-1)
        with pytest.raises(ValueError, match="true_bucket must be in"):
            discretizer.create_soft_labels(101)

    def test_bucket_width_calculation(self) -> None:
        """Test that bucket widths are correct."""
        config = BucketConfig(num_buckets=101, min_bps=-50.0, max_bps=50.0)
        discretizer = LogReturnDiscretizer(config)
        # With 101 buckets from -50 to +50, each bucket is 1 bps wide
        expected_width = 100.0 / 101.0
        diffs = np.diff(discretizer.boundaries)
        assert np.allclose(diffs, expected_width)

    def test_custom_config_discretization(self) -> None:
        """Test discretization with custom config."""
        config = BucketConfig(num_buckets=21, min_bps=-100.0, max_bps=100.0)
        discretizer = LogReturnDiscretizer(config)
        assert len(discretizer.centers) == 21
        assert discretizer.get_zero_bucket() == 10
        # 0 should map to center bucket
        assert discretizer.discretize(0.0) == 10


class TestVolatilityConfig:
    """Tests for VolatilityConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VolatilityConfig()
        assert config.num_buckets == 101
        assert config.min_vol == 5.0
        assert config.max_vol == 150.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = VolatilityConfig(num_buckets=51, min_vol=10.0, max_vol=100.0)
        assert config.num_buckets == 51
        assert config.min_vol == 10.0
        assert config.max_vol == 100.0

    def test_num_buckets_too_small(self) -> None:
        """Test that num_buckets must be at least 3."""
        with pytest.raises(ValueError, match="num_buckets must be at least 3"):
            VolatilityConfig(num_buckets=2)

    def test_min_vol_negative(self) -> None:
        """Test that min_vol must be non-negative."""
        with pytest.raises(ValueError, match="min_vol must be non-negative"):
            VolatilityConfig(min_vol=-5.0)

    def test_min_must_be_less_than_max(self) -> None:
        """Test that min_vol must be less than max_vol."""
        with pytest.raises(ValueError, match="min_vol must be less than max_vol"):
            VolatilityConfig(min_vol=100.0, max_vol=50.0)


class TestVolatilityDiscretizer:
    """Tests for VolatilityDiscretizer class."""

    def test_default_initialization(self) -> None:
        """Test initialization with default config."""
        discretizer = VolatilityDiscretizer()
        assert discretizer.config.num_buckets == 101
        assert len(discretizer.boundaries) == 102  # num_buckets + 1
        assert len(discretizer.centers) == 101

    def test_boundaries_are_evenly_spaced(self) -> None:
        """Test that bucket boundaries are evenly spaced."""
        discretizer = VolatilityDiscretizer()
        diffs = np.diff(discretizer.boundaries)
        assert np.allclose(diffs, diffs[0])

    def test_centers_are_midpoints(self) -> None:
        """Test that centers are midpoints of boundaries."""
        discretizer = VolatilityDiscretizer()
        expected_centers = (
            discretizer.boundaries[:-1] + discretizer.boundaries[1:]
        ) / 2.0
        assert np.allclose(discretizer.centers, expected_centers)

    def test_discretize_within_range(self) -> None:
        """Test discretization for values within range."""
        discretizer = VolatilityDiscretizer()
        # Middle volatility (around 77.5%) should be in middle bucket range
        bucket = discretizer.discretize(77.5)
        assert 40 < bucket < 60

    def test_discretize_clips_below_min(self) -> None:
        """Test that values below min_vol map to bucket 0."""
        discretizer = VolatilityDiscretizer()
        bucket = discretizer.discretize(1.0)  # Below 5%
        assert bucket == 0

    def test_discretize_clips_above_max(self) -> None:
        """Test that values above max_vol map to last bucket."""
        discretizer = VolatilityDiscretizer()
        bucket = discretizer.discretize(200.0)  # Above 150%
        assert bucket == 100

    def test_discretize_array(self) -> None:
        """Test discretization of array input."""
        discretizer = VolatilityDiscretizer()
        values = np.array([1.0, 20.0, 77.5, 130.0, 200.0])
        buckets = discretizer.discretize(values)
        assert buckets[0] == 0  # Clipped to min
        assert buckets[4] == 100  # Clipped to max
        # Values in between should be in increasing order
        assert buckets[1] < buckets[2] < buckets[3]

    def test_bucket_to_volatility_roundtrip(self) -> None:
        """Test that bucket centers match bucket_to_volatility output."""
        discretizer = VolatilityDiscretizer()
        for i in range(discretizer.config.num_buckets):
            vol = discretizer.bucket_to_volatility(i)
            assert np.isclose(vol, discretizer.centers[i])

    def test_bucket_to_volatility_array(self) -> None:
        """Test bucket_to_volatility with array input."""
        discretizer = VolatilityDiscretizer()
        indices = np.array([0, 50, 100])
        vols = discretizer.bucket_to_volatility(indices)
        assert len(vols) == 3
        # First bucket center should be close to min
        assert vols[0] < 10.0
        # Last bucket center should be close to max
        assert vols[2] > 140.0

    def test_bucket_to_volatility_rejects_invalid_index(self) -> None:
        """Test that invalid bucket indices raise error."""
        discretizer = VolatilityDiscretizer()
        with pytest.raises(ValueError, match="Bucket indices must be in"):
            discretizer.bucket_to_volatility(-1)
        with pytest.raises(ValueError, match="Bucket indices must be in"):
            discretizer.bucket_to_volatility(101)

    def test_create_soft_labels_sum_to_one(self) -> None:
        """Test that soft labels sum to 1."""
        discretizer = VolatilityDiscretizer()
        labels = discretizer.create_soft_labels(50)
        assert np.isclose(labels.sum(), 1.0)

    def test_create_soft_labels_peak_at_true_bucket(self) -> None:
        """Test that soft labels peak at true bucket."""
        discretizer = VolatilityDiscretizer()
        true_bucket = 30
        labels = discretizer.create_soft_labels(true_bucket)
        assert labels[true_bucket] == labels.max()

    def test_create_soft_labels_symmetric(self) -> None:
        """Test that soft labels are symmetric around true bucket."""
        discretizer = VolatilityDiscretizer()
        true_bucket = 50  # Center to avoid edge effects
        labels = discretizer.create_soft_labels(true_bucket)
        for offset in range(1, 40):
            assert np.isclose(labels[true_bucket - offset], labels[true_bucket + offset])

    def test_create_soft_labels_rejects_invalid_bucket(self) -> None:
        """Test that invalid true_bucket raises error."""
        discretizer = VolatilityDiscretizer()
        with pytest.raises(ValueError, match="true_bucket must be in"):
            discretizer.create_soft_labels(-1)
        with pytest.raises(ValueError, match="true_bucket must be in"):
            discretizer.create_soft_labels(101)

    def test_custom_config_discretization(self) -> None:
        """Test discretization with custom config."""
        config = VolatilityConfig(num_buckets=21, min_vol=10.0, max_vol=60.0)
        discretizer = VolatilityDiscretizer(config)
        assert len(discretizer.centers) == 21
        # 35% (midpoint) should be around bucket 10
        bucket = discretizer.discretize(35.0)
        assert 9 <= bucket <= 11


class TestVolumeConfig:
    """Tests for VolumeConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VolumeConfig()
        assert config.num_buckets == 101
        assert config.min_ratio == 0.1
        assert config.max_ratio == 5.0
        assert config.log_scale is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = VolumeConfig(num_buckets=51, min_ratio=0.2, max_ratio=3.0, log_scale=False)
        assert config.num_buckets == 51
        assert config.min_ratio == 0.2
        assert config.max_ratio == 3.0
        assert config.log_scale is False

    def test_num_buckets_too_small(self) -> None:
        """Test that num_buckets must be at least 3."""
        with pytest.raises(ValueError, match="num_buckets must be at least 3"):
            VolumeConfig(num_buckets=2)

    def test_min_ratio_must_be_positive(self) -> None:
        """Test that min_ratio must be positive."""
        with pytest.raises(ValueError, match="min_ratio must be positive"):
            VolumeConfig(min_ratio=0.0)
        with pytest.raises(ValueError, match="min_ratio must be positive"):
            VolumeConfig(min_ratio=-0.5)

    def test_min_must_be_less_than_max(self) -> None:
        """Test that min_ratio must be less than max_ratio."""
        with pytest.raises(ValueError, match="min_ratio must be less than max_ratio"):
            VolumeConfig(min_ratio=5.0, max_ratio=2.0)


class TestVolumeDiscretizer:
    """Tests for VolumeDiscretizer class."""

    def test_default_initialization(self) -> None:
        """Test initialization with default config (log scale)."""
        discretizer = VolumeDiscretizer()
        assert discretizer.config.num_buckets == 101
        assert len(discretizer.boundaries) == 102
        assert len(discretizer.centers) == 101

    def test_log_scale_boundaries_geometric(self) -> None:
        """Test that log scale boundaries are geometrically spaced."""
        discretizer = VolumeDiscretizer()
        # In log space, boundaries should be evenly spaced
        log_boundaries = np.log(discretizer.boundaries)
        diffs = np.diff(log_boundaries)
        assert np.allclose(diffs, diffs[0])

    def test_linear_scale_boundaries_arithmetic(self) -> None:
        """Test that linear scale boundaries are arithmetically spaced."""
        config = VolumeConfig(log_scale=False)
        discretizer = VolumeDiscretizer(config)
        diffs = np.diff(discretizer.boundaries)
        assert np.allclose(diffs, diffs[0])

    def test_log_scale_centers_geometric_mean(self) -> None:
        """Test that log scale centers are geometric means."""
        discretizer = VolumeDiscretizer()
        expected_centers = np.sqrt(
            discretizer.boundaries[:-1] * discretizer.boundaries[1:]
        )
        assert np.allclose(discretizer.centers, expected_centers)

    def test_linear_scale_centers_arithmetic_mean(self) -> None:
        """Test that linear scale centers are arithmetic means."""
        config = VolumeConfig(log_scale=False)
        discretizer = VolumeDiscretizer(config)
        expected_centers = (
            discretizer.boundaries[:-1] + discretizer.boundaries[1:]
        ) / 2.0
        assert np.allclose(discretizer.centers, expected_centers)

    def test_discretize_within_range(self) -> None:
        """Test discretization for values within range."""
        discretizer = VolumeDiscretizer()
        # 1.0 (average volume) should be somewhere in the middle
        bucket = discretizer.discretize(1.0)
        assert 20 < bucket < 80

    def test_discretize_clips_below_min(self) -> None:
        """Test that values below min_ratio map to bucket 0."""
        discretizer = VolumeDiscretizer()
        bucket = discretizer.discretize(0.01)  # Below 0.1
        assert bucket == 0

    def test_discretize_clips_above_max(self) -> None:
        """Test that values above max_ratio map to last bucket."""
        discretizer = VolumeDiscretizer()
        bucket = discretizer.discretize(10.0)  # Above 5.0
        assert bucket == 100

    def test_discretize_array(self) -> None:
        """Test discretization of array input."""
        discretizer = VolumeDiscretizer()
        values = np.array([0.01, 0.5, 1.0, 2.0, 10.0])
        buckets = discretizer.discretize(values)
        assert buckets[0] == 0  # Clipped to min
        assert buckets[4] == 100  # Clipped to max
        # Values in between should be in increasing order
        assert buckets[1] < buckets[2] < buckets[3]

    def test_get_average_bucket(self) -> None:
        """Test get_average_bucket returns bucket containing 1.0."""
        discretizer = VolumeDiscretizer()
        avg_bucket = discretizer.get_average_bucket()
        # Verify 1.0 maps to this bucket
        assert discretizer.discretize(1.0) == avg_bucket

    def test_get_average_bucket_outside_range_low(self) -> None:
        """Test get_average_bucket when 1.0 is below min_ratio."""
        config = VolumeConfig(min_ratio=2.0, max_ratio=10.0)
        discretizer = VolumeDiscretizer(config)
        assert discretizer.get_average_bucket() == 0

    def test_get_average_bucket_outside_range_high(self) -> None:
        """Test get_average_bucket when 1.0 is above max_ratio."""
        config = VolumeConfig(min_ratio=0.01, max_ratio=0.5)
        discretizer = VolumeDiscretizer(config)
        assert discretizer.get_average_bucket() == config.num_buckets - 1

    def test_bucket_to_ratio_roundtrip(self) -> None:
        """Test that bucket centers match bucket_to_ratio output."""
        discretizer = VolumeDiscretizer()
        for i in range(discretizer.config.num_buckets):
            ratio = discretizer.bucket_to_ratio(i)
            assert np.isclose(ratio, discretizer.centers[i])

    def test_bucket_to_ratio_array(self) -> None:
        """Test bucket_to_ratio with array input."""
        discretizer = VolumeDiscretizer()
        indices = np.array([0, 50, 100])
        ratios = discretizer.bucket_to_ratio(indices)
        assert len(ratios) == 3
        # First bucket center should be close to min
        assert ratios[0] < 0.2
        # Last bucket center should be close to max
        assert ratios[2] > 4.0

    def test_bucket_to_ratio_rejects_invalid_index(self) -> None:
        """Test that invalid bucket indices raise error."""
        discretizer = VolumeDiscretizer()
        with pytest.raises(ValueError, match="Bucket indices must be in"):
            discretizer.bucket_to_ratio(-1)
        with pytest.raises(ValueError, match="Bucket indices must be in"):
            discretizer.bucket_to_ratio(101)

    def test_create_soft_labels_sum_to_one(self) -> None:
        """Test that soft labels sum to 1."""
        discretizer = VolumeDiscretizer()
        labels = discretizer.create_soft_labels(50)
        assert np.isclose(labels.sum(), 1.0)

    def test_create_soft_labels_peak_at_true_bucket(self) -> None:
        """Test that soft labels peak at true bucket."""
        discretizer = VolumeDiscretizer()
        true_bucket = 30
        labels = discretizer.create_soft_labels(true_bucket)
        assert labels[true_bucket] == labels.max()

    def test_create_soft_labels_rejects_invalid_bucket(self) -> None:
        """Test that invalid true_bucket raises error."""
        discretizer = VolumeDiscretizer()
        with pytest.raises(ValueError, match="true_bucket must be in"):
            discretizer.create_soft_labels(-1)
        with pytest.raises(ValueError, match="true_bucket must be in"):
            discretizer.create_soft_labels(101)

    def test_log_vs_linear_scale_ratio_symmetry(self) -> None:
        """Test that log scale treats 0.5 and 2.0 as symmetric around 1.0."""
        config = VolumeConfig(log_scale=True)
        discretizer = VolumeDiscretizer(config)
        avg_bucket = discretizer.get_average_bucket()
        bucket_half = discretizer.discretize(0.5)
        bucket_double = discretizer.discretize(2.0)
        # Distance from average should be similar (in bucket units)
        dist_half = avg_bucket - bucket_half
        dist_double = bucket_double - avg_bucket
        # Allow for rounding differences but should be close
        assert abs(dist_half - dist_double) <= 2

    def test_custom_config_discretization(self) -> None:
        """Test discretization with custom config."""
        config = VolumeConfig(num_buckets=21, min_ratio=0.5, max_ratio=2.0, log_scale=True)
        discretizer = VolumeDiscretizer(config)
        assert len(discretizer.centers) == 21
        # 1.0 should be around the middle
        bucket = discretizer.discretize(1.0)
        assert 8 <= bucket <= 12
