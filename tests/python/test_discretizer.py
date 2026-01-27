"""Tests for the discretizer module."""

import numpy as np
import pytest

from data.processors.discretizer import BucketConfig, LogReturnDiscretizer


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
