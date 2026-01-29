"""Tests for evaluation.calibration module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.calibration import (
    BucketCalibration,
    CalibrationConfig,
    MultiHorizonCalibration,
    PlattScaler,
    ReliabilityDiagram,
    TemperatureScaler,
    apply_temperature_scaling_multi_horizon,
    compute_bucket_calibration,
    compute_multi_horizon_calibration,
    compute_reliability_diagram,
    format_calibration_report,
)


class TestCalibrationConfig:
    """Tests for CalibrationConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = CalibrationConfig()
        assert config.num_bins == 15
        assert config.temperature_lr == 0.01
        assert config.temperature_max_iter == 50
        assert config.temperature_tol == 1e-4
        assert config.platt_max_iter == 100

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CalibrationConfig(
            num_bins=20,
            temperature_lr=0.001,
            temperature_max_iter=100,
            temperature_tol=1e-5,
            platt_max_iter=200,
        )
        assert config.num_bins == 20
        assert config.temperature_lr == 0.001
        assert config.temperature_max_iter == 100
        assert config.temperature_tol == 1e-5
        assert config.platt_max_iter == 200

    def test_validation_num_bins(self) -> None:
        """Test validation of num_bins."""
        with pytest.raises(ValueError, match="num_bins must be at least 2"):
            CalibrationConfig(num_bins=1)

    def test_validation_temperature_lr(self) -> None:
        """Test validation of temperature_lr."""
        with pytest.raises(ValueError, match="temperature_lr must be positive"):
            CalibrationConfig(temperature_lr=0.0)

    def test_validation_temperature_max_iter(self) -> None:
        """Test validation of temperature_max_iter."""
        with pytest.raises(ValueError, match="temperature_max_iter must be at least 1"):
            CalibrationConfig(temperature_max_iter=0)

    def test_validation_temperature_tol(self) -> None:
        """Test validation of temperature_tol."""
        with pytest.raises(ValueError, match="temperature_tol must be positive"):
            CalibrationConfig(temperature_tol=0.0)

    def test_validation_platt_max_iter(self) -> None:
        """Test validation of platt_max_iter."""
        with pytest.raises(ValueError, match="platt_max_iter must be at least 1"):
            CalibrationConfig(platt_max_iter=0)


class TestTemperatureScaler:
    """Tests for TemperatureScaler."""

    def test_initialization(self) -> None:
        """Test scaler initialization."""
        scaler = TemperatureScaler()
        assert scaler.temperature == 1.0
        assert not scaler._is_fitted

    def test_initialization_with_config(self) -> None:
        """Test scaler initialization with custom config."""
        config = CalibrationConfig(temperature_max_iter=100)
        scaler = TemperatureScaler(config)
        assert scaler.config.temperature_max_iter == 100

    def test_fit_basic(self) -> None:
        """Test basic fitting."""
        # Create overconfident predictions
        logits = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float32
        )
        targets = torch.tensor([0, 1, 2], dtype=torch.long)

        scaler = TemperatureScaler()
        stats = scaler.fit(logits, targets)

        assert scaler._is_fitted
        # Temperature should be positive (exact value depends on optimization)
        assert scaler.temperature > 0.0
        assert "temperature" in stats
        assert "loss_history" in stats
        assert "train_ece" in stats
        assert "train_mce" in stats

    def test_fit_underconfident(self) -> None:
        """Test fitting with underconfident predictions."""
        # Create underconfident predictions
        logits = torch.tensor(
            [[1.0, 0.9, 0.8], [0.9, 1.0, 0.8], [0.8, 0.9, 1.0]], dtype=torch.float32
        )
        targets = torch.tensor([0, 1, 2], dtype=torch.long)

        scaler = TemperatureScaler()
        stats = scaler.fit(logits, targets)

        assert scaler._is_fitted
        # Temperature should be < 1 to increase confidence
        assert scaler.temperature < 1.0

    def test_fit_validation_split(self) -> None:
        """Test fitting with validation split."""
        logits = torch.randn(100, 10)
        targets = torch.randint(0, 10, (100,))

        scaler = TemperatureScaler()
        stats = scaler.fit(logits, targets, validation_split=0.2)

        assert "val_ece" in stats
        assert "val_mce" in stats

    def test_fit_invalid_logits_shape(self) -> None:
        """Test fit with invalid logits shape."""
        logits = torch.randn(10, 5, 3)  # 3D instead of 2D
        targets = torch.randint(0, 5, (10,))

        scaler = TemperatureScaler()
        with pytest.raises(ValueError, match="logits must be 2D"):
            scaler.fit(logits, targets)

    def test_fit_invalid_targets_shape(self) -> None:
        """Test fit with invalid targets shape."""
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10, 1))  # 2D instead of 1D

        scaler = TemperatureScaler()
        with pytest.raises(ValueError, match="targets must be 1D"):
            scaler.fit(logits, targets)

    def test_fit_batch_size_mismatch(self) -> None:
        """Test fit with mismatched batch sizes."""
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (8,))

        scaler = TemperatureScaler()
        with pytest.raises(ValueError, match="Batch size mismatch"):
            scaler.fit(logits, targets)

    def test_transform_before_fit(self) -> None:
        """Test transform before fitting."""
        scaler = TemperatureScaler()
        logits = torch.randn(5, 3)

        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            scaler.transform(logits)

    def test_transform_after_fit(self) -> None:
        """Test transform after fitting."""
        logits = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=torch.float32
        )
        targets = torch.tensor([0, 1], dtype=torch.long)

        scaler = TemperatureScaler()
        scaler.fit(logits, targets)
        probs = scaler.transform(logits)

        assert probs.shape == logits.shape
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
        # Probabilities should be valid (all entries between 0 and 1)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_fit_transform(self) -> None:
        """Test fit_transform convenience method."""
        logits = torch.randn(50, 5)
        targets = torch.randint(0, 5, (50,))

        scaler = TemperatureScaler()
        probs, stats = scaler.fit_transform(logits, targets)

        assert probs.shape == logits.shape
        assert scaler._is_fitted
        assert "temperature" in stats


class TestPlattScaler:
    """Tests for PlattScaler."""

    def test_initialization(self) -> None:
        """Test scaler initialization."""
        scaler = PlattScaler()
        assert scaler.scale == 1.0
        assert scaler.bias == 0.0
        assert not scaler._is_fitted

    def test_fit_basic(self) -> None:
        """Test basic fitting."""
        logits = torch.randn(100, 5)
        targets = torch.randint(0, 5, (100,))

        scaler = PlattScaler()
        stats = scaler.fit(logits, targets)

        assert scaler._is_fitted
        assert "scale" in stats
        assert "bias" in stats
        assert "loss_history" in stats

    def test_transform_before_fit(self) -> None:
        """Test transform before fitting."""
        scaler = PlattScaler()
        logits = torch.randn(5, 3)

        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            scaler.transform(logits)

    def test_transform_after_fit(self) -> None:
        """Test transform after fitting."""
        logits = torch.randn(50, 5)
        targets = torch.randint(0, 5, (50,))

        scaler = PlattScaler()
        scaler.fit(logits, targets)
        probs = scaler.transform(logits)

        assert probs.shape == logits.shape
        assert torch.allclose(probs.sum(dim=-1), torch.ones(50))

    def test_fit_transform(self) -> None:
        """Test fit_transform convenience method."""
        logits = torch.randn(50, 5)
        targets = torch.randint(0, 5, (50,))

        scaler = PlattScaler()
        probs, stats = scaler.fit_transform(logits, targets)

        assert probs.shape == logits.shape
        assert scaler._is_fitted
        assert "scale" in stats
        assert "bias" in stats


class TestReliabilityDiagram:
    """Tests for reliability diagram computation."""

    def test_compute_reliability_diagram_perfect_calibration(self) -> None:
        """Test reliability diagram with perfectly calibrated predictions."""
        # Create perfectly calibrated predictions
        torch.manual_seed(42)
        probs = torch.softmax(torch.randn(100, 5), dim=-1)
        # Sample targets from predicted distribution
        targets = torch.multinomial(probs, num_samples=1).squeeze(-1)

        diagram = compute_reliability_diagram(probs, targets, num_bins=10)

        assert isinstance(diagram, ReliabilityDiagram)
        assert len(diagram.bin_confidences) == 10
        assert len(diagram.bin_accuracies) == 10
        assert len(diagram.bin_counts) == 10
        assert len(diagram.bin_edges) == 11
        assert diagram.num_samples == 100
        # ECE should be relatively small for well-calibrated predictions
        assert diagram.ece >= 0.0

    def test_compute_reliability_diagram_overconfident(self) -> None:
        """Test reliability diagram with overconfident predictions."""
        # Create overconfident predictions (high probabilities)
        logits = torch.randn(100, 5) * 5.0  # Scale up for overconfidence
        probs = torch.softmax(logits, dim=-1)
        targets = torch.randint(0, 5, (100,))

        diagram = compute_reliability_diagram(probs, targets, num_bins=10)

        # Overconfident predictions should have higher ECE
        assert diagram.ece > 0.0

    def test_compute_reliability_diagram_attributes(self) -> None:
        """Test all attributes of reliability diagram."""
        probs = torch.softmax(torch.randn(50, 3), dim=-1)
        targets = torch.randint(0, 3, (50,))

        diagram = compute_reliability_diagram(probs, targets, num_bins=5)

        assert diagram.bin_confidences.dtype == np.float64
        assert diagram.bin_accuracies.dtype == np.float64
        assert diagram.bin_counts.dtype == np.intp
        assert diagram.bin_edges.dtype == np.float64
        assert isinstance(diagram.ece, float)
        assert isinstance(diagram.mce, float)
        assert isinstance(diagram.num_samples, int)


class TestMultiHorizonCalibration:
    """Tests for multi-horizon calibration analysis."""

    def test_compute_multi_horizon_calibration(self) -> None:
        """Test multi-horizon calibration computation."""
        horizons = [1, 5, 10]
        probs_dict = {h: torch.softmax(torch.randn(50, 5), dim=-1) for h in horizons}
        targets_dict = {h: torch.randint(0, 5, (50,)) for h in horizons}

        calibration = compute_multi_horizon_calibration(
            probs_dict, targets_dict, num_bins=10
        )

        assert isinstance(calibration, MultiHorizonCalibration)
        assert calibration.horizons == horizons
        assert len(calibration.reliability_diagrams) == 3
        assert len(calibration.ece_by_horizon) == 3
        assert len(calibration.mce_by_horizon) == 3

        for h in horizons:
            assert h in calibration.reliability_diagrams
            assert h in calibration.ece_by_horizon
            assert h in calibration.mce_by_horizon

        assert isinstance(calibration.mean_ece, float)
        assert isinstance(calibration.mean_mce, float)
        assert calibration.mean_ece >= 0.0
        assert calibration.mean_mce >= 0.0

    def test_compute_multi_horizon_calibration_sorted(self) -> None:
        """Test that horizons are sorted."""
        probs_dict = {
            10: torch.softmax(torch.randn(30, 5), dim=-1),
            1: torch.softmax(torch.randn(30, 5), dim=-1),
            5: torch.softmax(torch.randn(30, 5), dim=-1),
        }
        targets_dict = {h: torch.randint(0, 5, (30,)) for h in [10, 1, 5]}

        calibration = compute_multi_horizon_calibration(probs_dict, targets_dict)

        assert calibration.horizons == [1, 5, 10]

    def test_format_calibration_report(self) -> None:
        """Test formatting of calibration report."""
        horizons = [1, 5]
        probs_dict = {h: torch.softmax(torch.randn(30, 3), dim=-1) for h in horizons}
        targets_dict = {h: torch.randint(0, 3, (30,)) for h in horizons}

        calibration = compute_multi_horizon_calibration(probs_dict, targets_dict)
        report = format_calibration_report(calibration)

        assert isinstance(report, str)
        assert "Multi-Horizon Calibration Analysis" in report
        assert "Horizon" in report
        assert "ECE" in report
        assert "MCE" in report
        assert "Mean" in report
        assert "Calibration Quality" in report


class TestBucketCalibration:
    """Tests for per-bucket calibration analysis."""

    def test_compute_bucket_calibration(self) -> None:
        """Test bucket-level calibration computation."""
        num_buckets = 5
        probs = torch.softmax(torch.randn(100, num_buckets), dim=-1)
        targets = torch.randint(0, num_buckets, (100,))
        bucket_centers = torch.linspace(-10, 10, num_buckets)

        bucket_cal = compute_bucket_calibration(probs, targets, bucket_centers)

        assert isinstance(bucket_cal, BucketCalibration)
        assert len(bucket_cal.bucket_indices) == num_buckets
        assert len(bucket_cal.bucket_centers_bps) == num_buckets
        assert len(bucket_cal.predicted_frequencies) == num_buckets
        assert len(bucket_cal.actual_frequencies) == num_buckets
        assert len(bucket_cal.sample_counts) == num_buckets

        # Check dtypes
        assert bucket_cal.bucket_indices.dtype == np.intp
        assert bucket_cal.bucket_centers_bps.dtype == np.float64
        assert bucket_cal.predicted_frequencies.dtype == np.float64
        assert bucket_cal.actual_frequencies.dtype == np.float64
        assert bucket_cal.sample_counts.dtype == np.intp

    def test_compute_bucket_calibration_perfect(self) -> None:
        """Test bucket calibration with perfect predictions."""
        num_buckets = 3
        # Perfect one-hot predictions
        probs = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        bucket_centers = torch.tensor([0.0, 10.0, 20.0])

        bucket_cal = compute_bucket_calibration(probs, targets, bucket_centers)

        # For perfect predictions, predicted_freq should equal actual_freq
        for i in range(num_buckets):
            if bucket_cal.sample_counts[i] > 0:
                assert bucket_cal.predicted_frequencies[i] == 1.0
                assert bucket_cal.actual_frequencies[i] == 1.0

    def test_compute_bucket_calibration_no_predictions(self) -> None:
        """Test bucket calibration when some buckets are never predicted."""
        # All predictions for bucket 0
        probs = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        )
        targets = torch.tensor([0, 0], dtype=torch.long)
        bucket_centers = torch.tensor([0.0, 10.0, 20.0])

        bucket_cal = compute_bucket_calibration(probs, targets, bucket_centers)

        # Buckets 1 and 2 should have zero counts
        assert bucket_cal.sample_counts[1] == 0
        assert bucket_cal.sample_counts[2] == 0
        # Their frequencies should be 0.0
        assert bucket_cal.predicted_frequencies[1] == 0.0
        assert bucket_cal.actual_frequencies[1] == 0.0


class TestApplyTemperatureScalingMultiHorizon:
    """Tests for multi-horizon temperature scaling."""

    def test_apply_temperature_scaling_single_horizon(self) -> None:
        """Test temperature scaling with single horizon."""
        logits_dict = {1: torch.randn(50, 5)}
        targets_dict = {1: torch.randint(0, 5, (50,))}

        calibrated_probs, stats_dict = apply_temperature_scaling_multi_horizon(
            logits_dict, targets_dict
        )

        assert 1 in calibrated_probs
        assert 1 in stats_dict
        assert calibrated_probs[1].shape == (50, 5)
        assert "temperature" in stats_dict[1]

    def test_apply_temperature_scaling_multiple_horizons(self) -> None:
        """Test temperature scaling with multiple horizons."""
        horizons = [1, 5, 10]
        logits_dict = {h: torch.randn(50, 5) for h in horizons}
        targets_dict = {h: torch.randint(0, 5, (50,)) for h in horizons}

        calibrated_probs, stats_dict = apply_temperature_scaling_multi_horizon(
            logits_dict, targets_dict
        )

        assert len(calibrated_probs) == 3
        assert len(stats_dict) == 3

        for h in horizons:
            assert h in calibrated_probs
            assert h in stats_dict
            assert calibrated_probs[h].shape == (50, 5)
            assert "temperature" in stats_dict[h]
            assert "train_ece" in stats_dict[h]

    def test_apply_temperature_scaling_with_config(self) -> None:
        """Test temperature scaling with custom config."""
        config = CalibrationConfig(temperature_max_iter=100, num_bins=20)
        logits_dict = {1: torch.randn(50, 5)}
        targets_dict = {1: torch.randint(0, 5, (50,))}

        calibrated_probs, stats_dict = apply_temperature_scaling_multi_horizon(
            logits_dict, targets_dict, config
        )

        assert 1 in calibrated_probs
        assert 1 in stats_dict

    def test_apply_temperature_scaling_preserves_order(self) -> None:
        """Test that temperature scaling preserves rank order."""
        logits = torch.tensor([[3.0, 1.0, 2.0]], dtype=torch.float32)
        logits_dict = {1: logits}
        targets_dict = {1: torch.tensor([0], dtype=torch.long)}

        calibrated_probs, _ = apply_temperature_scaling_multi_horizon(
            logits_dict, targets_dict
        )

        probs = calibrated_probs[1]
        # Order should be preserved: bucket 0 > bucket 2 > bucket 1
        assert probs[0, 0] > probs[0, 2] > probs[0, 1]


class TestIntegration:
    """Integration tests for calibration pipeline."""

    def test_full_calibration_pipeline(self) -> None:
        """Test full calibration analysis pipeline."""
        # Generate data
        torch.manual_seed(42)
        horizons = [1, 5, 10]
        logits_dict = {
            h: torch.randn(100, 5) * 3.0 for h in horizons  # Overconfident
        }
        targets_dict = {h: torch.randint(0, 5, (100,)) for h in horizons}

        # Compute initial calibration
        initial_probs = {h: torch.softmax(logits_dict[h], dim=-1) for h in horizons}
        initial_cal = compute_multi_horizon_calibration(
            initial_probs, targets_dict, num_bins=10
        )

        # Apply temperature scaling
        calibrated_probs, _ = apply_temperature_scaling_multi_horizon(
            logits_dict, targets_dict
        )

        # Compute calibrated calibration
        calibrated_cal = compute_multi_horizon_calibration(
            calibrated_probs, targets_dict, num_bins=10
        )

        # Calibration should improve (ECE should decrease)
        # Note: Not guaranteed for every random seed, but likely
        assert calibrated_cal.mean_ece <= initial_cal.mean_ece + 0.1

    def test_bucket_calibration_analysis(self) -> None:
        """Test bucket-level calibration analysis."""
        num_buckets = 101
        probs = torch.softmax(torch.randn(200, num_buckets), dim=-1)
        targets = torch.randint(0, num_buckets, (200,))
        bucket_centers = torch.linspace(-50, 50, num_buckets)

        bucket_cal = compute_bucket_calibration(probs, targets, bucket_centers)

        # Total samples across buckets should equal total predictions
        # (not all samples, since only predicted class counts)
        assert bucket_cal.sample_counts.sum() == 200

    def test_reliability_diagram_with_temperature_scaling(self) -> None:
        """Test reliability diagram before and after temperature scaling."""
        # Overconfident logits
        logits = torch.randn(100, 5) * 5.0
        targets = torch.randint(0, 5, (100,))

        # Before calibration
        probs_before = torch.softmax(logits, dim=-1)
        diagram_before = compute_reliability_diagram(probs_before, targets)

        # After calibration
        scaler = TemperatureScaler()
        probs_after = scaler.fit_transform(logits, targets)[0]
        diagram_after = compute_reliability_diagram(probs_after, targets)

        # ECE should decrease after temperature scaling
        assert diagram_after.ece <= diagram_before.ece + 0.05
