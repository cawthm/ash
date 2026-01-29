"""Tests for the metrics module."""

import numpy as np
import pytest
import torch

from evaluation.metrics import (
    CalibrationResult,
    HorizonMetrics,
    MetricsConfig,
    MultiHorizonMetrics,
    PnLResult,
    brier_score,
    brier_score_per_bucket,
    compute_calibration,
    compute_horizon_metrics,
    compute_multi_horizon_metrics,
    directional_accuracy,
    entropy,
    expected_calibration_error,
    format_metrics_report,
    log_likelihood,
    mean_entropy,
    negative_log_likelihood,
    predicted_mean_bps,
    predicted_std_bps,
    sharpness,
    simulate_pnl,
)


class TestMetricsConfig:
    """Tests for MetricsConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MetricsConfig()
        assert config.num_buckets == 101
        assert config.bucket_centers_bps is None
        assert config.num_calibration_bins == 10
        assert config.eps == 1e-8

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        centers = tuple(np.linspace(-25, 25, 51))
        config = MetricsConfig(
            num_buckets=51,
            bucket_centers_bps=centers,
            num_calibration_bins=20,
            eps=1e-6,
        )
        assert config.num_buckets == 51
        assert config.bucket_centers_bps == centers
        assert config.num_calibration_bins == 20
        assert config.eps == 1e-6

    def test_num_buckets_too_small(self) -> None:
        """Test that num_buckets must be at least 3."""
        with pytest.raises(ValueError, match="num_buckets must be at least 3"):
            MetricsConfig(num_buckets=2)

    def test_bucket_centers_length_mismatch(self) -> None:
        """Test that bucket_centers_bps must match num_buckets."""
        with pytest.raises(ValueError, match="bucket_centers_bps length"):
            MetricsConfig(num_buckets=51, bucket_centers_bps=(1.0, 2.0, 3.0))

    def test_num_calibration_bins_too_small(self) -> None:
        """Test that num_calibration_bins must be at least 2."""
        with pytest.raises(ValueError, match="num_calibration_bins must be at least 2"):
            MetricsConfig(num_calibration_bins=1)

    def test_eps_must_be_positive(self) -> None:
        """Test that eps must be positive."""
        with pytest.raises(ValueError, match="eps must be positive"):
            MetricsConfig(eps=0.0)

    def test_get_bucket_centers_default(self) -> None:
        """Test default bucket centers."""
        config = MetricsConfig(num_buckets=101)
        centers = config.get_bucket_centers()
        assert centers.shape == (101,)
        assert np.isclose(centers[0], -50.0)
        assert np.isclose(centers[50], 0.0)
        assert np.isclose(centers[100], 50.0)

    def test_get_bucket_centers_custom(self) -> None:
        """Test custom bucket centers."""
        custom_centers = tuple(range(5))
        config = MetricsConfig(num_buckets=5, bucket_centers_bps=custom_centers)
        centers = config.get_bucket_centers()
        assert np.allclose(centers, custom_centers)


class TestLogLikelihood:
    """Tests for log_likelihood function."""

    def test_perfect_prediction(self) -> None:
        """Test log-likelihood with perfect predictions."""
        # Perfect prediction: all probability on true class
        probs = torch.eye(5)  # Each row is one-hot for corresponding class
        targets = torch.arange(5)

        ll = log_likelihood(probs, targets)
        # Log(1.0) = 0
        assert torch.allclose(ll, torch.zeros(5), atol=1e-6)

    def test_uniform_prediction(self) -> None:
        """Test log-likelihood with uniform predictions."""
        num_buckets = 10
        probs = torch.ones(3, num_buckets) / num_buckets
        targets = torch.tensor([0, 5, 9])

        ll = log_likelihood(probs, targets)
        expected = np.log(1.0 / num_buckets)
        assert torch.allclose(ll, torch.tensor([expected] * 3, dtype=torch.float32))

    def test_shape_preservation(self) -> None:
        """Test that output shape matches batch size."""
        probs = torch.rand(32, 101)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 101, (32,))

        ll = log_likelihood(probs, targets)
        assert ll.shape == (32,)


class TestNegativeLogLikelihood:
    """Tests for negative_log_likelihood function."""

    def test_nll_is_negative_of_ll(self) -> None:
        """Test that NLL is negative of LL."""
        probs = torch.rand(10, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 5, (10,))

        ll = log_likelihood(probs, targets)
        nll = negative_log_likelihood(probs, targets, reduction="none")

        assert torch.allclose(nll, -ll)

    def test_reduction_mean(self) -> None:
        """Test mean reduction."""
        probs = torch.rand(10, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 5, (10,))

        nll = negative_log_likelihood(probs, targets, reduction="mean")
        assert nll.shape == ()

    def test_reduction_sum(self) -> None:
        """Test sum reduction."""
        probs = torch.rand(10, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 5, (10,))

        nll_none = negative_log_likelihood(probs, targets, reduction="none")
        nll_sum = negative_log_likelihood(probs, targets, reduction="sum")

        assert torch.isclose(nll_sum, nll_none.sum())

    def test_reduction_none(self) -> None:
        """Test no reduction."""
        probs = torch.rand(10, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 5, (10,))

        nll = negative_log_likelihood(probs, targets, reduction="none")
        assert nll.shape == (10,)


class TestBrierScore:
    """Tests for brier_score function."""

    def test_perfect_prediction(self) -> None:
        """Test Brier score with perfect predictions."""
        probs = torch.eye(5)
        targets = torch.arange(5)

        bs = brier_score(probs, targets, reduction="mean")
        # Perfect prediction: Brier score = 0
        assert torch.isclose(bs, torch.tensor(0.0), atol=1e-6)

    def test_worst_prediction(self) -> None:
        """Test Brier score with worst predictions."""
        # Predict opposite class with probability 1
        probs = torch.eye(5)
        targets = torch.tensor([1, 0, 3, 2, 4])  # Wrong by one class

        bs = brier_score(probs, targets, reduction="mean")
        # Each sample has error (0-1)^2 for true class + (1-0)^2 for predicted
        # = 1 + 1 = 2 per sample
        assert bs > 0

    def test_uniform_prediction(self) -> None:
        """Test Brier score with uniform predictions."""
        num_buckets = 4
        probs = torch.ones(1, num_buckets) / num_buckets  # [0.25, 0.25, 0.25, 0.25]
        targets = torch.tensor([0])

        bs = brier_score(probs, targets, reduction="mean")
        # Expected: (0.25-1)^2 + 3*(0.25-0)^2 = 0.5625 + 0.1875 = 0.75
        expected = (0.25 - 1) ** 2 + 3 * (0.25 - 0) ** 2
        assert torch.isclose(bs, torch.tensor(expected), atol=1e-6)

    def test_reduction_options(self) -> None:
        """Test all reduction options."""
        probs = torch.rand(10, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 5, (10,))

        bs_mean = brier_score(probs, targets, reduction="mean")
        bs_sum = brier_score(probs, targets, reduction="sum")
        bs_none = brier_score(probs, targets, reduction="none")

        assert bs_mean.shape == ()
        assert bs_sum.shape == ()
        assert bs_none.shape == (10,)
        assert torch.isclose(bs_sum, bs_none.sum())
        assert torch.isclose(bs_mean, bs_none.mean())


class TestBrierScorePerBucket:
    """Tests for brier_score_per_bucket function."""

    def test_output_shape(self) -> None:
        """Test that output shape is (num_buckets,)."""
        num_buckets = 101
        probs = torch.rand(32, num_buckets)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, num_buckets, (32,))

        bs_per_bucket = brier_score_per_bucket(probs, targets)
        assert bs_per_bucket.shape == (num_buckets,)

    def test_all_values_non_negative(self) -> None:
        """Test that all per-bucket scores are non-negative."""
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 10, (100,))

        bs_per_bucket = brier_score_per_bucket(probs, targets)
        assert (bs_per_bucket >= 0).all()


class TestCalibration:
    """Tests for calibration functions."""

    def test_compute_calibration_result_type(self) -> None:
        """Test that compute_calibration returns CalibrationResult."""
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 10, (100,))

        result = compute_calibration(probs, targets, num_bins=5)
        assert isinstance(result, CalibrationResult)

    def test_calibration_result_shapes(self) -> None:
        """Test shapes of calibration result arrays."""
        num_bins = 5
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 10, (100,))

        result = compute_calibration(probs, targets, num_bins=num_bins)

        assert result.bin_confidences.shape == (num_bins,)
        assert result.bin_accuracies.shape == (num_bins,)
        assert result.bin_counts.shape == (num_bins,)

    def test_ece_bounded(self) -> None:
        """Test that ECE is bounded between 0 and 1."""
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 10, (100,))

        result = compute_calibration(probs, targets)
        assert 0 <= result.expected_calibration_error <= 1

    def test_mce_bounded(self) -> None:
        """Test that MCE is bounded between 0 and 1."""
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 10, (100,))

        result = compute_calibration(probs, targets)
        assert 0 <= result.maximum_calibration_error <= 1

    def test_expected_calibration_error_shortcut(self) -> None:
        """Test that expected_calibration_error matches compute_calibration."""
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 10, (100,))

        result = compute_calibration(probs, targets, num_bins=10)
        ece = expected_calibration_error(probs, targets, num_bins=10)

        assert np.isclose(ece, result.expected_calibration_error)


class TestPredictedMeanAndStd:
    """Tests for predicted_mean_bps and predicted_std_bps."""

    def test_predicted_mean_uniform(self) -> None:
        """Test predicted mean with uniform distribution."""
        probs = torch.ones(1, 5) / 5
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        mean = predicted_mean_bps(probs, centers)
        # Uniform over symmetric centers should give mean = 0
        assert torch.isclose(mean[0], torch.tensor(0.0), atol=1e-6)

    def test_predicted_mean_peaked(self) -> None:
        """Test predicted mean with peaked distribution."""
        probs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])  # All mass on center
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        mean = predicted_mean_bps(probs, centers)
        assert torch.isclose(mean[0], torch.tensor(0.0), atol=1e-6)

    def test_predicted_mean_biased(self) -> None:
        """Test predicted mean with biased distribution."""
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])  # All mass on first
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        mean = predicted_mean_bps(probs, centers)
        assert torch.isclose(mean[0], torch.tensor(-2.0), atol=1e-6)

    def test_predicted_std_uniform(self) -> None:
        """Test predicted std with uniform distribution."""
        probs = torch.ones(1, 5) / 5
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        std = predicted_std_bps(probs, centers)
        # Uniform has some non-zero std
        assert std[0] > 0

    def test_predicted_std_peaked(self) -> None:
        """Test predicted std with peaked (low uncertainty) distribution."""
        probs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        std = predicted_std_bps(probs, centers)
        # All mass on one point: std should be ~0
        assert torch.isclose(std[0], torch.tensor(0.0), atol=1e-4)

    def test_output_shapes(self) -> None:
        """Test output shapes."""
        batch_size = 32
        num_buckets = 101
        probs = torch.rand(batch_size, num_buckets)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        centers = torch.linspace(-50, 50, num_buckets)

        mean = predicted_mean_bps(probs, centers)
        std = predicted_std_bps(probs, centers)

        assert mean.shape == (batch_size,)
        assert std.shape == (batch_size,)


class TestDirectionalAccuracy:
    """Tests for directional_accuracy function."""

    def test_perfect_direction(self) -> None:
        """Test with predictions that always get direction right."""
        # Predict positive, actual is positive
        probs = torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5]])
        targets = torch.tensor([4])  # Positive actual
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        acc = directional_accuracy(probs, targets, centers)
        assert torch.isclose(acc, torch.tensor(1.0))

    def test_wrong_direction(self) -> None:
        """Test with predictions that get direction wrong."""
        # Predict positive, actual is negative
        probs = torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5]])
        targets = torch.tensor([0])  # Negative actual
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        acc = directional_accuracy(probs, targets, centers)
        assert torch.isclose(acc, torch.tensor(0.0))

    def test_mixed_directions(self) -> None:
        """Test with mixed correct/incorrect directions."""
        probs = torch.tensor([
            [0.0, 0.0, 0.0, 0.5, 0.5],  # Predict positive
            [0.5, 0.5, 0.0, 0.0, 0.0],  # Predict negative
        ])
        targets = torch.tensor([4, 4])  # Both positive actual
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        acc = directional_accuracy(probs, targets, centers)
        assert torch.isclose(acc, torch.tensor(0.5))

    def test_threshold_filtering(self) -> None:
        """Test that threshold filters out small predictions."""
        probs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])  # Predict ~0
        targets = torch.tensor([2])  # Actual ~0
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        # With high threshold, this should be filtered
        acc = directional_accuracy(probs, targets, centers, threshold_bps=0.5)
        assert torch.isclose(acc, torch.tensor(0.5))  # No valid samples


class TestSharpness:
    """Tests for sharpness function."""

    def test_peaked_is_sharp(self) -> None:
        """Test that peaked distribution has low sharpness (is sharp)."""
        probs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        sharp = sharpness(probs, centers)
        assert sharp < 0.1  # Very sharp

    def test_uniform_is_not_sharp(self) -> None:
        """Test that uniform distribution has high sharpness (not sharp)."""
        probs = torch.ones(1, 5) / 5
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        sharp = sharpness(probs, centers)
        assert sharp > 0.5  # Not sharp


class TestEntropy:
    """Tests for entropy functions."""

    def test_peaked_low_entropy(self) -> None:
        """Test that peaked distribution has low entropy."""
        probs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        ent = entropy(probs)
        assert ent[0] < 0.1

    def test_uniform_high_entropy(self) -> None:
        """Test that uniform distribution has high entropy."""
        num_classes = 10
        probs = torch.ones(1, num_classes) / num_classes
        ent = entropy(probs)
        # Max entropy for 10 classes is log(10) ≈ 2.3
        assert ent[0] > 2.0

    def test_mean_entropy(self) -> None:
        """Test mean_entropy function."""
        probs = torch.ones(5, 10) / 10
        ent = mean_entropy(probs)
        assert ent.shape == ()


class TestPnLSimulation:
    """Tests for simulate_pnl function."""

    def test_pnl_result_type(self) -> None:
        """Test that simulate_pnl returns PnLResult."""
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 10, (100,))
        centers = torch.linspace(-5, 5, 10)

        result = simulate_pnl(probs, targets, centers)
        assert isinstance(result, PnLResult)

    def test_perfect_prediction_positive_pnl(self) -> None:
        """Test that perfect predictions yield positive P&L."""
        # Predict exactly where price will go
        probs = torch.zeros(10, 5)
        targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        centers = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Set probability = 1 for true bucket
        for i in range(10):
            probs[i, targets[i]] = 1.0

        result = simulate_pnl(probs, targets, centers)
        # When prediction = actual, we always win
        # (position matches direction)
        assert result.total_pnl_bps >= 0

    def test_no_trades_with_high_threshold(self) -> None:
        """Test that high threshold results in no trades."""
        probs = torch.rand(10, 5)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 5, (10,))
        centers = torch.linspace(-5, 5, 5)

        result = simulate_pnl(
            probs, targets, centers, confidence_threshold=1.0  # Impossible to exceed
        )

        assert result.num_trades == 0
        assert result.total_pnl_bps == 0.0

    def test_pnl_series_length(self) -> None:
        """Test that pnl_series length matches num_trades."""
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 10, (100,))
        centers = torch.linspace(-5, 5, 10)

        result = simulate_pnl(probs, targets, centers)
        assert len(result.pnl_series) == result.num_trades


class TestHorizonMetrics:
    """Tests for compute_horizon_metrics function."""

    def test_returns_horizon_metrics(self) -> None:
        """Test that function returns HorizonMetrics."""
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 10, (100,))
        centers = torch.linspace(-5, 5, 10)

        result = compute_horizon_metrics(probs, targets, horizon=60, bucket_centers_bps=centers)
        assert isinstance(result, HorizonMetrics)
        assert result.horizon == 60

    def test_all_metrics_computed(self) -> None:
        """Test that all metrics are computed."""
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        targets = torch.randint(0, 10, (100,))
        centers = torch.linspace(-5, 5, 10)

        result = compute_horizon_metrics(probs, targets, horizon=60, bucket_centers_bps=centers)

        assert result.nll > 0  # NLL should be positive
        assert result.brier_score >= 0  # Brier is non-negative
        assert 0 <= result.ece <= 1  # ECE is bounded
        assert 0 <= result.directional_accuracy <= 1  # Accuracy is bounded
        assert result.sharpness > 0  # Sharpness (std) should be positive
        assert result.mean_entropy >= 0  # Entropy is non-negative


class TestMultiHorizonMetrics:
    """Tests for compute_multi_horizon_metrics function."""

    def test_returns_multi_horizon_metrics(self) -> None:
        """Test that function returns MultiHorizonMetrics."""
        horizons = [1, 5, 10]
        probs_dict = {h: torch.rand(50, 10) for h in horizons}
        for h in horizons:
            probs_dict[h] = probs_dict[h] / probs_dict[h].sum(dim=-1, keepdim=True)
        targets_dict = {h: torch.randint(0, 10, (50,)) for h in horizons}
        centers = torch.linspace(-5, 5, 10)

        result = compute_multi_horizon_metrics(probs_dict, targets_dict, centers)
        assert isinstance(result, MultiHorizonMetrics)

    def test_horizons_in_order(self) -> None:
        """Test that horizons are returned in sorted order."""
        horizons = [60, 5, 300, 1]  # Unsorted
        probs_dict = {h: torch.rand(50, 10) for h in horizons}
        for h in horizons:
            probs_dict[h] = probs_dict[h] / probs_dict[h].sum(dim=-1, keepdim=True)
        targets_dict = {h: torch.randint(0, 10, (50,)) for h in horizons}
        centers = torch.linspace(-5, 5, 10)

        result = compute_multi_horizon_metrics(probs_dict, targets_dict, centers)

        horizon_list = [m.horizon for m in result.horizons]
        assert horizon_list == sorted(horizons)

    def test_aggregate_metrics_are_means(self) -> None:
        """Test that aggregate metrics are means of per-horizon metrics."""
        horizons = [1, 5, 10]
        probs_dict = {h: torch.rand(50, 10) for h in horizons}
        for h in horizons:
            probs_dict[h] = probs_dict[h] / probs_dict[h].sum(dim=-1, keepdim=True)
        targets_dict = {h: torch.randint(0, 10, (50,)) for h in horizons}
        centers = torch.linspace(-5, 5, 10)

        result = compute_multi_horizon_metrics(probs_dict, targets_dict, centers)

        expected_mean_nll = np.mean([m.nll for m in result.horizons])
        expected_mean_brier = np.mean([m.brier_score for m in result.horizons])
        expected_mean_ece = np.mean([m.ece for m in result.horizons])
        expected_mean_da = np.mean([m.directional_accuracy for m in result.horizons])

        assert np.isclose(result.mean_nll, expected_mean_nll)
        assert np.isclose(result.mean_brier, expected_mean_brier)
        assert np.isclose(result.mean_ece, expected_mean_ece)
        assert np.isclose(result.mean_directional_accuracy, expected_mean_da)


class TestFormatMetricsReport:
    """Tests for format_metrics_report function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        horizons = [1, 5, 10]
        probs_dict = {h: torch.rand(50, 10) for h in horizons}
        for h in horizons:
            probs_dict[h] = probs_dict[h] / probs_dict[h].sum(dim=-1, keepdim=True)
        targets_dict = {h: torch.randint(0, 10, (50,)) for h in horizons}
        centers = torch.linspace(-5, 5, 10)

        metrics = compute_multi_horizon_metrics(probs_dict, targets_dict, centers)
        report = format_metrics_report(metrics)

        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_horizon_info(self) -> None:
        """Test that report contains horizon information."""
        horizons = [1, 60, 300]
        probs_dict = {h: torch.rand(50, 10) for h in horizons}
        for h in horizons:
            probs_dict[h] = probs_dict[h] / probs_dict[h].sum(dim=-1, keepdim=True)
        targets_dict = {h: torch.randint(0, 10, (50,)) for h in horizons}
        centers = torch.linspace(-5, 5, 10)

        metrics = compute_multi_horizon_metrics(probs_dict, targets_dict, centers)
        report = format_metrics_report(metrics)

        assert "1" in report
        assert "60" in report
        assert "300" in report

    def test_contains_metric_names(self) -> None:
        """Test that report contains metric names."""
        horizons = [1, 5]
        probs_dict = {h: torch.rand(50, 10) for h in horizons}
        for h in horizons:
            probs_dict[h] = probs_dict[h] / probs_dict[h].sum(dim=-1, keepdim=True)
        targets_dict = {h: torch.randint(0, 10, (50,)) for h in horizons}
        centers = torch.linspace(-5, 5, 10)

        metrics = compute_multi_horizon_metrics(probs_dict, targets_dict, centers)
        report = format_metrics_report(metrics)

        assert "NLL" in report
        assert "Brier" in report
        assert "ECE" in report
        assert "Dir Acc" in report
        assert "Sharpness" in report
