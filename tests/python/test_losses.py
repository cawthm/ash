"""Tests for the losses module."""

import pytest
import torch

from models.architectures.losses import (
    EMDLoss,
    FocalLoss,
    LossConfig,
    MultiHorizonLoss,
    SoftCrossEntropyLoss,
    compute_cdf,
    create_soft_labels,
    earth_movers_distance,
    get_loss_function,
)


class TestLossConfig:
    """Tests for LossConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = LossConfig()
        assert config.num_buckets == 101
        assert config.loss_type == "emd"
        assert config.label_smoothing_sigma == 1.0
        assert config.focal_gamma == 2.0
        assert config.emd_p == 1
        assert config.reduction == "mean"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = LossConfig(
            num_buckets=51,
            loss_type="soft_ce",
            label_smoothing_sigma=2.0,
            focal_gamma=3.0,
            emd_p=2,
            reduction="sum",
        )
        assert config.num_buckets == 51
        assert config.loss_type == "soft_ce"
        assert config.label_smoothing_sigma == 2.0
        assert config.focal_gamma == 3.0
        assert config.emd_p == 2
        assert config.reduction == "sum"

    def test_num_buckets_too_small(self) -> None:
        """Test that num_buckets must be at least 3."""
        with pytest.raises(ValueError, match="num_buckets must be at least 3"):
            LossConfig(num_buckets=2)

    def test_invalid_loss_type(self) -> None:
        """Test that invalid loss_type raises error."""
        with pytest.raises(ValueError, match="Unknown loss_type"):
            LossConfig(loss_type="invalid")  # type: ignore[arg-type]

    def test_invalid_sigma(self) -> None:
        """Test that non-positive sigma raises error."""
        with pytest.raises(ValueError, match="label_smoothing_sigma must be positive"):
            LossConfig(label_smoothing_sigma=0.0)

    def test_negative_focal_gamma(self) -> None:
        """Test that negative focal_gamma raises error."""
        with pytest.raises(ValueError, match="focal_gamma must be non-negative"):
            LossConfig(focal_gamma=-1.0)

    def test_invalid_emd_p(self) -> None:
        """Test that invalid emd_p raises error."""
        with pytest.raises(ValueError, match="emd_p must be 1 or 2"):
            LossConfig(emd_p=3)

    def test_invalid_reduction(self) -> None:
        """Test that invalid reduction raises error."""
        with pytest.raises(ValueError, match="Unknown reduction"):
            LossConfig(reduction="invalid")  # type: ignore[arg-type]


class TestComputeCDF:
    """Tests for compute_cdf function."""

    def test_simple_cdf(self) -> None:
        """Test CDF computation for simple distribution."""
        probs = torch.tensor([0.2, 0.3, 0.5])
        cdf = compute_cdf(probs)
        expected = torch.tensor([0.2, 0.5, 1.0])
        assert torch.allclose(cdf, expected)

    def test_uniform_cdf(self) -> None:
        """Test CDF for uniform distribution."""
        probs = torch.ones(5) / 5
        cdf = compute_cdf(probs)
        expected = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
        assert torch.allclose(cdf, expected)

    def test_batched_cdf(self) -> None:
        """Test CDF computation for batched input."""
        probs = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
        cdf = compute_cdf(probs)
        expected = torch.tensor([[0.5, 1.0], [0.25, 1.0]])
        assert torch.allclose(cdf, expected)

    def test_cdf_ends_at_one(self) -> None:
        """Test that CDF always ends at 1.0."""
        probs = torch.softmax(torch.randn(10, 50), dim=-1)
        cdf = compute_cdf(probs)
        assert torch.allclose(cdf[:, -1], torch.ones(10))


class TestEarthMoversDistance:
    """Tests for earth_movers_distance function."""

    def test_identical_distributions(self) -> None:
        """Test EMD between identical distributions is 0."""
        probs = torch.softmax(torch.randn(5, 10), dim=-1)
        emd = earth_movers_distance(probs, probs)
        assert torch.allclose(emd, torch.zeros(5), atol=1e-6)

    def test_one_hot_distributions(self) -> None:
        """Test EMD between one-hot distributions equals bucket distance."""
        num_buckets = 10

        # Bucket 0 vs bucket 9 (max distance)
        pred = torch.zeros(1, num_buckets)
        pred[0, 0] = 1.0
        target = torch.zeros(1, num_buckets)
        target[0, 9] = 1.0

        emd_l1 = earth_movers_distance(pred, target, p=1)
        # L1 EMD: sum of abs CDF differences
        # pred CDF: [1,1,1,1,1,1,1,1,1,1]
        # target CDF: [0,0,0,0,0,0,0,0,0,1]
        # diff: [1,1,1,1,1,1,1,1,1,0] -> sum = 9
        assert torch.isclose(emd_l1, torch.tensor(9.0))

    def test_adjacent_buckets(self) -> None:
        """Test EMD between adjacent buckets is 1."""
        num_buckets = 10
        pred = torch.zeros(1, num_buckets)
        pred[0, 4] = 1.0
        target = torch.zeros(1, num_buckets)
        target[0, 5] = 1.0

        emd_l1 = earth_movers_distance(pred, target, p=1)
        assert torch.isclose(emd_l1, torch.tensor(1.0))

    def test_symmetric(self) -> None:
        """Test that EMD is symmetric."""
        pred = torch.softmax(torch.randn(5, 10), dim=-1)
        target = torch.softmax(torch.randn(5, 10), dim=-1)

        emd_forward = earth_movers_distance(pred, target)
        emd_backward = earth_movers_distance(target, pred)
        assert torch.allclose(emd_forward, emd_backward)

    def test_l2_emd(self) -> None:
        """Test L2 EMD calculation."""
        num_buckets = 5
        pred = torch.zeros(1, num_buckets)
        pred[0, 0] = 1.0
        target = torch.zeros(1, num_buckets)
        target[0, 2] = 1.0

        emd_l2 = earth_movers_distance(pred, target, p=2)
        # pred CDF: [1,1,1,1,1]
        # target CDF: [0,0,1,1,1]
        # diff squared: [1,1,0,0,0] -> sum = 2 -> sqrt = sqrt(2)
        expected = torch.sqrt(torch.tensor(2.0))
        assert torch.isclose(emd_l2, expected)


class TestCreateSoftLabels:
    """Tests for create_soft_labels function."""

    def test_center_bucket(self) -> None:
        """Test soft labels for center bucket."""
        targets = torch.tensor([5])
        soft = create_soft_labels(targets, num_buckets=11, sigma=1.0)

        assert soft.shape == (1, 11)
        # Maximum at target bucket
        assert soft[0, 5] == soft.max()
        # Sums to 1
        assert torch.isclose(soft.sum(), torch.tensor(1.0))
        # Symmetric around center
        assert torch.isclose(soft[0, 4], soft[0, 6])
        assert torch.isclose(soft[0, 3], soft[0, 7])

    def test_edge_bucket(self) -> None:
        """Test soft labels for edge bucket."""
        targets = torch.tensor([0])
        soft = create_soft_labels(targets, num_buckets=11, sigma=1.0)

        assert soft.shape == (1, 11)
        assert soft[0, 0] == soft.max()
        assert torch.isclose(soft.sum(), torch.tensor(1.0))

    def test_sigma_effect(self) -> None:
        """Test that larger sigma spreads probability more."""
        targets = torch.tensor([5])
        soft_narrow = create_soft_labels(targets, num_buckets=11, sigma=0.5)
        soft_wide = create_soft_labels(targets, num_buckets=11, sigma=2.0)

        # Narrow sigma should have higher peak
        assert soft_narrow[0, 5] > soft_wide[0, 5]
        # Wide sigma should have higher probability at distant buckets
        assert soft_wide[0, 0] > soft_narrow[0, 0]

    def test_batch_soft_labels(self) -> None:
        """Test soft labels for batch of targets."""
        targets = torch.tensor([2, 5, 8])
        soft = create_soft_labels(targets, num_buckets=11, sigma=1.0)

        assert soft.shape == (3, 11)
        assert soft[0].argmax() == 2
        assert soft[1].argmax() == 5
        assert soft[2].argmax() == 8


class TestEMDLoss:
    """Tests for EMDLoss class."""

    def test_initialization(self) -> None:
        """Test EMDLoss initialization."""
        loss_fn = EMDLoss()
        assert loss_fn.config.loss_type == "emd"

    def test_perfect_prediction(self) -> None:
        """Test loss when prediction matches target."""
        loss_fn = EMDLoss(LossConfig(num_buckets=11))
        # Logits that strongly predict bucket 5
        logits = torch.zeros(2, 11)
        logits[:, 5] = 10.0  # High logit at bucket 5
        targets = torch.tensor([5, 5])

        loss = loss_fn(logits, targets)
        # Should be close to 0
        assert loss.item() < 0.1

    def test_bad_prediction(self) -> None:
        """Test loss when prediction is far from target."""
        loss_fn = EMDLoss(LossConfig(num_buckets=11))
        # Logits that strongly predict bucket 0
        logits = torch.zeros(2, 11)
        logits[:, 0] = 10.0
        # Target is bucket 10 (far away)
        targets = torch.tensor([10, 10])

        loss = loss_fn(logits, targets)
        # Should be high (close to 10)
        assert loss.item() > 5.0

    def test_integer_targets(self) -> None:
        """Test EMD loss with integer target indices."""
        loss_fn = EMDLoss(LossConfig(num_buckets=11))
        logits = torch.randn(4, 11)
        targets = torch.tensor([0, 3, 7, 10])

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_probability_targets(self) -> None:
        """Test EMD loss with probability distribution targets."""
        loss_fn = EMDLoss(LossConfig(num_buckets=11))
        logits = torch.randn(4, 11)
        targets = torch.softmax(torch.randn(4, 11), dim=-1)

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_reduction_none(self) -> None:
        """Test EMD loss with no reduction."""
        config = LossConfig(num_buckets=11, reduction="none")
        loss_fn = EMDLoss(config)
        logits = torch.randn(4, 11)
        targets = torch.tensor([0, 3, 7, 10])

        loss = loss_fn(logits, targets)
        assert loss.shape == (4,)

    def test_reduction_sum(self) -> None:
        """Test EMD loss with sum reduction."""
        config = LossConfig(num_buckets=11, reduction="sum")
        loss_fn = EMDLoss(config)
        logits = torch.randn(4, 11)
        targets = torch.tensor([0, 3, 7, 10])

        loss_sum = loss_fn(logits, targets)

        # Compare to none reduction
        config_none = LossConfig(num_buckets=11, reduction="none")
        loss_fn_none = EMDLoss(config_none)
        loss_none = loss_fn_none(logits, targets)

        assert torch.isclose(loss_sum, loss_none.sum())


class TestSoftCrossEntropyLoss:
    """Tests for SoftCrossEntropyLoss class."""

    def test_initialization(self) -> None:
        """Test SoftCrossEntropyLoss initialization."""
        loss_fn = SoftCrossEntropyLoss()
        assert loss_fn.config.loss_type == "soft_ce"

    def test_perfect_prediction(self) -> None:
        """Test loss when prediction matches target."""
        loss_fn = SoftCrossEntropyLoss(LossConfig(num_buckets=11, loss_type="soft_ce"))
        logits = torch.zeros(2, 11)
        logits[:, 5] = 10.0
        targets = torch.tensor([5, 5])

        loss = loss_fn(logits, targets)
        # Loss is not zero due to soft labels spreading probability
        # But correct prediction should have lower loss than wrong prediction
        wrong_logits = torch.zeros(2, 11)
        wrong_logits[:, 0] = 10.0  # Predict bucket 0, target is 5
        wrong_loss = loss_fn(wrong_logits, targets)
        assert loss.item() < wrong_loss.item()

    def test_nearby_vs_far_prediction(self) -> None:
        """Test that nearby predictions have lower loss than far ones."""
        loss_fn = SoftCrossEntropyLoss(
            LossConfig(num_buckets=11, loss_type="soft_ce", label_smoothing_sigma=1.0)
        )
        target = torch.tensor([5])

        # Predict bucket 4 (nearby)
        logits_near = torch.zeros(1, 11)
        logits_near[0, 4] = 10.0
        loss_near = loss_fn(logits_near, target)

        # Predict bucket 0 (far)
        logits_far = torch.zeros(1, 11)
        logits_far[0, 0] = 10.0
        loss_far = loss_fn(logits_far, target)

        # Nearby should have lower loss
        assert loss_near.item() < loss_far.item()

    def test_sigma_effect_on_loss(self) -> None:
        """Test that sigma affects loss sensitivity."""
        target = torch.tensor([5])
        logits = torch.zeros(1, 11)
        logits[0, 3] = 10.0  # Predict bucket 3, target is 5

        # Small sigma: more sensitive to distance
        loss_narrow = SoftCrossEntropyLoss(
            LossConfig(num_buckets=11, loss_type="soft_ce", label_smoothing_sigma=0.5)
        )
        # Large sigma: less sensitive to distance
        loss_wide = SoftCrossEntropyLoss(
            LossConfig(num_buckets=11, loss_type="soft_ce", label_smoothing_sigma=3.0)
        )

        # Narrow sigma should penalize the error more
        assert loss_narrow(logits, target).item() > loss_wide(logits, target).item()

    def test_probability_targets(self) -> None:
        """Test soft CE loss with probability distribution targets."""
        loss_fn = SoftCrossEntropyLoss(
            LossConfig(num_buckets=11, loss_type="soft_ce")
        )
        logits = torch.randn(4, 11)
        targets = torch.softmax(torch.randn(4, 11), dim=-1)

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestFocalLoss:
    """Tests for FocalLoss class."""

    def test_initialization(self) -> None:
        """Test FocalLoss initialization."""
        loss_fn = FocalLoss()
        assert loss_fn.config.loss_type == "focal"

    def test_gamma_zero_equals_ce(self) -> None:
        """Test that gamma=0 gives standard cross-entropy."""
        config = LossConfig(num_buckets=11, loss_type="focal", focal_gamma=0.0)
        focal_loss = FocalLoss(config)

        logits = torch.randn(10, 11)
        targets = torch.randint(0, 11, (10,))

        focal = focal_loss(logits, targets)
        ce = torch.nn.functional.cross_entropy(logits, targets)

        assert torch.isclose(focal, ce, rtol=1e-4)

    def test_focal_reduces_easy_examples(self) -> None:
        """Test that focal loss down-weights confident correct predictions."""
        logits = torch.zeros(2, 11)
        # Easy example: very confident correct prediction
        logits[0, 5] = 10.0
        # Hard example: uncertain prediction
        logits[1, 5] = 0.5
        targets = torch.tensor([5, 5])

        # Standard CE
        ce_config = LossConfig(num_buckets=11, loss_type="focal", focal_gamma=0.0)
        ce_loss = FocalLoss(ce_config)
        ce = ce_loss(logits, targets)

        # Focal with gamma=2
        focal_config = LossConfig(num_buckets=11, loss_type="focal", focal_gamma=2.0)
        focal = FocalLoss(focal_config)
        focal_loss_val = focal(logits, targets)

        # Focal should have lower total loss because easy example is down-weighted
        assert focal_loss_val.item() < ce.item()

    def test_reduction_options(self) -> None:
        """Test focal loss reduction options."""
        logits = torch.randn(4, 11)
        targets = torch.randint(0, 11, (4,))

        config_none = LossConfig(num_buckets=11, loss_type="focal", reduction="none")
        loss_none = FocalLoss(config_none)(logits, targets)
        assert loss_none.shape == (4,)

        config_sum = LossConfig(num_buckets=11, loss_type="focal", reduction="sum")
        loss_sum = FocalLoss(config_sum)(logits, targets)
        assert torch.isclose(loss_sum, loss_none.sum())


class TestMultiHorizonLoss:
    """Tests for MultiHorizonLoss class."""

    def test_initialization(self) -> None:
        """Test MultiHorizonLoss initialization."""
        loss_fn = MultiHorizonLoss(num_horizons=8)
        assert loss_fn.num_horizons == 8
        weights = loss_fn.horizon_weights
        assert weights.shape == (8,)
        assert torch.isclose(weights.sum(), torch.tensor(1.0))

    def test_custom_weights(self) -> None:
        """Test MultiHorizonLoss with custom horizon weights."""
        weights = [1.0, 2.0, 3.0, 4.0]
        loss_fn = MultiHorizonLoss(num_horizons=4, horizon_weights=weights)

        # Weights should be normalized
        expected = torch.tensor(weights) / sum(weights)
        actual_weights = loss_fn.horizon_weights
        assert torch.allclose(actual_weights, expected)

    def test_wrong_number_of_weights(self) -> None:
        """Test error when weight count doesn't match horizons."""
        with pytest.raises(ValueError, match="Expected 4 weights"):
            MultiHorizonLoss(num_horizons=4, horizon_weights=[1.0, 2.0])

    def test_list_input(self) -> None:
        """Test multi-horizon loss with list inputs."""
        loss_fn = MultiHorizonLoss(
            LossConfig(num_buckets=11), num_horizons=4
        )

        logits = [torch.randn(8, 11) for _ in range(4)]
        targets = [torch.randint(0, 11, (8,)) for _ in range(4)]

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_dict_input(self) -> None:
        """Test multi-horizon loss with dict inputs."""
        loss_fn = MultiHorizonLoss(
            LossConfig(num_buckets=11), num_horizons=4
        )

        logits = {i: torch.randn(8, 11) for i in range(4)}
        targets = {i: torch.randint(0, 11, (8,)) for i in range(4)}

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_wrong_number_of_horizons(self) -> None:
        """Test error when input has wrong number of horizons."""
        loss_fn = MultiHorizonLoss(num_horizons=4)

        logits = [torch.randn(8, 101) for _ in range(3)]  # Only 3 horizons
        targets = [torch.randint(0, 101, (8,)) for _ in range(3)]

        with pytest.raises(ValueError, match="Expected 4 horizons"):
            loss_fn(logits, targets)

    def test_different_loss_types(self) -> None:
        """Test multi-horizon loss with different base loss types."""
        batch_size = 8
        num_buckets = 11
        num_horizons = 4

        for loss_type in ["emd", "soft_ce", "focal"]:
            config = LossConfig(
                num_buckets=num_buckets,
                loss_type=loss_type,  # type: ignore[arg-type]
            )
            loss_fn = MultiHorizonLoss(config, num_horizons=num_horizons)

            logits = [torch.randn(batch_size, num_buckets) for _ in range(num_horizons)]
            targets = [
                torch.randint(0, num_buckets, (batch_size,))
                for _ in range(num_horizons)
            ]

            loss = loss_fn(logits, targets)
            assert loss.dim() == 0
            assert loss.item() >= 0


class TestGetLossFunction:
    """Tests for get_loss_function factory."""

    def test_emd_loss(self) -> None:
        """Test creating EMD loss."""
        config = LossConfig(loss_type="emd")
        loss_fn = get_loss_function(config)
        assert isinstance(loss_fn, EMDLoss)

    def test_soft_ce_loss(self) -> None:
        """Test creating soft CE loss."""
        config = LossConfig(loss_type="soft_ce")
        loss_fn = get_loss_function(config)
        assert isinstance(loss_fn, SoftCrossEntropyLoss)

    def test_focal_loss(self) -> None:
        """Test creating focal loss."""
        config = LossConfig(loss_type="focal")
        loss_fn = get_loss_function(config)
        assert isinstance(loss_fn, FocalLoss)


class TestGradientFlow:
    """Tests for gradient flow through loss functions."""

    def test_emd_loss_gradients(self) -> None:
        """Test that EMD loss allows gradient flow."""
        loss_fn = EMDLoss(LossConfig(num_buckets=11))
        logits = torch.randn(4, 11, requires_grad=True)
        targets = torch.randint(0, 11, (4,))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_soft_ce_loss_gradients(self) -> None:
        """Test that soft CE loss allows gradient flow."""
        loss_fn = SoftCrossEntropyLoss(LossConfig(num_buckets=11, loss_type="soft_ce"))
        logits = torch.randn(4, 11, requires_grad=True)
        targets = torch.randint(0, 11, (4,))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_focal_loss_gradients(self) -> None:
        """Test that focal loss allows gradient flow."""
        loss_fn = FocalLoss(LossConfig(num_buckets=11, loss_type="focal"))
        logits = torch.randn(4, 11, requires_grad=True)
        targets = torch.randint(0, 11, (4,))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_multi_horizon_loss_gradients(self) -> None:
        """Test that multi-horizon loss allows gradient flow."""
        loss_fn = MultiHorizonLoss(LossConfig(num_buckets=11), num_horizons=4)
        logits = [torch.randn(4, 11, requires_grad=True) for _ in range(4)]
        targets = [torch.randint(0, 11, (4,)) for _ in range(4)]

        loss = loss_fn(logits, targets)
        loss.backward()

        for idx, logit_tensor in enumerate(logits):
            assert logit_tensor.grad is not None, f"Horizon {idx} has no gradient"
            assert not torch.isnan(logit_tensor.grad).any(), f"Horizon {idx} has NaN gradients"


class TestNumericalStability:
    """Tests for numerical stability of loss functions."""

    def test_emd_with_extreme_logits(self) -> None:
        """Test EMD loss with very large logits."""
        loss_fn = EMDLoss(LossConfig(num_buckets=11))
        logits = torch.randn(4, 11) * 100  # Very large logits
        targets = torch.randint(0, 11, (4,))

        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_soft_ce_with_extreme_logits(self) -> None:
        """Test soft CE loss with very large logits."""
        loss_fn = SoftCrossEntropyLoss(LossConfig(num_buckets=11, loss_type="soft_ce"))
        logits = torch.randn(4, 11) * 100
        targets = torch.randint(0, 11, (4,))

        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_focal_with_extreme_logits(self) -> None:
        """Test focal loss with very large logits."""
        loss_fn = FocalLoss(LossConfig(num_buckets=11, loss_type="focal"))
        logits = torch.randn(4, 11) * 100
        targets = torch.randint(0, 11, (4,))

        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_focal_with_high_confidence(self) -> None:
        """Test focal loss when prediction is very confident."""
        loss_fn = FocalLoss(LossConfig(num_buckets=11, loss_type="focal"))
        # Very confident correct prediction
        logits = torch.zeros(1, 11)
        logits[0, 5] = 50.0  # Very confident
        targets = torch.tensor([5])

        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)
        assert loss.item() >= 0
