"""Tests for the data_loader module."""

import numpy as np
import pytest
import torch

from models.training.data_loader import (
    DataLoaderBundle,
    DataLoaderConfig,
    PriceDataset,
    RollingWindowDataset,
    SequentialBatchSampler,
    SplitConfig,
    SplitIndices,
    TemporalSampler,
    collate_price_data,
    compute_split_indices,
    create_data_loaders,
    create_rolling_windows,
    get_split_name,
)


class TestSplitConfig:
    """Tests for SplitConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SplitConfig()
        assert config.train_ratio == 0.70
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15
        assert config.gap_samples == 0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SplitConfig(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            gap_samples=100,
        )
        assert config.train_ratio == 0.8
        assert config.val_ratio == 0.1
        assert config.test_ratio == 0.1
        assert config.gap_samples == 100

    def test_train_ratio_too_small(self) -> None:
        """Test that train_ratio must be positive."""
        with pytest.raises(ValueError, match="train_ratio must be in"):
            SplitConfig(train_ratio=0.0, val_ratio=0.5, test_ratio=0.5)

    def test_train_ratio_too_large(self) -> None:
        """Test that train_ratio must be less than 1."""
        with pytest.raises(ValueError, match="train_ratio must be in"):
            SplitConfig(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

    def test_val_ratio_negative(self) -> None:
        """Test that val_ratio must be non-negative."""
        with pytest.raises(ValueError, match="val_ratio must be in"):
            SplitConfig(train_ratio=0.8, val_ratio=-0.1, test_ratio=0.3)

    def test_test_ratio_negative(self) -> None:
        """Test that test_ratio must be non-negative."""
        with pytest.raises(ValueError, match="test_ratio must be in"):
            SplitConfig(train_ratio=0.8, val_ratio=0.3, test_ratio=-0.1)

    def test_ratios_must_sum_to_one(self) -> None:
        """Test that ratios must sum to 1.0."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_negative_gap_samples(self) -> None:
        """Test that gap_samples must be non-negative."""
        with pytest.raises(ValueError, match="gap_samples must be non-negative"):
            SplitConfig(gap_samples=-1)

    def test_no_validation_split(self) -> None:
        """Test configuration with no validation split."""
        config = SplitConfig(train_ratio=0.8, val_ratio=0.0, test_ratio=0.2)
        assert config.val_ratio == 0.0

    def test_no_test_split(self) -> None:
        """Test configuration with no test split."""
        config = SplitConfig(train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
        assert config.test_ratio == 0.0


class TestSplitIndices:
    """Tests for SplitIndices dataclass."""

    def test_split_sizes(self) -> None:
        """Test split size calculations."""
        splits = SplitIndices(
            train_start=0,
            train_end=700,
            val_start=700,
            val_end=850,
            test_start=850,
            test_end=1000,
        )
        assert splits.train_size == 700
        assert splits.val_size == 150
        assert splits.test_size == 150

    def test_empty_splits(self) -> None:
        """Test split sizes when some splits are empty."""
        splits = SplitIndices(
            train_start=0,
            train_end=800,
            val_start=800,
            val_end=800,  # Empty validation
            test_start=800,
            test_end=1000,
        )
        assert splits.train_size == 800
        assert splits.val_size == 0
        assert splits.test_size == 200


class TestComputeSplitIndices:
    """Tests for compute_split_indices function."""

    def test_basic_split(self) -> None:
        """Test basic 70/15/15 split."""
        config = SplitConfig()
        splits = compute_split_indices(1000, config)

        assert splits.train_start == 0
        assert splits.train_size == 700
        assert splits.val_size == 150
        assert splits.test_size == 150

    def test_split_with_gap(self) -> None:
        """Test split with gaps between splits."""
        config = SplitConfig(gap_samples=50)
        splits = compute_split_indices(1000, config)

        # Available = 1000 - 2*50 = 900
        # Train = 630, Val = 135, Test = 135
        assert splits.train_size == 630
        assert splits.val_start == splits.train_end + 50
        assert splits.test_start == splits.val_end + 50

    def test_insufficient_samples_for_gaps(self) -> None:
        """Test error when not enough samples for gaps."""
        config = SplitConfig(gap_samples=500)
        with pytest.raises(ValueError, match="Not enough samples"):
            compute_split_indices(1000, config)

    def test_minimum_samples(self) -> None:
        """Test with minimum viable sample count."""
        config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        splits = compute_split_indices(10, config)
        assert splits.train_size >= 1

    def test_custom_ratios(self) -> None:
        """Test with custom split ratios."""
        config = SplitConfig(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
        splits = compute_split_indices(1000, config)

        assert splits.train_size == 500
        assert splits.val_size == 250
        assert splits.test_size == 250


class TestDataLoaderConfig:
    """Tests for DataLoaderConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DataLoaderConfig()
        assert config.sequence_length == 256
        assert config.horizons == (1, 5, 10, 30, 60, 120, 300, 600)
        assert config.batch_size == 32
        assert config.drop_last is False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = DataLoaderConfig(
            sequence_length=128,
            horizons=(1, 5, 10),
            batch_size=64,
            drop_last=True,
        )
        assert config.sequence_length == 128
        assert config.horizons == (1, 5, 10)
        assert config.batch_size == 64
        assert config.drop_last is True

    def test_sequence_length_too_small(self) -> None:
        """Test that sequence_length must be positive."""
        with pytest.raises(ValueError, match="sequence_length must be at least 1"):
            DataLoaderConfig(sequence_length=0)

    def test_empty_horizons(self) -> None:
        """Test that horizons must have at least one element."""
        with pytest.raises(ValueError, match="horizons must have at least one"):
            DataLoaderConfig(horizons=())

    def test_negative_horizon(self) -> None:
        """Test that all horizons must be positive."""
        with pytest.raises(ValueError, match="All horizons must be positive"):
            DataLoaderConfig(horizons=(1, -5, 10))

    def test_zero_horizon(self) -> None:
        """Test that zero is not a valid horizon."""
        with pytest.raises(ValueError, match="All horizons must be positive"):
            DataLoaderConfig(horizons=(0, 5, 10))

    def test_batch_size_too_small(self) -> None:
        """Test that batch_size must be positive."""
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            DataLoaderConfig(batch_size=0)


class TestPriceDataset:
    """Tests for PriceDataset class."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray, DataLoaderConfig]:
        """Create sample data for testing."""
        num_timesteps = 500
        num_features = 16
        horizons = (1, 5, 10)
        num_horizons = len(horizons)

        features = np.random.randn(num_timesteps, num_features).astype(np.float64)
        targets = np.random.randint(0, 101, (num_timesteps, num_horizons)).astype(
            np.intp
        )

        config = DataLoaderConfig(
            sequence_length=50,
            horizons=horizons,
            batch_size=8,
        )

        return features, targets, config

    def test_dataset_creation(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test dataset creation with valid data."""
        features, targets, config = sample_data
        dataset = PriceDataset(features, targets, config)

        # Check valid indices are computed correctly
        max_horizon = max(config.horizons)
        expected_len = features.shape[0] - config.sequence_length - max_horizon + 1
        assert len(dataset) == expected_len

    def test_getitem_returns_correct_shapes(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test that __getitem__ returns correctly shaped tensors."""
        features, targets, config = sample_data
        dataset = PriceDataset(features, targets, config)

        seq_features, targets_dict = dataset[0]

        assert seq_features.shape == (config.sequence_length, features.shape[1])
        assert len(targets_dict) == len(config.horizons)
        for h in config.horizons:
            assert h in targets_dict
            assert targets_dict[h].shape == ()  # Scalar

    def test_getitem_different_indices(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test that different indices return different sequences."""
        features, targets, config = sample_data
        dataset = PriceDataset(features, targets, config)

        seq0, _ = dataset[0]
        seq1, _ = dataset[1]

        # Sequences should be different (shifted by 1)
        assert not torch.allclose(seq0, seq1)

    def test_features_must_be_2d(self) -> None:
        """Test that features must be 2D."""
        features = np.random.randn(100).astype(np.float64)  # 1D
        targets = np.random.randint(0, 101, (100, 3)).astype(np.intp)
        config = DataLoaderConfig(sequence_length=10, horizons=(1, 5, 10))

        with pytest.raises(ValueError, match="features must be 2D"):
            PriceDataset(features, targets, config)

    def test_targets_must_be_2d(self) -> None:
        """Test that targets must be 2D."""
        features = np.random.randn(100, 16).astype(np.float64)
        targets = np.random.randint(0, 101, (100,)).astype(np.intp)  # 1D
        config = DataLoaderConfig(sequence_length=10, horizons=(1, 5, 10))

        with pytest.raises(ValueError, match="targets must be 2D"):
            PriceDataset(features, targets, config)

    def test_shape_mismatch(self) -> None:
        """Test that features and targets must have same length."""
        features = np.random.randn(100, 16).astype(np.float64)
        targets = np.random.randint(0, 101, (90, 3)).astype(np.intp)
        config = DataLoaderConfig(sequence_length=10, horizons=(1, 5, 10))

        with pytest.raises(ValueError, match="same length"):
            PriceDataset(features, targets, config)

    def test_wrong_number_of_horizons(self) -> None:
        """Test that targets must have correct number of horizons."""
        features = np.random.randn(100, 16).astype(np.float64)
        targets = np.random.randint(0, 101, (100, 2)).astype(np.intp)  # 2 not 3
        config = DataLoaderConfig(sequence_length=10, horizons=(1, 5, 10))

        with pytest.raises(ValueError, match="3 horizons"):
            PriceDataset(features, targets, config)

    def test_insufficient_timesteps(self) -> None:
        """Test that there must be enough timesteps."""
        features = np.random.randn(20, 16).astype(np.float64)
        targets = np.random.randint(0, 101, (20, 3)).astype(np.intp)
        config = DataLoaderConfig(
            sequence_length=15,
            horizons=(1, 5, 10),  # max=10
        )
        # Need 15 + 10 = 25 timesteps minimum

        with pytest.raises(ValueError, match="Not enough timesteps"):
            PriceDataset(features, targets, config)


class TestTemporalSampler:
    """Tests for TemporalSampler class."""

    def test_yields_sequential_indices(self) -> None:
        """Test that sampler yields indices in order."""
        sampler = TemporalSampler(10)
        indices = list(sampler)
        assert indices == list(range(10))

    def test_length(self) -> None:
        """Test that sampler reports correct length."""
        sampler = TemporalSampler(100)
        assert len(sampler) == 100

    def test_iteration_multiple_times(self) -> None:
        """Test that sampler can be iterated multiple times."""
        sampler = TemporalSampler(5)
        first = list(sampler)
        second = list(sampler)
        assert first == second == [0, 1, 2, 3, 4]


class TestSequentialBatchSampler:
    """Tests for SequentialBatchSampler class."""

    def test_yields_contiguous_batches(self) -> None:
        """Test that batches contain contiguous indices."""
        sampler = SequentialBatchSampler(10, batch_size=3)
        batches = list(sampler)

        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]  # Last incomplete batch

    def test_drop_last(self) -> None:
        """Test that drop_last=True drops incomplete batch."""
        sampler = SequentialBatchSampler(10, batch_size=3, drop_last=True)
        batches = list(sampler)

        assert len(batches) == 3
        assert batches[-1] == [6, 7, 8]

    def test_exact_batches(self) -> None:
        """Test when dataset size is exact multiple of batch size."""
        sampler = SequentialBatchSampler(9, batch_size=3)
        batches = list(sampler)

        assert len(batches) == 3
        assert all(len(b) == 3 for b in batches)

    def test_length_with_remainder(self) -> None:
        """Test length calculation with remainder."""
        sampler = SequentialBatchSampler(10, batch_size=3)
        assert len(sampler) == 4

    def test_length_drop_last(self) -> None:
        """Test length calculation with drop_last."""
        sampler = SequentialBatchSampler(10, batch_size=3, drop_last=True)
        assert len(sampler) == 3


class TestCollateFunction:
    """Tests for collate_price_data function."""

    def test_collate_batches_correctly(self) -> None:
        """Test that collate stacks features and collates targets."""
        batch = [
            (
                torch.randn(10, 4),
                {1: torch.tensor(5), 5: torch.tensor(10)},
            ),
            (
                torch.randn(10, 4),
                {1: torch.tensor(6), 5: torch.tensor(11)},
            ),
        ]

        features, targets = collate_price_data(batch)

        assert features.shape == (2, 10, 4)
        assert targets[1].shape == (2,)
        assert targets[5].shape == (2,)
        assert targets[1][0] == 5
        assert targets[1][1] == 6


class TestCreateDataLoaders:
    """Tests for create_data_loaders function."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample data for testing."""
        num_timesteps = 1000
        num_features = 16
        num_horizons = 3

        features = np.random.randn(num_timesteps, num_features).astype(np.float64)
        targets = np.random.randint(0, 101, (num_timesteps, num_horizons)).astype(
            np.intp
        )

        return features, targets

    def test_creates_all_loaders(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that all data loaders are created."""
        features, targets = sample_data
        config = DataLoaderConfig(
            sequence_length=50,
            horizons=(1, 5, 10),
            batch_size=16,
        )

        bundle = create_data_loaders(features, targets, config)

        assert bundle.train_loader is not None
        assert bundle.val_loader is not None
        assert bundle.test_loader is not None

    def test_returns_split_indices(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that split indices are included in bundle."""
        features, targets = sample_data
        config = DataLoaderConfig(
            sequence_length=50,
            horizons=(1, 5, 10),
        )

        bundle = create_data_loaders(features, targets, config)

        assert bundle.split_indices.train_size > 0
        assert bundle.split_indices.val_size > 0
        assert bundle.split_indices.test_size > 0

    def test_loaders_yield_correct_shapes(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that loaders yield correctly shaped batches."""
        features, targets = sample_data
        config = DataLoaderConfig(
            sequence_length=50,
            horizons=(1, 5, 10),
            batch_size=16,
        )

        bundle = create_data_loaders(features, targets, config)

        # Get one batch
        batch_features, batch_targets = next(iter(bundle.train_loader))

        assert batch_features.shape[0] == 16  # batch size
        assert batch_features.shape[1] == 50  # sequence length
        assert batch_features.shape[2] == features.shape[1]  # num features
        assert 1 in batch_targets and 5 in batch_targets and 10 in batch_targets

    def test_no_validation_split(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test with no validation split."""
        features, targets = sample_data
        loader_config = DataLoaderConfig(
            sequence_length=50,
            horizons=(1, 5, 10),
        )
        split_config = SplitConfig(
            train_ratio=0.8,
            val_ratio=0.0,
            test_ratio=0.2,
        )

        bundle = create_data_loaders(
            features, targets, loader_config, split_config
        )

        assert bundle.val_loader is None
        assert bundle.val_dataset is None

    def test_custom_split_config(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test with custom split configuration."""
        features, targets = sample_data
        loader_config = DataLoaderConfig(
            sequence_length=50,
            horizons=(1, 5, 10),
        )
        split_config = SplitConfig(
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            gap_samples=10,
        )

        bundle = create_data_loaders(
            features, targets, loader_config, split_config
        )

        # Verify gaps are applied
        splits = bundle.split_indices
        assert splits.val_start == splits.train_end + 10
        assert splits.test_start == splits.val_end + 10


class TestRollingWindowDataset:
    """Tests for RollingWindowDataset class."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray, DataLoaderConfig]:
        """Create sample data for testing."""
        num_timesteps = 500
        num_features = 16
        horizons = (1, 5, 10)
        num_horizons = len(horizons)

        features = np.random.randn(num_timesteps, num_features).astype(np.float64)
        targets = np.random.randint(0, 101, (num_timesteps, num_horizons)).astype(
            np.intp
        )

        config = DataLoaderConfig(
            sequence_length=50,
            horizons=horizons,
            batch_size=8,
        )

        return features, targets, config

    def test_window_creation(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test creating a rolling window dataset."""
        features, targets, config = sample_data
        window = RollingWindowDataset(
            features, targets, window_start=100, window_end=300, config=config
        )

        assert window.window_start == 100
        assert window.window_end == 300
        assert len(window) > 0

    def test_window_getitem(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test getting items from rolling window."""
        features, targets, config = sample_data
        window = RollingWindowDataset(
            features, targets, window_start=0, window_end=200, config=config
        )

        seq_features, targets_dict = window[0]

        assert seq_features.shape == (config.sequence_length, features.shape[1])
        assert len(targets_dict) == len(config.horizons)

    def test_invalid_window_start(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test that negative window_start raises error."""
        features, targets, config = sample_data
        with pytest.raises(ValueError, match="window_start must be non-negative"):
            RollingWindowDataset(
                features, targets, window_start=-1, window_end=100, config=config
            )

    def test_invalid_window_order(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test that window_end must be greater than window_start."""
        features, targets, config = sample_data
        with pytest.raises(ValueError, match="window_end must be greater"):
            RollingWindowDataset(
                features, targets, window_start=100, window_end=50, config=config
            )

    def test_window_exceeds_data(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test that window cannot exceed data length."""
        features, targets, config = sample_data
        with pytest.raises(ValueError, match="window_end exceeds data length"):
            RollingWindowDataset(
                features, targets, window_start=0, window_end=1000, config=config
            )

    def test_window_too_small(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test that window must be large enough."""
        features, targets, config = sample_data
        with pytest.raises(ValueError, match="Window too small"):
            RollingWindowDataset(
                features, targets, window_start=0, window_end=50, config=config
            )


class TestCreateRollingWindows:
    """Tests for create_rolling_windows function."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray, DataLoaderConfig]:
        """Create sample data for testing."""
        num_timesteps = 1000
        num_features = 16
        horizons = (1, 5, 10)
        num_horizons = len(horizons)

        features = np.random.randn(num_timesteps, num_features).astype(np.float64)
        targets = np.random.randint(0, 101, (num_timesteps, num_horizons)).astype(
            np.intp
        )

        config = DataLoaderConfig(
            sequence_length=50,
            horizons=horizons,
        )

        return features, targets, config

    def test_creates_windows(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test that rolling windows are created."""
        features, targets, config = sample_data
        windows = create_rolling_windows(
            features, targets, config, window_size=200, step_size=100
        )

        assert len(windows) > 0
        assert all(isinstance(w, RollingWindowDataset) for w in windows)

    def test_window_count(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test that correct number of windows is created."""
        features, targets, config = sample_data
        windows = create_rolling_windows(
            features, targets, config, window_size=200, step_size=200
        )

        # 1000 timesteps, 200 window size, 200 step = 5 windows
        assert len(windows) == 5

    def test_window_step(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test that windows advance by step_size."""
        features, targets, config = sample_data
        windows = create_rolling_windows(
            features, targets, config, window_size=200, step_size=100
        )

        assert windows[0].window_start == 0
        assert windows[1].window_start == 100
        assert windows[2].window_start == 200

    def test_min_samples_filter(
        self, sample_data: tuple[np.ndarray, np.ndarray, DataLoaderConfig]
    ) -> None:
        """Test that windows with too few samples are filtered."""
        features, targets, config = sample_data
        # With very large min_samples, some windows might be filtered
        windows = create_rolling_windows(
            features, targets, config, window_size=200, step_size=100, min_samples=1000
        )

        # Windows have ~140 samples each, so with min_samples=1000, all filtered
        assert len(windows) == 0


class TestGetSplitName:
    """Tests for get_split_name function."""

    def test_train_name(self) -> None:
        """Test training split name."""
        assert get_split_name("train") == "Training"

    def test_val_name(self) -> None:
        """Test validation split name."""
        assert get_split_name("val") == "Validation"

    def test_test_name(self) -> None:
        """Test test split name."""
        assert get_split_name("test") == "Test"


class TestDataLoaderIntegration:
    """Integration tests for data loading pipeline."""

    def test_full_training_iteration(self) -> None:
        """Test iterating through all training batches."""
        num_timesteps = 500
        num_features = 8
        horizons = (1, 5)

        features = np.random.randn(num_timesteps, num_features).astype(np.float64)
        targets = np.random.randint(0, 101, (num_timesteps, 2)).astype(np.intp)

        config = DataLoaderConfig(
            sequence_length=20,
            horizons=horizons,
            batch_size=8,
        )
        split_config = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        bundle = create_data_loaders(features, targets, config, split_config)

        batch_count = 0
        for batch_features, batch_targets in bundle.train_loader:
            assert batch_features.dim() == 3
            assert 1 in batch_targets and 5 in batch_targets
            batch_count += 1

        assert batch_count > 0

    def test_dtype_preservation(self) -> None:
        """Test that dtypes are preserved correctly."""
        features = np.random.randn(200, 8).astype(np.float64)
        targets = np.random.randint(0, 101, (200, 2)).astype(np.intp)

        config = DataLoaderConfig(
            sequence_length=20,
            horizons=(1, 5),
            batch_size=8,
        )

        bundle = create_data_loaders(features, targets, config)

        batch_features, batch_targets = next(iter(bundle.train_loader))

        assert batch_features.dtype == torch.float32
        assert batch_targets[1].dtype == torch.long

    def test_temporal_order_preserved(self) -> None:
        """Test that temporal order is preserved in batches."""
        # Create features with increasing values to track order
        num_timesteps = 100
        features = np.arange(num_timesteps).reshape(-1, 1).astype(np.float64)
        targets = np.zeros((num_timesteps, 1), dtype=np.intp)

        config = DataLoaderConfig(
            sequence_length=10,
            horizons=(1,),
            batch_size=4,
        )

        bundle = create_data_loaders(features, targets, config)

        prev_start = -1
        for batch_features, _ in bundle.train_loader:
            # First element of first sequence in batch
            first_val = batch_features[0, 0, 0].item()
            assert first_val > prev_start
            prev_start = first_val
