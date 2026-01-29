"""Temporal data loading for training price prediction models.

This module implements data loading with proper temporal separation:
- Chronological train/validation/test splits (no shuffling)
- Gap between splits to prevent lookahead bias
- Rolling window support for time series sequences
- Batch construction for multi-horizon predictions
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for temporal data splits.

    Attributes:
        train_ratio: Fraction of data for training (default 0.70).
        val_ratio: Fraction of data for validation (default 0.15).
        test_ratio: Fraction of data for testing (default 0.15).
        gap_samples: Number of samples to skip between splits for lookahead prevention.
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    gap_samples: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.train_ratio <= 0 or self.train_ratio >= 1:
            raise ValueError("train_ratio must be in (0, 1)")
        if self.val_ratio < 0 or self.val_ratio >= 1:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.test_ratio < 0 or self.test_ratio >= 1:
            raise ValueError("test_ratio must be in [0, 1)")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f}"
            )
        if self.gap_samples < 0:
            raise ValueError("gap_samples must be non-negative")


@dataclass
class SplitIndices:
    """Indices for train/validation/test splits.

    Attributes:
        train_start: Start index of training data (inclusive).
        train_end: End index of training data (exclusive).
        val_start: Start index of validation data (inclusive).
        val_end: End index of validation data (exclusive).
        test_start: Start index of test data (inclusive).
        test_end: End index of test data (exclusive).
    """

    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        """Number of samples in training split."""
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        """Number of samples in validation split."""
        return self.val_end - self.val_start

    @property
    def test_size(self) -> int:
        """Number of samples in test split."""
        return self.test_end - self.test_start


def compute_split_indices(
    num_samples: int,
    config: SplitConfig,
) -> SplitIndices:
    """Compute indices for chronological train/val/test splits.

    Splits are strictly chronological with optional gaps between them
    to prevent information leakage.

    Args:
        num_samples: Total number of samples in the dataset.
        config: Split configuration.

    Returns:
        SplitIndices with start/end indices for each split.

    Raises:
        ValueError: If num_samples is too small for the splits and gaps.
    """
    # Compute raw split sizes (before gaps)
    total_gap = 2 * config.gap_samples  # Gap after train and after val
    available = num_samples - total_gap

    if available <= 0:
        raise ValueError(
            f"Not enough samples ({num_samples}) for gaps "
            f"({total_gap} total gap samples)"
        )

    # Allocate samples proportionally
    train_size = int(available * config.train_ratio)
    val_size = int(available * config.val_ratio)
    test_size = available - train_size - val_size  # Remainder to test

    if train_size < 1:
        raise ValueError("Not enough samples for training split")

    # Compute indices
    train_start = 0
    train_end = train_size
    val_start = train_end + config.gap_samples
    val_end = val_start + val_size
    test_start = val_end + config.gap_samples
    test_end = test_start + test_size

    return SplitIndices(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )


@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration for temporal data loader.

    Attributes:
        sequence_length: Number of time steps in each input sequence.
        horizons: Prediction horizons in time steps (must match model).
        batch_size: Number of sequences per batch.
        drop_last: Whether to drop the last incomplete batch.
    """

    sequence_length: int = 256
    horizons: tuple[int, ...] = (1, 5, 10, 30, 60, 120, 300, 600)
    batch_size: int = 32
    drop_last: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if len(self.horizons) < 1:
            raise ValueError("horizons must have at least one element")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("All horizons must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


class PriceDataset(Dataset[tuple[Tensor, dict[int, Tensor]]]):
    """Dataset for price prediction with multi-horizon targets.

    This dataset constructs sequences from feature arrays and target buckets,
    handling temporal alignment for multiple prediction horizons.

    Attributes:
        features: Feature array of shape (num_timesteps, num_features).
        targets: Target bucket indices of shape (num_timesteps, num_horizons).
        sequence_length: Number of time steps per input sequence.
        horizons: Prediction horizon time steps.
        valid_indices: Indices where complete sequences and targets exist.
    """

    def __init__(
        self,
        features: NDArray[np.floating[Any]],
        targets: NDArray[np.intp],
        config: DataLoaderConfig,
    ) -> None:
        """Initialize the dataset.

        Args:
            features: Feature array of shape (num_timesteps, num_features).
            targets: Target bucket indices of shape (num_timesteps, num_horizons).
                targets[t, h] is the bucket index for horizon h at time t.
            config: Data loader configuration.

        Raises:
            ValueError: If shapes are invalid or insufficient data.
        """
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}")
        if targets.ndim != 2:
            raise ValueError(f"targets must be 2D, got shape {targets.shape}")
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"features and targets must have same length, "
                f"got {features.shape[0]} and {targets.shape[0]}"
            )
        if targets.shape[1] != len(config.horizons):
            raise ValueError(
                f"targets must have {len(config.horizons)} horizons, "
                f"got {targets.shape[1]}"
            )

        self.features = features
        self.targets = targets
        self.sequence_length = config.sequence_length
        self.horizons = config.horizons
        self.config = config

        # Compute valid starting indices for sequences
        # A sequence starting at index i uses features[i:i+seq_len]
        # and needs targets at time i+seq_len-1 (last time step of sequence)
        # The target at that time must have valid future values for all horizons
        max_horizon = max(self.horizons)
        num_timesteps = features.shape[0]

        # Last valid starting index: need seq_len for features + max_horizon for targets
        last_valid = num_timesteps - self.sequence_length - max_horizon

        if last_valid < 0:
            raise ValueError(
                f"Not enough timesteps ({num_timesteps}) for "
                f"sequence_length ({self.sequence_length}) and "
                f"max_horizon ({max_horizon})"
            )

        self.valid_indices = np.arange(last_valid + 1)

    def __len__(self) -> int:
        """Return the number of valid sequences."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[int, Tensor]]:
        """Get a sequence and its targets.

        Args:
            idx: Index into valid_indices.

        Returns:
            Tuple of:
                - features: Tensor of shape (sequence_length, num_features)
                - targets: Dict mapping horizon -> target bucket index (scalar)
        """
        start = self.valid_indices[idx]
        end = start + self.sequence_length

        # Get feature sequence
        seq_features = torch.from_numpy(
            self.features[start:end].astype(np.float32)
        )

        # Get targets at the end of the sequence
        target_time = end - 1
        targets_dict: dict[int, Tensor] = {}
        for i, horizon in enumerate(self.horizons):
            # Target is the bucket at time (target_time + horizon)
            # which is stored in targets[target_time + horizon, i]
            target_idx = target_time + horizon
            bucket = self.targets[target_idx, i]
            targets_dict[horizon] = torch.tensor(bucket, dtype=torch.long)

        return seq_features, targets_dict


class TemporalSampler(Sampler[int]):
    """Sampler that yields indices in temporal order (no shuffling).

    For time series data, shuffling would break temporal dependencies
    and potentially leak future information. This sampler yields indices
    in strict chronological order.

    Attributes:
        dataset_size: Number of samples in the dataset.
    """

    def __init__(self, dataset_size: int) -> None:
        """Initialize the sampler.

        Args:
            dataset_size: Number of samples in the dataset.
        """
        self.dataset_size = dataset_size

    def __iter__(self) -> Iterator[int]:
        """Iterate over indices in temporal order."""
        return iter(range(self.dataset_size))

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.dataset_size


class SequentialBatchSampler(Sampler[list[int]]):
    """Batch sampler that yields contiguous batches in temporal order.

    This sampler creates batches of consecutive indices, maintaining
    temporal locality within each batch.

    Attributes:
        dataset_size: Number of samples in the dataset.
        batch_size: Number of samples per batch.
        drop_last: Whether to drop the last incomplete batch.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        """Initialize the batch sampler.

        Args:
            dataset_size: Number of samples in the dataset.
            batch_size: Number of samples per batch.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batches of indices."""
        batch: list[int] = []
        for idx in range(self.dataset_size):
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size


def collate_price_data(
    batch: list[tuple[Tensor, dict[int, Tensor]]],
) -> tuple[Tensor, dict[int, Tensor]]:
    """Collate function for PriceDataset.

    Stacks features and collects targets per horizon.

    Args:
        batch: List of (features, targets_dict) tuples from PriceDataset.

    Returns:
        Tuple of:
            - features: Tensor of shape (batch_size, sequence_length, num_features)
            - targets: Dict mapping horizon -> Tensor of shape (batch_size,)
    """
    features_list = [item[0] for item in batch]
    targets_dicts = [item[1] for item in batch]

    # Stack features
    features = torch.stack(features_list, dim=0)

    # Collate targets per horizon
    horizons = targets_dicts[0].keys()
    targets: dict[int, Tensor] = {}
    for h in horizons:
        targets[h] = torch.stack([d[h] for d in targets_dicts], dim=0)

    return features, targets


@dataclass
class DataLoaderBundle:
    """Bundle of data loaders for train/val/test splits.

    Attributes:
        train_dataset: Training dataset.
        val_dataset: Validation dataset (may be None if val_ratio=0).
        test_dataset: Test dataset (may be None if test_ratio=0).
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation (may be None).
        test_loader: DataLoader for testing (may be None).
        split_indices: The split indices used.
    """

    train_dataset: PriceDataset
    val_dataset: PriceDataset | None
    test_dataset: PriceDataset | None
    train_loader: torch.utils.data.DataLoader[tuple[Tensor, dict[int, Tensor]]]
    val_loader: torch.utils.data.DataLoader[tuple[Tensor, dict[int, Tensor]]] | None
    test_loader: torch.utils.data.DataLoader[tuple[Tensor, dict[int, Tensor]]] | None
    split_indices: SplitIndices


def create_data_loaders(
    features: NDArray[np.floating[Any]],
    targets: NDArray[np.intp],
    loader_config: DataLoaderConfig,
    split_config: SplitConfig | None = None,
    num_workers: int = 0,
) -> DataLoaderBundle:
    """Create data loaders with temporal train/val/test splits.

    Args:
        features: Feature array of shape (num_timesteps, num_features).
        targets: Target bucket indices of shape (num_timesteps, num_horizons).
        loader_config: Data loader configuration.
        split_config: Split configuration. Uses defaults if None.
        num_workers: Number of worker processes for data loading.

    Returns:
        DataLoaderBundle with data loaders for each split.
    """
    split_config = split_config or SplitConfig()

    # Compute split indices
    num_samples = features.shape[0]
    splits = compute_split_indices(num_samples, split_config)

    # Create datasets for each split
    train_features = features[splits.train_start:splits.train_end]
    train_targets = targets[splits.train_start:splits.train_end]
    train_dataset = PriceDataset(train_features, train_targets, loader_config)

    val_dataset: PriceDataset | None = None
    test_dataset: PriceDataset | None = None

    if splits.val_size > 0:
        val_features = features[splits.val_start:splits.val_end]
        val_targets = targets[splits.val_start:splits.val_end]
        val_dataset = PriceDataset(val_features, val_targets, loader_config)

    if splits.test_size > 0:
        test_features = features[splits.test_start:splits.test_end]
        test_targets = targets[splits.test_start:splits.test_end]
        test_dataset = PriceDataset(test_features, test_targets, loader_config)

    # Create data loaders
    train_sampler = SequentialBatchSampler(
        len(train_dataset),
        loader_config.batch_size,
        drop_last=loader_config.drop_last,
    )
    train_loader: torch.utils.data.DataLoader[tuple[Tensor, dict[int, Tensor]]] = (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_price_data,
            num_workers=num_workers,
        )
    )

    val_loader: torch.utils.data.DataLoader[tuple[Tensor, dict[int, Tensor]]] | None = None
    if val_dataset is not None:
        val_sampler = SequentialBatchSampler(
            len(val_dataset),
            loader_config.batch_size,
            drop_last=False,  # Never drop last for validation
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=collate_price_data,
            num_workers=num_workers,
        )

    test_loader: torch.utils.data.DataLoader[tuple[Tensor, dict[int, Tensor]]] | None = None
    if test_dataset is not None:
        test_sampler = SequentialBatchSampler(
            len(test_dataset),
            loader_config.batch_size,
            drop_last=False,  # Never drop last for testing
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=collate_price_data,
            num_workers=num_workers,
        )

    return DataLoaderBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        split_indices=splits,
    )


class RollingWindowDataset(Dataset[tuple[Tensor, dict[int, Tensor]]]):
    """Dataset for rolling window validation.

    This dataset supports rolling window cross-validation where the training
    window advances through time, useful for evaluating model stability.

    Attributes:
        features: Full feature array.
        targets: Full target array.
        window_start: Start index of the current window.
        window_end: End index of the current window.
        config: Data loader configuration.
    """

    def __init__(
        self,
        features: NDArray[np.floating[Any]],
        targets: NDArray[np.intp],
        window_start: int,
        window_end: int,
        config: DataLoaderConfig,
    ) -> None:
        """Initialize rolling window dataset.

        Args:
            features: Full feature array of shape (num_timesteps, num_features).
            targets: Full target array of shape (num_timesteps, num_horizons).
            window_start: Start index of the window (inclusive).
            window_end: End index of the window (exclusive).
            config: Data loader configuration.
        """
        if window_start < 0:
            raise ValueError("window_start must be non-negative")
        if window_end <= window_start:
            raise ValueError("window_end must be greater than window_start")
        if window_end > features.shape[0]:
            raise ValueError("window_end exceeds data length")

        # Extract window data
        self.features = features[window_start:window_end]
        self.targets = targets[window_start:window_end]
        self.window_start = window_start
        self.window_end = window_end
        self.config = config
        self.sequence_length = config.sequence_length
        self.horizons = config.horizons

        # Compute valid indices within the window
        max_horizon = max(self.horizons)
        window_length = window_end - window_start
        last_valid = window_length - self.sequence_length - max_horizon

        if last_valid < 0:
            raise ValueError(
                f"Window too small ({window_length}) for "
                f"sequence_length ({self.sequence_length}) and "
                f"max_horizon ({max_horizon})"
            )

        self.valid_indices = np.arange(last_valid + 1)

    def __len__(self) -> int:
        """Return the number of valid sequences in the window."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[int, Tensor]]:
        """Get a sequence and its targets from the window.

        Args:
            idx: Index into valid_indices.

        Returns:
            Tuple of (features, targets_dict).
        """
        start = self.valid_indices[idx]
        end = start + self.sequence_length

        seq_features = torch.from_numpy(
            self.features[start:end].astype(np.float32)
        )

        target_time = end - 1
        targets_dict: dict[int, Tensor] = {}
        for i, horizon in enumerate(self.horizons):
            target_idx = target_time + horizon
            bucket = self.targets[target_idx, i]
            targets_dict[horizon] = torch.tensor(bucket, dtype=torch.long)

        return seq_features, targets_dict


def create_rolling_windows(
    features: NDArray[np.floating[Any]],
    targets: NDArray[np.intp],
    config: DataLoaderConfig,
    window_size: int,
    step_size: int,
    min_samples: int = 1,
) -> list[RollingWindowDataset]:
    """Create rolling window datasets for cross-validation.

    Args:
        features: Feature array of shape (num_timesteps, num_features).
        targets: Target array of shape (num_timesteps, num_horizons).
        config: Data loader configuration.
        window_size: Size of each training window in timesteps.
        step_size: Number of timesteps to advance between windows.
        min_samples: Minimum number of valid samples required per window.

    Returns:
        List of RollingWindowDataset instances.
    """
    num_timesteps = features.shape[0]
    windows: list[RollingWindowDataset] = []

    start = 0
    while start + window_size <= num_timesteps:
        end = start + window_size
        try:
            window = RollingWindowDataset(
                features, targets, start, end, config
            )
            if len(window) >= min_samples:
                windows.append(window)
        except ValueError:
            # Window too small for sequence + horizon, skip
            pass
        start += step_size

    return windows


def get_split_name(split: Literal["train", "val", "test"]) -> str:
    """Get display name for a split.

    Args:
        split: Split identifier.

    Returns:
        Human-readable split name.
    """
    names = {"train": "Training", "val": "Validation", "test": "Test"}
    return names[split]
