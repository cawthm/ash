"""Tests for trainer.py - training loop and infrastructure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation.metrics import MetricsConfig
from models.architectures.losses import LossConfig, MultiHorizonLoss
from models.architectures.price_transformer import PriceTransformer, TransformerConfig
from models.training.data_loader import (
    DataLoaderConfig,
    PriceDataset,
    SequentialBatchSampler,
    collate_price_data,
)
from models.training.trainer import (
    CheckpointInfo,
    CosineWarmupScheduler,
    Trainer,
    TrainerConfig,
    TrainingState,
    create_trainer,
)

# Fixtures


@pytest.fixture
def device() -> torch.device:
    """Fixture for device."""
    return torch.device("cpu")


@pytest.fixture
def horizons() -> tuple[int, ...]:
    """Fixture for prediction horizons."""
    return (1, 5, 10)


@pytest.fixture
def num_buckets() -> int:
    """Fixture for number of buckets."""
    return 101


@pytest.fixture
def simple_model(horizons: tuple[int, ...], num_buckets: int) -> nn.Module:
    """Fixture for simple test model."""
    config = TransformerConfig(
        input_dim=16,
        embedding_dim=32,
        num_layers=1,
        num_heads=2,
        ff_dim=64,
        max_seq_len=10,
        horizons=horizons,
        num_buckets=num_buckets,
    )
    return PriceTransformer(config)


@pytest.fixture
def loss_fn(num_buckets: int, horizons: tuple[int, ...]) -> nn.Module:
    """Fixture for loss function."""
    config = LossConfig(num_buckets=num_buckets, loss_type="emd")
    return MultiHorizonLoss(config, num_horizons=len(horizons))


@pytest.fixture
def sample_data(
    horizons: tuple[int, ...], num_buckets: int
) -> tuple[NDArray[np.floating[Any]], NDArray[np.intp]]:
    """Fixture for sample training data."""
    num_timesteps = 200
    num_features = 16
    num_horizons = len(horizons)

    features = np.random.randn(num_timesteps, num_features).astype(np.float32)
    targets = np.random.randint(0, num_buckets, size=(num_timesteps, num_horizons))

    return features, targets


@pytest.fixture
def train_loader(
    sample_data: tuple[NDArray[np.floating[Any]], NDArray[np.intp]],
    horizons: tuple[int, ...],
) -> DataLoader[tuple[Tensor, dict[int, Tensor]]]:
    """Fixture for training data loader."""
    features, targets = sample_data
    config = DataLoaderConfig(
        sequence_length=10, horizons=horizons, batch_size=4, drop_last=False
    )
    dataset = PriceDataset(features[:100], targets[:100], config)
    sampler = SequentialBatchSampler(len(dataset), config.batch_size, drop_last=False)
    return DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_price_data
    )


@pytest.fixture
def val_loader(
    sample_data: tuple[NDArray[np.floating[Any]], NDArray[np.intp]],
    horizons: tuple[int, ...],
) -> DataLoader[tuple[Tensor, dict[int, Tensor]]]:
    """Fixture for validation data loader."""
    features, targets = sample_data
    config = DataLoaderConfig(
        sequence_length=10, horizons=horizons, batch_size=4, drop_last=False
    )
    dataset = PriceDataset(features[100:], targets[100:], config)
    sampler = SequentialBatchSampler(len(dataset), config.batch_size, drop_last=False)
    return DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_price_data
    )


# TrainerConfig tests


def test_trainer_config_defaults() -> None:
    """Test TrainerConfig default values."""
    config = TrainerConfig()
    assert config.learning_rate == 0.0001
    assert config.weight_decay == 0.01
    assert config.warmup_steps == 1000
    assert config.max_grad_norm == 1.0
    assert config.max_epochs == 100
    assert config.patience == 10
    assert config.fp16 is True
    assert config.checkpoint_dir == "checkpoints"
    assert config.save_every == 1
    assert config.keep_best == 3
    assert config.log_interval == 100
    assert config.val_interval == 1


def test_trainer_config_custom() -> None:
    """Test TrainerConfig with custom values."""
    config = TrainerConfig(
        learning_rate=0.001,
        weight_decay=0.001,
        warmup_steps=500,
        max_grad_norm=0.5,
        max_epochs=50,
        patience=5,
        fp16=False,
        checkpoint_dir="my_checkpoints",
        save_every=2,
        keep_best=5,
        log_interval=50,
        val_interval=2,
    )
    assert config.learning_rate == 0.001
    assert config.weight_decay == 0.001
    assert config.warmup_steps == 500
    assert config.max_grad_norm == 0.5
    assert config.max_epochs == 50
    assert config.patience == 5
    assert config.fp16 is False
    assert config.checkpoint_dir == "my_checkpoints"
    assert config.save_every == 2
    assert config.keep_best == 5
    assert config.log_interval == 50
    assert config.val_interval == 2


def test_trainer_config_validation() -> None:
    """Test TrainerConfig validation."""
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        TrainerConfig(learning_rate=0.0)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        TrainerConfig(learning_rate=-0.001)

    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        TrainerConfig(weight_decay=-0.01)

    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        TrainerConfig(warmup_steps=-100)

    with pytest.raises(ValueError, match="max_grad_norm must be non-negative"):
        TrainerConfig(max_grad_norm=-1.0)

    with pytest.raises(ValueError, match="max_epochs must be at least 1"):
        TrainerConfig(max_epochs=0)

    with pytest.raises(ValueError, match="patience must be at least 1"):
        TrainerConfig(patience=0)

    with pytest.raises(ValueError, match="save_every must be non-negative"):
        TrainerConfig(save_every=-1)

    with pytest.raises(ValueError, match="keep_best must be non-negative"):
        TrainerConfig(keep_best=-1)

    with pytest.raises(ValueError, match="log_interval must be at least 1"):
        TrainerConfig(log_interval=0)

    with pytest.raises(ValueError, match="val_interval must be at least 1"):
        TrainerConfig(val_interval=0)


# TrainingState tests


def test_training_state_defaults() -> None:
    """Test TrainingState initialization."""
    state = TrainingState()
    assert state.epoch == 0
    assert state.step == 0
    assert state.best_val_loss == float("inf")
    assert state.epochs_without_improvement == 0
    assert state.train_loss_history == []
    assert state.val_loss_history == []
    assert state.learning_rate_history == []


def test_training_state_mutable_fields() -> None:
    """Test TrainingState mutable fields."""
    state = TrainingState()
    state.train_loss_history.append(0.5)
    state.val_loss_history.append(0.6)
    state.learning_rate_history.append(0.0001)

    assert state.train_loss_history == [0.5]
    assert state.val_loss_history == [0.6]
    assert state.learning_rate_history == [0.0001]


# CosineWarmupScheduler tests


def test_cosine_warmup_scheduler_warmup() -> None:
    """Test learning rate during warmup phase."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=100, total_steps=1000, min_lr=0.00001
    )

    # At step 0 (initial, after __init__)
    # get_last_lr() returns the LR computed for the current last_epoch
    initial_lr = scheduler.get_last_lr()[0]
    assert initial_lr < 0.001  # Should be near 0 during warmup

    # During warmup
    scheduler.step()  # Step 1
    assert 0.0 < scheduler.get_last_lr()[0] < 0.001

    # At halfway through warmup
    for _ in range(49):
        scheduler.step()
    # Now at step 50
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0005, rel=5e-2)


def test_cosine_warmup_scheduler_decay() -> None:
    """Test learning rate during cosine decay phase."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=100, total_steps=1000, min_lr=0.00001
    )

    # Skip warmup
    for _ in range(100):
        scheduler.step()

    # At end of warmup, should be close to max LR
    lr_after_warmup = scheduler.get_last_lr()[0]
    assert lr_after_warmup == pytest.approx(0.001, rel=1e-3)

    # During decay
    for _ in range(450):  # Halfway through decay
        scheduler.step()

    lr_mid_decay = scheduler.get_last_lr()[0]
    assert 0.00001 < lr_mid_decay < 0.001


def test_cosine_warmup_scheduler_end() -> None:
    """Test learning rate at end of schedule."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=100, total_steps=1000, min_lr=0.00001
    )

    # Run full schedule
    for _ in range(1000):
        scheduler.step()

    # Should approach min_lr
    final_lr = scheduler.get_last_lr()[0]
    assert final_lr > 0.00001  # Not exactly min_lr but close


# Trainer tests


def test_trainer_initialization(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
) -> None:
    """Test Trainer initialization."""
    config = TrainerConfig(max_epochs=10)
    trainer = Trainer(
        simple_model, loss_fn, train_loader, config=config, device=device
    )

    assert trainer.model is simple_model
    assert trainer.loss_fn is loss_fn
    assert trainer.train_loader is train_loader
    assert trainer.device == device
    assert trainer.config.max_epochs == 10
    assert trainer.state.epoch == 0
    assert trainer.state.step == 0


def test_trainer_device_auto_detection(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
) -> None:
    """Test Trainer device auto-detection."""
    trainer = Trainer(simple_model, loss_fn, train_loader, device=None)
    assert trainer.device.type in ("cpu", "cuda")


def test_trainer_count_parameters(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
) -> None:
    """Test parameter counting."""
    trainer = Trainer(simple_model, loss_fn, train_loader, device=device)
    num_params = trainer._count_parameters()
    assert num_params > 0


def test_trainer_train_epoch(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
) -> None:
    """Test training for one epoch."""
    config = TrainerConfig(fp16=False, log_interval=1000)  # No FP16 for CPU
    trainer = Trainer(
        simple_model, loss_fn, train_loader, config=config, device=device
    )

    initial_step = trainer.state.step
    avg_loss = trainer.train_epoch()

    assert isinstance(avg_loss, float)
    assert avg_loss > 0  # Loss should be positive
    assert trainer.state.step > initial_step  # Step counter should increase


def test_trainer_validate(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    val_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
) -> None:
    """Test validation."""
    config = TrainerConfig(fp16=False)
    trainer = Trainer(
        simple_model, loss_fn, train_loader, val_loader, config=config, device=device
    )

    val_loss, metrics = trainer.validate()

    assert isinstance(val_loss, float)
    assert val_loss > 0
    # metrics will be None because bucket_centers_bps not set
    assert metrics is None


def test_trainer_validate_with_metrics(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    val_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
    num_buckets: int,
) -> None:
    """Test validation with metrics computation."""
    config = TrainerConfig(fp16=False)
    bucket_centers = tuple(np.linspace(-50.0, 50.0, num_buckets).tolist())
    metrics_config = MetricsConfig(
        num_buckets=num_buckets, bucket_centers_bps=bucket_centers
    )
    trainer = Trainer(
        simple_model,
        loss_fn,
        train_loader,
        val_loader,
        config=config,
        metrics_config=metrics_config,
        device=device,
    )

    val_loss, metrics = trainer.validate()

    assert isinstance(val_loss, float)
    assert val_loss > 0
    assert metrics is not None
    assert len(metrics.horizons) == 3  # 3 horizons in fixture


def test_trainer_validate_no_loader(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
) -> None:
    """Test validation without validation loader."""
    trainer = Trainer(simple_model, loss_fn, train_loader, device=device)

    val_loss, metrics = trainer.validate()

    assert val_loss == float("inf")
    assert metrics is None


def test_trainer_save_checkpoint(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test checkpoint saving."""
    config = TrainerConfig(checkpoint_dir=str(tmp_path))
    trainer = Trainer(
        simple_model, loss_fn, train_loader, config=config, device=device
    )

    checkpoint_path = trainer.save_checkpoint(epoch=1, val_loss=0.5, is_best=True)

    assert checkpoint_path.exists()
    assert checkpoint_path.name == "best_model.pt"

    # Load and verify checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["val_loss"] == 0.5
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_trainer_save_periodic_checkpoint(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test periodic checkpoint saving."""
    config = TrainerConfig(checkpoint_dir=str(tmp_path))
    trainer = Trainer(
        simple_model, loss_fn, train_loader, config=config, device=device
    )
    trainer.state.step = 100

    checkpoint_path = trainer.save_checkpoint(epoch=5, val_loss=0.5, is_best=False)

    assert checkpoint_path.exists()
    assert "epoch5" in checkpoint_path.name
    assert "step100" in checkpoint_path.name
    assert len(trainer.checkpoints) == 1


def test_trainer_cleanup_checkpoints(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test checkpoint cleanup."""
    config = TrainerConfig(checkpoint_dir=str(tmp_path), keep_best=2)
    trainer = Trainer(
        simple_model, loss_fn, train_loader, config=config, device=device
    )

    # Create 4 checkpoints with different losses
    trainer.save_checkpoint(epoch=1, val_loss=0.5, is_best=False)
    trainer.save_checkpoint(epoch=2, val_loss=0.4, is_best=False)
    trainer.save_checkpoint(epoch=3, val_loss=0.6, is_best=False)
    trainer.save_checkpoint(epoch=4, val_loss=0.3, is_best=False)

    # Should keep only 2 best (0.3 and 0.4)
    assert len(trainer.checkpoints) == 2
    assert trainer.checkpoints[0].val_loss == 0.3
    assert trainer.checkpoints[1].val_loss == 0.4


def test_trainer_load_checkpoint(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test checkpoint loading."""
    config = TrainerConfig(checkpoint_dir=str(tmp_path))
    trainer = Trainer(
        simple_model, loss_fn, train_loader, config=config, device=device
    )

    # Train a bit and save
    trainer.train_epoch()
    checkpoint_path = trainer.save_checkpoint(epoch=1, val_loss=0.5, is_best=True)

    # Create new trainer and load
    assert hasattr(simple_model, "config")
    new_model = PriceTransformer(simple_model.config)  # type: ignore[arg-type]
    new_trainer = Trainer(
        new_model, loss_fn, train_loader, config=config, device=device
    )
    new_trainer.load_checkpoint(checkpoint_path)

    assert new_trainer.state.epoch == 1
    assert new_trainer.state.step == trainer.state.step


def test_trainer_load_checkpoint_not_found(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
) -> None:
    """Test loading non-existent checkpoint."""
    trainer = Trainer(simple_model, loss_fn, train_loader, device=device)

    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint(Path("nonexistent.pt"))


def test_trainer_full_training(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    val_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test full training loop."""
    config = TrainerConfig(
        max_epochs=3,
        patience=2,
        fp16=False,
        checkpoint_dir=str(tmp_path),
        log_interval=1000,
    )
    trainer = Trainer(
        simple_model, loss_fn, train_loader, val_loader, config=config, device=device
    )

    trainer.train()

    # Check training completed (should finish 3 epochs or hit patience limit)
    # epoch is 0-indexed, so epoch >= 2 means 3 epochs completed (0, 1, 2)
    assert trainer.state.epoch >= 2 or trainer.state.epochs_without_improvement >= 2
    assert len(trainer.state.train_loss_history) > 0
    assert len(trainer.state.val_loss_history) > 0

    # Check best model saved
    best_checkpoint = tmp_path / "best_model.pt"
    assert best_checkpoint.exists()

    # Check training history saved
    history_file = tmp_path / "training_history.json"
    assert history_file.exists()
    with open(history_file) as f:
        history = json.load(f)
    assert "train_loss" in history
    assert "val_loss" in history


def test_trainer_early_stopping(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    val_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test early stopping mechanism."""
    config = TrainerConfig(
        max_epochs=100,
        patience=2,
        fp16=False,
        checkpoint_dir=str(tmp_path),
        log_interval=1000,
    )
    trainer = Trainer(
        simple_model, loss_fn, train_loader, val_loader, config=config, device=device
    )

    # Mock validate to return constant loss (no improvement)
    def mock_validate() -> tuple[float, None]:
        return (1.0, None)

    trainer.validate = mock_validate  # type: ignore[method-assign]

    trainer.train()

    # Should stop early due to patience
    assert trainer.state.epoch < 100
    assert trainer.state.epochs_without_improvement >= config.patience


# Factory function tests


def test_create_trainer(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
) -> None:
    """Test create_trainer factory function."""
    config_dict = {"max_epochs": 50, "learning_rate": 0.001}
    trainer = create_trainer(
        simple_model, loss_fn, train_loader, config_dict=config_dict, device=device
    )

    assert trainer.config.max_epochs == 50
    assert trainer.config.learning_rate == 0.001


def test_create_trainer_with_defaults(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
) -> None:
    """Test create_trainer with default config."""
    trainer = create_trainer(simple_model, loss_fn, train_loader, device=device)

    assert trainer.config.max_epochs == 100  # Default
    assert trainer.config.learning_rate == 0.0001  # Default


def test_create_trainer_with_metrics_config(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
    num_buckets: int,
) -> None:
    """Test create_trainer with metrics config."""
    metrics_dict = {"num_buckets": num_buckets, "num_calibration_bins": 20}
    trainer = create_trainer(
        simple_model,
        loss_fn,
        train_loader,
        metrics_config_dict=metrics_dict,
        device=device,
    )

    assert trainer.metrics_config.num_buckets == num_buckets
    assert trainer.metrics_config.num_calibration_bins == 20


# CheckpointInfo tests


def test_checkpoint_info() -> None:
    """Test CheckpointInfo dataclass."""
    info = CheckpointInfo(
        epoch=5, step=1000, val_loss=0.25, path=Path("/path/to/checkpoint.pt")
    )

    assert info.epoch == 5
    assert info.step == 1000
    assert info.val_loss == 0.25
    assert info.path == Path("/path/to/checkpoint.pt")


# Edge cases and error handling


def test_trainer_with_no_gradient_clipping(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
) -> None:
    """Test training with gradient clipping disabled."""
    config = TrainerConfig(max_grad_norm=0.0, fp16=False, log_interval=1000)
    trainer = Trainer(
        simple_model, loss_fn, train_loader, config=config, device=device
    )

    avg_loss = trainer.train_epoch()
    assert isinstance(avg_loss, float)


def test_trainer_save_history(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test saving training history."""
    config = TrainerConfig(checkpoint_dir=str(tmp_path))
    trainer = Trainer(
        simple_model, loss_fn, train_loader, config=config, device=device
    )

    # Add some history
    trainer.state.train_loss_history = [0.5, 0.4, 0.3]
    trainer.state.val_loss_history = [0.6, 0.5, 0.4]
    trainer.state.learning_rate_history = [0.001, 0.0009, 0.0008]
    trainer.state.best_val_loss = 0.4
    trainer.state.epoch = 2
    trainer.state.step = 300

    trainer._save_training_history()

    history_file = tmp_path / "training_history.json"
    assert history_file.exists()

    with open(history_file) as f:
        history = json.load(f)

    assert history["train_loss"] == [0.5, 0.4, 0.3]
    assert history["val_loss"] == [0.6, 0.5, 0.4]
    assert history["learning_rate"] == [0.001, 0.0009, 0.0008]
    assert history["best_val_loss"] == 0.4
    assert history["epochs_trained"] == 3
    assert history["total_steps"] == 300


def test_trainer_checkpoint_directory_creation(
    simple_model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test checkpoint directory is created if it doesn't exist."""
    checkpoint_dir = tmp_path / "nested" / "dir" / "checkpoints"
    config = TrainerConfig(checkpoint_dir=str(checkpoint_dir))

    _trainer = Trainer(
        simple_model, loss_fn, train_loader, config=config, device=device
    )

    assert checkpoint_dir.exists()
    assert checkpoint_dir.is_dir()
