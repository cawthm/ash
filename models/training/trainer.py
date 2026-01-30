"""Training loop for price prediction models.

This module implements the training infrastructure including:
- AdamW optimizer with learning rate warmup and cosine decay
- Gradient clipping for stable training
- Early stopping on validation loss
- Checkpointing best models with state saving/loading
- Mixed precision training (FP16) support
- Comprehensive logging and metric tracking
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from evaluation.metrics import (
    MetricsConfig,
    MultiHorizonMetrics,
    compute_multi_horizon_metrics,
    format_metrics_report,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for model training.

    Attributes:
        learning_rate: Initial learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization) coefficient.
        warmup_steps: Number of steps for learning rate warmup.
        max_grad_norm: Maximum gradient norm for clipping (0 to disable).
        max_epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        fp16: Whether to use mixed precision training.
        checkpoint_dir: Directory to save model checkpoints.
        save_every: Save checkpoint every N epochs (0 to save only best).
        keep_best: Number of best checkpoints to keep (0 to keep all).
        log_interval: Log training progress every N steps.
        val_interval: Validate every N epochs.
    """

    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    max_epochs: int = 100
    patience: int = 10
    fp16: bool = True
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1
    keep_best: int = 3
    log_interval: int = 100
    val_interval: int = 1

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.save_every < 0:
            raise ValueError("save_every must be non-negative")
        if self.keep_best < 0:
            raise ValueError("keep_best must be non-negative")
        if self.log_interval < 1:
            raise ValueError("log_interval must be at least 1")
        if self.val_interval < 1:
            raise ValueError("val_interval must be at least 1")


@dataclass
class TrainingState:
    """State of training progress.

    Attributes:
        epoch: Current epoch number.
        step: Global training step number.
        best_val_loss: Best validation loss observed.
        epochs_without_improvement: Epochs since last improvement.
        train_loss_history: Training loss history.
        val_loss_history: Validation loss history.
        learning_rate_history: Learning rate history.
    """

    epoch: int = 0
    step: int = 0
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0
    train_loss_history: list[float] = None  # type: ignore[assignment]
    val_loss_history: list[float] = None  # type: ignore[assignment]
    learning_rate_history: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize mutable fields."""
        if self.train_loss_history is None:
            self.train_loss_history = []
        if self.val_loss_history is None:
            self.val_loss_history = []
        if self.learning_rate_history is None:
            self.learning_rate_history = []


class CosineWarmupScheduler(LRScheduler):
    """Learning rate scheduler with warmup and cosine decay.

    Linearly increases LR from 0 to max over warmup_steps,
    then cosine anneals to min_lr over remaining steps.

    Attributes:
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate (end of cosine decay).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """Initialize scheduler.

        Args:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of warmup steps.
            total_steps: Total training steps.
            min_lr: Minimum learning rate.
            last_epoch: Index of last epoch for resuming.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        """Compute learning rate for current step."""
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            lr_scale = step / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr_scale = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
            # Scale from max to min_lr
            base_lr_value = float(self.base_lrs[0])
            lr_scale = self.min_lr / base_lr_value + lr_scale * (
                1.0 - self.min_lr / base_lr_value
            )

        return [float(base_lr) * lr_scale for base_lr in self.base_lrs]


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint.

    Attributes:
        epoch: Epoch number.
        step: Global step number.
        val_loss: Validation loss at checkpoint.
        path: Path to checkpoint file.
    """

    epoch: int
    step: int
    val_loss: float
    path: Path


class Trainer:
    """Trainer for price prediction models.

    Handles the complete training loop including optimization,
    validation, checkpointing, and early stopping.

    Attributes:
        model: The model to train.
        loss_fn: Loss function module.
        train_loader: Training data loader.
        val_loader: Validation data loader (optional).
        config: Training configuration.
        metrics_config: Metrics configuration.
        device: Device to train on (CPU or CUDA).
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
        scaler: Gradient scaler for mixed precision (if fp16 enabled).
        state: Training state tracker.
        checkpoint_dir: Path to checkpoint directory.
        checkpoints: List of saved checkpoint info.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
        val_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]] | None = None,
        config: TrainerConfig | None = None,
        metrics_config: MetricsConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train.
            loss_fn: Loss function module.
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            config: Training configuration.
            metrics_config: Metrics configuration for evaluation.
            device: Device to train on. Auto-detects if None.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainerConfig()
        self.metrics_config = metrics_config or MetricsConfig()

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        logger.info(f"Training on device: {self.device}")

        # Move model to device
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup learning rate scheduler
        total_steps = self.config.max_epochs * len(self.train_loader)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
            min_lr=self.config.learning_rate * 0.01,
        )

        # Setup mixed precision training
        if self.config.fp16:
            self.scaler: GradScaler | None = GradScaler("cuda")
        else:
            self.scaler = None

        # Initialize training state
        self.state = TrainingState()

        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: list[CheckpointInfo] = []

        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Model parameters: {self._count_parameters():,}")

    def _count_parameters(self) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (features, targets) in enumerate(self.train_loader):
            # Move data to device
            features = features.to(self.device)
            targets = {h: t.to(self.device) for h, t in targets.items()}

            # Forward pass with optional mixed precision
            if self.config.fp16 and self.scaler is not None:
                with autocast("cuda"):
                    logits = self.model(features)
                    loss = self.loss_fn(logits, targets)
            else:
                logits = self.model(features)
                loss = self.loss_fn(logits, targets)

            # Backward pass
            self.optimizer.zero_grad()

            if self.config.fp16 and self.scaler is not None:
                self.scaler.scale(loss).backward()

                # Gradient clipping (unscale first)
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1
            self.state.step += 1

            # Log progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {self.state.epoch+1} "
                    f"[{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.4f} "
                    f"LR: {lr:.6f}"
                )

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> tuple[float, MultiHorizonMetrics | None]:
        """Validate on validation set.

        Returns:
            Tuple of (average validation loss, detailed metrics).
            Metrics is None if metrics_config.bucket_centers_bps is not set.
        """
        if self.val_loader is None:
            return float("inf"), None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Collect predictions and targets for metrics
        all_probs: dict[int, list[Tensor]] = {}
        all_targets: dict[int, list[Tensor]] = {}

        for features, targets in self.val_loader:
            # Move data to device
            features = features.to(self.device)
            targets = {h: t.to(self.device) for h, t in targets.items()}

            # Forward pass
            if self.config.fp16:
                with autocast("cuda"):
                    logits = self.model(features)
                    loss = self.loss_fn(logits, targets)
            else:
                logits = self.model(features)
                loss = self.loss_fn(logits, targets)

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions and targets
            probs = {h: torch.softmax(logits[h], dim=-1) for h in logits}
            for h in probs:
                if h not in all_probs:
                    all_probs[h] = []
                    all_targets[h] = []
                all_probs[h].append(probs[h])
                all_targets[h].append(targets[h])

        avg_loss = total_loss / max(1, num_batches)

        # Compute detailed metrics if bucket centers available
        metrics: MultiHorizonMetrics | None = None
        if self.metrics_config.bucket_centers_bps is not None:
            # Concatenate all batches
            probs_dict = {h: torch.cat(all_probs[h], dim=0) for h in all_probs}
            targets_dict = {h: torch.cat(all_targets[h], dim=0) for h in all_targets}

            # Get bucket centers as tensor
            bucket_centers = torch.tensor(
                self.metrics_config.bucket_centers_bps,
                device=self.device,
                dtype=torch.float32,
            )

            metrics = compute_multi_horizon_metrics(
                probs_dict,
                targets_dict,
                bucket_centers,
                self.metrics_config.num_calibration_bins,
            )

        return avg_loss, metrics

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> Path:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss at checkpoint.
            is_best: Whether this is the best model so far.

        Returns:
            Path to saved checkpoint file.
        """
        # Create checkpoint filename
        if is_best:
            filename = "best_model.pt"
        else:
            filename = f"checkpoint_epoch{epoch}_step{self.state.step}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "step": self.state.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "training_state": {
                "best_val_loss": self.state.best_val_loss,
                "epochs_without_improvement": self.state.epochs_without_improvement,
                "train_loss_history": self.state.train_loss_history,
                "val_loss_history": self.state.val_loss_history,
                "learning_rate_history": self.state.learning_rate_history,
            },
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Track checkpoint
        if not is_best:
            info = CheckpointInfo(
                epoch=epoch,
                step=self.state.step,
                val_loss=val_loss,
                path=checkpoint_path,
            )
            self.checkpoints.append(info)

            # Clean up old checkpoints if needed
            if self.config.keep_best > 0:
                self._cleanup_checkpoints()

        return checkpoint_path

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the best N."""
        if len(self.checkpoints) <= self.config.keep_best:
            return

        # Sort by validation loss (ascending)
        self.checkpoints.sort(key=lambda x: x.val_loss)

        # Remove worst checkpoints
        to_remove = self.checkpoints[self.config.keep_best :]
        for info in to_remove:
            if info.path.exists():
                info.path.unlink()
                logger.info(f"Removed checkpoint: {info.path}")

        self.checkpoints = self.checkpoints[: self.config.keep_best]

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Restore model and optimizer
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore training state
        self.state.epoch = checkpoint["epoch"]
        self.state.step = checkpoint["step"]

        if "training_state" in checkpoint:
            ts = checkpoint["training_state"]
            self.state.best_val_loss = ts["best_val_loss"]
            self.state.epochs_without_improvement = ts["epochs_without_improvement"]
            self.state.train_loss_history = ts["train_loss_history"]
            self.state.val_loss_history = ts["val_loss_history"]
            self.state.learning_rate_history = ts["learning_rate_history"]

        logger.info(f"Resumed from epoch {self.state.epoch}, step {self.state.step}")

    def train(self) -> None:
        """Run full training loop with validation and early stopping."""
        logger.info("Starting training...")
        logger.info(f"Max epochs: {self.config.max_epochs}")
        logger.info(f"Patience: {self.config.patience}")
        logger.info(f"Training batches per epoch: {len(self.train_loader)}")
        if self.val_loader is not None:
            logger.info(f"Validation batches: {len(self.val_loader)}")

        for epoch in range(self.state.epoch, self.config.max_epochs):
            self.state.epoch = epoch

            # Train epoch
            train_loss = self.train_epoch()
            self.state.train_loss_history.append(train_loss)

            # Get current learning rate
            lr = float(self.scheduler.get_last_lr()[0])
            self.state.learning_rate_history.append(lr)

            logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"LR: {lr:.6f}"
            )

            # Validate if needed
            if (epoch + 1) % self.config.val_interval == 0:
                val_loss, metrics = self.validate()
                self.state.val_loss_history.append(val_loss)

                logger.info(f"Validation Loss: {val_loss:.4f}")

                if metrics is not None:
                    logger.info("\n" + format_metrics_report(metrics))

                # Check for improvement
                is_best = val_loss < self.state.best_val_loss
                if is_best:
                    self.state.best_val_loss = val_loss
                    self.state.epochs_without_improvement = 0
                    logger.info(f"New best validation loss: {val_loss:.4f}")

                    # Save best model
                    self.save_checkpoint(epoch + 1, val_loss, is_best=True)
                else:
                    self.state.epochs_without_improvement += 1
                    logger.info(
                        f"No improvement for {self.state.epochs_without_improvement} "
                        f"epoch(s)"
                    )

                # Save periodic checkpoint
                if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(epoch + 1, val_loss, is_best=False)

                # Early stopping check
                if self.state.epochs_without_improvement >= self.config.patience:
                    logger.info(
                        f"Early stopping triggered after {epoch+1} epochs "
                        f"(patience: {self.config.patience})"
                    )
                    break

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.state.best_val_loss:.4f}")

        # Save final training history
        self._save_training_history()

    def _save_training_history(self) -> None:
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir / "training_history.json"
        history = {
            "train_loss": self.state.train_loss_history,
            "val_loss": self.state.val_loss_history,
            "learning_rate": self.state.learning_rate_history,
            "best_val_loss": self.state.best_val_loss,
            "epochs_trained": self.state.epoch + 1,
            "total_steps": self.state.step,
        }

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved training history: {history_path}")


def create_trainer(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]],
    val_loader: DataLoader[tuple[Tensor, dict[int, Tensor]]] | None = None,
    config_dict: dict[str, Any] | None = None,
    metrics_config_dict: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
) -> Trainer:
    """Factory function to create trainer from configuration dictionaries.

    Args:
        model: Model to train.
        loss_fn: Loss function module.
        train_loader: Training data loader.
        val_loader: Validation data loader (optional).
        config_dict: Training configuration dictionary.
        metrics_config_dict: Metrics configuration dictionary.
        device: Device to train on. Auto-detects if None.

    Returns:
        Initialized Trainer instance.

    Example:
        >>> config = {
        ...     "learning_rate": 0.0001,
        ...     "max_epochs": 100,
        ...     "patience": 10,
        ... }
        >>> trainer = create_trainer(model, loss_fn, train_loader, config_dict=config)
    """
    config = TrainerConfig(**config_dict) if config_dict else None
    metrics_config = (
        MetricsConfig(**metrics_config_dict) if metrics_config_dict else None
    )

    return Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        metrics_config=metrics_config,
        device=device,
    )
