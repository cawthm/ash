"""Transformer model for multi-horizon probabilistic price prediction.

This module implements a lightweight transformer architecture that outputs
probability distributions over price buckets at multiple time horizons
simultaneously. The architecture is optimized for sub-10ms inference latency.

Architecture:
    Input Sequence (features × time steps)
        ↓
    Feature Embedding Layer
        ↓
    Positional Encoding (learned)
        ↓
    Transformer Encoder (N layers)
        ↓
    [CLS] Token Representation
        ↓
    Multi-Horizon Classification Heads (parallel)
        → P(bucket | t+1s), P(bucket | t+5s), ..., P(bucket | t+600s)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the price prediction transformer.

    Attributes:
        input_dim: Dimension of input features at each time step.
        embedding_dim: Dimension of embedded representations (128-256 for <10ms).
        num_layers: Number of transformer encoder layers (2-4).
        num_heads: Number of attention heads (4-8).
        ff_dim: Feed-forward network hidden dimension (512-1024).
        dropout: Dropout rate for regularization.
        max_seq_len: Maximum sequence length (number of time steps).
        horizons: Prediction horizons in seconds.
        num_buckets: Number of output buckets per horizon.
        use_cls_token: Whether to use [CLS] token for classification.
        attention_type: Type of attention ("causal" or "bidirectional").
    """

    input_dim: int
    embedding_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256
    horizons: tuple[int, ...] = (1, 5, 10, 30, 60, 120, 300, 600)
    num_buckets: int = 101
    use_cls_token: bool = True
    attention_type: Literal["causal", "bidirectional"] = "causal"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.input_dim < 1:
            raise ValueError("input_dim must be at least 1")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ff_dim < 1:
            raise ValueError("ff_dim must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")
        if len(self.horizons) < 1:
            raise ValueError("horizons must have at least one element")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("All horizons must be positive")
        if self.num_buckets < 3:
            raise ValueError("num_buckets must be at least 3")
        if self.attention_type not in ("causal", "bidirectional"):
            raise ValueError(f"Unknown attention_type: {self.attention_type}")


class PositionalEncoding(nn.Module):
    """Learned positional encoding for sequence positions.

    Unlike sinusoidal encoding, learned positions can capture task-specific
    temporal patterns. The encoding is added to the embedded features.

    Attributes:
        config: Transformer configuration.
        position_embedding: Learnable position embeddings.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize positional encoding.

        Args:
            config: Transformer configuration.
        """
        super().__init__()
        self.config = config

        # Account for CLS token if used
        max_positions = config.max_seq_len + (1 if config.use_cls_token else 0)

        # Learned position embeddings
        self.position_embedding = nn.Embedding(max_positions, config.embedding_dim)

        # Initialize with small values for stable training
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim).

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        position_embeddings: Tensor = self.position_embedding(positions)
        result: Tensor = x + position_embeddings.unsqueeze(0)  # Broadcast over batch
        return result


class FeatureEmbedding(nn.Module):
    """Projects input features to embedding dimension.

    This layer maps the raw feature vectors to the transformer's hidden
    dimension with layer normalization for stable training.

    Attributes:
        config: Transformer configuration.
        projection: Linear projection layer.
        norm: Layer normalization.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize feature embedding.

        Args:
            config: Transformer configuration.
        """
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.input_dim, config.embedding_dim)
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Embed input features.

        Args:
            x: Input features of shape (batch, seq_len, input_dim).

        Returns:
            Embedded features of shape (batch, seq_len, embedding_dim).
        """
        result: Tensor = self.norm(self.projection(x))
        return result


class MultiHorizonHead(nn.Module):
    """Classification heads for multiple prediction horizons.

    Each horizon has its own linear head that produces logits over
    price buckets. All heads share the same input representation.

    Attributes:
        config: Transformer configuration.
        heads: ModuleDict mapping horizon (str) to classification head.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize multi-horizon classification heads.

        Args:
            config: Transformer configuration.
        """
        super().__init__()
        self.config = config

        # Create a classification head for each horizon
        self.heads = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.Linear(config.embedding_dim, config.ff_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.ff_dim, config.num_buckets),
                )
                for horizon in config.horizons
            }
        )

    def forward(self, x: Tensor) -> dict[int, Tensor]:
        """Compute logits for all horizons.

        Args:
            x: Representation to classify, shape (batch, embedding_dim).

        Returns:
            Dictionary mapping horizon (int) to logits (batch, num_buckets).
        """
        return {
            int(horizon): head(x) for horizon, head in self.heads.items()
        }


class PriceTransformer(nn.Module):
    """Transformer model for multi-horizon probabilistic price prediction.

    This model takes a sequence of feature vectors and outputs probability
    distributions over price buckets for multiple future time horizons.
    The architecture is designed for low-latency inference (<10ms).

    Attributes:
        config: Transformer configuration.
        feature_embedding: Projects input features to embedding space.
        cls_token: Learnable [CLS] token for classification (if enabled).
        positional_encoding: Adds position information to embeddings.
        encoder: Stack of transformer encoder layers.
        horizon_heads: Multi-horizon classification heads.
    """

    cls_token: Tensor | None  # Buffer type annotation

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize price transformer.

        Args:
            config: Transformer configuration.
        """
        super().__init__()
        self.config = config

        # Feature embedding layer
        self.feature_embedding = FeatureEmbedding(config)

        # [CLS] token for classification
        if config.use_cls_token:
            cls_param = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
            self.register_parameter("cls_token", cls_param)
            nn.init.normal_(cls_param, mean=0.0, std=0.02)
        else:
            self.cls_token = None

        # Positional encoding
        self.positional_encoding = PositionalEncoding(config)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,  # For compatibility
        )

        # Multi-horizon classification heads
        self.horizon_heads = MultiHorizonHead(config)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _create_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tensor | None:
        """Create attention mask based on attention type.

        Args:
            seq_len: Sequence length (including CLS token if used).
            device: Device to create mask on.

        Returns:
            Attention mask or None for bidirectional attention.
            For causal attention, returns upper triangular mask where True
            indicates positions to mask (prevent attending to).
        """
        if self.config.attention_type == "bidirectional":
            return None

        # Causal mask: prevent attending to future positions
        # True = masked (cannot attend), False = allowed
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

        # If using CLS token, allow it to attend to all positions
        # CLS token is at position 0
        if self.config.use_cls_token:
            mask[0, :] = False  # CLS can attend to everything

        return mask

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> dict[int, Tensor]:
        """Forward pass through the transformer.

        Args:
            x: Input features of shape (batch, seq_len, input_dim).
            padding_mask: Boolean mask of shape (batch, seq_len) where True
                indicates padded positions to ignore. Optional.

        Returns:
            Dictionary mapping horizon (int) to logits (batch, num_buckets).
        """
        batch_size = x.size(0)
        device = x.device

        # Embed features
        x = self.feature_embedding(x)  # (batch, seq_len, embedding_dim)

        # Prepend [CLS] token
        if self.config.use_cls_token and self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, embed_dim)

            # Extend padding mask for CLS token (CLS is never padded)
            if padding_mask is not None:
                cls_mask = torch.zeros(
                    batch_size, 1, device=device, dtype=torch.bool
                )
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Create attention mask
        seq_len = x.size(1)
        attn_mask = self._create_attention_mask(seq_len, device)

        # Pass through transformer encoder
        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=padding_mask)

        # Extract representation for classification
        if self.config.use_cls_token:
            # Use [CLS] token representation
            cls_repr = x[:, 0, :]  # (batch, embedding_dim)
        else:
            # Use mean pooling over sequence
            if padding_mask is not None:
                # Mask out padded positions
                mask_expanded = padding_mask.unsqueeze(-1)  # (batch, seq, 1)
                x = x.masked_fill(mask_expanded, 0.0)
                lengths = (~padding_mask).sum(dim=1, keepdim=True).float()
                cls_repr = x.sum(dim=1) / lengths.clamp(min=1.0)
            else:
                cls_repr = x.mean(dim=1)  # (batch, embedding_dim)

        # Compute logits for all horizons
        logits: dict[int, Tensor] = self.horizon_heads(cls_repr)

        return logits

    def get_probabilities(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> dict[int, Tensor]:
        """Get probability distributions for all horizons.

        Convenience method that applies softmax to logits.

        Args:
            x: Input features of shape (batch, seq_len, input_dim).
            padding_mask: Boolean mask for padded positions. Optional.

        Returns:
            Dictionary mapping horizon (int) to probabilities (batch, num_buckets).
        """
        logits = self.forward(x, padding_mask)
        return {
            horizon: torch.softmax(horizon_logits, dim=-1)
            for horizon, horizon_logits in logits.items()
        }

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_horizons(self) -> int:
        """Number of prediction horizons."""
        return len(self.config.horizons)


def create_model(config: TransformerConfig) -> PriceTransformer:
    """Factory function to create price transformer model.

    Args:
        config: Transformer configuration.

    Returns:
        Initialized PriceTransformer model.
    """
    return PriceTransformer(config)


def create_model_from_dict(config_dict: dict[str, Any]) -> PriceTransformer:
    """Create model from configuration dictionary.

    This is useful for loading configuration from YAML files.

    Args:
        config_dict: Dictionary with configuration values.
            Required keys: input_dim
            Optional keys match TransformerConfig attributes.

    Returns:
        Initialized PriceTransformer model.

    Example:
        >>> config_dict = {
        ...     "input_dim": 64,
        ...     "embedding_dim": 128,
        ...     "num_layers": 2,
        ... }
        >>> model = create_model_from_dict(config_dict)
    """
    # Handle horizons tuple conversion (YAML loads as list)
    if "horizons" in config_dict and isinstance(config_dict["horizons"], list):
        config_dict = dict(config_dict)  # Copy to avoid mutation
        config_dict["horizons"] = tuple(config_dict["horizons"])

    config = TransformerConfig(**config_dict)
    return PriceTransformer(config)
