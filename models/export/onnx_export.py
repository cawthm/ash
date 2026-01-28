"""ONNX export utilities for PriceTransformer model.

This module provides functionality to export trained PyTorch models to ONNX
format for production inference using ONNX Runtime (particularly the Rust
`ort` crate).

Key features:
- Export PriceTransformer to ONNX with configurable opset version
- Graph optimization (constant folding, operator fusion)
- Input/output naming for multi-horizon predictions
- Metadata embedding for model configuration
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from models.architectures.price_transformer import PriceTransformer, TransformerConfig


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for ONNX export.

    Attributes:
        opset_version: ONNX opset version. Version 17 is well-supported by ort.
        do_constant_folding: Whether to apply constant folding optimization.
        dynamic_axes: Whether to use dynamic batch and sequence dimensions.
        input_names: Names for model inputs.
        output_prefix: Prefix for output names (will be suffixed with horizon).
    """

    opset_version: int = 17
    do_constant_folding: bool = True
    dynamic_axes: bool = True
    input_names: tuple[str, ...] = ("features",)
    output_prefix: str = "logits"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.opset_version < 11:
            raise ValueError("opset_version must be at least 11 for transformer ops")
        if self.opset_version > 21:
            raise ValueError("opset_version must be at most 21")
        if len(self.input_names) < 1:
            raise ValueError("input_names must have at least one element")


@dataclass
class ExportResult:
    """Result of ONNX export operation.

    Attributes:
        output_path: Path where ONNX model was saved.
        input_names: List of input tensor names.
        output_names: List of output tensor names (one per horizon).
        horizons: Prediction horizons in seconds.
        opset_version: ONNX opset version used.
        model_size_bytes: Size of exported ONNX file in bytes.
    """

    output_path: Path
    input_names: list[str]
    output_names: list[str]
    horizons: tuple[int, ...]
    opset_version: int
    model_size_bytes: int


class ONNXExporter:
    """Export PriceTransformer models to ONNX format.

    This class handles the conversion of trained PyTorch models to ONNX
    format with appropriate optimizations for production inference.

    Attributes:
        config: Export configuration.
    """

    def __init__(self, config: ExportConfig | None = None) -> None:
        """Initialize ONNX exporter.

        Args:
            config: Export configuration. Uses defaults if not provided.
        """
        self.config = config or ExportConfig()

    def export(
        self,
        model: PriceTransformer,
        output_path: str | Path,
        sample_batch_size: int = 1,
        sample_seq_len: int | None = None,
    ) -> ExportResult:
        """Export PriceTransformer to ONNX format.

        Args:
            model: Trained PriceTransformer model to export.
            output_path: Path to save the ONNX model.
            sample_batch_size: Batch size for sample input during tracing.
            sample_seq_len: Sequence length for sample input. Uses model's
                max_seq_len if not provided.

        Returns:
            ExportResult with export metadata.

        Raises:
            RuntimeError: If export fails.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare model for export
        model.eval()
        device = next(model.parameters()).device

        # Create sample input
        seq_len = sample_seq_len or model.config.max_seq_len
        sample_input = torch.randn(
            sample_batch_size,
            seq_len,
            model.config.input_dim,
            device=device,
        )

        # Prepare input/output names
        input_names = list(self.config.input_names)
        output_names = [
            f"{self.config.output_prefix}_{h}s" for h in model.config.horizons
        ]

        # Prepare dynamic axes if enabled
        dynamic_axes: dict[str, dict[int, str]] | None = None
        if self.config.dynamic_axes:
            dynamic_axes = {
                input_names[0]: {0: "batch_size", 1: "seq_len"},
            }
            for name in output_names:
                dynamic_axes[name] = {0: "batch_size"}

        # Create wrapper for ordered output with ONNX-exportable encoder
        wrapper = _OrderedOutputWrapper(model)

        # Export to ONNX using legacy TorchScript-based exporter
        # The newer dynamo-based exporter has issues with transformer attention masks
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (sample_input,),
                str(output_path),
                export_params=True,
                opset_version=self.config.opset_version,
                do_constant_folding=self.config.do_constant_folding,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=False,  # Use legacy TorchScript exporter for compatibility
            )

        # Get file size
        model_size = output_path.stat().st_size

        return ExportResult(
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            horizons=model.config.horizons,
            opset_version=self.config.opset_version,
            model_size_bytes=model_size,
        )

    def export_with_metadata(
        self,
        model: PriceTransformer,
        output_path: str | Path,
        sample_batch_size: int = 1,
        sample_seq_len: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ExportResult:
        """Export model with embedded metadata.

        Exports the model and adds metadata properties to the ONNX graph
        for documentation and versioning purposes.

        Args:
            model: Trained PriceTransformer model to export.
            output_path: Path to save the ONNX model.
            sample_batch_size: Batch size for sample input during tracing.
            sample_seq_len: Sequence length for sample input.
            metadata: Additional metadata to embed in the model.

        Returns:
            ExportResult with export metadata.
        """
        import onnx

        # First export without metadata
        result = self.export(
            model,
            output_path,
            sample_batch_size=sample_batch_size,
            sample_seq_len=sample_seq_len,
        )

        # Load and add metadata
        onnx_model = onnx.load(str(result.output_path))

        # Add model configuration metadata
        config_metadata = _config_to_metadata(model.config)
        for key, value in config_metadata.items():
            onnx_model.metadata_props.append(
                onnx.StringStringEntryProto(key=f"config.{key}", value=value)
            )

        # Add custom metadata
        if metadata:
            for key, value in metadata.items():
                onnx_model.metadata_props.append(
                    onnx.StringStringEntryProto(key=key, value=value)
                )

        # Save with metadata
        onnx.save(onnx_model, str(result.output_path))

        return result


class _ExportableMultiheadAttention(torch.nn.Module):
    """ONNX-exportable multi-head attention.

    Manually implements multi-head attention to avoid the
    fused kernel (aten::_native_multi_head_attention).
    """

    def __init__(self, mha: torch.nn.MultiheadAttention) -> None:
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = mha.head_dim
        self.dropout = mha.dropout
        self.batch_first = mha.batch_first

        # Copy weight parameters
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with manual attention computation."""
        if self.batch_first:
            # (batch, seq, embed) -> (seq, batch, embed)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        seq_len, batch_size, embed_dim = query.shape

        # Linear projections (Q, K, V)
        # in_proj_weight is (3*embed_dim, embed_dim)
        qkv = torch.nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        # (seq, batch, embed) -> (seq, batch, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Scaled dot-product attention
        scale = float(self.head_dim) ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply attention mask
        if attn_mask is not None:
            # attn_mask is (seq, seq) or (batch*num_heads, seq, seq)
            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))
            else:
                attn_weights = attn_weights + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask is (batch, seq)
            # Expand to (batch, 1, 1, seq) for broadcasting
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))

        # Softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        # (batch, num_heads, seq, head_dim) -> (seq, batch, num_heads, head_dim) -> (seq, batch, embed)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch_size, embed_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            # (seq, batch, embed) -> (batch, seq, embed)
            attn_output = attn_output.transpose(0, 1)

        result: Tensor = attn_output
        return result


class _ExportableTransformerEncoderLayer(torch.nn.Module):
    """ONNX-exportable transformer encoder layer.

    Manually implements the encoder layer forward pass to avoid the
    fused kernel (aten::_transformer_encoder_layer_fwd).
    """

    def __init__(self, layer: torch.nn.TransformerEncoderLayer) -> None:
        super().__init__()
        # Wrap attention with exportable version
        self.self_attn = _ExportableMultiheadAttention(layer.self_attn)
        self.linear1 = layer.linear1
        self.linear2 = layer.linear2
        self.norm1 = layer.norm1
        self.norm2 = layer.norm2
        self.dropout = layer.dropout
        self.dropout1 = layer.dropout1
        self.dropout2 = layer.dropout2
        self.activation = layer.activation
        self.norm_first = layer.norm_first

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass without fused kernels."""
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
    ) -> Tensor:
        """Self-attention block."""
        attn_output: Tensor = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        result: Tensor = self.dropout1(attn_output)
        return result

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed-forward block."""
        ff_output: Tensor = self.linear2(self.dropout(self.activation(self.linear1(x))))
        result: Tensor = self.dropout2(ff_output)
        return result


class _ExportableTransformerEncoder(torch.nn.Module):
    """ONNX-exportable wrapper for TransformerEncoder.

    PyTorch's TransformerEncoder uses fused kernels (aten::_transformer_encoder_layer_fwd)
    that are not supported by ONNX. This wrapper wraps each layer with an
    exportable version that manually implements the forward pass.
    """

    def __init__(self, encoder: torch.nn.TransformerEncoder) -> None:
        super().__init__()
        # Wrap each layer with exportable version
        self.layers = torch.nn.ModuleList([
            _ExportableTransformerEncoderLayer(layer)
            for layer in encoder.layers
        ])
        self.norm = encoder.norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through transformer encoder layers.

        Args:
            src: Input tensor.
            mask: Attention mask.
            src_key_padding_mask: Key padding mask.

        Returns:
            Encoded tensor.
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class _OrderedOutputWrapper(torch.nn.Module):
    """Wrapper to ensure ordered tuple output for ONNX export.

    ONNX export requires outputs to be tensors or tuples of tensors,
    not dictionaries. This wrapper converts the horizon dict to an
    ordered tuple based on the model's horizon configuration.

    It also replaces the TransformerEncoder with an ONNX-exportable
    version that avoids fused kernels.
    """

    def __init__(self, model: PriceTransformer) -> None:
        """Initialize wrapper.

        Args:
            model: PriceTransformer model to wrap.
        """
        super().__init__()
        self.horizons = model.config.horizons
        self.config = model.config

        # Copy model components
        self.feature_embedding = model.feature_embedding
        self.cls_token = model.cls_token
        self.positional_encoding = model.positional_encoding
        self.horizon_heads = model.horizon_heads

        # Replace encoder with ONNX-exportable version
        self.encoder = _ExportableTransformerEncoder(model.encoder)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Forward pass returning ordered tuple of logits.

        Args:
            x: Input features of shape (batch, seq_len, input_dim).

        Returns:
            Tuple of logits tensors, one per horizon in config order.
        """
        batch_size = x.size(0)
        device = x.device

        # Embed features
        x = self.feature_embedding(x)

        # Prepend [CLS] token
        if self.config.use_cls_token and self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Create attention mask
        seq_len = x.size(1)
        attn_mask = self._create_attention_mask(seq_len, device)

        # Pass through transformer encoder
        x = self.encoder(x, mask=attn_mask)

        # Extract representation for classification
        cls_repr = x[:, 0, :] if self.config.use_cls_token else x.mean(dim=1)

        # Compute logits for all horizons
        logits_dict = self.horizon_heads(cls_repr)

        return tuple(logits_dict[h] for h in self.horizons)

    def _create_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tensor | None:
        """Create attention mask based on attention type."""
        if self.config.attention_type == "bidirectional":
            return None

        # Causal mask: prevent attending to future positions
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

        # If using CLS token, allow it to attend to all positions
        if self.config.use_cls_token:
            mask[0, :] = False

        return mask


def _config_to_metadata(config: TransformerConfig) -> dict[str, str]:
    """Convert TransformerConfig to string metadata dict.

    Args:
        config: Model configuration.

    Returns:
        Dictionary with string keys and values for ONNX metadata.
    """
    return {
        "input_dim": str(config.input_dim),
        "embedding_dim": str(config.embedding_dim),
        "num_layers": str(config.num_layers),
        "num_heads": str(config.num_heads),
        "ff_dim": str(config.ff_dim),
        "dropout": str(config.dropout),
        "max_seq_len": str(config.max_seq_len),
        "horizons": ",".join(str(h) for h in config.horizons),
        "num_buckets": str(config.num_buckets),
        "use_cls_token": str(config.use_cls_token),
        "attention_type": config.attention_type,
    }


def export_model(
    model: PriceTransformer,
    output_path: str | Path,
    config: ExportConfig | None = None,
    sample_batch_size: int = 1,
    sample_seq_len: int | None = None,
) -> ExportResult:
    """Convenience function to export a PriceTransformer model.

    Args:
        model: Trained PriceTransformer model to export.
        output_path: Path to save the ONNX model.
        config: Export configuration. Uses defaults if not provided.
        sample_batch_size: Batch size for sample input during tracing.
        sample_seq_len: Sequence length for sample input.

    Returns:
        ExportResult with export metadata.

    Example:
        >>> from models.architectures.price_transformer import (
        ...     PriceTransformer, TransformerConfig
        ... )
        >>> config = TransformerConfig(input_dim=64)
        >>> model = PriceTransformer(config)
        >>> result = export_model(model, "model.onnx")
        >>> print(result.output_names)
        ['logits_1s', 'logits_5s', ...]
    """
    exporter = ONNXExporter(config)
    return exporter.export(
        model,
        output_path,
        sample_batch_size=sample_batch_size,
        sample_seq_len=sample_seq_len,
    )


def load_onnx_model(model_path: str | Path) -> Any:
    """Load an ONNX model and return the model proto.

    This is a convenience function for loading exported models
    for inspection or further processing.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        ONNX ModelProto object.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    import onnx

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return onnx.load(str(model_path))


def get_model_metadata(model_path: str | Path) -> dict[str, str]:
    """Extract metadata from an exported ONNX model.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        Dictionary of metadata key-value pairs.
    """
    onnx_model = load_onnx_model(model_path)
    return {prop.key: prop.value for prop in onnx_model.metadata_props}


def check_model(model_path: str | Path) -> bool:
    """Validate an ONNX model file.

    Performs ONNX's built-in model validation to check
    for structural and semantic errors.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        True if model is valid.

    Raises:
        onnx.checker.ValidationError: If model is invalid.
        FileNotFoundError: If model file doesn't exist.
    """
    import onnx

    onnx_model = load_onnx_model(model_path)
    onnx.checker.check_model(onnx_model)
    return True


def optimize_model(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Apply ONNX graph optimizations to a model.

    Uses ONNX Runtime's session optimization to apply graph simplification,
    constant folding, and operator fusion. The optimized model is saved
    with the full optimization level applied.

    Args:
        input_path: Path to input ONNX model.
        output_path: Path to save optimized model. If not provided,
            saves to input_path with "_optimized" suffix.

    Returns:
        Path to optimized model.
    """
    import onnxruntime as ort

    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_optimized")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ONNX Runtime session to optimize and save
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(output_path)

    # Creating the session triggers optimization and saves the model
    ort.InferenceSession(str(input_path), sess_options)

    return output_path
