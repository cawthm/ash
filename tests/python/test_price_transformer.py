"""Tests for the price_transformer module."""

import pytest
import torch

from models.architectures.price_transformer import (
    FeatureEmbedding,
    MultiHorizonHead,
    PositionalEncoding,
    PriceTransformer,
    TransformerConfig,
    create_model,
    create_model_from_dict,
)


class TestTransformerConfig:
    """Tests for TransformerConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TransformerConfig(input_dim=64)
        assert config.input_dim == 64
        assert config.embedding_dim == 128
        assert config.num_layers == 2
        assert config.num_heads == 4
        assert config.ff_dim == 512
        assert config.dropout == 0.1
        assert config.max_seq_len == 256
        assert config.horizons == (1, 5, 10, 30, 60, 120, 300, 600)
        assert config.num_buckets == 101
        assert config.use_cls_token is True
        assert config.attention_type == "causal"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TransformerConfig(
            input_dim=32,
            embedding_dim=256,
            num_layers=4,
            num_heads=8,
            ff_dim=1024,
            dropout=0.2,
            max_seq_len=512,
            horizons=(1, 5, 10),
            num_buckets=51,
            use_cls_token=False,
            attention_type="bidirectional",
        )
        assert config.input_dim == 32
        assert config.embedding_dim == 256
        assert config.num_layers == 4
        assert config.num_heads == 8
        assert config.ff_dim == 1024
        assert config.dropout == 0.2
        assert config.max_seq_len == 512
        assert config.horizons == (1, 5, 10)
        assert config.num_buckets == 51
        assert config.use_cls_token is False
        assert config.attention_type == "bidirectional"

    def test_input_dim_too_small(self) -> None:
        """Test that input_dim must be at least 1."""
        with pytest.raises(ValueError, match="input_dim must be at least 1"):
            TransformerConfig(input_dim=0)

    def test_embedding_dim_too_small(self) -> None:
        """Test that embedding_dim must be at least 1."""
        with pytest.raises(ValueError, match="embedding_dim must be at least 1"):
            TransformerConfig(input_dim=64, embedding_dim=0)

    def test_num_layers_too_small(self) -> None:
        """Test that num_layers must be at least 1."""
        with pytest.raises(ValueError, match="num_layers must be at least 1"):
            TransformerConfig(input_dim=64, num_layers=0)

    def test_num_heads_too_small(self) -> None:
        """Test that num_heads must be at least 1."""
        with pytest.raises(ValueError, match="num_heads must be at least 1"):
            TransformerConfig(input_dim=64, num_heads=0)

    def test_embedding_dim_not_divisible_by_heads(self) -> None:
        """Test that embedding_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="embedding_dim.*must be divisible by"):
            TransformerConfig(input_dim=64, embedding_dim=128, num_heads=3)

    def test_ff_dim_too_small(self) -> None:
        """Test that ff_dim must be at least 1."""
        with pytest.raises(ValueError, match="ff_dim must be at least 1"):
            TransformerConfig(input_dim=64, ff_dim=0)

    def test_dropout_invalid(self) -> None:
        """Test that dropout must be in [0.0, 1.0)."""
        with pytest.raises(ValueError, match="dropout must be in"):
            TransformerConfig(input_dim=64, dropout=1.0)
        with pytest.raises(ValueError, match="dropout must be in"):
            TransformerConfig(input_dim=64, dropout=-0.1)

    def test_max_seq_len_too_small(self) -> None:
        """Test that max_seq_len must be at least 1."""
        with pytest.raises(ValueError, match="max_seq_len must be at least 1"):
            TransformerConfig(input_dim=64, max_seq_len=0)

    def test_horizons_empty(self) -> None:
        """Test that horizons must have at least one element."""
        with pytest.raises(ValueError, match="horizons must have at least one"):
            TransformerConfig(input_dim=64, horizons=())

    def test_horizons_non_positive(self) -> None:
        """Test that all horizons must be positive."""
        with pytest.raises(ValueError, match="All horizons must be positive"):
            TransformerConfig(input_dim=64, horizons=(1, 5, 0))
        with pytest.raises(ValueError, match="All horizons must be positive"):
            TransformerConfig(input_dim=64, horizons=(1, -5, 10))

    def test_num_buckets_too_small(self) -> None:
        """Test that num_buckets must be at least 3."""
        with pytest.raises(ValueError, match="num_buckets must be at least 3"):
            TransformerConfig(input_dim=64, num_buckets=2)

    def test_invalid_attention_type(self) -> None:
        """Test that invalid attention_type raises error."""
        with pytest.raises(ValueError, match="Unknown attention_type"):
            TransformerConfig(
                input_dim=64, attention_type="invalid"  # type: ignore[arg-type]
            )


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input shape."""
        config = TransformerConfig(input_dim=64)
        pe = PositionalEncoding(config)

        x = torch.randn(4, 100, 128)  # (batch, seq_len, embed_dim)
        output = pe(x)

        assert output.shape == x.shape

    def test_adds_position_info(self) -> None:
        """Test that positional encoding modifies the input."""
        config = TransformerConfig(input_dim=64)
        pe = PositionalEncoding(config)

        x = torch.zeros(2, 50, 128)
        output = pe(x)

        # Output should not be all zeros
        assert not torch.allclose(output, x)

    def test_different_positions_have_different_encoding(self) -> None:
        """Test that different positions have different encodings."""
        config = TransformerConfig(input_dim=64)
        pe = PositionalEncoding(config)

        x = torch.zeros(1, 10, 128)
        output = pe(x)

        # Each position should be different
        for i in range(9):
            assert not torch.allclose(output[0, i], output[0, i + 1])

    def test_accounts_for_cls_token(self) -> None:
        """Test that positional encoding accounts for CLS token."""
        config_with_cls = TransformerConfig(input_dim=64, use_cls_token=True)
        config_without_cls = TransformerConfig(input_dim=64, use_cls_token=False)

        pe_with_cls = PositionalEncoding(config_with_cls)
        pe_without_cls = PositionalEncoding(config_without_cls)

        # With CLS token, should have one more position
        assert pe_with_cls.position_embedding.num_embeddings == (
            pe_without_cls.position_embedding.num_embeddings + 1
        )

    def test_batch_consistency(self) -> None:
        """Test that same positions get same encoding across batch."""
        config = TransformerConfig(input_dim=64)
        pe = PositionalEncoding(config)

        x = torch.zeros(4, 20, 128)
        output = pe(x)

        # Same positions should have same encoding
        for i in range(4):
            assert torch.allclose(output[0], output[i])


class TestFeatureEmbedding:
    """Tests for FeatureEmbedding module."""

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        config = TransformerConfig(input_dim=64, embedding_dim=128)
        embed = FeatureEmbedding(config)

        x = torch.randn(4, 100, 64)  # (batch, seq_len, input_dim)
        output = embed(x)

        assert output.shape == (4, 100, 128)

    def test_projection_works(self) -> None:
        """Test that projection changes the dimension."""
        config = TransformerConfig(input_dim=32, embedding_dim=64)
        embed = FeatureEmbedding(config)

        x = torch.randn(2, 50, 32)
        output = embed(x)

        assert output.shape == (2, 50, 64)

    def test_layer_norm_applied(self) -> None:
        """Test that layer normalization is applied."""
        config = TransformerConfig(input_dim=64, embedding_dim=128)
        embed = FeatureEmbedding(config)

        x = torch.randn(4, 100, 64) * 10  # Large values
        output = embed(x)

        # After layer norm, values should be more normalized
        # Check mean and std are reasonable per position
        output_std = output.std(dim=-1)
        assert output_std.mean() < 5  # Should be normalized


class TestMultiHorizonHead:
    """Tests for MultiHorizonHead module."""

    def test_output_structure(self) -> None:
        """Test that output is a dict with correct keys."""
        config = TransformerConfig(input_dim=64, horizons=(1, 5, 10, 30))
        head = MultiHorizonHead(config)

        x = torch.randn(4, 128)  # (batch, embedding_dim)
        output = head(x)

        assert isinstance(output, dict)
        assert set(output.keys()) == {1, 5, 10, 30}

    def test_output_shapes(self) -> None:
        """Test that each horizon output has correct shape."""
        config = TransformerConfig(input_dim=64, horizons=(1, 5, 10), num_buckets=51)
        head = MultiHorizonHead(config)

        x = torch.randn(8, 128)
        output = head(x)

        for horizon in (1, 5, 10):
            assert output[horizon].shape == (8, 51)

    def test_heads_are_independent(self) -> None:
        """Test that different horizons produce different outputs."""
        config = TransformerConfig(input_dim=64, horizons=(1, 5, 10))
        head = MultiHorizonHead(config)

        x = torch.randn(4, 128)
        output = head(x)

        # Different horizons should produce different outputs
        # (unless by extreme chance)
        assert not torch.allclose(output[1], output[5])
        assert not torch.allclose(output[5], output[10])


class TestPriceTransformer:
    """Tests for PriceTransformer main model."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        config = TransformerConfig(input_dim=64)
        model = PriceTransformer(config)

        assert model.config == config
        assert model.num_horizons == len(config.horizons)

    def test_forward_pass_shape(self) -> None:
        """Test forward pass output shapes."""
        config = TransformerConfig(
            input_dim=64, horizons=(1, 5, 10), num_buckets=51
        )
        model = PriceTransformer(config)

        x = torch.randn(4, 100, 64)  # (batch, seq_len, input_dim)
        output = model(x)

        assert isinstance(output, dict)
        assert len(output) == 3
        for horizon in (1, 5, 10):
            assert output[horizon].shape == (4, 51)

    def test_cls_token_integration(self) -> None:
        """Test that CLS token is properly integrated."""
        config = TransformerConfig(input_dim=64, use_cls_token=True)
        model = PriceTransformer(config)

        # CLS token should be a learnable parameter
        assert model.cls_token is not None
        assert model.cls_token.shape == (1, 1, 128)
        assert model.cls_token.requires_grad

    def test_no_cls_token(self) -> None:
        """Test model without CLS token uses mean pooling."""
        config = TransformerConfig(input_dim=64, use_cls_token=False)
        model = PriceTransformer(config)

        assert model.cls_token is None

        x = torch.randn(4, 50, 64)
        output = model(x)

        # Should still produce valid output
        assert len(output) == len(config.horizons)

    def test_causal_attention(self) -> None:
        """Test causal attention mask creation without CLS token."""
        config = TransformerConfig(
            input_dim=64, attention_type="causal", use_cls_token=False
        )
        model = PriceTransformer(config)

        mask = model._create_attention_mask(10, torch.device("cpu"))

        assert mask is not None
        # Upper triangular should be True (masked)
        assert mask[0, 1]  # Position 0 cannot attend to position 1
        assert mask[5, 9]  # Position 5 cannot attend to position 9
        # Lower triangular should be False (allowed)
        assert not mask[5, 3]  # Position 5 can attend to position 3
        # Diagonal is allowed (self-attention)
        assert not mask[3, 3]

    def test_bidirectional_attention(self) -> None:
        """Test bidirectional attention has no mask."""
        config = TransformerConfig(input_dim=64, attention_type="bidirectional")
        model = PriceTransformer(config)

        mask = model._create_attention_mask(10, torch.device("cpu"))
        assert mask is None

    def test_causal_mask_cls_token(self) -> None:
        """Test CLS token can attend to all positions in causal mode."""
        config = TransformerConfig(
            input_dim=64, attention_type="causal", use_cls_token=True
        )
        model = PriceTransformer(config)

        mask = model._create_attention_mask(10, torch.device("cpu"))

        assert mask is not None
        # CLS at position 0 should attend to everything
        assert not mask[0, :].any()  # All False = can attend to all

    def test_padding_mask(self) -> None:
        """Test that padding mask is handled correctly."""
        config = TransformerConfig(input_dim=64)
        model = PriceTransformer(config)

        batch_size = 4
        seq_len = 50
        x = torch.randn(batch_size, seq_len, 64)

        # Create padding mask (last 10 positions are padded)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        padding_mask[:, -10:] = True

        output = model(x, padding_mask=padding_mask)

        # Should produce valid output
        for horizon in config.horizons:
            assert output[horizon].shape == (batch_size, 101)
            assert not torch.isnan(output[horizon]).any()

    def test_get_probabilities(self) -> None:
        """Test probability output method."""
        config = TransformerConfig(input_dim=64, horizons=(1, 5))
        model = PriceTransformer(config)

        x = torch.randn(4, 50, 64)
        probs = model.get_probabilities(x)

        for horizon in (1, 5):
            p = probs[horizon]
            # Should be valid probabilities
            assert (p >= 0).all()
            assert (p <= 1).all()
            # Should sum to 1
            assert torch.allclose(p.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_num_parameters(self) -> None:
        """Test num_parameters property."""
        config = TransformerConfig(input_dim=64)
        model = PriceTransformer(config)

        num_params = model.num_parameters
        assert isinstance(num_params, int)
        assert num_params > 0

        # Verify count manually
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params == manual_count

    def test_num_horizons(self) -> None:
        """Test num_horizons property."""
        config = TransformerConfig(input_dim=64, horizons=(1, 5, 10, 30))
        model = PriceTransformer(config)

        assert model.num_horizons == 4


class TestCreateModel:
    """Tests for create_model factory function."""

    def test_creates_model(self) -> None:
        """Test that factory creates a model."""
        config = TransformerConfig(input_dim=64)
        model = create_model(config)

        assert isinstance(model, PriceTransformer)
        assert model.config == config

    def test_model_is_functional(self) -> None:
        """Test that created model can do forward pass."""
        config = TransformerConfig(input_dim=32)
        model = create_model(config)

        x = torch.randn(2, 20, 32)
        output = model(x)

        assert len(output) == len(config.horizons)


class TestCreateModelFromDict:
    """Tests for create_model_from_dict factory function."""

    def test_creates_model_from_dict(self) -> None:
        """Test creating model from dictionary."""
        config_dict = {
            "input_dim": 64,
            "embedding_dim": 128,
            "num_layers": 2,
        }
        model = create_model_from_dict(config_dict)

        assert isinstance(model, PriceTransformer)
        assert model.config.input_dim == 64
        assert model.config.embedding_dim == 128
        assert model.config.num_layers == 2

    def test_handles_horizons_as_list(self) -> None:
        """Test that horizons can be provided as list (from YAML)."""
        config_dict = {
            "input_dim": 64,
            "horizons": [1, 5, 10],  # List, not tuple
        }
        model = create_model_from_dict(config_dict)

        assert model.config.horizons == (1, 5, 10)

    def test_does_not_mutate_input(self) -> None:
        """Test that input dict is not mutated."""
        config_dict = {
            "input_dim": 64,
            "horizons": [1, 5, 10],
        }
        original_horizons = config_dict["horizons"]

        create_model_from_dict(config_dict)

        # Original dict should not be changed
        assert config_dict["horizons"] is original_horizons
        assert config_dict["horizons"] == [1, 5, 10]


class TestGradientFlow:
    """Tests for gradient flow through the model."""

    def test_basic_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        config = TransformerConfig(input_dim=64, horizons=(1, 5))
        model = PriceTransformer(config)

        x = torch.randn(4, 50, 64, requires_grad=True)
        output = model(x)

        # Sum all outputs and backprop
        loss = sum(o.sum() for o in output.values())
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flow_with_padding(self) -> None:
        """Test gradient flow with padding mask."""
        config = TransformerConfig(input_dim=64, horizons=(1, 5))
        model = PriceTransformer(config)

        x = torch.randn(4, 50, 64, requires_grad=True)
        padding_mask = torch.zeros(4, 50, dtype=torch.bool)
        padding_mask[:, -10:] = True

        output = model(x, padding_mask=padding_mask)

        loss = sum(o.sum() for o in output.values())
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flow_no_cls(self) -> None:
        """Test gradient flow without CLS token."""
        config = TransformerConfig(input_dim=64, horizons=(1,), use_cls_token=False)
        model = PriceTransformer(config)

        x = torch.randn(4, 50, 64, requires_grad=True)
        output = model(x)

        loss = output[1].sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestNumericalStability:
    """Tests for numerical stability of the model."""

    def test_large_input_values(self) -> None:
        """Test model with large input values."""
        config = TransformerConfig(input_dim=64, horizons=(1,))
        model = PriceTransformer(config)

        x = torch.randn(4, 50, 64) * 100
        output = model(x)

        assert not torch.isnan(output[1]).any()
        assert not torch.isinf(output[1]).any()

    def test_small_input_values(self) -> None:
        """Test model with small input values."""
        config = TransformerConfig(input_dim=64, horizons=(1,))
        model = PriceTransformer(config)

        x = torch.randn(4, 50, 64) * 1e-6
        output = model(x)

        assert not torch.isnan(output[1]).any()
        assert not torch.isinf(output[1]).any()

    def test_zero_input(self) -> None:
        """Test model with zero input."""
        config = TransformerConfig(input_dim=64, horizons=(1,))
        model = PriceTransformer(config)

        x = torch.zeros(4, 50, 64)
        output = model(x)

        assert not torch.isnan(output[1]).any()
        assert not torch.isinf(output[1]).any()

    def test_probability_outputs_valid(self) -> None:
        """Test that probabilities are always valid."""
        config = TransformerConfig(input_dim=64, horizons=(1, 5, 10))
        model = PriceTransformer(config)

        # Test with various inputs
        for _ in range(5):
            x = torch.randn(4, 50, 64) * torch.randint(1, 100, (1,)).float()
            probs = model.get_probabilities(x)

            for horizon, p in probs.items():
                assert (p >= 0).all(), f"Negative probability at horizon {horizon}"
                assert (p <= 1).all(), f"Probability > 1 at horizon {horizon}"
                sums = p.sum(dim=-1)
                assert torch.allclose(
                    sums, torch.ones_like(sums), atol=1e-5
                ), f"Probabilities don't sum to 1 at horizon {horizon}"


class TestModelDimensions:
    """Tests for various model dimension configurations."""

    def test_small_model(self) -> None:
        """Test small model configuration."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1,),
            num_buckets=11,
        )
        model = PriceTransformer(config)

        x = torch.randn(2, 10, 16)
        output = model(x)

        assert output[1].shape == (2, 11)

    def test_larger_model(self) -> None:
        """Test larger model configuration."""
        config = TransformerConfig(
            input_dim=128,
            embedding_dim=256,
            num_layers=4,
            num_heads=8,
            ff_dim=1024,
            horizons=(1, 5, 10, 30, 60),
            num_buckets=201,
        )
        model = PriceTransformer(config)

        x = torch.randn(2, 100, 128)
        output = model(x)

        assert len(output) == 5
        for horizon in (1, 5, 10, 30, 60):
            assert output[horizon].shape == (2, 201)

    def test_single_head_attention(self) -> None:
        """Test model with single attention head."""
        config = TransformerConfig(
            input_dim=64, embedding_dim=64, num_heads=1  # Single head
        )
        model = PriceTransformer(config)

        x = torch.randn(4, 50, 64)
        output = model(x)

        assert len(output) == 8  # Default horizons

    def test_many_horizons(self) -> None:
        """Test model with many prediction horizons."""
        horizons = tuple(range(1, 21))  # 20 horizons
        config = TransformerConfig(input_dim=64, horizons=horizons)
        model = PriceTransformer(config)

        x = torch.randn(2, 50, 64)
        output = model(x)

        assert len(output) == 20
        for h in horizons:
            assert h in output


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_timestep(self) -> None:
        """Test with single timestep input."""
        config = TransformerConfig(input_dim=64, horizons=(1,))
        model = PriceTransformer(config)

        x = torch.randn(4, 1, 64)  # Single timestep
        output = model(x)

        assert output[1].shape == (4, 101)

    def test_batch_size_one(self) -> None:
        """Test with batch size of 1."""
        config = TransformerConfig(input_dim=64, horizons=(1,))
        model = PriceTransformer(config)

        x = torch.randn(1, 50, 64)
        output = model(x)

        assert output[1].shape == (1, 101)

    def test_max_sequence_length(self) -> None:
        """Test with max sequence length."""
        config = TransformerConfig(input_dim=64, max_seq_len=100, horizons=(1,))
        model = PriceTransformer(config)

        # Exactly max sequence length
        x = torch.randn(2, 100, 64)
        output = model(x)

        assert output[1].shape == (2, 101)

    def test_all_padded_except_one(self) -> None:
        """Test with almost all positions padded."""
        config = TransformerConfig(input_dim=64, horizons=(1,), use_cls_token=False)
        model = PriceTransformer(config)

        x = torch.randn(2, 50, 64)
        # Only first position is not padded
        padding_mask = torch.ones(2, 50, dtype=torch.bool)
        padding_mask[:, 0] = False

        output = model(x, padding_mask=padding_mask)

        assert not torch.isnan(output[1]).any()


class TestDeviceCompatibility:
    """Tests for device compatibility."""

    def test_cpu_inference(self) -> None:
        """Test inference on CPU."""
        config = TransformerConfig(input_dim=64, horizons=(1,))
        model = PriceTransformer(config)
        model.cpu()

        x = torch.randn(4, 50, 64)
        output = model(x)

        assert output[1].device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self) -> None:
        """Test inference on CUDA."""
        config = TransformerConfig(input_dim=64, horizons=(1,))
        model = PriceTransformer(config)
        model.cuda()

        x = torch.randn(4, 50, 64).cuda()
        output = model(x)

        assert output[1].device.type == "cuda"


class TestTrainEvalModes:
    """Tests for train and eval mode behavior."""

    def test_dropout_in_train_mode(self) -> None:
        """Test that dropout is applied in train mode."""
        config = TransformerConfig(input_dim=64, dropout=0.5, horizons=(1,))
        model = PriceTransformer(config)
        model.train()

        x = torch.randn(4, 50, 64)

        # Run multiple forward passes
        outputs = [model(x)[1] for _ in range(10)]

        # Outputs should vary due to dropout
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout not being applied in train mode"

    def test_deterministic_in_eval_mode(self) -> None:
        """Test that eval mode is deterministic."""
        config = TransformerConfig(input_dim=64, dropout=0.5, horizons=(1,))
        model = PriceTransformer(config)
        model.eval()

        x = torch.randn(4, 50, 64)

        # Run multiple forward passes
        with torch.no_grad():
            outputs = [model(x)[1] for _ in range(5)]

        # All outputs should be identical
        for i, o in enumerate(outputs[1:], 1):
            assert torch.allclose(outputs[0], o), f"Output {i} differs in eval mode"
