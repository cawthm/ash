"""Tests for the onnx_export module."""

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from models.architectures.price_transformer import PriceTransformer, TransformerConfig
from models.export.onnx_export import (
    ExportConfig,
    ExportResult,
    ONNXExporter,
    check_model,
    export_model,
    get_model_metadata,
    load_onnx_model,
    optimize_model,
)


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ExportConfig()
        assert config.opset_version == 17
        assert config.do_constant_folding is True
        assert config.dynamic_axes is True
        assert config.input_names == ("features",)
        assert config.output_prefix == "logits"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ExportConfig(
            opset_version=14,
            do_constant_folding=False,
            dynamic_axes=False,
            input_names=("input",),
            output_prefix="output",
        )
        assert config.opset_version == 14
        assert config.do_constant_folding is False
        assert config.dynamic_axes is False
        assert config.input_names == ("input",)
        assert config.output_prefix == "output"

    def test_opset_version_too_low(self) -> None:
        """Test that opset_version must be at least 11."""
        with pytest.raises(ValueError, match="opset_version must be at least 11"):
            ExportConfig(opset_version=10)

    def test_opset_version_too_high(self) -> None:
        """Test that opset_version must be at most 21."""
        with pytest.raises(ValueError, match="opset_version must be at most 21"):
            ExportConfig(opset_version=22)

    def test_empty_input_names(self) -> None:
        """Test that input_names must have at least one element."""
        with pytest.raises(ValueError, match="input_names must have at least one"):
            ExportConfig(input_names=())


class TestONNXExporter:
    """Tests for ONNXExporter class."""

    @pytest.fixture
    def model(self) -> PriceTransformer:
        """Create a small test model."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1, 5),
            num_buckets=11,
        )
        return PriceTransformer(config)

    @pytest.fixture
    def temp_dir(self) -> tempfile.TemporaryDirectory[str]:
        """Create a temporary directory for test outputs."""
        return tempfile.TemporaryDirectory()

    def test_export_creates_file(
        self, model: PriceTransformer, temp_dir: tempfile.TemporaryDirectory[str]
    ) -> None:
        """Test that export creates an ONNX file."""
        output_path = Path(temp_dir.name) / "model.onnx"

        exporter = ONNXExporter()
        result = exporter.export(model, output_path)

        assert output_path.exists()
        assert result.output_path == output_path
        assert result.model_size_bytes > 0

    def test_export_result_metadata(
        self, model: PriceTransformer, temp_dir: tempfile.TemporaryDirectory[str]
    ) -> None:
        """Test that export result contains correct metadata."""
        output_path = Path(temp_dir.name) / "model.onnx"

        exporter = ONNXExporter()
        result = exporter.export(model, output_path)

        assert result.input_names == ["features"]
        assert result.output_names == ["logits_1s", "logits_5s"]
        assert result.horizons == (1, 5)
        assert result.opset_version == 17

    def test_export_with_custom_config(
        self, model: PriceTransformer, temp_dir: tempfile.TemporaryDirectory[str]
    ) -> None:
        """Test export with custom configuration."""
        output_path = Path(temp_dir.name) / "model.onnx"

        export_config = ExportConfig(
            opset_version=14,
            input_names=("input_features",),
            output_prefix="pred",
        )
        exporter = ONNXExporter(export_config)
        result = exporter.export(model, output_path)

        assert result.input_names == ["input_features"]
        assert result.output_names == ["pred_1s", "pred_5s"]
        assert result.opset_version == 14

    def test_export_creates_parent_directories(
        self, model: PriceTransformer, temp_dir: tempfile.TemporaryDirectory[str]
    ) -> None:
        """Test that export creates parent directories if needed."""
        output_path = Path(temp_dir.name) / "subdir" / "nested" / "model.onnx"

        exporter = ONNXExporter()
        exporter.export(model, output_path)

        assert output_path.exists()

    def test_export_with_different_seq_len(
        self, model: PriceTransformer, temp_dir: tempfile.TemporaryDirectory[str]
    ) -> None:
        """Test export with custom sequence length."""
        output_path = Path(temp_dir.name) / "model.onnx"

        exporter = ONNXExporter()
        result = exporter.export(model, output_path, sample_seq_len=50)

        assert result.output_path.exists()


class TestExportValidation:
    """Tests for validating exported ONNX models."""

    @pytest.fixture
    def model(self) -> PriceTransformer:
        """Create a small test model."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1, 5, 10),
            num_buckets=11,
        )
        return PriceTransformer(config)

    @pytest.fixture
    def exported_model(
        self, model: PriceTransformer
    ) -> tuple[Path, PriceTransformer, tempfile.TemporaryDirectory[str]]:
        """Export model and return paths."""
        temp_dir = tempfile.TemporaryDirectory()
        output_path = Path(temp_dir.name) / "model.onnx"

        export_model(model, output_path)

        return output_path, model, temp_dir

    def test_check_model_valid(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test that check_model passes for valid model."""
        model_path, _, temp_dir = exported_model
        assert check_model(model_path)
        temp_dir.cleanup()

    def test_load_onnx_model(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test loading an ONNX model."""
        model_path, _, temp_dir = exported_model
        onnx_model = load_onnx_model(model_path)

        assert isinstance(onnx_model, onnx.ModelProto)
        temp_dir.cleanup()

    def test_load_nonexistent_model(self) -> None:
        """Test loading a nonexistent model raises error."""
        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_onnx_model("/nonexistent/path/model.onnx")

    def test_onnx_runtime_can_load(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test that ONNX Runtime can load the exported model."""
        model_path, _, temp_dir = exported_model
        session = ort.InferenceSession(str(model_path))

        # Check inputs
        inputs = session.get_inputs()
        assert len(inputs) == 1
        assert inputs[0].name == "features"

        # Check outputs
        outputs = session.get_outputs()
        assert len(outputs) == 3  # 3 horizons
        output_names = [o.name for o in outputs]
        assert "logits_1s" in output_names
        assert "logits_5s" in output_names
        assert "logits_10s" in output_names

        temp_dir.cleanup()


class TestPyTorchONNXParity:
    """Tests to verify PyTorch and ONNX outputs match."""

    @pytest.fixture
    def model(self) -> PriceTransformer:
        """Create a small test model."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1, 5),
            num_buckets=11,
        )
        model = PriceTransformer(config)
        model.eval()
        return model

    @pytest.fixture
    def exported_model(
        self, model: PriceTransformer
    ) -> tuple[Path, PriceTransformer, tempfile.TemporaryDirectory[str]]:
        """Export model and return paths."""
        temp_dir = tempfile.TemporaryDirectory()
        output_path = Path(temp_dir.name) / "model.onnx"

        export_model(model, output_path)

        return output_path, model, temp_dir

    def test_output_parity_single_sample(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test that ONNX output matches PyTorch for single sample."""
        model_path, pytorch_model, temp_dir = exported_model

        # Create test input
        test_input = torch.randn(1, 50, 16)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)

        # ONNX inference
        session = ort.InferenceSession(str(model_path))
        onnx_output = session.run(
            None, {"features": test_input.numpy()}
        )

        # Compare outputs
        for i, horizon in enumerate([1, 5]):
            pytorch_logits = pytorch_output[horizon].numpy()
            onnx_logits = onnx_output[i]

            # Note: Manual attention implementation has small numerical differences
            # from PyTorch's fused implementation, so we use relaxed tolerances
            np.testing.assert_allclose(
                pytorch_logits,
                onnx_logits,
                rtol=1e-2,
                atol=1e-3,
                err_msg=f"Mismatch at horizon {horizon}",
            )

        temp_dir.cleanup()

    def test_output_parity_batch(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test that ONNX output matches PyTorch for batch input."""
        model_path, pytorch_model, temp_dir = exported_model

        # Create batch test input
        test_input = torch.randn(4, 50, 16)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)

        # ONNX inference
        session = ort.InferenceSession(str(model_path))
        onnx_output = session.run(
            None, {"features": test_input.numpy()}
        )

        # Compare outputs
        for i, horizon in enumerate([1, 5]):
            pytorch_logits = pytorch_output[horizon].numpy()
            onnx_logits = onnx_output[i]

            # Note: Manual attention implementation has small numerical differences
            # from PyTorch's fused implementation, so we use relaxed tolerances
            np.testing.assert_allclose(
                pytorch_logits,
                onnx_logits,
                rtol=1e-2,
                atol=1e-3,
                err_msg=f"Mismatch at horizon {horizon}",
            )

        temp_dir.cleanup()

    def test_output_parity_dynamic_seq_len(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test parity with different sequence lengths (dynamic axes)."""
        model_path, pytorch_model, temp_dir = exported_model

        for seq_len in [10, 30, 100]:
            test_input = torch.randn(2, seq_len, 16)

            # PyTorch inference
            with torch.no_grad():
                pytorch_output = pytorch_model(test_input)

            # ONNX inference
            session = ort.InferenceSession(str(model_path))
            onnx_output = session.run(
                None, {"features": test_input.numpy()}
            )

            # Compare outputs
            for i, horizon in enumerate([1, 5]):
                pytorch_logits = pytorch_output[horizon].numpy()
                onnx_logits = onnx_output[i]

                # Note: Manual attention implementation has small numerical differences
                # from PyTorch's fused implementation, so we use relaxed tolerances
                np.testing.assert_allclose(
                    pytorch_logits,
                    onnx_logits,
                    rtol=1e-2,
                    atol=1e-3,
                    err_msg=f"Mismatch at horizon {horizon}, seq_len {seq_len}",
                )

        temp_dir.cleanup()

    def test_probability_parity(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test that softmax probabilities match."""
        model_path, pytorch_model, temp_dir = exported_model

        test_input = torch.randn(4, 50, 16)

        # PyTorch probabilities
        with torch.no_grad():
            pytorch_probs = pytorch_model.get_probabilities(test_input)

        # ONNX logits
        session = ort.InferenceSession(str(model_path))
        onnx_logits = session.run(None, {"features": test_input.numpy()})

        # Apply softmax to ONNX output (inline to avoid type annotation complexity)
        def apply_softmax(x: np.ndarray[tuple[int, ...], np.dtype[np.float32]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            result: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
            return result

        for i, horizon in enumerate([1, 5]):
            pytorch_prob = pytorch_probs[horizon].numpy()
            onnx_prob = apply_softmax(onnx_logits[i])

            # Probabilities should be close even with relaxed tolerances
            # since softmax normalizes the differences
            np.testing.assert_allclose(
                pytorch_prob,
                onnx_prob,
                rtol=1e-3,
                atol=1e-4,
                err_msg=f"Probability mismatch at horizon {horizon}",
            )

        temp_dir.cleanup()


class TestExportWithMetadata:
    """Tests for exporting with metadata."""

    @pytest.fixture
    def model(self) -> PriceTransformer:
        """Create a small test model."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1, 5),
            num_buckets=11,
            attention_type="causal",
        )
        return PriceTransformer(config)

    def test_export_with_metadata(self, model: PriceTransformer) -> None:
        """Test exporting with custom metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"

            exporter = ONNXExporter()
            exporter.export_with_metadata(
                model,
                output_path,
                metadata={"version": "1.0.0", "author": "test"},
            )

            # Verify metadata
            metadata = get_model_metadata(output_path)
            assert metadata.get("version") == "1.0.0"
            assert metadata.get("author") == "test"

    def test_config_metadata_embedded(self, model: PriceTransformer) -> None:
        """Test that model config is embedded as metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"

            exporter = ONNXExporter()
            exporter.export_with_metadata(model, output_path)

            metadata = get_model_metadata(output_path)

            # Check config values
            assert metadata.get("config.input_dim") == "16"
            assert metadata.get("config.embedding_dim") == "32"
            assert metadata.get("config.num_layers") == "1"
            assert metadata.get("config.num_heads") == "2"
            assert metadata.get("config.num_buckets") == "11"
            assert metadata.get("config.horizons") == "1,5"
            assert metadata.get("config.attention_type") == "causal"


class TestOptimizeModel:
    """Tests for model optimization."""

    @pytest.fixture
    def model(self) -> PriceTransformer:
        """Create a small test model."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1,),
            num_buckets=11,
        )
        return PriceTransformer(config)

    def test_optimize_creates_file(self, model: PriceTransformer) -> None:
        """Test that optimization creates an output file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "model.onnx"
            output_path = Path(temp_dir) / "model_opt.onnx"

            export_model(model, input_path)
            result_path = optimize_model(input_path, output_path)

            assert result_path == output_path
            assert output_path.exists()

    def test_optimize_default_output_path(self, model: PriceTransformer) -> None:
        """Test optimization with default output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "model.onnx"

            export_model(model, input_path)
            result_path = optimize_model(input_path)

            expected_path = Path(temp_dir) / "model_optimized.onnx"
            assert result_path == expected_path
            assert expected_path.exists()

    def test_optimized_model_works(self, model: PriceTransformer) -> None:
        """Test that optimized model produces correct outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "model.onnx"
            output_path = Path(temp_dir) / "model_opt.onnx"

            export_model(model, input_path)
            optimize_model(input_path, output_path)

            # Run inference on optimized model
            test_input = torch.randn(2, 30, 16)
            session = ort.InferenceSession(str(output_path))
            outputs = session.run(None, {"features": test_input.numpy()})

            # Should have output for horizon 1
            assert len(outputs) == 1
            assert outputs[0].shape == (2, 11)


class TestExportModelFunction:
    """Tests for the export_model convenience function."""

    def test_export_model_basic(self) -> None:
        """Test basic export_model usage."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1, 5),
            num_buckets=11,
        )
        model = PriceTransformer(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            result = export_model(model, output_path)

            assert isinstance(result, ExportResult)
            assert output_path.exists()

    def test_export_model_with_config(self) -> None:
        """Test export_model with custom export config."""
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

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_config = ExportConfig(opset_version=14)
            result = export_model(model, output_path, config=export_config)

            assert result.opset_version == 14


class TestDifferentModelConfigurations:
    """Tests for exporting models with different configurations."""

    def test_export_bidirectional_attention(self) -> None:
        """Test export with bidirectional attention."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1,),
            num_buckets=11,
            attention_type="bidirectional",
        )
        model = PriceTransformer(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            result = export_model(model, output_path)

            assert result.output_path.exists()

            # Verify it runs
            session = ort.InferenceSession(str(output_path))
            test_input = np.random.randn(1, 20, 16).astype(np.float32)
            outputs = session.run(None, {"features": test_input})
            assert outputs[0].shape == (1, 11)

    def test_export_without_cls_token(self) -> None:
        """Test export without CLS token (mean pooling)."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1,),
            num_buckets=11,
            use_cls_token=False,
        )
        model = PriceTransformer(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            result = export_model(model, output_path)

            assert result.output_path.exists()

            # Verify it runs
            session = ort.InferenceSession(str(output_path))
            test_input = np.random.randn(1, 20, 16).astype(np.float32)
            outputs = session.run(None, {"features": test_input})
            assert outputs[0].shape == (1, 11)

    def test_export_many_horizons(self) -> None:
        """Test export with many prediction horizons."""
        horizons = (1, 5, 10, 30, 60, 120, 300, 600)
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=horizons,
            num_buckets=11,
        )
        model = PriceTransformer(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            result = export_model(model, output_path)

            assert len(result.output_names) == 8
            expected_names = [f"logits_{h}s" for h in horizons]
            assert result.output_names == expected_names

    def test_export_larger_model(self) -> None:
        """Test export with larger model configuration."""
        config = TransformerConfig(
            input_dim=64,
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            ff_dim=512,
            horizons=(1, 5, 10),
            num_buckets=101,
        )
        model = PriceTransformer(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            result = export_model(model, output_path)

            assert result.output_path.exists()
            assert result.model_size_bytes > 100_000  # Should be reasonably sized


class TestEdgeCases:
    """Tests for edge cases in ONNX export."""

    def test_export_with_string_path(self) -> None:
        """Test export accepts string path."""
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

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = f"{temp_dir}/model.onnx"
            result = export_model(model, output_path)

            assert Path(output_path).exists()
            assert result.output_path == Path(output_path)

    def test_export_single_horizon(self) -> None:
        """Test export with single horizon."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(60,),  # Single horizon
            num_buckets=11,
        )
        model = PriceTransformer(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            result = export_model(model, output_path)

            assert result.output_names == ["logits_60s"]

    def test_export_preserves_eval_mode(self) -> None:
        """Test that export puts model in eval mode."""
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
        model.train()  # Start in train mode

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            # Model should now be in eval mode
            assert not model.training
