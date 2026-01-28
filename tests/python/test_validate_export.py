"""Tests for the validate_export module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from models.architectures.price_transformer import PriceTransformer, TransformerConfig
from models.export.onnx_export import export_model
from models.export.validate_export import (
    ExportValidator,
    HorizonValidationResult,
    ValidationConfig,
    ValidationResult,
    check_tolerance,
    compute_max_diff,
    format_validation_report,
    validate_export,
)


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ValidationConfig()
        assert config.rtol == 1e-2
        assert config.atol == 1e-3
        assert config.num_samples == 10
        assert config.batch_sizes == (1, 4, 8)
        assert config.seq_lengths is None
        assert config.check_probabilities is True
        assert config.probability_rtol == 1e-3
        assert config.probability_atol == 1e-4
        assert config.random_seed == 42

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ValidationConfig(
            rtol=1e-3,
            atol=1e-4,
            num_samples=5,
            batch_sizes=(2, 4),
            seq_lengths=(10, 20, 30),
            check_probabilities=False,
            random_seed=123,
        )
        assert config.rtol == 1e-3
        assert config.atol == 1e-4
        assert config.num_samples == 5
        assert config.batch_sizes == (2, 4)
        assert config.seq_lengths == (10, 20, 30)
        assert config.check_probabilities is False
        assert config.random_seed == 123

    def test_negative_rtol(self) -> None:
        """Test that negative rtol raises error."""
        with pytest.raises(ValueError, match="rtol must be non-negative"):
            ValidationConfig(rtol=-0.1)

    def test_negative_atol(self) -> None:
        """Test that negative atol raises error."""
        with pytest.raises(ValueError, match="atol must be non-negative"):
            ValidationConfig(atol=-0.1)

    def test_zero_num_samples(self) -> None:
        """Test that zero num_samples raises error."""
        with pytest.raises(ValueError, match="num_samples must be at least 1"):
            ValidationConfig(num_samples=0)

    def test_empty_batch_sizes(self) -> None:
        """Test that empty batch_sizes raises error."""
        with pytest.raises(ValueError, match="batch_sizes must have at least one"):
            ValidationConfig(batch_sizes=())

    def test_non_positive_batch_size(self) -> None:
        """Test that non-positive batch size raises error."""
        with pytest.raises(ValueError, match="All batch sizes must be positive"):
            ValidationConfig(batch_sizes=(1, 0, 2))

    def test_empty_seq_lengths(self) -> None:
        """Test that empty seq_lengths raises error."""
        with pytest.raises(ValueError, match="seq_lengths must have at least one"):
            ValidationConfig(seq_lengths=())

    def test_non_positive_seq_length(self) -> None:
        """Test that non-positive sequence length raises error."""
        with pytest.raises(ValueError, match="All sequence lengths must be positive"):
            ValidationConfig(seq_lengths=(10, 0, 20))

    def test_negative_probability_rtol(self) -> None:
        """Test that negative probability_rtol raises error."""
        with pytest.raises(ValueError, match="probability_rtol must be non-negative"):
            ValidationConfig(probability_rtol=-0.1)

    def test_negative_probability_atol(self) -> None:
        """Test that negative probability_atol raises error."""
        with pytest.raises(ValueError, match="probability_atol must be non-negative"):
            ValidationConfig(probability_atol=-0.1)

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable."""
        config = ValidationConfig()
        with pytest.raises(AttributeError):
            config.rtol = 0.5  # type: ignore[misc]


class TestHorizonValidationResult:
    """Tests for HorizonValidationResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a horizon result."""
        result = HorizonValidationResult(
            horizon=60,
            max_abs_diff=0.001,
            max_rel_diff=0.01,
            mean_abs_diff=0.0005,
            passed=True,
            failed_samples=0,
        )
        assert result.horizon == 60
        assert result.max_abs_diff == 0.001
        assert result.max_rel_diff == 0.01
        assert result.mean_abs_diff == 0.0005
        assert result.passed is True
        assert result.failed_samples == 0

    def test_failed_result(self) -> None:
        """Test creating a failed horizon result."""
        result = HorizonValidationResult(
            horizon=300,
            max_abs_diff=0.1,
            max_rel_diff=0.5,
            mean_abs_diff=0.05,
            passed=False,
            failed_samples=5,
        )
        assert result.passed is False
        assert result.failed_samples == 5


class TestExportValidator:
    """Tests for ExportValidator class."""

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
            max_seq_len=100,
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

    def test_default_validator(self) -> None:
        """Test creating validator with default config."""
        validator = ExportValidator()
        assert validator.config.rtol == 1e-2
        assert validator.config.atol == 1e-3

    def test_custom_validator(self) -> None:
        """Test creating validator with custom config."""
        config = ValidationConfig(rtol=1e-4, atol=1e-5)
        validator = ExportValidator(config)
        assert validator.config.rtol == 1e-4
        assert validator.config.atol == 1e-5

    def test_validate_passes(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test that validation passes for valid export."""
        model_path, pytorch_model, temp_dir = exported_model

        config = ValidationConfig(num_samples=2, batch_sizes=(1, 2))
        validator = ExportValidator(config)
        result = validator.validate(pytorch_model, model_path)

        assert result.passed is True
        assert result.model_path == model_path
        assert result.num_samples_tested > 0
        assert result.error_message is None
        assert len(result.horizon_results) == 2  # 2 horizons

        temp_dir.cleanup()

    def test_validate_nonexistent_file(self, model: PriceTransformer) -> None:
        """Test validation with nonexistent ONNX file."""
        validator = ExportValidator()
        result = validator.validate(model, "/nonexistent/path/model.onnx")

        assert result.passed is False
        assert result.num_samples_tested == 0
        assert result.error_message is not None
        assert "not found" in result.error_message

    def test_validate_with_probability_check(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test validation with probability checking enabled."""
        model_path, pytorch_model, temp_dir = exported_model

        config = ValidationConfig(
            num_samples=2,
            batch_sizes=(1,),
            check_probabilities=True,
        )
        validator = ExportValidator(config)
        result = validator.validate(pytorch_model, model_path)

        assert result.probability_validation_passed is not None
        assert result.probability_max_diff is not None
        assert result.probability_max_diff >= 0

        temp_dir.cleanup()

    def test_validate_without_probability_check(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test validation without probability checking."""
        model_path, pytorch_model, temp_dir = exported_model

        config = ValidationConfig(
            num_samples=2,
            batch_sizes=(1,),
            check_probabilities=False,
        )
        validator = ExportValidator(config)
        result = validator.validate(pytorch_model, model_path)

        assert result.probability_validation_passed is None
        assert result.probability_max_diff is None

        temp_dir.cleanup()

    def test_validate_single_input(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test validation with a single specific input."""
        model_path, pytorch_model, temp_dir = exported_model

        validator = ExportValidator()
        input_tensor = torch.randn(2, 30, 16)
        result = validator.validate_single_input(
            pytorch_model, model_path, input_tensor
        )

        assert result.passed is True
        assert result.num_samples_tested == 1

        temp_dir.cleanup()

    def test_validate_custom_seq_lengths(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test validation with custom sequence lengths."""
        model_path, pytorch_model, temp_dir = exported_model

        config = ValidationConfig(
            num_samples=1,
            batch_sizes=(1,),
            seq_lengths=(15, 30),
        )
        validator = ExportValidator(config)
        result = validator.validate(pytorch_model, model_path)

        assert result.passed is True
        # Should have tested 1 batch_size * 2 seq_lengths * 1 sample = 2
        assert result.num_samples_tested == 2

        temp_dir.cleanup()

    def test_validate_reproducibility(
        self,
        exported_model: tuple[
            Path, PriceTransformer, tempfile.TemporaryDirectory[str]
        ],
    ) -> None:
        """Test that validation with seed is reproducible."""
        model_path, pytorch_model, temp_dir = exported_model

        config = ValidationConfig(
            num_samples=3,
            batch_sizes=(1,),
            random_seed=42,
        )
        validator = ExportValidator(config)

        result1 = validator.validate(pytorch_model, model_path)
        result2 = validator.validate(pytorch_model, model_path)

        # Both should pass
        assert result1.passed is True
        assert result2.passed is True

        # Same seed should give same max diffs
        for hr1, hr2 in zip(
            result1.horizon_results, result2.horizon_results, strict=True
        ):
            assert hr1.max_abs_diff == hr2.max_abs_diff

        temp_dir.cleanup()


class TestValidateExportFunction:
    """Tests for the validate_export convenience function."""

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
        model = PriceTransformer(config)
        model.eval()
        return model

    def test_validate_export_basic(self, model: PriceTransformer) -> None:
        """Test basic validate_export usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(model, output_path)

            assert isinstance(result, ValidationResult)
            assert result.passed is True

    def test_validate_export_with_config(self, model: PriceTransformer) -> None:
        """Test validate_export with custom config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            config = ValidationConfig(num_samples=2, batch_sizes=(1,))
            result = validate_export(model, output_path, config)

            assert result.passed is True
            assert result.config == config

    def test_validate_export_with_string_path(self, model: PriceTransformer) -> None:
        """Test validate_export accepts string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = f"{temp_dir}/model.onnx"
            export_model(model, output_path)

            result = validate_export(model, output_path)

            assert result.passed is True
            assert result.model_path == Path(output_path)


class TestComputeMaxDiff:
    """Tests for compute_max_diff utility function."""

    def test_compute_max_diff_zero(self) -> None:
        """Test max diff is zero for identical outputs."""
        horizons = (1, 5)
        pytorch_output = {
            1: torch.tensor([[1.0, 2.0, 3.0]]),
            5: torch.tensor([[4.0, 5.0, 6.0]]),
        }
        onnx_output = [
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
        ]

        result = compute_max_diff(pytorch_output, onnx_output, horizons)

        assert result[1] == pytest.approx(0.0, abs=1e-6)
        assert result[5] == pytest.approx(0.0, abs=1e-6)

    def test_compute_max_diff_nonzero(self) -> None:
        """Test max diff for different outputs."""
        horizons = (1, 5)
        pytorch_output = {
            1: torch.tensor([[1.0, 2.0, 3.0]]),
            5: torch.tensor([[4.0, 5.0, 6.0]]),
        }
        onnx_output = [
            np.array([[1.1, 2.0, 3.0]], dtype=np.float32),  # diff 0.1
            np.array([[4.0, 5.5, 6.0]], dtype=np.float32),  # diff 0.5
        ]

        result = compute_max_diff(pytorch_output, onnx_output, horizons)

        assert result[1] == pytest.approx(0.1, abs=1e-5)
        assert result[5] == pytest.approx(0.5, abs=1e-5)

    def test_compute_max_diff_batch(self) -> None:
        """Test max diff with batch dimension."""
        horizons = (60,)
        pytorch_output = {
            60: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
        onnx_output = [
            np.array([[1.2, 2.0], [3.0, 4.3]], dtype=np.float32),  # max diff 0.3
        ]

        result = compute_max_diff(pytorch_output, onnx_output, horizons)

        assert result[60] == pytest.approx(0.3, abs=1e-5)


class TestCheckTolerance:
    """Tests for check_tolerance utility function."""

    def test_all_pass(self) -> None:
        """Test when all horizons pass tolerance."""
        max_diffs = {1: 0.0001, 5: 0.0005, 10: 0.001}
        passed, failed = check_tolerance(max_diffs, atol=0.01)

        assert passed is True
        assert failed == []

    def test_some_fail(self) -> None:
        """Test when some horizons fail tolerance."""
        max_diffs = {1: 0.0001, 5: 0.05, 10: 0.001}  # 5 fails
        passed, failed = check_tolerance(max_diffs, atol=0.01)

        assert passed is False
        assert failed == [5]

    def test_all_fail(self) -> None:
        """Test when all horizons fail tolerance."""
        max_diffs = {1: 0.1, 5: 0.2, 10: 0.3}
        passed, failed = check_tolerance(max_diffs, atol=0.01)

        assert passed is False
        assert set(failed) == {1, 5, 10}

    def test_exact_tolerance_boundary(self) -> None:
        """Test behavior at exact tolerance boundary."""
        max_diffs = {1: 0.01}  # Exactly at tolerance
        passed, failed = check_tolerance(max_diffs, atol=0.01)

        assert passed is True  # At boundary should pass
        assert failed == []


class TestFormatValidationReport:
    """Tests for format_validation_report utility function."""

    def test_format_passed_report(self) -> None:
        """Test formatting a passed validation result."""
        result = ValidationResult(
            passed=True,
            model_path=Path("/path/to/model.onnx"),
            num_samples_tested=10,
            horizon_results=[
                HorizonValidationResult(
                    horizon=1,
                    max_abs_diff=0.001,
                    max_rel_diff=0.01,
                    mean_abs_diff=0.0005,
                    passed=True,
                    failed_samples=0,
                ),
                HorizonValidationResult(
                    horizon=5,
                    max_abs_diff=0.002,
                    max_rel_diff=0.02,
                    mean_abs_diff=0.001,
                    passed=True,
                    failed_samples=0,
                ),
            ],
            probability_validation_passed=True,
            probability_max_diff=0.0001,
            error_message=None,
            config=ValidationConfig(),
        )

        report = format_validation_report(result)

        assert "PASSED" in report
        assert "model.onnx" in report
        assert "10" in report  # samples tested
        assert "Horizon   1s" in report
        assert "Horizon   5s" in report
        assert "[PASS]" in report
        assert "Probability validation" in report

    def test_format_failed_report(self) -> None:
        """Test formatting a failed validation result."""
        result = ValidationResult(
            passed=False,
            model_path=Path("/path/to/model.onnx"),
            num_samples_tested=10,
            horizon_results=[
                HorizonValidationResult(
                    horizon=60,
                    max_abs_diff=0.5,
                    max_rel_diff=0.5,
                    mean_abs_diff=0.25,
                    passed=False,
                    failed_samples=5,
                ),
            ],
            probability_validation_passed=None,
            probability_max_diff=None,
            error_message=None,
            config=ValidationConfig(),
        )

        report = format_validation_report(result)

        assert "FAILED" in report
        assert "[FAIL]" in report
        assert "Failed samples: 5" in report

    def test_format_error_report(self) -> None:
        """Test formatting a report with error message."""
        result = ValidationResult(
            passed=False,
            model_path=Path("/path/to/model.onnx"),
            num_samples_tested=0,
            horizon_results=[],
            probability_validation_passed=None,
            probability_max_diff=None,
            error_message="Model file not found",
            config=ValidationConfig(),
        )

        report = format_validation_report(result)

        assert "FAILED" in report
        assert "Error: Model file not found" in report


class TestDifferentModelConfigurations:
    """Tests for validation with different model configurations."""

    def test_validate_causal_attention(self) -> None:
        """Test validation with causal attention model."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1,),
            num_buckets=11,
            attention_type="causal",
        )
        model = PriceTransformer(config)
        model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(num_samples=2, batch_sizes=(1,)),
            )

            assert result.passed is True

    def test_validate_bidirectional_attention(self) -> None:
        """Test validation with bidirectional attention model."""
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
        model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(num_samples=2, batch_sizes=(1,)),
            )

            assert result.passed is True

    def test_validate_without_cls_token(self) -> None:
        """Test validation with mean pooling (no CLS token)."""
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
        model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(num_samples=2, batch_sizes=(1,)),
            )

            assert result.passed is True

    def test_validate_many_horizons(self) -> None:
        """Test validation with many prediction horizons."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(1, 5, 10, 30, 60, 120, 300, 600),
            num_buckets=11,
        )
        model = PriceTransformer(config)
        model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(num_samples=2, batch_sizes=(1,)),
            )

            assert result.passed is True
            assert len(result.horizon_results) == 8

    def test_validate_larger_model(self) -> None:
        """Test validation with larger model configuration."""
        config = TransformerConfig(
            input_dim=64,
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            ff_dim=512,
            horizons=(1, 5),
            num_buckets=101,
        )
        model = PriceTransformer(config)
        model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(num_samples=2, batch_sizes=(1,)),
            )

            assert result.passed is True


class TestEdgeCases:
    """Tests for edge cases in validation."""

    def test_validate_single_horizon(self) -> None:
        """Test validation with single horizon."""
        config = TransformerConfig(
            input_dim=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dim=64,
            horizons=(60,),
            num_buckets=11,
        )
        model = PriceTransformer(config)
        model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(num_samples=2, batch_sizes=(1,)),
            )

            assert result.passed is True
            assert len(result.horizon_results) == 1
            assert result.horizon_results[0].horizon == 60

    def test_validate_minimal_input(self) -> None:
        """Test validation with minimal input size."""
        config = TransformerConfig(
            input_dim=4,
            embedding_dim=16,
            num_layers=1,
            num_heads=2,
            ff_dim=32,
            horizons=(1,),
            num_buckets=5,
        )
        model = PriceTransformer(config)
        model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(
                    num_samples=1,
                    batch_sizes=(1,),
                    seq_lengths=(5,),
                ),
            )

            assert result.passed is True

    def test_validate_large_batch(self) -> None:
        """Test validation with larger batch size."""
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
        model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(
                    num_samples=1,
                    batch_sizes=(16,),
                    seq_lengths=(50,),
                ),
            )

            assert result.passed is True

    def test_validate_no_seed(self) -> None:
        """Test validation without random seed (non-deterministic)."""
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
        model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(
                    num_samples=2,
                    batch_sizes=(1,),
                    random_seed=None,
                ),
            )

            assert result.passed is True


class TestValidationResultDetails:
    """Tests for detailed ValidationResult information."""

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
        model = PriceTransformer(config)
        model.eval()
        return model

    def test_result_contains_all_horizons(self, model: PriceTransformer) -> None:
        """Test that result contains all horizon results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(num_samples=2, batch_sizes=(1,)),
            )

            horizons = {hr.horizon for hr in result.horizon_results}
            assert horizons == {1, 5, 10}

    def test_result_diff_values_reasonable(self, model: PriceTransformer) -> None:
        """Test that diff values are reasonable (non-negative, bounded)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            result = validate_export(
                model,
                output_path,
                ValidationConfig(num_samples=5, batch_sizes=(1, 2)),
            )

            for hr in result.horizon_results:
                assert hr.max_abs_diff >= 0
                assert hr.max_rel_diff >= 0
                assert hr.mean_abs_diff >= 0
                assert hr.mean_abs_diff <= hr.max_abs_diff

    def test_result_config_preserved(self, model: PriceTransformer) -> None:
        """Test that config is preserved in result."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            custom_config = ValidationConfig(
                rtol=1e-4,
                atol=1e-5,
                num_samples=3,
            )
            result = validate_export(model, output_path, custom_config)

            assert result.config == custom_config
            assert result.config.rtol == 1e-4
            assert result.config.atol == 1e-5


class TestStrictTolerance:
    """Tests for strict tolerance checking scenarios."""

    def test_very_strict_tolerance_may_fail(self) -> None:
        """Test that very strict tolerance may report failures.

        Note: The manual attention implementation has small numerical
        differences from PyTorch's fused implementation, so extremely
        strict tolerances may fail. This test documents expected behavior.
        """
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
        model.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            # Very strict tolerance
            strict_config = ValidationConfig(
                rtol=1e-8,
                atol=1e-8,
                num_samples=5,
                batch_sizes=(1, 4),
            )
            result = validate_export(model, output_path, strict_config)

            # This may pass or fail depending on numerical precision
            # The important thing is we get a result without errors
            assert result.error_message is None
            assert result.num_samples_tested > 0

    def test_relaxed_tolerance_passes(self) -> None:
        """Test that relaxed tolerance passes for typical exports."""
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

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.onnx"
            export_model(model, output_path)

            # Relaxed tolerance matching test_onnx_export.py
            relaxed_config = ValidationConfig(
                rtol=1e-2,
                atol=1e-3,
                num_samples=5,
                batch_sizes=(1, 4, 8),
            )
            result = validate_export(model, output_path, relaxed_config)

            assert result.passed is True
