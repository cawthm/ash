"""ONNX export validation utilities.

This module provides functionality to validate that exported ONNX models
produce outputs matching the original PyTorch model within acceptable
tolerances. This is critical for ensuring no train/serve skew in production.

Key features:
- Configurable tolerance thresholds for numerical comparison
- Statistical validation across multiple random inputs
- Per-horizon validation with detailed diagnostics
- Edge case testing (different sequence lengths, batch sizes)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from torch import Tensor

from models.architectures.price_transformer import PriceTransformer


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for ONNX validation.

    Attributes:
        rtol: Relative tolerance for numerical comparison.
        atol: Absolute tolerance for numerical comparison.
        num_samples: Number of random samples to validate.
        batch_sizes: Batch sizes to test with.
        seq_lengths: Sequence lengths to test (None uses model default).
        check_probabilities: Whether to also validate softmax probabilities.
        probability_rtol: Relative tolerance for probability comparison.
        probability_atol: Absolute tolerance for probability comparison.
        random_seed: Random seed for reproducibility. None for non-deterministic.
    """

    rtol: float = 1e-2
    atol: float = 1e-3
    num_samples: int = 10
    batch_sizes: tuple[int, ...] = (1, 4, 8)
    seq_lengths: tuple[int, ...] | None = None
    check_probabilities: bool = True
    probability_rtol: float = 1e-3
    probability_atol: float = 1e-4
    random_seed: int | None = 42

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.rtol < 0:
            raise ValueError("rtol must be non-negative")
        if self.atol < 0:
            raise ValueError("atol must be non-negative")
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if len(self.batch_sizes) < 1:
            raise ValueError("batch_sizes must have at least one element")
        if any(b < 1 for b in self.batch_sizes):
            raise ValueError("All batch sizes must be positive")
        if self.seq_lengths is not None:
            if len(self.seq_lengths) < 1:
                raise ValueError("seq_lengths must have at least one element")
            if any(s < 1 for s in self.seq_lengths):
                raise ValueError("All sequence lengths must be positive")
        if self.probability_rtol < 0:
            raise ValueError("probability_rtol must be non-negative")
        if self.probability_atol < 0:
            raise ValueError("probability_atol must be non-negative")


@dataclass
class HorizonValidationResult:
    """Validation result for a single horizon.

    Attributes:
        horizon: Prediction horizon in seconds.
        max_abs_diff: Maximum absolute difference across all samples.
        max_rel_diff: Maximum relative difference across all samples.
        mean_abs_diff: Mean absolute difference.
        passed: Whether validation passed for this horizon.
        failed_samples: Number of samples that failed tolerance check.
    """

    horizon: int
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    passed: bool
    failed_samples: int


@dataclass
class ValidationResult:
    """Result of ONNX validation.

    Attributes:
        passed: Whether all validations passed.
        model_path: Path to the validated ONNX model.
        num_samples_tested: Total number of test samples.
        horizon_results: Per-horizon validation results.
        probability_validation_passed: Whether probability validation passed.
        probability_max_diff: Maximum probability difference (if checked).
        error_message: Error message if validation failed.
        config: Validation configuration used.
    """

    passed: bool
    model_path: Path
    num_samples_tested: int
    horizon_results: list[HorizonValidationResult]
    probability_validation_passed: bool | None
    probability_max_diff: float | None
    error_message: str | None
    config: ValidationConfig


@dataclass
class _TestCase:
    """Internal test case specification.

    Attributes:
        batch_size: Batch size for this test.
        seq_len: Sequence length for this test.
        input_tensor: Generated input tensor.
    """

    batch_size: int
    seq_len: int
    input_tensor: Tensor


class ExportValidator:
    """Validates ONNX exports against PyTorch models.

    This class performs comprehensive validation to ensure that exported
    ONNX models produce outputs matching the original PyTorch model
    within specified tolerances.

    Attributes:
        config: Validation configuration.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize export validator.

        Args:
            config: Validation configuration. Uses defaults if not provided.
        """
        self.config = config or ValidationConfig()

    def validate(
        self,
        pytorch_model: PriceTransformer,
        onnx_path: str | Path,
    ) -> ValidationResult:
        """Validate ONNX model against PyTorch model.

        Runs multiple test cases with different batch sizes and sequence
        lengths, comparing outputs between PyTorch and ONNX Runtime.

        Args:
            pytorch_model: Original PyTorch model.
            onnx_path: Path to exported ONNX model.

        Returns:
            ValidationResult with detailed validation outcomes.
        """
        onnx_path = Path(onnx_path)

        # Verify ONNX file exists
        if not onnx_path.exists():
            return ValidationResult(
                passed=False,
                model_path=onnx_path,
                num_samples_tested=0,
                horizon_results=[],
                probability_validation_passed=None,
                probability_max_diff=None,
                error_message=f"ONNX model not found: {onnx_path}",
                config=self.config,
            )

        # Set up random seed
        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        # Set PyTorch model to eval mode
        pytorch_model.eval()

        # Load ONNX model
        try:
            session = ort.InferenceSession(str(onnx_path))
        except Exception as e:
            return ValidationResult(
                passed=False,
                model_path=onnx_path,
                num_samples_tested=0,
                horizon_results=[],
                probability_validation_passed=None,
                probability_max_diff=None,
                error_message=f"Failed to load ONNX model: {e}",
                config=self.config,
            )

        # Generate test cases
        test_cases = self._generate_test_cases(pytorch_model)

        # Collect results per horizon
        horizon_diffs: dict[int, list[tuple[float, float]]] = {
            h: [] for h in pytorch_model.config.horizons
        }
        probability_diffs: list[float] = []

        # Run validation
        total_samples = 0
        try:
            for test_case in test_cases:
                for _ in range(self.config.num_samples):
                    # Generate fresh random input
                    input_tensor = torch.randn_like(test_case.input_tensor)

                    # PyTorch inference
                    with torch.no_grad():
                        pytorch_output = pytorch_model(input_tensor)

                    # ONNX inference
                    input_name = session.get_inputs()[0].name
                    onnx_output = session.run(
                        None, {input_name: input_tensor.numpy()}
                    )

                    # Compare outputs per horizon
                    for i, horizon in enumerate(pytorch_model.config.horizons):
                        pytorch_logits = pytorch_output[horizon].numpy()
                        onnx_logits = onnx_output[i]

                        abs_diff = np.abs(pytorch_logits - onnx_logits)
                        max_abs = float(np.max(abs_diff))

                        # Compute relative difference safely
                        denominator = np.maximum(np.abs(pytorch_logits), 1e-8)
                        rel_diff = abs_diff / denominator
                        max_rel = float(np.max(rel_diff))

                        horizon_diffs[horizon].append((max_abs, max_rel))

                    # Check probabilities if configured
                    if self.config.check_probabilities:
                        with torch.no_grad():
                            pytorch_probs = pytorch_model.get_probabilities(
                                input_tensor
                            )

                        for i, horizon in enumerate(pytorch_model.config.horizons):
                            pytorch_prob = pytorch_probs[horizon].numpy()
                            onnx_prob = _softmax(onnx_output[i])
                            prob_diff = float(np.max(np.abs(pytorch_prob - onnx_prob)))
                            probability_diffs.append(prob_diff)

                    total_samples += 1

        except Exception as e:
            return ValidationResult(
                passed=False,
                model_path=onnx_path,
                num_samples_tested=total_samples,
                horizon_results=[],
                probability_validation_passed=None,
                probability_max_diff=None,
                error_message=f"Validation error: {e}",
                config=self.config,
            )

        # Compute per-horizon results
        horizon_results = []
        all_passed = True

        for horizon in pytorch_model.config.horizons:
            diffs = horizon_diffs[horizon]
            if not diffs:
                continue

            max_abs_values = [d[0] for d in diffs]
            max_rel_values = [d[1] for d in diffs]

            max_abs_diff = max(max_abs_values)
            max_rel_diff = max(max_rel_values)
            mean_abs_diff = sum(max_abs_values) / len(max_abs_values)

            # Count failures
            failed = sum(
                1
                for abs_d, rel_d in diffs
                if abs_d > self.config.atol
                and rel_d > self.config.rtol
            )

            passed = failed == 0

            horizon_results.append(
                HorizonValidationResult(
                    horizon=horizon,
                    max_abs_diff=max_abs_diff,
                    max_rel_diff=max_rel_diff,
                    mean_abs_diff=mean_abs_diff,
                    passed=passed,
                    failed_samples=failed,
                )
            )

            if not passed:
                all_passed = False

        # Check probability validation
        probability_passed: bool | None = None
        probability_max: float | None = None

        if self.config.check_probabilities and probability_diffs:
            probability_max = max(probability_diffs)
            probability_passed = probability_max <= max(
                self.config.probability_atol,
                self.config.probability_rtol * 1.0,  # Max prob is ~1.0
            )
            if not probability_passed:
                all_passed = False

        return ValidationResult(
            passed=all_passed,
            model_path=onnx_path,
            num_samples_tested=total_samples,
            horizon_results=horizon_results,
            probability_validation_passed=probability_passed,
            probability_max_diff=probability_max,
            error_message=None,
            config=self.config,
        )

    def _generate_test_cases(
        self,
        model: PriceTransformer,
    ) -> list[_TestCase]:
        """Generate test cases for validation.

        Args:
            model: PyTorch model to generate test cases for.

        Returns:
            List of test cases with different configurations.
        """
        test_cases: list[_TestCase] = []
        device = next(model.parameters()).device

        # Determine sequence lengths to test
        if self.config.seq_lengths is not None:
            seq_lengths = self.config.seq_lengths
        else:
            # Use a variety of lengths up to max_seq_len
            max_len = model.config.max_seq_len
            seq_lengths = tuple(
                min(length, max_len)
                for length in (10, 50, 100, max_len)
            )
            # Remove duplicates while preserving order
            seen: set[int] = set()
            unique_lengths: list[int] = []
            for length in seq_lengths:
                if length not in seen:
                    seen.add(length)
                    unique_lengths.append(length)
            seq_lengths = tuple(unique_lengths)

        # Generate test cases
        for batch_size in self.config.batch_sizes:
            for seq_len in seq_lengths:
                input_tensor = torch.randn(
                    batch_size,
                    seq_len,
                    model.config.input_dim,
                    device=device,
                )
                test_cases.append(
                    _TestCase(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        input_tensor=input_tensor,
                    )
                )

        return test_cases

    def validate_single_input(
        self,
        pytorch_model: PriceTransformer,
        onnx_path: str | Path,
        input_tensor: Tensor,
    ) -> ValidationResult:
        """Validate with a specific input tensor.

        Useful for debugging or testing specific edge cases.

        Args:
            pytorch_model: Original PyTorch model.
            onnx_path: Path to exported ONNX model.
            input_tensor: Specific input to test.

        Returns:
            ValidationResult for this single input.
        """
        onnx_path = Path(onnx_path)

        pytorch_model.eval()

        try:
            session = ort.InferenceSession(str(onnx_path))
        except Exception as e:
            return ValidationResult(
                passed=False,
                model_path=onnx_path,
                num_samples_tested=0,
                horizon_results=[],
                probability_validation_passed=None,
                probability_max_diff=None,
                error_message=f"Failed to load ONNX model: {e}",
                config=self.config,
            )

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(input_tensor)

        # ONNX inference
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: input_tensor.numpy()})

        # Build results
        horizon_results = []
        all_passed = True

        for i, horizon in enumerate(pytorch_model.config.horizons):
            pytorch_logits = pytorch_output[horizon].numpy()
            onnx_logits = onnx_output[i]

            abs_diff = np.abs(pytorch_logits - onnx_logits)
            max_abs = float(np.max(abs_diff))
            mean_abs = float(np.mean(abs_diff))

            denominator = np.maximum(np.abs(pytorch_logits), 1e-8)
            rel_diff = abs_diff / denominator
            max_rel = float(np.max(rel_diff))

            passed = max_abs <= self.config.atol or max_rel <= self.config.rtol

            horizon_results.append(
                HorizonValidationResult(
                    horizon=horizon,
                    max_abs_diff=max_abs,
                    max_rel_diff=max_rel,
                    mean_abs_diff=mean_abs,
                    passed=passed,
                    failed_samples=0 if passed else 1,
                )
            )

            if not passed:
                all_passed = False

        # Check probabilities
        probability_passed: bool | None = None
        probability_max: float | None = None

        if self.config.check_probabilities:
            with torch.no_grad():
                pytorch_probs = pytorch_model.get_probabilities(input_tensor)

            prob_diffs = []
            for i, horizon in enumerate(pytorch_model.config.horizons):
                pytorch_prob = pytorch_probs[horizon].numpy()
                onnx_prob = _softmax(onnx_output[i])
                prob_diff = float(np.max(np.abs(pytorch_prob - onnx_prob)))
                prob_diffs.append(prob_diff)

            probability_max = max(prob_diffs)
            probability_passed = probability_max <= max(
                self.config.probability_atol,
                self.config.probability_rtol * 1.0,
            )
            if not probability_passed:
                all_passed = False

        return ValidationResult(
            passed=all_passed,
            model_path=onnx_path,
            num_samples_tested=1,
            horizon_results=horizon_results,
            probability_validation_passed=probability_passed,
            probability_max_diff=probability_max,
            error_message=None,
            config=self.config,
        )


def _softmax(x: NDArray[Any]) -> NDArray[Any]:
    """Compute softmax over last axis.

    Args:
        x: Input array.

    Returns:
        Softmax probabilities.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    result: NDArray[Any] = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return result


def validate_export(
    pytorch_model: PriceTransformer,
    onnx_path: str | Path,
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """Convenience function to validate an ONNX export.

    Args:
        pytorch_model: Original PyTorch model.
        onnx_path: Path to exported ONNX model.
        config: Validation configuration. Uses defaults if not provided.

    Returns:
        ValidationResult with detailed validation outcomes.

    Example:
        >>> from models.architectures.price_transformer import (
        ...     PriceTransformer, TransformerConfig
        ... )
        >>> from models.export.onnx_export import export_model
        >>> config = TransformerConfig(input_dim=64)
        >>> model = PriceTransformer(config)
        >>> export_model(model, "model.onnx")
        >>> result = validate_export(model, "model.onnx")
        >>> print(f"Validation passed: {result.passed}")
    """
    validator = ExportValidator(config)
    return validator.validate(pytorch_model, onnx_path)


def compute_max_diff(
    pytorch_output: dict[int, Tensor],
    onnx_output: list[NDArray[Any]],
    horizons: tuple[int, ...],
) -> dict[int, float]:
    """Compute maximum absolute difference per horizon.

    Utility function for quick comparison checks.

    Args:
        pytorch_output: PyTorch model output dict.
        onnx_output: ONNX model output list.
        horizons: Prediction horizons in order.

    Returns:
        Dictionary mapping horizon to maximum absolute difference.
    """
    result: dict[int, float] = {}
    for i, horizon in enumerate(horizons):
        pytorch_arr = pytorch_output[horizon].detach().numpy()
        onnx_arr = onnx_output[i]
        max_diff = float(np.max(np.abs(pytorch_arr - onnx_arr)))
        result[horizon] = max_diff
    return result


def check_tolerance(
    max_diffs: dict[int, float],
    atol: float = 1e-3,
) -> tuple[bool, list[int]]:
    """Check if maximum differences are within tolerance.

    Uses absolute tolerance only since max_diffs contains absolute
    differences without reference values for relative comparison.

    Args:
        max_diffs: Maximum differences per horizon.
        atol: Absolute tolerance threshold.

    Returns:
        Tuple of (all_passed, list of failed horizons).
    """
    failed_horizons: list[int] = []
    for horizon, diff in max_diffs.items():
        if diff > atol:
            failed_horizons.append(horizon)

    return len(failed_horizons) == 0, failed_horizons


def format_validation_report(result: ValidationResult) -> str:
    """Format validation result as human-readable report.

    Args:
        result: Validation result to format.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        "ONNX Export Validation Report",
        "=" * 60,
        f"Model: {result.model_path}",
        f"Samples tested: {result.num_samples_tested}",
        f"Overall result: {'PASSED' if result.passed else 'FAILED'}",
        "",
    ]

    if result.error_message:
        lines.append(f"Error: {result.error_message}")
        lines.append("")

    if result.horizon_results:
        lines.append("Per-Horizon Results:")
        lines.append("-" * 40)
        for hr in result.horizon_results:
            status = "PASS" if hr.passed else "FAIL"
            lines.append(
                f"  Horizon {hr.horizon:>3}s: [{status}] "
                f"max_abs={hr.max_abs_diff:.2e}, "
                f"max_rel={hr.max_rel_diff:.2e}, "
                f"mean_abs={hr.mean_abs_diff:.2e}"
            )
            if not hr.passed:
                lines.append(f"             Failed samples: {hr.failed_samples}")
        lines.append("")

    if result.probability_validation_passed is not None:
        prob_status = "PASS" if result.probability_validation_passed else "FAIL"
        lines.append(f"Probability validation: [{prob_status}]")
        if result.probability_max_diff is not None:
            lines.append(f"  Max probability difference: {result.probability_max_diff:.2e}")
        lines.append("")

    lines.append("Configuration:")
    lines.append(f"  rtol: {result.config.rtol}")
    lines.append(f"  atol: {result.config.atol}")
    lines.append(f"  num_samples: {result.config.num_samples}")
    lines.append(f"  batch_sizes: {result.config.batch_sizes}")

    lines.append("=" * 60)

    return "\n".join(lines)
