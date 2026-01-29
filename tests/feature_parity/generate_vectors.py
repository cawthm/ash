"""Generate test vectors for Python/Rust feature parity validation.

This script generates test vectors by running the Python feature builders on
synthetic and edge-case data, then saves the inputs and expected outputs to
JSON files that Rust tests can load and validate against.

Critical: This is the source of truth for feature computation. Rust must match
these outputs exactly to avoid train/serve skew.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data.processors.feature_builder import (
    PriceFeatureBuilder,
    PriceFeatureConfig,
    VolatilityFeatureBuilder,
    VolatilityFeatureConfig,
)


def generate_test_vectors() -> dict[str, Any]:
    """Generate comprehensive test vectors for feature parity validation.

    Returns:
        Dictionary containing test cases with inputs and expected outputs.
    """
    test_vectors = {
        "version": "1.0",
        "description": "Test vectors for Python/Rust feature parity validation",
        "test_cases": [],
    }

    # Test Case 1: Basic log returns
    prices_1 = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    volumes_1 = [1000.0, 1100.0, 1200.0, 1000.0, 900.0, 1050.0]

    price_builder = PriceFeatureBuilder(
        PriceFeatureConfig(
            return_windows=(1, 2),
            include_vwap=False,
            sample_interval=1,
        )
    )

    returns_multi = price_builder.compute_returns_multi_window(np.array(prices_1))

    test_vectors["test_cases"].append(
        {
            "name": "basic_log_returns",
            "description": "Basic log returns at windows 1 and 2",
            "inputs": {"prices": prices_1, "windows": [1, 2]},
            "expected": {
                "returns_window_1": _serialize_array(returns_multi[:, 0]),
                "returns_window_2": _serialize_array(returns_multi[:, 1]),
            },
        }
    )

    # Test Case 2: VWAP calculation
    prices_2 = [100.0, 101.0, 100.5, 102.0, 101.5]
    volumes_2 = [1000.0, 2000.0, 1500.0, 1800.0, 1200.0]

    vwap = price_builder.compute_vwap(
        np.array(prices_2), np.array(volumes_2), window=3
    )
    vwap_dev = price_builder.compute_vwap_deviation(
        np.array(prices_2), np.array(volumes_2), window=3
    )

    test_vectors["test_cases"].append(
        {
            "name": "vwap_calculation",
            "description": "VWAP and VWAP deviation with window=3",
            "inputs": {"prices": prices_2, "volumes": volumes_2, "window": 3},
            "expected": {
                "vwap": _serialize_array(vwap),
                "vwap_deviation": _serialize_array(vwap_dev),
            },
        }
    )

    # Test Case 3: Realized volatility
    # Generate synthetic log returns with known volatility
    np.random.seed(42)
    log_returns = np.random.normal(0.0, 0.01, 100)  # 1% daily volatility

    vol_builder = VolatilityFeatureBuilder(
        VolatilityFeatureConfig(
            rv_windows=(10, 30),
            include_iv_rv_spread=False,
            include_vol_of_vol=False,
        )
    )

    rv_multi = vol_builder.compute_rv_multi_window(log_returns, sample_interval=1)

    test_vectors["test_cases"].append(
        {
            "name": "realized_volatility",
            "description": "Realized volatility at windows 10 and 30",
            "inputs": {
                "log_returns": _serialize_array(log_returns),
                "windows": [10, 30],
                "annualization_factor": vol_builder.config.annualization_factor,
            },
            "expected": {
                "rv_window_10": _serialize_array(rv_multi[:, 0]),
                "rv_window_30": _serialize_array(rv_multi[:, 1]),
            },
        }
    )

    # Test Case 4: Edge case - zero prices
    prices_edge = [100.0, 0.0, 102.0, 103.0]
    returns_edge = price_builder.compute_log_returns(np.array(prices_edge), window=1)

    test_vectors["test_cases"].append(
        {
            "name": "zero_price_handling",
            "description": "Log returns with zero price (should produce NaN)",
            "inputs": {"prices": prices_edge, "window": 1},
            "expected": {"returns": _serialize_array(returns_edge)},
        }
    )

    # Test Case 5: Edge case - negative prices
    prices_neg = [100.0, -101.0, 102.0]
    returns_neg = price_builder.compute_log_returns(np.array(prices_neg), window=1)

    test_vectors["test_cases"].append(
        {
            "name": "negative_price_handling",
            "description": "Log returns with negative price (should produce NaN)",
            "inputs": {"prices": prices_neg, "window": 1},
            "expected": {"returns": _serialize_array(returns_neg)},
        }
    )

    # Test Case 6: Edge case - insufficient data
    prices_short = [100.0, 101.0]
    returns_short = price_builder.compute_log_returns(
        np.array(prices_short), window=5
    )

    test_vectors["test_cases"].append(
        {
            "name": "insufficient_data",
            "description": "Log returns with window > data length (should be all NaN)",
            "inputs": {"prices": prices_short, "window": 5},
            "expected": {"returns": _serialize_array(returns_short)},
        }
    )

    # Test Case 7: Realistic price movement
    # Simulate realistic intraday price movement
    t = np.arange(300)  # 5 minutes at 1 Hz
    base_price = 150.0
    drift = 0.0001 * t  # Slight upward drift
    noise = np.random.RandomState(123).normal(0, 0.002, len(t))
    prices_realistic = base_price * (1 + drift + noise)
    volumes_realistic = np.random.RandomState(456).uniform(800, 1200, len(t))

    price_builder_full = PriceFeatureBuilder(
        PriceFeatureConfig(
            return_windows=(1, 5, 10, 30, 60),
            include_vwap=True,
            sample_interval=1,
        )
    )

    features_realistic = price_builder_full.compute_features(
        prices_realistic, volumes_realistic, vwap_window=60
    )

    test_vectors["test_cases"].append(
        {
            "name": "realistic_price_movement",
            "description": "Realistic intraday price movement with full feature set",
            "inputs": {
                "prices": _serialize_array(prices_realistic),
                "volumes": _serialize_array(volumes_realistic),
                "vwap_window": 60,
                "return_windows": [1, 5, 10, 30, 60],
            },
            "expected": {
                "features": _serialize_2d_array(features_realistic),
                "feature_names": price_builder_full.get_feature_names(),
            },
        }
    )

    return test_vectors


def _serialize_array(arr: np.ndarray) -> list[float | None]:
    """Serialize NumPy array to JSON-compatible list.

    NaN values are converted to None for JSON compatibility.
    """
    result = []
    for val in arr.flat:
        if np.isnan(val):
            result.append(None)
        elif np.isinf(val):
            result.append("inf" if val > 0 else "-inf")
        else:
            result.append(float(val))
    return result


def _serialize_2d_array(arr: np.ndarray) -> list[list[float | None]]:
    """Serialize 2D NumPy array to JSON-compatible nested list."""
    return [_serialize_array(row) for row in arr]


def main() -> None:
    """Generate test vectors and save to JSON file."""
    # Generate test vectors
    test_vectors = generate_test_vectors()

    # Save to JSON file
    output_path = Path(__file__).parent / "test_vectors.json"
    with open(output_path, "w") as f:
        json.dump(test_vectors, f, indent=2)

    print(f"Generated {len(test_vectors['test_cases'])} test cases")
    print(f"Saved to: {output_path}")

    # Print summary
    for tc in test_vectors["test_cases"]:
        print(f"  - {tc['name']}: {tc['description']}")


if __name__ == "__main__":
    main()
