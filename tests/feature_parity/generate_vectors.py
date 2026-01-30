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
    OrderFlowFeatureBuilder,
    OrderFlowFeatureConfig,
    OptionsFeatureBuilder,
    OptionsFeatureConfig,
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
    test_vectors: dict[str, Any] = {
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

    # Test Case 8: Order flow - trade direction inference
    trade_prices = [100.0, 101.0, 100.5, 100.5, 101.5, 101.0]
    trade_sizes = [10.0, 20.0, 15.0, 12.0, 18.0, 25.0]
    trade_timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    orderflow_builder = OrderFlowFeatureBuilder(
        OrderFlowFeatureConfig(
            imbalance_window=3,
            include_size_distribution=True,
            include_arrival_rate=True,
            size_quantiles=(0.25, 0.5, 0.75),
            arrival_rate_window=3,
        )
    )

    directions = orderflow_builder.infer_trade_direction(np.array(trade_prices))
    imbalance = orderflow_builder.compute_trade_imbalance(
        directions, np.array(trade_sizes), window=3
    )
    size_quantiles = orderflow_builder.compute_size_quantiles(
        np.array(trade_sizes), window=3
    )
    arrival_rate = orderflow_builder.compute_arrival_rate(
        np.array(trade_timestamps), window=3
    )

    test_vectors["test_cases"].append(
        {
            "name": "orderflow_basic",
            "description": "Order flow features: direction, imbalance, size quantiles, arrival rate",
            "inputs": {
                "prices": trade_prices,
                "sizes": trade_sizes,
                "timestamps": trade_timestamps,
                "imbalance_window": 3,
                "quantiles": [0.25, 0.5, 0.75],
                "arrival_rate_window": 3,
            },
            "expected": {
                "directions": [int(d) for d in directions],
                "imbalance": _serialize_array(imbalance),
                "size_q25": _serialize_array(size_quantiles[:, 0]),
                "size_q50": _serialize_array(size_quantiles[:, 1]),
                "size_q75": _serialize_array(size_quantiles[:, 2]),
                "arrival_rate": _serialize_array(arrival_rate),
            },
        }
    )

    # Test Case 9: Order flow - edge case with zero ticks
    zero_tick_prices = [100.0, 100.0, 100.0, 101.0, 101.0, 99.0]
    zero_tick_sizes = [5.0, 10.0, 8.0, 12.0, 6.0, 15.0]

    directions_zero = orderflow_builder.infer_trade_direction(
        np.array(zero_tick_prices)
    )
    imbalance_zero = orderflow_builder.compute_trade_imbalance(
        directions_zero, np.array(zero_tick_sizes), window=4
    )

    test_vectors["test_cases"].append(
        {
            "name": "orderflow_zero_ticks",
            "description": "Order flow with zero ticks (direction propagation)",
            "inputs": {
                "prices": zero_tick_prices,
                "sizes": zero_tick_sizes,
                "window": 4,
            },
            "expected": {
                "directions": [int(d) for d in directions_zero],
                "imbalance": _serialize_array(imbalance_zero),
            },
        }
    )

    # Test Case 10: Order flow - realistic trading scenario
    np.random.seed(789)
    n_trades = 100
    realistic_trade_prices = 150.0 + np.cumsum(
        np.random.normal(0, 0.05, n_trades)
    )  # Random walk
    realistic_trade_sizes = np.random.lognormal(3.0, 0.5, n_trades)  # Log-normal sizes
    realistic_timestamps = np.sort(
        np.random.uniform(0, 60, n_trades)
    )  # Random arrival times over 60s

    orderflow_realistic = OrderFlowFeatureBuilder(
        OrderFlowFeatureConfig(
            imbalance_window=20,
            include_size_distribution=True,
            include_arrival_rate=True,
            size_quantiles=(0.25, 0.5, 0.75),
            arrival_rate_window=20,
        )
    )

    features_orderflow = orderflow_realistic.compute_features(
        realistic_trade_prices, realistic_trade_sizes, realistic_timestamps
    )

    test_vectors["test_cases"].append(
        {
            "name": "orderflow_realistic",
            "description": "Realistic order flow scenario with full feature set",
            "inputs": {
                "prices": _serialize_array(realistic_trade_prices),
                "sizes": _serialize_array(realistic_trade_sizes),
                "timestamps": _serialize_array(realistic_timestamps),
                "imbalance_window": 20,
                "quantiles": [0.25, 0.5, 0.75],
                "arrival_rate_window": 20,
            },
            "expected": {
                "features": _serialize_2d_array(features_orderflow),
                "feature_names": orderflow_realistic.get_feature_names(),
            },
        }
    )

    # Test Case 11: Options - basic moneyness and days to expiry
    spot_price = 100.0
    current_time = 0.0
    strikes = np.array([95.0, 100.0, 105.0])
    expirations = np.array([
        7.0 * 24 * 3600,  # 7 days
        30.0 * 24 * 3600,  # 30 days
        60.0 * 24 * 3600,  # 60 days
    ])

    options_builder = OptionsFeatureBuilder()

    moneyness = options_builder.compute_moneyness(strikes, spot_price)
    dte = options_builder.compute_days_to_expiry(expirations, current_time)

    test_vectors["test_cases"].append(
        {
            "name": "options_moneyness_dte",
            "description": "Options moneyness and days to expiry calculation",
            "inputs": {
                "strikes": strikes.tolist(),
                "spot_price": spot_price,
                "expirations": expirations.tolist(),
                "current_time": current_time,
            },
            "expected": {
                "moneyness": _serialize_array(moneyness),
                "days_to_expiry": _serialize_array(dte),
            },
        }
    )

    # Test Case 12: Options - aggregated Greeks
    deltas = np.array([0.5, -0.3, 0.6])
    gammas = np.array([0.01, 0.02, 0.015])
    vegas = np.array([10.0, 12.0, 11.0])
    thetas = np.array([-0.5, -0.6, -0.4])
    open_interests = np.array([100.0, 200.0, 100.0])
    option_types = np.array([1, -1, 1], dtype=np.int8)

    greeks = options_builder.compute_aggregated_greeks(
        deltas, gammas, vegas, thetas, open_interests, option_types
    )

    test_vectors["test_cases"].append(
        {
            "name": "options_greeks",
            "description": "Aggregated Greeks weighted by open interest",
            "inputs": {
                "deltas": deltas.tolist(),
                "gammas": gammas.tolist(),
                "vegas": vegas.tolist(),
                "thetas": thetas.tolist(),
                "open_interests": open_interests.tolist(),
                "option_types": option_types.tolist(),
            },
            "expected": {"greeks": _serialize_array(greeks)},
        }
    )

    # Test Case 13: Options - put/call ratios
    option_types_pc = np.array([1, 1, -1, -1, 1], dtype=np.int8)  # 3 calls, 2 puts
    volumes_pc = np.array([100.0, 150.0, 200.0, 100.0, 50.0])
    ois_pc = np.array([500.0, 500.0, 600.0, 400.0, 100.0])

    ratios = options_builder.compute_put_call_ratios(option_types_pc, volumes_pc, ois_pc)

    test_vectors["test_cases"].append(
        {
            "name": "options_put_call_ratios",
            "description": "Put/call volume and OI ratios",
            "inputs": {
                "option_types": option_types_pc.tolist(),
                "volumes": volumes_pc.tolist(),
                "open_interests": ois_pc.tolist(),
            },
            "expected": {"ratios": _serialize_array(ratios)},
        }
    )

    # Test Case 14: Options - term structure slope
    atm_ivs = np.array([0.20, 0.22, 0.24, 0.26])
    slope = options_builder.compute_term_structure_slope(atm_ivs)

    test_vectors["test_cases"].append(
        {
            "name": "options_term_structure",
            "description": "IV term structure slope via linear regression",
            "inputs": {
                "atm_ivs_by_expiry": atm_ivs.tolist(),
                "expiry_buckets_days": list(
                    options_builder.config.expiry_buckets_days
                ),
            },
            "expected": {"slope": float(slope)},
        }
    )

    # Test Case 15: Options - ATM IV by expiry
    # Create realistic options chain
    np.random.seed(999)
    n_options = 50
    strikes_chain = np.random.uniform(90, 110, n_options)
    expirations_chain = np.array(
        [
            np.random.choice([7, 14, 30, 45, 60, 90]) * 24 * 3600.0
            for _ in range(n_options)
        ]
    )
    ivs_chain = np.random.uniform(0.15, 0.35, n_options)

    atm_ivs_by_expiry = options_builder.compute_atm_iv_by_expiry(
        strikes_chain, expirations_chain, ivs_chain, spot_price=100.0, current_time=0.0
    )

    test_vectors["test_cases"].append(
        {
            "name": "options_atm_iv_by_expiry",
            "description": "ATM implied volatility across expiry buckets",
            "inputs": {
                "strikes": _serialize_array(strikes_chain),
                "expirations": _serialize_array(expirations_chain),
                "ivs": _serialize_array(ivs_chain),
                "spot_price": 100.0,
                "current_time": 0.0,
                "expiry_buckets_days": list(
                    options_builder.config.expiry_buckets_days
                ),
            },
            "expected": {"atm_ivs": _serialize_array(atm_ivs_by_expiry)},
        }
    )

    # Test Case 16: Options - IV surface features
    iv_surface = options_builder.compute_iv_surface_features(
        strikes_chain, expirations_chain, ivs_chain, spot_price=100.0, current_time=0.0
    )

    test_vectors["test_cases"].append(
        {
            "name": "options_iv_surface",
            "description": "IV surface features across moneyness and expiry grid",
            "inputs": {
                "strikes": _serialize_array(strikes_chain),
                "expirations": _serialize_array(expirations_chain),
                "ivs": _serialize_array(ivs_chain),
                "spot_price": 100.0,
                "current_time": 0.0,
                "moneyness_buckets": list(options_builder.config.iv_moneyness_buckets),
                "expiry_buckets_days": list(
                    options_builder.config.expiry_buckets_days
                ),
            },
            "expected": {"iv_surface": _serialize_array(iv_surface)},
        }
    )

    return test_vectors


def _serialize_array(arr: np.ndarray) -> list[float | None | str]:
    """Serialize NumPy array to JSON-compatible list.

    NaN values are converted to None for JSON compatibility.
    Infinity values are converted to "inf" or "-inf" strings.
    """
    result: list[float | None | str] = []
    for val in arr.flat:
        if np.isnan(val):
            result.append(None)
        elif np.isinf(val):
            result.append("inf" if val > 0 else "-inf")
        else:
            result.append(float(val))
    return result


def _serialize_2d_array(arr: np.ndarray) -> list[list[float | None | str]]:
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
