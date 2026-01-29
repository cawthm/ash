# Rust Inference Latency Benchmarks

**Date**: 2026-01-29
**System**: Linux 6.8.0-60-generic
**Compiler**: rustc 1.85.0 (release profile with LTO)

## Executive Summary

All feature computations are **well below the 10ms latency requirement** for real-time trading. The most expensive feature computation (size quantiles) takes only **90 microseconds (0.09ms)**, leaving ample headroom for model inference.

**Key Finding**: Feature computation overhead is negligible (~0.1ms total). The critical path for meeting the <10ms target is ONNX model inference speed, which will be benchmarked once a trained model is available.

---

## Buffer Operations

| Benchmark | Latency | Throughput |
|-----------|---------|------------|
| `buffer_update` | 8.8 ns | 113M ops/sec |
| `rolling_buffer_push` | 2.4 ns | 417M ops/sec |
| `rolling_buffer_to_vec` | 118 ns | 8.5M ops/sec |

**Analysis**: Buffer operations are extremely fast, with single-digit nanosecond latencies. State management overhead is negligible.

---

## Price & Volatility Features

| Benchmark | Latency | Notes |
|-----------|---------|-------|
| `feature_returns_multi_window` | 33.3 µs | 7 windows, 300 samples |
| `feature_vwap` | 1.25 µs | 60-second window |
| `feature_vwap_deviation` | 6.26 µs | Includes VWAP computation |
| `feature_realized_volatility` | 2.70 µs | Single window RV calculation |

**Analysis**: Multi-window returns computation is the most expensive price feature at 33µs, but still well within budget. VWAP calculation is highly optimized at 1.25µs.

---

## OrderFlow Features

| Benchmark | Latency | Notes |
|-----------|---------|-------|
| `feature_trade_direction` | 389 ns | Tick rule inference |
| `feature_trade_imbalance` | 1.48 µs | 30-sample window |
| `feature_size_quantiles` | **90.5 µs** | 3 quantiles, 30-sample window |
| `feature_arrival_rate` | 329 ns | Trades per second |

**Analysis**: Size quantiles is the slowest orderflow feature due to sorting operations. Consider optimization if used in real-time path. Trade direction and arrival rate are sub-microsecond.

---

## Options Features

| Benchmark | Latency | Notes |
|-----------|---------|-------|
| `feature_moneyness` | 34.0 ns | Strike/spot ratio |
| `feature_aggregated_greeks` | 63.5 ns | OI-weighted Greeks |
| `feature_put_call_ratios` | 3.04 ns | Volume/OI ratios |
| `feature_term_structure_slope` | 124 ns | IV term structure |
| `feature_atm_iv_by_expiry` | 161 ns | ATM IV selection |
| `feature_iv_surface` | 208 ns | IV surface features |

**Analysis**: All options features are extremely fast (<1µs). Moneyness and Greeks computations are highly optimized.

---

## Sequence Preparation

| Benchmark | Latency | Sequence Length |
|-----------|---------|-----------------|
| `prepare_input/60` | 83.0 ns | 60 seconds |
| `prepare_input/120` | 94.9 ns | 120 seconds |
| `prepare_input/300` | 240 ns | 300 seconds (5 min) |

**Analysis**: Input preparation overhead grows sub-linearly with sequence length. 300-second sequences take only 240ns.

---

## Estimated Total Feature Computation Time

Based on current implementation (`predictor.rs:prepare_input()`):

| Component | Latency | Count | Total |
|-----------|---------|-------|-------|
| Multi-window returns | 33.3 µs | 1 | 33.3 µs |
| VWAP deviation | 6.3 µs | 1 | 6.3 µs |
| Multi-window RV | 2.7 µs | 3 | 8.1 µs |
| Buffer operations | 8.8 ns | 1 | 0.009 µs |
| **Total Feature Computation** | | | **~48 µs** |

**Conservative estimate**: 50-100 µs (0.05-0.1ms) for complete feature preparation.

---

## Latency Budget Analysis

For **sub-10ms inference target**:

```
Total Budget:               10,000 µs (10ms)
Feature Computation:           ~50 µs (0.05ms)
Buffer/State Management:       ~10 µs (0.01ms)
Overhead (copying, etc.):      ~40 µs (0.04ms)
-------------------------------------------------
Available for ONNX Inference:  9,900 µs (9.9ms)
```

**Conclusion**: Feature computation uses <1% of the latency budget. Model inference has **9.9ms available**, which is sufficient for the lightweight transformer architecture specified in Phase 4.

---

## Next Steps

1. ✅ **COMPLETE**: All feature computations benchmarked and verified <10ms
2. **TODO**: Train lightweight ONNX model (Phase 4-5)
3. **TODO**: Benchmark end-to-end inference with real ONNX model
4. **TODO**: Validate total latency <10ms with production configuration

---

## Benchmark Reproduction

```bash
# Run all benchmarks
cargo bench --bench latency

# Run specific benchmark
cargo bench --bench latency -- buffer_update

# With real ONNX model (place model at benches/test_model.onnx)
cargo bench --bench latency -- end_to_end
```

---

## Hardware Configuration

- **CPU**: (auto-detected by criterion.rs)
- **Optimization Level**: `opt-level = 3`, LTO enabled, single codegen unit
- **Target**: x86_64-unknown-linux-gnu
- **Allocator**: System default (jemalloc recommended for production)

---

*Generated from criterion.rs benchmark results*
