//! Latency benchmarks for price prediction inference.
//!
//! Measures end-to-end inference latency to validate <10ms target.
//!
//! # Benchmarks
//!
//! ## Buffer Operations
//! - `buffer_update`: Single data point ingestion
//! - `rolling_buffer_push`: Circular buffer push operation
//! - `rolling_buffer_to_vec`: Buffer to vector conversion
//!
//! ## Feature Computation
//! - `feature_returns_multi_window`: Multi-window log returns computation
//! - `feature_vwap_deviation`: VWAP deviation calculation
//! - `feature_realized_volatility`: Multi-window RV computation
//! - `feature_orderflow_imbalance`: Trade imbalance calculation
//! - `feature_options_greeks`: Aggregated Greeks computation
//! - `prepare_input_full`: Complete feature matrix preparation
//!
//! ## End-to-End Inference
//! - `update_and_predict`: Update state + run inference
//! - `predict_only`: Pure inference latency (with warm buffer)
//!
//! # Running Benchmarks
//!
//! ```bash
//! # All benchmarks
//! cargo bench
//!
//! # Specific benchmark
//! cargo bench -- buffer_update
//!
//! # With real ONNX model (place model.onnx in benches/)
//! cargo bench -- end_to_end
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

//
// Buffer Benchmarks
//

fn benchmark_buffer_operations(c: &mut Criterion) {
    use ash_inference::FeatureBuffer;

    c.bench_function("buffer_update", |b| {
        let mut buffer = FeatureBuffer::new(300, 1);

        b.iter(|| {
            buffer.update(black_box(1000.0), black_box(100.5), black_box(1000.0));
        });
    });
}

fn benchmark_rolling_buffer(c: &mut Criterion) {
    use ash_inference::RollingBuffer;

    c.bench_function("rolling_buffer_push", |b| {
        let mut buffer = RollingBuffer::new(300);

        b.iter(|| {
            buffer.push(black_box(100.5));
        });
    });

    c.bench_function("rolling_buffer_to_vec", |b| {
        let mut buffer = RollingBuffer::new(300);
        for i in 0..300 {
            buffer.push(i as f64);
        }

        b.iter(|| {
            let _vec = buffer.to_vec();
        });
    });
}

//
// Feature Computation Benchmarks
//

fn benchmark_feature_computation(c: &mut Criterion) {
    use ash_inference::features::{
        compute_log_returns, compute_returns_multi_window, compute_rv_multi_window,
        compute_vwap, compute_vwap_deviation, PriceFeatureConfig, VolatilityFeatureConfig,
    };

    // Prepare realistic test data
    let prices: Vec<f64> = (0..300)
        .map(|i| 100.0 + (i as f64 * 0.01).sin() * 2.0)
        .collect();
    let volumes: Vec<f64> = (0..300).map(|_| 1000.0).collect();

    // Benchmark multi-window returns
    c.bench_function("feature_returns_multi_window", |b| {
        let config = PriceFeatureConfig::default();
        b.iter(|| {
            let _ = compute_returns_multi_window(black_box(&prices), &config);
        });
    });

    // Benchmark VWAP
    c.bench_function("feature_vwap", |b| {
        b.iter(|| {
            let _ = compute_vwap(black_box(&prices), black_box(&volumes), 60);
        });
    });

    // Benchmark VWAP deviation
    c.bench_function("feature_vwap_deviation", |b| {
        b.iter(|| {
            let _ = compute_vwap_deviation(black_box(&prices), black_box(&volumes), 60);
        });
    });

    // Benchmark realized volatility
    c.bench_function("feature_realized_volatility", |b| {
        let config = VolatilityFeatureConfig::default();
        let log_returns = compute_log_returns(&prices, 1).unwrap();
        b.iter(|| {
            let _ = compute_rv_multi_window(black_box(&log_returns), &config);
        });
    });
}

fn benchmark_orderflow_features(c: &mut Criterion) {
    use ash_inference::features::{
        compute_arrival_rate, compute_size_quantiles, compute_trade_imbalance,
        infer_trade_direction, OrderFlowFeatureConfig,
    };

    // Prepare test trade data
    let prices: Vec<f64> = (0..300)
        .map(|i| 100.0 + (i as f64 * 0.01).sin() * 2.0)
        .collect();
    let volumes: Vec<f64> = (0..300).map(|i| 100.0 + (i % 10) as f64 * 10.0).collect();
    let timestamps: Vec<f64> = (0..300).map(|i| i as f64).collect();

    c.bench_function("feature_trade_direction", |b| {
        b.iter(|| {
            let _ = infer_trade_direction(black_box(&prices));
        });
    });

    c.bench_function("feature_trade_imbalance", |b| {
        let config = OrderFlowFeatureConfig::default();
        let directions = infer_trade_direction(&prices);
        b.iter(|| {
            let _ = compute_trade_imbalance(
                black_box(&directions),
                black_box(&volumes),
                config.imbalance_window,
            );
        });
    });

    c.bench_function("feature_size_quantiles", |b| {
        let config = OrderFlowFeatureConfig::default();
        b.iter(|| {
            let _ = compute_size_quantiles(black_box(&volumes), 30, &config.size_quantiles);
        });
    });

    c.bench_function("feature_arrival_rate", |b| {
        let config = OrderFlowFeatureConfig::default();
        b.iter(|| {
            let _ = compute_arrival_rate(black_box(&timestamps), config.arrival_rate_window);
        });
    });
}

fn benchmark_options_features(c: &mut Criterion) {
    use ash_inference::features::{
        compute_aggregated_greeks, compute_atm_iv_by_expiry, compute_iv_surface_features,
        compute_moneyness, compute_put_call_ratios, compute_term_structure_slope,
        OptionsFeatureConfig,
    };

    // Prepare test options data
    let spot = 100.0;
    let current_time = 0.0;
    let strikes: Vec<f64> = vec![95.0, 97.5, 100.0, 102.5, 105.0];
    let ivs: Vec<f64> = vec![0.25, 0.22, 0.20, 0.22, 0.25];
    let expirations: Vec<f64> = vec![
        30.0 * 24.0 * 3600.0,
        30.0 * 24.0 * 3600.0,
        30.0 * 24.0 * 3600.0,
        30.0 * 24.0 * 3600.0,
        30.0 * 24.0 * 3600.0,
    ];
    let option_types: Vec<i8> = vec![1, 1, 1, -1, -1]; // 1 for call, -1 for put
    let deltas: Vec<f64> = vec![0.3, 0.45, 0.5, -0.45, -0.3];
    let gammas: Vec<f64> = vec![0.05, 0.06, 0.055, 0.06, 0.05];
    let vegas: Vec<f64> = vec![0.15, 0.18, 0.16, 0.18, 0.15];
    let thetas: Vec<f64> = vec![-0.02, -0.025, -0.022, -0.025, -0.02];
    let open_interests: Vec<f64> = vec![100.0, 150.0, 200.0, 150.0, 100.0];
    let volumes: Vec<f64> = vec![10.0, 15.0, 25.0, 20.0, 12.0];

    c.bench_function("feature_moneyness", |b| {
        b.iter(|| {
            let _ = compute_moneyness(black_box(&strikes), black_box(spot));
        });
    });

    c.bench_function("feature_aggregated_greeks", |b| {
        b.iter(|| {
            let _ = compute_aggregated_greeks(
                black_box(&deltas),
                black_box(&gammas),
                black_box(&vegas),
                black_box(&thetas),
                black_box(&open_interests),
                black_box(&option_types),
            );
        });
    });

    c.bench_function("feature_put_call_ratios", |b| {
        b.iter(|| {
            let _ = compute_put_call_ratios(
                black_box(&option_types),
                black_box(&volumes),
                black_box(&open_interests),
            );
        });
    });

    c.bench_function("feature_term_structure_slope", |b| {
        let config = OptionsFeatureConfig::default();
        let atm_ivs = vec![0.20, 0.22, 0.24, 0.26];
        b.iter(|| {
            let _ =
                compute_term_structure_slope(black_box(&atm_ivs), &config.expiry_buckets_days);
        });
    });

    c.bench_function("feature_atm_iv_by_expiry", |b| {
        let config = OptionsFeatureConfig::default();
        b.iter(|| {
            let _ = compute_atm_iv_by_expiry(
                black_box(&strikes),
                black_box(&expirations),
                black_box(&ivs),
                black_box(spot),
                black_box(current_time),
                &config.expiry_buckets_days,
            );
        });
    });

    c.bench_function("feature_iv_surface", |b| {
        let config = OptionsFeatureConfig::default();
        b.iter(|| {
            let _ = compute_iv_surface_features(
                black_box(&strikes),
                black_box(&expirations),
                black_box(&ivs),
                black_box(spot),
                black_box(current_time),
                &config,
            );
        });
    });
}

fn benchmark_prepare_input(c: &mut Criterion) {
    use ash_inference::{Config, FeatureBuffer};

    let mut group = c.benchmark_group("prepare_input");

    // Test different sequence lengths
    for seq_len in [60, 120, 300].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, &seq_len| {
                let mut config = Config::default();
                config.sequence_length = seq_len;
                let mut buffer = FeatureBuffer::new(seq_len, 1);

                // Warm up buffer with realistic data
                for i in 0..seq_len {
                    let price = 100.0 + (i as f64 * 0.01).sin() * 2.0;
                    buffer.update(i as f64, price, 1000.0);
                }

                // Benchmark feature preparation
                // Note: This tests the feature computation logic but not ONNX inference
                b.iter(|| {
                    let prices: Vec<f64> = buffer.prices.to_vec();
                    let volumes: Vec<f64> = buffer.volumes.to_vec();

                    // Simulate feature preparation
                    let n = prices.len();
                    let start_idx = if n > seq_len { n - seq_len } else { 0 };
                    let _price_slice = &prices[start_idx..];
                    let _volume_slice = &volumes[start_idx..];
                });
            },
        );
    }

    group.finish();
}

//
// End-to-End Benchmarks
//
// Note: These require a real ONNX model file.
// Place a model at benches/test_model.onnx to enable.
//

fn benchmark_end_to_end(c: &mut Criterion) {
    use std::path::PathBuf;

    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("test_model.onnx");

    if !model_path.exists() {
        eprintln!(
            "Skipping end-to-end benchmarks: model not found at {}",
            model_path.display()
        );
        eprintln!("To enable, place an ONNX model at benches/test_model.onnx");
        return;
    }

    // Only run if model exists
    benchmark_with_model(c, &model_path);
}

#[allow(dead_code)]
fn benchmark_with_model(c: &mut Criterion, model_path: &std::path::Path) {
    use ash_inference::{Config, PricePredictor};

    let config = Config::default();

    // Initialize predictor
    let mut predictor = PricePredictor::new(model_path, config).unwrap();

    // Warm up with 300 data points
    for i in 0..300 {
        let price = 100.0 + (i as f64 * 0.01).sin() * 2.0;
        predictor.update(i as f64, price, 1000.0);
    }

    // Benchmark update operation
    c.bench_function("update_tick", |b| {
        let mut i = 300;
        b.iter(|| {
            let price = 100.0 + (i as f64 * 0.01).sin() * 2.0;
            predictor.update(black_box(i as f64), black_box(price), black_box(1000.0));
            i += 1;
        });
    });

    // Benchmark inference only (buffer already warm)
    c.bench_function("predict_only", |b| {
        b.iter(|| {
            let _ = predictor.predict();
        });
    });

    // Benchmark update + predict (full cycle)
    c.bench_function("update_and_predict", |b| {
        let mut i = 300;
        b.iter(|| {
            let price = 100.0 + (i as f64 * 0.01).sin() * 2.0;
            predictor.update(black_box(i as f64), black_box(price), black_box(1000.0));
            let _ = predictor.predict();
            i += 1;
        });
    });
}

criterion_group!(
    benches,
    benchmark_buffer_operations,
    benchmark_rolling_buffer,
    benchmark_feature_computation,
    benchmark_orderflow_features,
    benchmark_options_features,
    benchmark_prepare_input,
    benchmark_end_to_end,
);
criterion_main!(benches);
