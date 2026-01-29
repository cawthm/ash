//! Production inference for probabilistic price prediction.
//!
//! This crate provides low-latency ONNX model inference for real-time trading decisions.
//! Target: <10ms end-to-end latency.

pub mod buffer;
pub mod features;
pub mod predictor;

pub use buffer::{FeatureBuffer, RollingBuffer};
pub use features::{PriceFeatureConfig, VolatilityFeatureConfig};
pub use predictor::{Config, Horizon, PricePredictor, PredictionResult};

/// Library-wide error type.
pub type Result<T> = anyhow::Result<T>;
