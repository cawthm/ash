//! Price prediction inference interface.
//!
//! Provides the main PricePredictor struct for loading ONNX models
//! and generating probability distributions over future prices.

use crate::buffer::FeatureBuffer;
use crate::features::{
    compute_log_returns, compute_returns_multi_window, compute_rv_multi_window,
    compute_vwap_deviation, PriceFeatureConfig, VolatilityFeatureConfig,
};
use crate::Result;
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session, SessionOutputs};
use ort::value::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Time horizon for price predictions (in seconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Horizon {
    OneSecond = 1,
    FiveSeconds = 5,
    TenSeconds = 10,
    ThirtySeconds = 30,
    SixtySeconds = 60,
    TwoMinutes = 120,
    FiveMinutes = 300,
    TenMinutes = 600,
}

impl Horizon {
    /// Get all horizons in order.
    pub fn all() -> Vec<Horizon> {
        vec![
            Horizon::OneSecond,
            Horizon::FiveSeconds,
            Horizon::TenSeconds,
            Horizon::ThirtySeconds,
            Horizon::SixtySeconds,
            Horizon::TwoMinutes,
            Horizon::FiveMinutes,
            Horizon::TenMinutes,
        ]
    }

    /// Get the horizon value in seconds.
    pub fn as_seconds(&self) -> u32 {
        *self as u32
    }
}

/// Configuration for the price predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Number of input features (auto-computed from feature configs)
    pub num_features: usize,

    /// Sequence length (number of time steps)
    pub sequence_length: usize,

    /// Number of price buckets (discretization)
    pub num_buckets: usize,

    /// Lookback window in seconds
    pub lookback_seconds: usize,

    /// Data frequency in Hz
    pub frequency_hz: usize,

    /// Enable graph optimization
    pub optimize_graph: bool,

    /// Price feature configuration
    pub price_features: PriceFeatureConfig,

    /// Volatility feature configuration
    pub volatility_features: VolatilityFeatureConfig,
}

impl Default for Config {
    fn default() -> Self {
        let price_features = PriceFeatureConfig::default();
        let volatility_features = VolatilityFeatureConfig::default();

        // Compute total feature count:
        // - Multi-window returns: 7 windows
        // - VWAP deviation: 1 feature
        // - Multi-window RV: 3 windows
        // Total: 7 + 1 + 3 = 11 features
        let num_features = price_features.return_windows.len()
            + if price_features.include_vwap { 1 } else { 0 }
            + volatility_features.rv_windows.len();

        Self {
            num_features,
            sequence_length: 300,     // 300 seconds at 1 Hz
            num_buckets: 101,         // -50 to +50 bps
            lookback_seconds: 300,    // 5 minutes
            frequency_hz: 1,          // 1 Hz sampling
            optimize_graph: true,
            price_features,
            volatility_features,
        }
    }
}

/// Result of a price prediction containing probability distributions.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Probability distribution for each horizon
    pub distributions: HashMap<Horizon, Vec<f32>>,

    /// Timestamp when prediction was made
    pub timestamp: f64,
}

impl PredictionResult {
    /// Get the predicted distribution for a specific horizon.
    pub fn get_distribution(&self, horizon: Horizon) -> Option<&Vec<f32>> {
        self.distributions.get(&horizon)
    }

    /// Get the predicted mean (in basis points) for a horizon.
    ///
    /// Assumes buckets are centered at -50, -49, ..., 0, ..., 49, 50 bps.
    pub fn predicted_mean_bps(&self, horizon: Horizon, num_buckets: usize) -> Option<f64> {
        let dist = self.get_distribution(horizon)?;

        // Bucket centers from -50 to +50 bps
        let min_bps = -50.0;
        let max_bps = 50.0;
        let range = max_bps - min_bps;

        let mut mean = 0.0;
        for (i, &prob) in dist.iter().enumerate() {
            let bucket_center = min_bps + (i as f64 * range) / (num_buckets - 1) as f64;
            mean += bucket_center * prob as f64;
        }

        Some(mean)
    }
}

/// Main price predictor using ONNX model inference.
pub struct PricePredictor {
    /// ONNX Runtime session
    session: Session,

    /// Feature buffer for maintaining state
    feature_buffer: FeatureBuffer,

    /// Configuration
    config: Config,
}

impl PricePredictor {
    /// Load ONNX model and initialize predictor.
    ///
    /// # Arguments
    /// * `model_path` - Path to ONNX model file
    /// * `config` - Predictor configuration
    ///
    /// # Example
    /// ```no_run
    /// use ash_inference::{PricePredictor, Config};
    /// use std::path::Path;
    ///
    /// let config = Config::default();
    /// let predictor = PricePredictor::new(
    ///     Path::new("model.onnx"),
    ///     config
    /// ).unwrap();
    /// ```
    pub fn new(model_path: &Path, config: Config) -> Result<Self> {
        // Initialize ONNX Runtime session
        let session = Session::builder()?
            .with_optimization_level(if config.optimize_graph {
                GraphOptimizationLevel::Level3
            } else {
                GraphOptimizationLevel::Level1
            })?
            .with_intra_threads(1)? // Single-threaded for determinism
            .commit_from_file(model_path)?;

        // Initialize feature buffer
        let feature_buffer =
            FeatureBuffer::new(config.lookback_seconds, config.frequency_hz);

        Ok(Self {
            session,
            feature_buffer,
            config,
        })
    }

    /// Update internal state with new tick data.
    ///
    /// # Arguments
    /// * `timestamp` - Unix timestamp in seconds
    /// * `price` - Current price
    /// * `volume` - Current volume
    pub fn update(&mut self, timestamp: f64, price: f64, volume: f64) {
        self.feature_buffer.update(timestamp, price, volume);
    }

    /// Generate price predictions for all horizons.
    ///
    /// Returns `None` if insufficient data is buffered.
    ///
    /// # Returns
    /// Probability distributions for each prediction horizon
    pub fn predict(&mut self) -> Result<Option<PredictionResult>> {
        // Check if we have sufficient data
        if !self
            .feature_buffer
            .has_sufficient_data(self.config.sequence_length)
        {
            return Ok(None);
        }

        // Prepare input tensor (ndarray)
        let input_array = self.prepare_input()?;

        // Convert to ort Value
        let input_tensor = Value::from_array(input_array)?;

        // Run inference
        let outputs = self.session.run(ort::inputs![input_tensor])?;

        // Parse outputs into distributions
        let distributions = Self::parse_outputs(&outputs)?;

        Ok(Some(PredictionResult {
            distributions,
            timestamp: self
                .feature_buffer
                .timestamps
                .iter()
                .last()
                .copied()
                .unwrap_or(0.0),
        }))
    }

    /// Clear the feature buffer (e.g., on market open/close).
    pub fn clear(&mut self) {
        self.feature_buffer.clear();
    }

    /// Get the current number of buffered data points.
    pub fn buffer_len(&self) -> usize {
        self.feature_buffer.len()
    }

    /// Prepare input tensor from feature buffer.
    ///
    /// Computes features using integrated feature computation from features.rs:
    /// - Multi-window log returns
    /// - VWAP deviation
    /// - Multi-window realized volatility
    fn prepare_input(&self) -> Result<Array2<f32>> {
        let seq_len = self.config.sequence_length;
        let num_features = self.config.num_features;

        // Get last sequence_length data points
        let all_prices: Vec<f64> = self.feature_buffer.prices.to_vec();
        let all_volumes: Vec<f64> = self.feature_buffer.volumes.to_vec();

        let n = all_prices.len();
        let start_idx = if n > seq_len { n - seq_len } else { 0 };

        let prices: Vec<f64> = all_prices[start_idx..].to_vec();
        let volumes: Vec<f64> = all_volumes[start_idx..].to_vec();

        // Initialize feature matrix
        let mut features = Array2::<f32>::zeros((seq_len, num_features));

        // Track feature column index
        let mut col = 0;

        // 1. Multi-window log returns
        let returns_multi = compute_returns_multi_window(&prices, &self.config.price_features)?;
        for window_returns in returns_multi.iter() {
            for (i, &ret) in window_returns.iter().enumerate() {
                if i < seq_len {
                    features[[i, col]] = if ret.is_finite() { ret as f32 } else { 0.0 };
                }
            }
            col += 1;
        }

        // 2. VWAP deviation (if enabled)
        if self.config.price_features.include_vwap {
            // Use 60-second VWAP window (configurable in future)
            let vwap_window = 60;
            let vwap_dev = compute_vwap_deviation(&prices, &volumes, vwap_window)?;
            for (i, &dev) in vwap_dev.iter().enumerate() {
                if i < seq_len {
                    features[[i, col]] = if dev.is_finite() { dev as f32 } else { 0.0 };
                }
            }
            col += 1;
        }

        // 3. Multi-window realized volatility
        // First compute 1-second log returns for RV calculation
        let log_returns = compute_log_returns(&prices, 1)?;
        let rv_multi = compute_rv_multi_window(&log_returns, &self.config.volatility_features)?;
        for window_rv in rv_multi.iter() {
            for (i, &rv) in window_rv.iter().enumerate() {
                if i < seq_len {
                    features[[i, col]] = if rv.is_finite() { rv as f32 } else { 0.0 };
                }
            }
            col += 1;
        }

        Ok(features)
    }

    /// Parse ONNX model outputs into probability distributions.
    fn parse_outputs(outputs: &SessionOutputs) -> Result<HashMap<Horizon, Vec<f32>>> {
        let mut distributions = HashMap::new();

        // Model outputs one tensor per horizon
        // Output names: "output_1s", "output_5s", etc.
        for horizon in Horizon::all() {
            let output_name = format!("output_{}s", horizon.as_seconds());

            if let Some(output) = outputs.get(&output_name) {
                let tensor_data = output.try_extract_tensor::<f32>()?;
                let dist: Vec<f32> = tensor_data.1.to_vec();

                // Verify distribution sums to ~1.0
                let sum: f32 = dist.iter().sum();
                if (sum - 1.0).abs() > 0.01 {
                    anyhow::bail!(
                        "Invalid probability distribution for {}: sum = {}",
                        output_name,
                        sum
                    );
                }

                distributions.insert(horizon, dist);
            }
        }

        if distributions.is_empty() {
            anyhow::bail!("No valid model outputs found");
        }

        Ok(distributions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizon_all() {
        let horizons = Horizon::all();
        assert_eq!(horizons.len(), 8);
        assert_eq!(horizons[0], Horizon::OneSecond);
        assert_eq!(horizons[7], Horizon::TenMinutes);
    }

    #[test]
    fn test_horizon_as_seconds() {
        assert_eq!(Horizon::OneSecond.as_seconds(), 1);
        assert_eq!(Horizon::FiveMinutes.as_seconds(), 300);
        assert_eq!(Horizon::TenMinutes.as_seconds(), 600);
    }

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.num_buckets, 101);
        assert_eq!(config.lookback_seconds, 300);
        assert_eq!(config.sequence_length, 300);
        // Verify num_features = 7 (returns) + 1 (vwap) + 3 (rv) = 11
        assert_eq!(config.num_features, 11);
        assert_eq!(config.price_features.return_windows.len(), 7);
        assert_eq!(config.volatility_features.rv_windows.len(), 3);
    }

    #[test]
    fn test_prediction_result_mean() {
        let mut distributions = HashMap::new();

        // Create a distribution centered at 0 (bucket 50)
        let mut dist = vec![0.0f32; 101];
        dist[50] = 1.0; // All probability at center bucket

        distributions.insert(Horizon::OneSecond, dist);

        let result = PredictionResult {
            distributions,
            timestamp: 1000.0,
        };

        let mean = result.predicted_mean_bps(Horizon::OneSecond, 101).unwrap();
        assert!((mean - 0.0).abs() < 0.01); // Should be ~0 bps
    }

    #[test]
    fn test_prediction_result_mean_positive() {
        let mut distributions = HashMap::new();

        // Create a distribution at +10 bps (bucket 60)
        let mut dist = vec![0.0f32; 101];
        dist[60] = 1.0;

        distributions.insert(Horizon::OneSecond, dist);

        let result = PredictionResult {
            distributions,
            timestamp: 1000.0,
        };

        let mean = result.predicted_mean_bps(Horizon::OneSecond, 101).unwrap();
        assert!((mean - 10.0).abs() < 1.0); // Should be ~10 bps
    }
}
