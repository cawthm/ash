//! Feature computation for model inputs.
//!
//! Implements feature calculation logic ported from Python (data/processors/feature_builder.py).
//! This module computes price history and volatility features from tick data.
//!
//! **Critical for correctness**: Feature computation must match Python implementation exactly
//! to avoid train/serve skew. See tests/feature_parity for validation tests.

use crate::Result;
use serde::{Deserialize, Serialize};

/// Configuration for price history features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceFeatureConfig {
    /// Lookback windows (in seconds) for computing log returns
    pub return_windows: Vec<u32>,

    /// Whether to include VWAP deviation feature
    pub include_vwap: bool,

    /// Sampling interval in seconds
    pub sample_interval: u32,
}

impl Default for PriceFeatureConfig {
    fn default() -> Self {
        Self {
            return_windows: vec![1, 5, 10, 30, 60, 120, 300],
            include_vwap: true,
            sample_interval: 1,
        }
    }
}

/// Configuration for volatility features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityFeatureConfig {
    /// Windows (in seconds) for realized volatility calculation
    pub rv_windows: Vec<u32>,

    /// Whether to include IV/RV spread feature
    pub include_iv_rv_spread: bool,

    /// Whether to include volatility of volatility
    pub include_vol_of_vol: bool,

    /// Annualization factor for volatility (sqrt of trading seconds per year)
    /// Default: sqrt(252 * 6.5 * 3600) ≈ 3039.1
    pub annualization_factor: f64,
}

impl Default for VolatilityFeatureConfig {
    fn default() -> Self {
        Self {
            rv_windows: vec![60, 300, 600],
            include_iv_rv_spread: true,
            include_vol_of_vol: true,
            // 252 trading days * 6.5 hours * 3600 seconds
            annualization_factor: (252.0_f64 * 6.5 * 3600.0).sqrt(),
        }
    }
}

/// Compute log returns over a lookback window.
///
/// Matches Python: `PriceFeatureBuilder.compute_log_returns()`
///
/// # Arguments
/// * `prices` - Price array, ordered from oldest to newest
/// * `window` - Lookback window in number of samples
///
/// # Returns
/// Log returns where window is valid, NaN elsewhere
pub fn compute_log_returns(prices: &[f64], window: usize) -> Result<Vec<f64>> {
    if window < 1 {
        anyhow::bail!("window must be at least 1");
    }

    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    if window >= n {
        return Ok(result);
    }

    // Current prices and lagged prices
    for i in window..n {
        let current = prices[i];
        let lagged = prices[i - window];

        // Log return = ln(current / lagged)
        // Handle zero/negative prices by leaving as NaN
        if current > 0.0 && lagged > 0.0 {
            result[i] = (current / lagged).ln();
        }
    }

    Ok(result)
}

/// Compute log returns at multiple lookback windows.
///
/// Matches Python: `PriceFeatureBuilder.compute_returns_multi_window()`
///
/// # Arguments
/// * `prices` - Price array
/// * `config` - Price feature configuration
///
/// # Returns
/// 2D array (time x windows) with log returns for each configured window
pub fn compute_returns_multi_window(
    prices: &[f64],
    config: &PriceFeatureConfig,
) -> Result<Vec<Vec<f64>>> {
    let n = prices.len();
    let num_windows = config.return_windows.len();
    let mut result = vec![vec![f64::NAN; n]; num_windows];

    for (i, &window_seconds) in config.return_windows.iter().enumerate() {
        let window_samples = (window_seconds / config.sample_interval) as usize;
        if window_samples >= 1 {
            result[i] = compute_log_returns(prices, window_samples)?;
        }
    }

    Ok(result)
}

/// Compute Volume Weighted Average Price over a rolling window.
///
/// Matches Python: `PriceFeatureBuilder.compute_vwap()`
///
/// # Arguments
/// * `prices` - Price array
/// * `volumes` - Volume array, same length as prices
/// * `window` - Rolling window size in samples
///
/// # Returns
/// VWAP values, NaN where window extends before data start
pub fn compute_vwap(prices: &[f64], volumes: &[f64], window: usize) -> Result<Vec<f64>> {
    if prices.len() != volumes.len() {
        anyhow::bail!("prices and volumes must have same length");
    }
    if window < 1 {
        anyhow::bail!("window must be at least 1");
    }

    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    if window > n {
        return Ok(result);
    }

    // Price * Volume for each sample
    let pv: Vec<f64> = prices
        .iter()
        .zip(volumes.iter())
        .map(|(&p, &v)| p * v)
        .collect();

    // Rolling sums using cumsum for efficiency
    let mut cumsum_pv = vec![0.0; n];
    let mut cumsum_v = vec![0.0; n];

    cumsum_pv[0] = pv[0];
    cumsum_v[0] = volumes[0];
    for i in 1..n {
        cumsum_pv[i] = cumsum_pv[i - 1] + pv[i];
        cumsum_v[i] = cumsum_v[i - 1] + volumes[i];
    }

    // First valid position
    let valid_v = cumsum_v[window - 1].max(1e-10);
    result[window - 1] = cumsum_pv[window - 1] / valid_v;

    // Remaining positions
    if window < n {
        for i in window..n {
            let rolling_pv = cumsum_pv[i] - cumsum_pv[i - window];
            let rolling_v = cumsum_v[i] - cumsum_v[i - window];
            let valid_v = rolling_v.max(1e-10);
            result[i] = rolling_pv / valid_v;
        }
    }

    Ok(result)
}

/// Compute price deviation from VWAP as a log ratio.
///
/// Matches Python: `PriceFeatureBuilder.compute_vwap_deviation()`
///
/// # Arguments
/// * `prices` - Price array
/// * `volumes` - Volume array
/// * `window` - Rolling window for VWAP calculation
///
/// # Returns
/// Log ratio of current price to VWAP: ln(price / vwap)
pub fn compute_vwap_deviation(
    prices: &[f64],
    volumes: &[f64],
    window: usize,
) -> Result<Vec<f64>> {
    let vwap = compute_vwap(prices, volumes, window)?;

    let mut result = vec![f64::NAN; prices.len()];
    for i in 0..prices.len() {
        if vwap[i] > 0.0 && prices[i] > 0.0 && !vwap[i].is_nan() {
            result[i] = (prices[i] / vwap[i]).ln();
        }
    }

    Ok(result)
}

/// Compute realized volatility over a rolling window.
///
/// Uses standard deviation of log returns, annualized.
///
/// Matches Python: `VolatilityFeatureBuilder.compute_realized_volatility()`
///
/// # Arguments
/// * `log_returns` - Log returns (not percentage)
/// * `window` - Rolling window size in samples
/// * `annualization_factor` - Factor to annualize volatility
///
/// # Returns
/// Annualized realized volatility as percentage (e.g., 20.0 for 20%)
pub fn compute_realized_volatility(
    log_returns: &[f64],
    window: usize,
    annualization_factor: f64,
) -> Result<Vec<f64>> {
    if window < 2 {
        anyhow::bail!("window must be at least 2 for volatility calculation");
    }

    let n = log_returns.len();
    let mut result = vec![f64::NAN; n];

    if window > n {
        return Ok(result);
    }

    // Rolling standard deviation using cumsum approach
    // Var(X) = E[X^2] - E[X]^2
    let mut cumsum = vec![0.0; n];
    let mut cumsum_sq = vec![0.0; n];

    cumsum[0] = log_returns[0];
    cumsum_sq[0] = log_returns[0] * log_returns[0];
    for i in 1..n {
        cumsum[i] = cumsum[i - 1] + log_returns[i];
        cumsum_sq[i] = cumsum_sq[i - 1] + log_returns[i] * log_returns[i];
    }

    // First valid position
    let mean_first = cumsum[window - 1] / window as f64;
    let mean_sq_first = cumsum_sq[window - 1] / window as f64;
    let var_first = (mean_sq_first - mean_first * mean_first).max(0.0);
    result[window - 1] = var_first.sqrt() * annualization_factor * 100.0;

    // Remaining positions
    if window < n {
        for i in window..n {
            let rolling_sum = cumsum[i] - cumsum[i - window];
            let rolling_sum_sq = cumsum_sq[i] - cumsum_sq[i - window];
            let rolling_mean = rolling_sum / window as f64;
            let rolling_mean_sq = rolling_sum_sq / window as f64;
            let rolling_var = (rolling_mean_sq - rolling_mean * rolling_mean).max(0.0);
            result[i] = rolling_var.sqrt() * annualization_factor * 100.0;
        }
    }

    Ok(result)
}

/// Compute realized volatility at multiple windows.
///
/// Matches Python: `VolatilityFeatureBuilder.compute_rv_multi_window()`
///
/// # Arguments
/// * `log_returns` - Log returns array
/// * `config` - Volatility feature configuration
///
/// # Returns
/// 2D array (time x windows) with RV values for each configured window
pub fn compute_rv_multi_window(
    log_returns: &[f64],
    config: &VolatilityFeatureConfig,
) -> Result<Vec<Vec<f64>>> {
    let n = log_returns.len();
    let num_windows = config.rv_windows.len();
    let mut result = vec![vec![f64::NAN; n]; num_windows];

    for (i, &window_seconds) in config.rv_windows.iter().enumerate() {
        let window_samples = window_seconds as usize; // Assuming 1s sample interval
        if window_samples >= 2 {
            result[i] = compute_realized_volatility(
                log_returns,
                window_samples,
                config.annualization_factor,
            )?;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_log_returns_basic() {
        let prices = vec![100.0, 101.0, 102.0, 101.5];
        let returns = compute_log_returns(&prices, 1).unwrap();

        assert!(returns[0].is_nan());
        assert!((returns[1] - (101.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[2] - (102.0_f64 / 101.0).ln()).abs() < 1e-10);
        assert!((returns[3] - (101.5_f64 / 102.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_log_returns_window() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let returns = compute_log_returns(&prices, 2).unwrap();

        assert!(returns[0].is_nan());
        assert!(returns[1].is_nan());
        assert!((returns[2] - (102.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[3] - (103.0_f64 / 101.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_log_returns_zero_price() {
        let prices = vec![100.0, 0.0, 102.0];
        let returns = compute_log_returns(&prices, 1).unwrap();

        assert!(returns[0].is_nan());
        assert!(returns[1].is_nan()); // Zero price should be NaN
        assert!(returns[2].is_nan()); // Previous is zero
    }

    #[test]
    fn test_compute_vwap_basic() {
        let prices = vec![100.0, 101.0, 102.0];
        let volumes = vec![10.0, 20.0, 30.0];
        let vwap = compute_vwap(&prices, &volumes, 2).unwrap();

        assert!(vwap[0].is_nan());
        // Window [100, 101] with volumes [10, 20]
        // VWAP = (100*10 + 101*20) / (10 + 20) = 3020 / 30 = 100.666...
        let expected1 = (100.0 * 10.0 + 101.0 * 20.0) / 30.0;
        assert!((vwap[1] - expected1).abs() < 1e-10);
    }

    #[test]
    fn test_compute_vwap_deviation() {
        let prices = vec![100.0, 101.0, 102.0];
        let volumes = vec![10.0, 10.0, 10.0];
        let dev = compute_vwap_deviation(&prices, &volumes, 2).unwrap();

        assert!(dev[0].is_nan());
        // VWAP at index 1 = (100 + 101) / 2 = 100.5
        // Deviation = ln(101 / 100.5)
        let expected_vwap1 = 100.5;
        let expected_dev1 = (101.0_f64 / expected_vwap1).ln();
        assert!((dev[1] - expected_dev1).abs() < 1e-10);
    }

    #[test]
    fn test_compute_realized_volatility() {
        // Simple test with known returns
        let returns = vec![0.01, -0.01, 0.02, -0.02, 0.01];
        let rv = compute_realized_volatility(&returns, 3, 1.0).unwrap();

        assert!(rv[0].is_nan());
        assert!(rv[1].is_nan());
        assert!(rv[2] > 0.0); // Should be positive
    }

    #[test]
    fn test_compute_returns_multi_window() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let config = PriceFeatureConfig {
            return_windows: vec![1, 2],
            include_vwap: false,
            sample_interval: 1,
        };

        let returns = compute_returns_multi_window(&prices, &config).unwrap();

        assert_eq!(returns.len(), 2); // Two windows
        assert_eq!(returns[0].len(), 5); // Same length as prices

        // Window 1: should match compute_log_returns(prices, 1)
        assert!(returns[0][0].is_nan());
        assert!((returns[0][1] - (101.0_f64 / 100.0).ln()).abs() < 1e-10);

        // Window 2: should match compute_log_returns(prices, 2)
        assert!(returns[1][0].is_nan());
        assert!(returns[1][1].is_nan());
        assert!((returns[1][2] - (102.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_price_feature_config_default() {
        let config = PriceFeatureConfig::default();
        assert_eq!(config.return_windows.len(), 7);
        assert!(config.include_vwap);
        assert_eq!(config.sample_interval, 1);
    }

    #[test]
    fn test_volatility_feature_config_default() {
        let config = VolatilityFeatureConfig::default();
        assert_eq!(config.rv_windows.len(), 3);
        assert!(config.include_iv_rv_spread);
        assert!(config.include_vol_of_vol);
        // Check annualization factor is approximately correct
        let expected = (252.0_f64 * 6.5 * 3600.0).sqrt();
        assert!((config.annualization_factor - expected).abs() < 1.0);
    }
}
