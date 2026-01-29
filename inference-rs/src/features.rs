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

/// Configuration for order flow features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowFeatureConfig {
    /// Window (in samples) for trade direction imbalance
    pub imbalance_window: usize,

    /// Whether to include trade size distribution features
    pub include_size_distribution: bool,

    /// Whether to include trade arrival rate
    pub include_arrival_rate: bool,

    /// Quantiles for trade size distribution (as fractions)
    pub size_quantiles: Vec<f64>,

    /// Window (in samples) for arrival rate calculation
    pub arrival_rate_window: usize,
}

impl Default for OrderFlowFeatureConfig {
    fn default() -> Self {
        Self {
            imbalance_window: 30,
            include_size_distribution: true,
            include_arrival_rate: true,
            size_quantiles: vec![0.25, 0.5, 0.75],
            arrival_rate_window: 60,
        }
    }
}

/// Configuration for options-derived features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionsFeatureConfig {
    /// Whether to include IV surface features (ATM IV by expiry)
    pub include_iv_surface: bool,

    /// Whether to include aggregated Greeks features
    pub include_greeks: bool,

    /// Whether to include put/call volume/OI ratios
    pub include_put_call_ratio: bool,

    /// Whether to include IV term structure slope
    pub include_term_structure: bool,

    /// Moneyness levels for IV surface (as ratios to spot)
    pub iv_moneyness_buckets: Vec<f64>,

    /// Days to expiration buckets for term structure
    pub expiry_buckets_days: Vec<u32>,
}

impl Default for OptionsFeatureConfig {
    fn default() -> Self {
        Self {
            include_iv_surface: true,
            include_greeks: true,
            include_put_call_ratio: true,
            include_term_structure: true,
            iv_moneyness_buckets: vec![0.95, 0.975, 1.0, 1.025, 1.05],
            expiry_buckets_days: vec![7, 30, 60, 90],
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

/// Infer trade direction using the tick rule.
///
/// Matches Python: `OrderFlowFeatureBuilder.infer_trade_direction()`
///
/// The tick rule classifies trades as:
/// - Buy (+1): if price > previous price (uptick)
/// - Sell (-1): if price < previous price (downtick)
/// - Unknown (0): if price == previous price (zero tick)
///
/// For zero ticks, we propagate the last known direction.
///
/// # Arguments
/// * `prices` - 1D array of trade prices
///
/// # Returns
/// Array of trade directions: +1 (buy), -1 (sell), 0 (unknown)
pub fn infer_trade_direction(prices: &[f64]) -> Vec<i8> {
    let n = prices.len();
    if n == 0 {
        return vec![];
    }

    let mut direction = vec![0i8; n];

    if n > 1 {
        // Compare each price to previous
        for i in 1..n {
            let diff = prices[i] - prices[i - 1];
            direction[i] = if diff > 0.0 {
                1
            } else if diff < 0.0 {
                -1
            } else {
                0
            };
        }

        // Propagate last known direction for zero ticks
        let mut last_dir: i8 = 0;
        for i in 0..n {
            if direction[i] != 0 {
                last_dir = direction[i];
            } else {
                direction[i] = last_dir;
            }
        }
    }

    direction
}

/// Compute volume-weighted trade direction imbalance.
///
/// Matches Python: `OrderFlowFeatureBuilder.compute_trade_imbalance()`
///
/// Imbalance = (buy_volume - sell_volume) / total_volume
///
/// # Arguments
/// * `directions` - Trade directions (+1 buy, -1 sell)
/// * `sizes` - Trade sizes (volumes)
/// * `window` - Rolling window size
///
/// # Returns
/// Imbalance values in range [-1, 1]. NaN where window is invalid.
pub fn compute_trade_imbalance(
    directions: &[i8],
    sizes: &[f64],
    window: usize,
) -> Result<Vec<f64>> {
    if directions.len() != sizes.len() {
        anyhow::bail!("directions and sizes must have same length");
    }
    if window < 1 {
        anyhow::bail!("window must be at least 1");
    }

    let n = directions.len();
    let mut result = vec![f64::NAN; n];

    if window > n {
        return Ok(result);
    }

    // Signed volume: positive for buys, negative for sells
    let signed_volume: Vec<f64> = directions
        .iter()
        .zip(sizes.iter())
        .map(|(&d, &s)| d as f64 * s)
        .collect();

    // Rolling sums using cumsum
    let mut cumsum_signed = vec![0.0; n];
    let mut cumsum_total = vec![0.0; n];

    cumsum_signed[0] = signed_volume[0];
    cumsum_total[0] = sizes[0].abs();
    for i in 1..n {
        cumsum_signed[i] = cumsum_signed[i - 1] + signed_volume[i];
        cumsum_total[i] = cumsum_total[i - 1] + sizes[i].abs();
    }

    // First valid position
    if cumsum_total[window - 1] > 0.0 {
        result[window - 1] = cumsum_signed[window - 1] / cumsum_total[window - 1];
    }

    // Remaining positions
    if window < n {
        for i in window..n {
            let rolling_signed = cumsum_signed[i] - cumsum_signed[i - window];
            let rolling_total = cumsum_total[i] - cumsum_total[i - window];
            if rolling_total > 0.0 {
                result[i] = rolling_signed / rolling_total;
            }
        }
    }

    Ok(result)
}

/// Compute rolling quantiles of trade sizes.
///
/// Matches Python: `OrderFlowFeatureBuilder.compute_size_quantiles()`
///
/// # Arguments
/// * `sizes` - 1D array of trade sizes
/// * `window` - Rolling window size
/// * `quantiles` - Quantiles to compute (e.g., [0.25, 0.5, 0.75])
///
/// # Returns
/// 2D array (time x quantiles) with quantile values. NaN where window is invalid.
pub fn compute_size_quantiles(
    sizes: &[f64],
    window: usize,
    quantiles: &[f64],
) -> Result<Vec<Vec<f64>>> {
    if window < 1 {
        anyhow::bail!("window must be at least 1");
    }

    let n = sizes.len();
    let num_quantiles = quantiles.len();
    let mut result = vec![vec![f64::NAN; num_quantiles]; n];

    if window > n {
        return Ok(result);
    }

    // For each position, compute quantiles of the window
    for i in (window - 1)..n {
        let start_idx = i + 1 - window; // Avoid underflow: i >= window - 1, so i + 1 >= window
        let mut window_data: Vec<f64> = sizes[start_idx..=i].to_vec();
        window_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (j, &q) in quantiles.iter().enumerate() {
            // Linear interpolation for quantile
            let pos = q * (window_data.len() - 1) as f64;
            let lower_idx = pos.floor() as usize;
            let upper_idx = pos.ceil() as usize;
            let frac = pos - lower_idx as f64;

            if lower_idx == upper_idx {
                result[i][j] = window_data[lower_idx];
            } else {
                result[i][j] =
                    window_data[lower_idx] * (1.0 - frac) + window_data[upper_idx] * frac;
            }
        }
    }

    Ok(result)
}

/// Compute trade arrival rate (trades per second).
///
/// Matches Python: `OrderFlowFeatureBuilder.compute_arrival_rate()`
///
/// # Arguments
/// * `timestamps` - 1D array of trade timestamps (in seconds)
/// * `window` - Number of trades to use for rate calculation
///
/// # Returns
/// Arrival rate in trades per second. NaN where window is invalid.
pub fn compute_arrival_rate(timestamps: &[f64], window: usize) -> Result<Vec<f64>> {
    if window < 2 {
        anyhow::bail!("window must be at least 2 for rate calculation");
    }

    let n = timestamps.len();
    let mut result = vec![f64::NAN; n];

    if window > n {
        return Ok(result);
    }

    // Rate = (window - 1) trades / time_elapsed
    for i in (window - 1)..n {
        let start_idx = i + 1 - window; // Avoid underflow
        let time_elapsed = timestamps[i] - timestamps[start_idx];
        if time_elapsed > 0.0 {
            result[i] = (window - 1) as f64 / time_elapsed;
        }
    }

    Ok(result)
}

/// Compute moneyness ratio for each strike.
///
/// Matches Python: `OptionsFeatureBuilder.compute_moneyness()`
///
/// # Arguments
/// * `strikes` - Array of strike prices
/// * `spot_price` - Current underlying spot price
///
/// # Returns
/// Moneyness ratios (strike / spot)
pub fn compute_moneyness(strikes: &[f64], spot_price: f64) -> Result<Vec<f64>> {
    if spot_price <= 0.0 {
        anyhow::bail!("spot_price must be positive");
    }
    Ok(strikes.iter().map(|&s| s / spot_price).collect())
}

/// Compute days to expiration.
///
/// Matches Python: `OptionsFeatureBuilder.compute_days_to_expiry()`
///
/// # Arguments
/// * `expirations` - Array of expiration timestamps (Unix seconds)
/// * `current_time` - Current timestamp (same units as expirations)
///
/// # Returns
/// Days to expiration for each option (non-negative)
pub fn compute_days_to_expiry(expirations: &[f64], current_time: f64) -> Vec<f64> {
    expirations
        .iter()
        .map(|&exp| {
            let dte = (exp - current_time) / (24.0 * 3600.0);
            dte.max(0.0)
        })
        .collect()
}

/// Compute ATM implied volatility for each expiry bucket.
///
/// Matches Python: `OptionsFeatureBuilder.compute_atm_iv_by_expiry()`
///
/// Selects options closest to ATM (moneyness ~1.0) and averages IV within each expiry bucket.
///
/// # Arguments
/// * `strikes` - Array of strike prices
/// * `expirations` - Array of expiration timestamps
/// * `ivs` - Array of implied volatilities (as decimals, e.g., 0.25 for 25%)
/// * `spot_price` - Current underlying spot price
/// * `current_time` - Current timestamp
/// * `expiry_buckets_days` - Expiry buckets in days
///
/// # Returns
/// Array of ATM IVs for each expiry bucket. NaN for buckets with no valid options.
pub fn compute_atm_iv_by_expiry(
    strikes: &[f64],
    expirations: &[f64],
    ivs: &[f64],
    spot_price: f64,
    current_time: f64,
    expiry_buckets_days: &[u32],
) -> Result<Vec<f64>> {
    let moneyness = compute_moneyness(strikes, spot_price)?;
    let dte = compute_days_to_expiry(expirations, current_time);

    let mut result = vec![f64::NAN; expiry_buckets_days.len()];

    // ATM is within 2% of spot
    let atm_mask: Vec<bool> = moneyness.iter().map(|&m| (m - 1.0).abs() < 0.02).collect();

    for (i, &bucket_days) in expiry_buckets_days.iter().enumerate() {
        // Options within ±7 days of bucket target (or 30% of bucket)
        let bucket_margin = (bucket_days as f64 * 0.3).min(7.0);

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for j in 0..strikes.len() {
            let dte_match = (dte[j] - bucket_days as f64).abs() <= bucket_margin;
            let is_valid = atm_mask[j] && dte_match && !ivs[j].is_nan();

            if is_valid {
                // Weight by how close to ATM
                let weight = 1.0 / ((moneyness[j] - 1.0).abs() + 0.001);
                weighted_sum += ivs[j] * weight;
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            result[i] = weighted_sum / weight_sum;
        }
    }

    Ok(result)
}

/// Compute IV surface features across moneyness and expiry.
///
/// Matches Python: `OptionsFeatureBuilder.compute_iv_surface_features()`
///
/// # Arguments
/// * `strikes` - Array of strike prices
/// * `expirations` - Array of expiration timestamps
/// * `ivs` - Array of implied volatilities
/// * `spot_price` - Current underlying spot price
/// * `current_time` - Current timestamp
/// * `config` - Options feature configuration
///
/// # Returns
/// Flattened array of IV values: [expiry1_m1, expiry1_m2, ..., expiry2_m1, ...]
/// Shape: (num_expiry_buckets * num_moneyness_buckets,)
pub fn compute_iv_surface_features(
    strikes: &[f64],
    expirations: &[f64],
    ivs: &[f64],
    spot_price: f64,
    current_time: f64,
    config: &OptionsFeatureConfig,
) -> Result<Vec<f64>> {
    let moneyness = compute_moneyness(strikes, spot_price)?;
    let dte = compute_days_to_expiry(expirations, current_time);

    let num_expiry = config.expiry_buckets_days.len();
    let num_moneyness = config.iv_moneyness_buckets.len();
    let mut result = vec![f64::NAN; num_expiry * num_moneyness];

    for (i, &bucket_days) in config.expiry_buckets_days.iter().enumerate() {
        let bucket_margin = (bucket_days as f64 * 0.3).min(7.0);

        for (j, &target_moneyness) in config.iv_moneyness_buckets.iter().enumerate() {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for k in 0..strikes.len() {
                let dte_match = (dte[k] - bucket_days as f64).abs() <= bucket_margin;
                let moneyness_match = (moneyness[k] - target_moneyness).abs() < 0.015;
                let is_valid = dte_match && moneyness_match && !ivs[k].is_nan();

                if is_valid {
                    // Weight by proximity to target moneyness
                    let weight = 1.0 / ((moneyness[k] - target_moneyness).abs() + 0.001);
                    weighted_sum += ivs[k] * weight;
                    weight_sum += weight;
                }
            }

            if weight_sum > 0.0 {
                result[i * num_moneyness + j] = weighted_sum / weight_sum;
            }
        }
    }

    Ok(result)
}

/// Compute aggregated Greeks weighted by open interest.
///
/// Matches Python: `OptionsFeatureBuilder.compute_aggregated_greeks()`
///
/// Greeks are aggregated as OI-weighted averages, providing a measure of market-wide exposure.
///
/// # Arguments
/// * `deltas` - Array of option deltas
/// * `gammas` - Array of option gammas
/// * `vegas` - Array of option vegas
/// * `thetas` - Array of option thetas
/// * `open_interests` - Array of open interest values
/// * `option_types` - Array of option types (1 for call, -1 for put)
///
/// # Returns
/// Array of [net_delta, total_gamma, total_vega, total_theta]
pub fn compute_aggregated_greeks(
    deltas: &[f64],
    gammas: &[f64],
    vegas: &[f64],
    thetas: &[f64],
    open_interests: &[f64],
    _option_types: &[i8], // Not used in calculation but kept for signature match
) -> Result<Vec<f64>> {
    if deltas.len() != gammas.len()
        || gammas.len() != vegas.len()
        || vegas.len() != thetas.len()
        || thetas.len() != open_interests.len()
    {
        anyhow::bail!("All arrays must have same length");
    }

    let total_oi: f64 = open_interests.iter().sum();

    if total_oi <= 0.0 {
        return Ok(vec![f64::NAN; 4]);
    }

    let mut result = vec![0.0; 4];

    // Net delta (weighted by OI)
    result[0] = deltas
        .iter()
        .zip(open_interests.iter())
        .map(|(&d, &oi)| d * oi)
        .sum::<f64>()
        / total_oi;

    // Total gamma (absolute, both calls and puts contribute positively)
    result[1] = gammas
        .iter()
        .zip(open_interests.iter())
        .map(|(&g, &oi)| g.abs() * oi)
        .sum::<f64>()
        / total_oi;

    // Total vega (positive for both calls and puts)
    result[2] = vegas
        .iter()
        .zip(open_interests.iter())
        .map(|(&v, &oi)| v * oi)
        .sum::<f64>()
        / total_oi;

    // Total theta (typically negative, time decay)
    result[3] = thetas
        .iter()
        .zip(open_interests.iter())
        .map(|(&t, &oi)| t * oi)
        .sum::<f64>()
        / total_oi;

    Ok(result)
}

/// Compute put/call ratios for volume and open interest.
///
/// Matches Python: `OptionsFeatureBuilder.compute_put_call_ratios()`
///
/// # Arguments
/// * `option_types` - Array of option types (1 for call, -1 for put)
/// * `volumes` - Array of trading volumes
/// * `open_interests` - Array of open interest values
///
/// # Returns
/// Array of [volume_pc_ratio, oi_pc_ratio]
/// Ratio is put/call, so >1 means more put activity.
pub fn compute_put_call_ratios(
    option_types: &[i8],
    volumes: &[f64],
    open_interests: &[f64],
) -> Result<Vec<f64>> {
    if option_types.len() != volumes.len() || volumes.len() != open_interests.len() {
        anyhow::bail!("All arrays must have same length");
    }

    let mut result = vec![f64::NAN; 2];

    // Volume ratio
    let call_volume: f64 = option_types
        .iter()
        .zip(volumes.iter())
        .filter(|(&t, _)| t == 1)
        .map(|(_, &v)| v)
        .sum();

    let put_volume: f64 = option_types
        .iter()
        .zip(volumes.iter())
        .filter(|(&t, _)| t == -1)
        .map(|(_, &v)| v)
        .sum();

    if call_volume > 0.0 {
        result[0] = put_volume / call_volume;
    }

    // OI ratio
    let call_oi: f64 = option_types
        .iter()
        .zip(open_interests.iter())
        .filter(|(&t, _)| t == 1)
        .map(|(_, &oi)| oi)
        .sum();

    let put_oi: f64 = option_types
        .iter()
        .zip(open_interests.iter())
        .filter(|(&t, _)| t == -1)
        .map(|(_, &oi)| oi)
        .sum();

    if call_oi > 0.0 {
        result[1] = put_oi / call_oi;
    }

    Ok(result)
}

/// Compute IV term structure slope.
///
/// Matches Python: `OptionsFeatureBuilder.compute_term_structure_slope()`
///
/// Fits a linear regression of IV vs days-to-expiry and returns the slope coefficient.
///
/// # Arguments
/// * `atm_ivs_by_expiry` - ATM IVs for each expiry bucket
/// * `expiry_days` - Days to expiry for each bucket
///
/// # Returns
/// Slope of IV term structure (IV change per day).
/// Positive means contango, negative means backwardation.
pub fn compute_term_structure_slope(atm_ivs_by_expiry: &[f64], expiry_days: &[u32]) -> f64 {
    // Filter out NaN values
    let valid_pairs: Vec<(f64, f64)> = expiry_days
        .iter()
        .zip(atm_ivs_by_expiry.iter())
        .filter(|(_, &iv)| !iv.is_nan())
        .map(|(&days, &iv)| (days as f64, iv))
        .collect();

    if valid_pairs.len() < 2 {
        return f64::NAN;
    }

    let (valid_days, valid_ivs): (Vec<f64>, Vec<f64>) = valid_pairs.into_iter().unzip();

    // Simple linear regression: slope = cov(x,y) / var(x)
    let mean_days: f64 = valid_days.iter().sum::<f64>() / valid_days.len() as f64;
    let mean_ivs: f64 = valid_ivs.iter().sum::<f64>() / valid_ivs.len() as f64;

    let numerator: f64 = valid_days
        .iter()
        .zip(valid_ivs.iter())
        .map(|(&d, &iv)| (d - mean_days) * (iv - mean_ivs))
        .sum();

    let denominator: f64 = valid_days.iter().map(|&d| (d - mean_days).powi(2)).sum();

    if denominator < 1e-10 {
        return f64::NAN;
    }

    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_trade_direction_basic() {
        let prices = vec![100.0, 101.0, 100.5, 100.5, 101.0];
        let directions = infer_trade_direction(&prices);

        assert_eq!(directions[0], 0); // First trade is unknown
        assert_eq!(directions[1], 1); // Uptick
        assert_eq!(directions[2], -1); // Downtick
        assert_eq!(directions[3], -1); // Zero tick, propagate last direction
        assert_eq!(directions[4], 1); // Uptick
    }

    #[test]
    fn test_infer_trade_direction_empty() {
        let prices: Vec<f64> = vec![];
        let directions = infer_trade_direction(&prices);
        assert_eq!(directions.len(), 0);
    }

    #[test]
    fn test_compute_trade_imbalance_basic() {
        let directions = vec![1i8, 1, -1, 1, -1];
        let sizes = vec![10.0, 20.0, 15.0, 10.0, 5.0];
        let imbalance = compute_trade_imbalance(&directions, &sizes, 3).unwrap();

        assert!(imbalance[0].is_nan());
        assert!(imbalance[1].is_nan());
        // Window [1, 1, -1] with sizes [10, 20, 15]
        // Signed: 10 + 20 - 15 = 15, Total: 45
        let expected2 = 15.0 / 45.0;
        assert!((imbalance[2] - expected2).abs() < 1e-10);
    }

    #[test]
    fn test_compute_size_quantiles_basic() {
        let sizes = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let quantiles = vec![0.0, 0.5, 1.0];
        let result = compute_size_quantiles(&sizes, 3, &quantiles).unwrap();

        assert!(result[0][0].is_nan());
        assert!(result[1][0].is_nan());
        // Window [10, 20, 30]: min=10, median=20, max=30
        assert!((result[2][0] - 10.0).abs() < 1e-10);
        assert!((result[2][1] - 20.0).abs() < 1e-10);
        assert!((result[2][2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_arrival_rate_basic() {
        let timestamps = vec![0.0, 1.0, 2.0, 3.0, 5.0];
        let rate = compute_arrival_rate(&timestamps, 3).unwrap();

        assert!(rate[0].is_nan());
        assert!(rate[1].is_nan());
        // Window [0.0, 1.0, 2.0]: 2 trades over 2 seconds = 1.0 trades/sec
        assert!((rate[2] - 1.0).abs() < 1e-10);
        // Window [1.0, 2.0, 3.0]: 2 trades over 2 seconds = 1.0 trades/sec
        assert!((rate[3] - 1.0).abs() < 1e-10);
        // Window [2.0, 3.0, 5.0]: 2 trades over 3 seconds = 0.666... trades/sec
        assert!((rate[4] - (2.0 / 3.0)).abs() < 1e-10);
    }

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

    #[test]
    fn test_compute_moneyness_basic() {
        let strikes = vec![95.0, 100.0, 105.0];
        let spot_price = 100.0;
        let moneyness = compute_moneyness(&strikes, spot_price).unwrap();

        assert!((moneyness[0] - 0.95).abs() < 1e-10);
        assert!((moneyness[1] - 1.0).abs() < 1e-10);
        assert!((moneyness[2] - 1.05).abs() < 1e-10);
    }

    #[test]
    fn test_compute_days_to_expiry_basic() {
        let current_time = 0.0;
        let expirations = vec![
            7.0 * 24.0 * 3600.0,   // 7 days
            30.0 * 24.0 * 3600.0,  // 30 days
            -1.0 * 24.0 * 3600.0,  // Past expiration
        ];
        let dte = compute_days_to_expiry(&expirations, current_time);

        assert!((dte[0] - 7.0).abs() < 1e-10);
        assert!((dte[1] - 30.0).abs() < 1e-10);
        assert!((dte[2] - 0.0).abs() < 1e-10); // Negative DTE becomes 0
    }

    #[test]
    fn test_compute_aggregated_greeks_basic() {
        let deltas = vec![0.5, -0.3, 0.6];
        let gammas = vec![0.01, 0.02, 0.015];
        let vegas = vec![10.0, 12.0, 11.0];
        let thetas = vec![-0.5, -0.6, -0.4];
        let ois = vec![100.0, 200.0, 100.0]; // Total OI = 400
        let types = vec![1i8, -1, 1];

        let greeks = compute_aggregated_greeks(&deltas, &gammas, &vegas, &thetas, &ois, &types).unwrap();

        // Net delta = (0.5*100 + (-0.3)*200 + 0.6*100) / 400 = (50 - 60 + 60) / 400 = 50/400 = 0.125
        assert!((greeks[0] - 0.125).abs() < 1e-10);

        // Total gamma (absolute) = (0.01*100 + 0.02*200 + 0.015*100) / 400 = (1 + 4 + 1.5) / 400 = 6.5/400
        assert!((greeks[1] - 6.5 / 400.0).abs() < 1e-10);

        // Total vega = (10*100 + 12*200 + 11*100) / 400 = (1000 + 2400 + 1100) / 400 = 4500/400 = 11.25
        assert!((greeks[2] - 11.25).abs() < 1e-10);

        // Total theta = (-0.5*100 + (-0.6)*200 + (-0.4)*100) / 400 = (-50 - 120 - 40) / 400 = -210/400 = -0.525
        assert!((greeks[3] - (-0.525)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_put_call_ratios_basic() {
        let types = vec![1i8, 1, -1, -1, 1]; // 3 calls, 2 puts
        let volumes = vec![100.0, 150.0, 200.0, 100.0, 50.0]; // Call vol: 300, Put vol: 300
        let ois = vec![500.0, 500.0, 600.0, 400.0, 100.0]; // Call OI: 1100, Put OI: 1000

        let ratios = compute_put_call_ratios(&types, &volumes, &ois).unwrap();

        // Volume ratio: 300 / 300 = 1.0
        assert!((ratios[0] - 1.0).abs() < 1e-10);

        // OI ratio: 1000 / 1100 = 0.909...
        assert!((ratios[1] - (1000.0 / 1100.0)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_term_structure_slope_basic() {
        let atm_ivs = vec![0.20, 0.22, 0.24, 0.26]; // Increasing IV
        let expiry_days = vec![7, 30, 60, 90];

        let slope = compute_term_structure_slope(&atm_ivs, &expiry_days);

        // Should be positive since IV increases with time
        assert!(slope > 0.0);
        assert!(!slope.is_nan());
    }

    #[test]
    fn test_compute_term_structure_slope_with_nans() {
        let atm_ivs = vec![0.20, f64::NAN, 0.24, f64::NAN];
        let expiry_days = vec![7, 30, 60, 90];

        let slope = compute_term_structure_slope(&atm_ivs, &expiry_days);

        // Should still compute with valid values only
        assert!(!slope.is_nan());
        assert!(slope > 0.0);
    }

    #[test]
    fn test_compute_term_structure_slope_insufficient_data() {
        let atm_ivs = vec![0.20, f64::NAN, f64::NAN, f64::NAN];
        let expiry_days = vec![7, 30, 60, 90];

        let slope = compute_term_structure_slope(&atm_ivs, &expiry_days);

        // Should return NaN with insufficient valid data
        assert!(slope.is_nan());
    }

    #[test]
    fn test_options_feature_config_default() {
        let config = OptionsFeatureConfig::default();
        assert!(config.include_iv_surface);
        assert!(config.include_greeks);
        assert!(config.include_put_call_ratio);
        assert!(config.include_term_structure);
        assert_eq!(config.iv_moneyness_buckets.len(), 5);
        assert_eq!(config.expiry_buckets_days.len(), 4);
    }
}
