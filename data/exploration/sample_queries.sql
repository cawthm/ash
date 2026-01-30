-- Sample Queries for options_data and stock_trades Tables
-- Database: datawarehouse (PostgreSQL with TimescaleDB)
-- Generated: 2026-01-29

-- ==============================================================================
-- TABLE INSPECTION QUERIES
-- ==============================================================================

-- Query 1: Check table schemas
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'options_data'
ORDER BY ordinal_position;

SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'stock_trades'
ORDER BY ordinal_position;

-- Query 2: Check existing indexes (important for performance)
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename IN ('options_data', 'stock_trades')
ORDER BY tablename, indexname;

-- Query 3: Approximate row counts (fast, uses statistics)
SELECT relname AS table_name,
       reltuples::BIGINT AS approx_row_count
FROM pg_class
WHERE relname IN ('options_data', 'stock_trades');

-- ==============================================================================
-- SYMBOL DISCOVERY QUERIES
-- ==============================================================================

-- Query 4: Get unique underlying tickers from options data
SELECT DISTINCT underlying_ticker
FROM options_data
ORDER BY underlying_ticker
LIMIT 100;

-- Query 5: Get unique stock tickers
SELECT DISTINCT ticker
FROM stock_trades
ORDER BY ticker
LIMIT 100;

-- Query 6: Find symbol overlap between tables (may be slow on large tables)
-- Use a specific time range to speed up
SELECT DISTINCT st.ticker
FROM stock_trades st
WHERE EXISTS (
    SELECT 1
    FROM options_data od
    WHERE od.underlying_ticker = st.ticker
    LIMIT 1
)
LIMIT 100;

-- ==============================================================================
-- TIME RANGE QUERIES (Use specific symbols for speed)
-- ==============================================================================

-- Query 7: Get time range for a specific stock ticker
SELECT
    ticker,
    MIN(time) AS earliest_trade,
    MAX(time) AS latest_trade,
    COUNT(*) AS num_trades
FROM stock_trades
WHERE ticker = 'SPY'  -- Replace with your symbol
GROUP BY ticker;

-- Query 8: Get time range for a specific underlying ticker in options
SELECT
    underlying_ticker,
    MIN(time) AS earliest_snapshot,
    MAX(time) AS latest_snapshot,
    COUNT(*) AS num_snapshots
FROM options_data
WHERE underlying_ticker = 'SPY'  -- Replace with your symbol
GROUP BY underlying_ticker;

-- Query 9: Check data availability for a specific date range
SELECT
    DATE(time) AS trade_date,
    COUNT(*) AS num_trades,
    MIN(time) AS first_trade,
    MAX(time) AS last_trade
FROM stock_trades
WHERE ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-31'
GROUP BY DATE(time)
ORDER BY trade_date;

-- ==============================================================================
-- DATA QUALITY QUERIES
-- ==============================================================================

-- Query 10: Check for null values in critical columns (stock_trades)
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN time IS NULL THEN 1 ELSE 0 END) AS null_time,
    SUM(CASE WHEN ticker IS NULL THEN 1 ELSE 0 END) AS null_ticker,
    SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) AS null_price,
    SUM(CASE WHEN size IS NULL THEN 1 ELSE 0 END) AS null_size,
    SUM(CASE WHEN gkyz_vol_1min_annual IS NULL THEN 1 ELSE 0 END) AS null_vol
FROM stock_trades
WHERE ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-02'
LIMIT 1000000;  -- Limit for safety

-- Query 11: Check for null values in critical columns (options_data)
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN time IS NULL THEN 1 ELSE 0 END) AS null_time,
    SUM(CASE WHEN underlying_ticker IS NULL THEN 1 ELSE 0 END) AS null_ticker,
    SUM(CASE WHEN option_price IS NULL THEN 1 ELSE 0 END) AS null_option_price,
    SUM(CASE WHEN implied_vol IS NULL THEN 1 ELSE 0 END) AS null_iv,
    SUM(CASE WHEN delta IS NULL THEN 1 ELSE 0 END) AS null_delta
FROM options_data
WHERE underlying_ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-02'
LIMIT 1000000;  -- Limit for safety

-- Query 12: Analyze options data lag (time alignment quality)
SELECT
    underlying_ticker,
    AVG(lag_ms) AS avg_lag_ms,
    MIN(lag_ms) AS min_lag_ms,
    MAX(lag_ms) AS max_lag_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lag_ms) AS median_lag_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY lag_ms) AS p95_lag_ms
FROM options_data
WHERE underlying_ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-02'
  AND lag_ms IS NOT NULL
GROUP BY underlying_ticker;

-- Query 13: Check volatility ranges (data sanity)
SELECT
    ticker,
    AVG(gkyz_vol_1min_annual) AS avg_rv_1min,
    AVG(gkyz_vol_5min_annual) AS avg_rv_5min,
    MIN(gkyz_vol_1min_annual) AS min_rv,
    MAX(gkyz_vol_1min_annual) AS max_rv
FROM stock_trades
WHERE ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-02'
  AND gkyz_vol_1min_annual IS NOT NULL
GROUP BY ticker;

-- Query 14: Check implied volatility ranges
SELECT
    underlying_ticker,
    option_type,
    AVG(implied_vol) AS avg_iv,
    MIN(implied_vol) AS min_iv,
    MAX(implied_vol) AS max_iv,
    COUNT(*) AS observations
FROM options_data
WHERE underlying_ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-02'
  AND implied_vol IS NOT NULL
  AND implied_vol > 0
GROUP BY underlying_ticker, option_type;

-- ==============================================================================
-- FEATURE EXTRACTION QUERIES
-- ==============================================================================

-- Query 15: Extract recent price history (for price features)
SELECT
    time,
    ticker,
    price,
    size,
    vwap_exp_3min,
    vwap_exp_10min,
    gkyz_vol_1min_annual,
    gkyz_vol_5min_annual
FROM stock_trades
WHERE ticker = 'SPY'
  AND time BETWEEN '2024-01-01 10:00:00' AND '2024-01-01 10:05:00'
ORDER BY time;

-- Query 16: Extract IV surface snapshot (for options features)
SELECT
    time,
    underlying_ticker,
    option_type,
    strike_price,
    moneyness,
    days_to_expiry,
    implied_vol,
    delta,
    gamma,
    vega,
    option_volume
FROM options_data
WHERE underlying_ticker = 'SPY'
  AND time >= '2024-01-01 10:00:00'
  AND time < '2024-01-01 10:00:01'  -- 1-second snapshot
  AND implied_vol IS NOT NULL
ORDER BY option_type, moneyness;

-- Query 17: Build ATM IV term structure (for term structure features)
WITH atm_options AS (
    SELECT
        time,
        underlying_ticker,
        option_type,
        days_to_expiry,
        implied_vol,
        ABS(moneyness - 1.0) AS distance_from_atm,
        ROW_NUMBER() OVER (
            PARTITION BY time, option_type, days_to_expiry
            ORDER BY ABS(moneyness - 1.0)
        ) AS rn
    FROM options_data
    WHERE underlying_ticker = 'SPY'
      AND time >= '2024-01-01 10:00:00'
      AND time < '2024-01-01 10:00:01'
      AND implied_vol IS NOT NULL
      AND days_to_expiry > 0
)
SELECT
    time,
    underlying_ticker,
    option_type,
    days_to_expiry,
    implied_vol AS atm_iv
FROM atm_options
WHERE rn = 1  -- Closest to ATM
ORDER BY option_type, days_to_expiry;

-- Query 18: Calculate put/call volume ratio (for options features)
SELECT
    time,
    underlying_ticker,
    SUM(CASE WHEN option_type = 'call' THEN option_volume ELSE 0 END) AS call_volume,
    SUM(CASE WHEN option_type = 'put' THEN option_volume ELSE 0 END) AS put_volume,
    SUM(CASE WHEN option_type = 'put' THEN option_volume ELSE 0 END) /
        NULLIF(SUM(CASE WHEN option_type = 'call' THEN option_volume ELSE 0 END), 0)
        AS put_call_ratio
FROM options_data
WHERE underlying_ticker = 'SPY'
  AND time BETWEEN '2024-01-01 10:00:00' AND '2024-01-01 10:05:00'
  AND option_volume IS NOT NULL
  AND option_volume > 0
GROUP BY time, underlying_ticker
ORDER BY time;

-- Query 19: Aggregate Greeks by moneyness and expiry (for options features)
SELECT
    time,
    underlying_ticker,
    ROUND(moneyness, 2) AS moneyness_bucket,
    ROUND(days_to_expiry / 7) * 7 AS days_bucket,  -- Weekly buckets
    AVG(delta) AS avg_delta,
    AVG(gamma) AS avg_gamma,
    AVG(vega) AS avg_vega,
    SUM(option_volume) AS total_volume
FROM options_data
WHERE underlying_ticker = 'SPY'
  AND time >= '2024-01-01 10:00:00'
  AND time < '2024-01-01 10:00:01'
  AND delta IS NOT NULL
  AND option_volume > 0
GROUP BY time, underlying_ticker, moneyness_bucket, days_bucket
ORDER BY moneyness_bucket, days_bucket;

-- ==============================================================================
-- JOINED QUERIES (Stock Trades + Options Data)
-- ==============================================================================

-- Query 20: Join options with underlying trades (temporal alignment)
SELECT
    o.time AS option_time,
    o.underlying_ticker,
    o.implied_vol,
    o.delta,
    o.option_price,
    t.time AS trade_time,
    t.price AS underlying_price,
    t.gkyz_vol_1min_annual AS realized_vol,
    t.vwap_exp_3min,
    EXTRACT(EPOCH FROM (o.time - t.time)) AS time_diff_seconds
FROM options_data o
JOIN stock_trades t
    ON o.underlying_ticker = t.ticker
    AND ABS(EXTRACT(EPOCH FROM (o.time - t.time))) < 1.0  -- Within 1 second
WHERE o.underlying_ticker = 'SPY'
  AND o.time BETWEEN '2024-01-01 10:00:00' AND '2024-01-01 10:01:00'
  AND o.implied_vol IS NOT NULL
LIMIT 100;

-- Query 21: Compare IV vs RV (volatility spread feature)
WITH iv_data AS (
    SELECT
        time,
        underlying_ticker,
        AVG(implied_vol) AS avg_iv
    FROM options_data
    WHERE underlying_ticker = 'SPY'
      AND time BETWEEN '2024-01-01 10:00:00' AND '2024-01-01 11:00:00'
      AND implied_vol IS NOT NULL
      AND ABS(moneyness - 1.0) < 0.05  -- Near ATM
    GROUP BY time, underlying_ticker
),
rv_data AS (
    SELECT
        time,
        ticker,
        gkyz_vol_1min_annual AS rv
    FROM stock_trades
    WHERE ticker = 'SPY'
      AND time BETWEEN '2024-01-01 10:00:00' AND '2024-01-01 11:00:00'
      AND gkyz_vol_1min_annual IS NOT NULL
)
SELECT
    iv.time,
    iv.underlying_ticker,
    iv.avg_iv,
    rv.rv,
    (iv.avg_iv - rv.rv) AS iv_rv_spread
FROM iv_data iv
JOIN rv_data rv
    ON iv.underlying_ticker = rv.ticker
    AND ABS(EXTRACT(EPOCH FROM (iv.time - rv.time))) < 1.0
ORDER BY iv.time
LIMIT 100;

-- ==============================================================================
-- TRAINING DATA EXTRACTION QUERIES
-- ==============================================================================

-- Query 22: Extract training sequence for a single time point
-- (Input features: last 300 seconds of trades)
SELECT
    time,
    ticker,
    price,
    size,
    vwap_exp_3min,
    gkyz_vol_1min_annual
FROM stock_trades
WHERE ticker = 'SPY'
  AND time >= '2024-01-01 10:00:00' - INTERVAL '300 seconds'
  AND time <= '2024-01-01 10:00:00'
ORDER BY time;

-- Query 23: Extract target labels for multiple horizons
-- (Target: prices at t+1s, t+5s, t+10s, ..., t+600s)
WITH reference_time AS (
    SELECT TIMESTAMP '2024-01-01 10:00:00' AS t0
)
SELECT
    EXTRACT(EPOCH FROM (st.time - rt.t0)) AS seconds_ahead,
    st.price
FROM stock_trades st, reference_time rt
WHERE st.ticker = 'SPY'
  AND st.time >= rt.t0
  AND st.time <= rt.t0 + INTERVAL '600 seconds'
ORDER BY st.time;

-- Query 24: Market hours filter (regular trading hours only)
-- US Eastern Time: 9:30 AM - 4:00 PM
SELECT
    time,
    ticker,
    price,
    size
FROM stock_trades
WHERE ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-02'
  AND EXTRACT(HOUR FROM time AT TIME ZONE 'America/New_York') >= 9
  AND EXTRACT(HOUR FROM time AT TIME ZONE 'America/New_York') < 16
  AND NOT (
      EXTRACT(HOUR FROM time AT TIME ZONE 'America/New_York') = 9
      AND EXTRACT(MINUTE FROM time AT TIME ZONE 'America/New_York') < 30
  )
ORDER BY time
LIMIT 1000;

-- ==============================================================================
-- PERFORMANCE OPTIMIZATION TIPS
-- ==============================================================================

-- Tip 1: Always use time-based WHERE clauses to avoid full table scans
-- Tip 2: Use LIMIT during development and exploration
-- Tip 3: Use EXPLAIN ANALYZE to understand query performance:
--        EXPLAIN ANALYZE <your_query>;
-- Tip 4: Consider creating indexes if they don't exist:
--        CREATE INDEX idx_stock_trades_ticker_time ON stock_trades(ticker, time);
--        CREATE INDEX idx_options_data_ticker_time ON options_data(underlying_ticker, time);
-- Tip 5: Use approximate counts (pg_class) instead of COUNT(*) on large tables
-- Tip 6: Partition queries by symbol to reduce scan size
-- Tip 7: Use TimescaleDB time_bucket() if tables are hypertables:
--        SELECT time_bucket('1 minute', time), AVG(price)
--        FROM stock_trades WHERE ticker = 'SPY' GROUP BY 1;

-- ==============================================================================
-- END OF SAMPLE QUERIES
-- ==============================================================================
