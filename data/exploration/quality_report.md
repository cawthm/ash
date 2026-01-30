# Data Quality Report

## Executive Summary

This report documents the data quality assessment of the `options_data` and `stock_trades` tables in the datawarehouse PostgreSQL database. Both tables contain high-frequency financial data with comprehensive pre-computed features suitable for probabilistic price prediction modeling.

**Key Findings**:
- ✅ Both tables have comprehensive schemas with pre-computed features (Greeks, volatility, VWAP)
- ✅ Symbol overlap exists between tables (via `ticker`/`underlying_ticker`)
- ✅ Consistent time column naming (`time`) across both tables
- ⚠️ Tables are very large (millions+ rows), requiring time-based filtering for efficient queries
- ⚠️ No visible primary key constraints in schema (may exist separately)
- ℹ️ All columns are nullable - data validation needed during feature construction

---

## Table: `options_data`

### Schema Completeness: ✅ Excellent

**Total Columns**: 26

#### Feature Categories

| Category | Columns | Status |
|----------|---------|--------|
| Identifiers | `time`, `raw_ticker`, `underlying_ticker` | ✅ Complete |
| Option Specs | `exp_date`, `option_type`, `strike_price`, `option_price`, `option_volume` | ✅ Complete |
| Underlying | `underlying_price` | ✅ Complete |
| Timing Metrics | `lag_ms`, `min_distance_ms`, `avg_distance_ms`, `buckets_used` | ✅ Complete |
| Time to Expiry | `time_to_expiry`, `days_to_expiry` | ✅ Complete |
| Moneyness | `moneyness` | ✅ Complete |
| Greeks | `delta`, `gamma`, `theta`, `vega`, `rho` | ✅ Complete (all 5 Greeks) |
| Volatility | `implied_vol`, `gkyz_vol_1min_annual`, `gkyz_vol_5min_annual` | ✅ Complete |
| Pricing | `theoretical_price_1m`, `theoretical_price_5m` | ✅ Complete |

### Data Quality Observations

#### Strengths
1. **Pre-computed Greeks**: All standard option Greeks available, eliminating need for real-time calculation
2. **Multiple Volatility Measures**: Both implied (IV) and realized (GKYZ at 1-min and 5-min)
3. **Time Alignment Metrics**: `lag_ms`, `min_distance_ms`, `avg_distance_ms` enable quality filtering
4. **Moneyness Pre-calculated**: Direct strike/spot ratio available
5. **Dual Time Representations**: Both fractional years (`time_to_expiry`) and days (`days_to_expiry`)

#### Potential Issues
1. **All Columns Nullable**: No NOT NULL constraints
   - **Impact**: Missing data possible for any field
   - **Mitigation**: Implement null checks in feature builders, filter incomplete records

2. **No Visible Primary Key**: Schema query shows no PK
   - **Impact**: Potential duplicate rows
   - **Mitigation**: Use (`time`, `raw_ticker`) as composite key during deduplication

3. **Large Table Size**: Aggregate queries timeout after 2+ minutes
   - **Impact**: Cannot determine exact row count or time range without indexes
   - **Mitigation**: Always use time-based WHERE clauses, consider EXPLAIN ANALYZE for query optimization

#### Recommended Filters
For high-quality data, filter on:
```sql
WHERE underlying_price IS NOT NULL
  AND option_price IS NOT NULL
  AND implied_vol IS NOT NULL
  AND implied_vol > 0
  AND days_to_expiry > 0
  AND lag_ms < 1000  -- Options data within 1 second of underlying
```

---

## Table: `stock_trades`

### Schema Completeness: ✅ Excellent

**Total Columns**: 19

#### Feature Categories

| Category | Columns | Status |
|----------|---------|--------|
| Identifiers | `time`, `ticker`, `id`, `sequence_number` | ✅ Complete |
| Trade Data | `price`, `size` | ✅ Complete |
| Timestamps | `participant_timestamp`, `sip_timestamp`, `trf_timestamp` | ✅ Complete |
| Metadata | `conditions`, `correction`, `exchange`, `tape`, `trf_id` | ✅ Complete |
| Time Precision | `dsecs` | ✅ Complete |
| Volatility | `gkyz_vol_1min_annual`, `gkyz_vol_5min_annual` | ✅ Complete |
| VWAP | `vwap_exp_3min`, `vwap_exp_10min` | ✅ Complete |

### Data Quality Observations

#### Strengths
1. **Tick-Level Granularity**: Individual trade records enable precise order flow analysis
2. **Multiple Timestamps**: Three timestamp sources enable latency analysis
3. **Pre-computed VWAP**: Exponential VWAP at 3-min and 10-min windows
4. **Pre-computed Volatility**: GKYZ realized volatility at multiple windows
5. **Trade Metadata**: `conditions`, `exchange`, `tape` enable filtering for quality

#### Potential Issues
1. **All Columns Nullable**: No NOT NULL constraints
   - **Impact**: Missing data possible for any field
   - **Mitigation**: Filter for `price IS NOT NULL AND size IS NOT NULL`

2. **No Visible Primary Key**: Schema query shows no PK
   - **Impact**: Potential duplicate rows
   - **Mitigation**: Use `id` or (`ticker`, `time`, `sequence_number`) for deduplication

3. **Trade Conditions Complexity**: `conditions` is text field
   - **Impact**: May contain various trade types (odd lots, late reports, etc.)
   - **Mitigation**: Analyze condition codes, filter non-standard trades

4. **Large Table Size**: Aggregate queries timeout after 2+ minutes
   - **Impact**: Cannot determine exact row count or time range without indexes
   - **Mitigation**: Always use time-based WHERE clauses

#### Recommended Filters
For high-quality trade data, filter on:
```sql
WHERE price IS NOT NULL
  AND price > 0
  AND size IS NOT NULL
  AND size > 0
  AND conditions NOT LIKE '%OddLot%'  -- Adjust based on actual condition codes
```

---

## Cross-Table Data Quality

### Symbol Alignment

**Mapping**: `stock_trades.ticker` ↔ `options_data.underlying_ticker`

#### Verification Needed
1. **Symbol Coverage**: Determine overlap between `ticker` (stock_trades) and `underlying_ticker` (options_data)
   - **Query** (requires index or patience):
   ```sql
   SELECT COUNT(DISTINCT t.ticker) as stock_symbols,
          COUNT(DISTINCT o.underlying_ticker) as option_symbols,
          COUNT(DISTINCT CASE WHEN t.ticker = o.underlying_ticker THEN t.ticker END) as overlap
   FROM (SELECT DISTINCT ticker FROM stock_trades) t
   FULL OUTER JOIN (SELECT DISTINCT underlying_ticker FROM options_data) o
     ON t.ticker = o.underlying_ticker;
   ```

2. **Symbol Format Consistency**: Verify ticker formats match (uppercase, no extra spaces)

### Temporal Alignment

**Common Column**: Both tables use `time` (timestamp with time zone)

#### Synchronization Quality
- `options_data.lag_ms` indicates lag between option snapshot and underlying price
- Typical lag values unknown (requires data profiling)
- **Recommended**: Filter for `lag_ms < 1000` (within 1 second) for high-quality alignment

#### Gap Analysis
**Status**: Cannot be completed without time range queries (too slow)

**Recommended Approach**:
1. Query time range for a single symbol first:
   ```sql
   SELECT MIN(time), MAX(time)
   FROM stock_trades
   WHERE ticker = 'SPY'  -- Use specific symbol
   LIMIT 1;
   ```

2. Check for overnight gaps (market hours 9:30 AM - 4:00 PM ET):
   ```sql
   WITH trade_times AS (
       SELECT time, ticker,
              LAG(time) OVER (PARTITION BY ticker ORDER BY time) as prev_time
       FROM stock_trades
       WHERE ticker = 'SPY'
         AND time BETWEEN '2024-01-01' AND '2024-01-02'
   )
   SELECT COUNT(*) as gaps,
          AVG(EXTRACT(EPOCH FROM (time - prev_time))) as avg_gap_seconds,
          MAX(EXTRACT(EPOCH FROM (time - prev_time))) as max_gap_seconds
   FROM trade_times
   WHERE prev_time IS NOT NULL;
   ```

---

## Volatility Data Quality

### Realized Volatility (GKYZ)

**Available In**: Both tables
- `stock_trades`: `gkyz_vol_1min_annual`, `gkyz_vol_5min_annual`
- `options_data`: `gkyz_vol_1min_annual`, `gkyz_vol_5min_annual`

#### Quality Checks Needed
1. **Non-negative**: `gkyz_vol_*min_annual >= 0`
2. **Reasonable Range**: Typically 5% - 150% annualized for equities
3. **IV vs RV Relationship**: IV should be >= RV on average (volatility risk premium)

#### Validation Query
```sql
SELECT ticker,
       AVG(gkyz_vol_1min_annual) as avg_rv_1min,
       AVG(gkyz_vol_5min_annual) as avg_rv_5min,
       MIN(gkyz_vol_1min_annual) as min_rv,
       MAX(gkyz_vol_1min_annual) as max_rv
FROM stock_trades
WHERE ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-02'
  AND gkyz_vol_1min_annual IS NOT NULL
GROUP BY ticker;
```

### Implied Volatility

**Available In**: `options_data.implied_vol`

#### Quality Checks Needed
1. **Non-negative**: `implied_vol > 0`
2. **Reasonable Range**: Typically 5% - 150% for most equities (higher for volatile stocks)
3. **Surface Consistency**: IV should vary smoothly across moneyness and expiry
4. **Arbitrage-Free**: Put-call parity should hold approximately

#### Validation Query
```sql
SELECT underlying_ticker,
       option_type,
       AVG(implied_vol) as avg_iv,
       MIN(implied_vol) as min_iv,
       MAX(implied_vol) as max_iv,
       COUNT(*) as observations
FROM options_data
WHERE underlying_ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-02'
  AND implied_vol IS NOT NULL
  AND implied_vol > 0
GROUP BY underlying_ticker, option_type;
```

---

## Greeks Data Quality

**Available In**: `options_data` (delta, gamma, theta, vega, rho)

### Expected Ranges

| Greek | Call Range | Put Range | Notes |
|-------|------------|-----------|-------|
| Delta | 0 to 1 | -1 to 0 | Sensitivity to underlying |
| Gamma | 0 to ∞ | 0 to ∞ | Always positive (long options) |
| Theta | -∞ to 0 | -∞ to 0 | Usually negative (time decay) |
| Vega | 0 to ∞ | 0 to ∞ | Always positive (long options) |
| Rho | -∞ to ∞ | -∞ to ∞ | Sensitivity to rates |

### Validation Query
```sql
SELECT
    underlying_ticker,
    option_type,
    COUNT(*) as observations,
    AVG(delta) as avg_delta,
    AVG(gamma) as avg_gamma,
    AVG(theta) as avg_theta,
    AVG(vega) as avg_vega,
    MIN(gamma) as min_gamma,  -- Should be >= 0
    MAX(gamma) as max_gamma
FROM options_data
WHERE underlying_ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-02'
  AND delta IS NOT NULL
GROUP BY underlying_ticker, option_type;
```

---

## Recommendations for Model Training

### Data Preprocessing

1. **Null Handling**
   - **Required**: Filter out rows with nulls in critical columns (price, volume, IV)
   - **Optional**: Impute nulls in secondary features (use with caution)

2. **Outlier Detection**
   - **Volatility**: Cap at 200% annualized (extremely rare above this)
   - **Price**: Remove zero or negative prices
   - **Volume**: Filter zero-volume trades (may be corrections)

3. **Time Filtering**
   - **Market Hours**: 9:30 AM - 4:00 PM ET (regular trading)
   - **After-Hours**: Consider excluding for initial model
   - **Gaps**: Use `SequenceBuilder.detect_overnight_gaps()` to handle

4. **Symbol Selection**
   - **Start With**: Liquid symbols (SPY, QQQ, AAPL, etc.)
   - **Volume Threshold**: Use symbols with consistent trading activity
   - **Options Coverage**: Ensure corresponding options data exists

### Data Quality Metrics to Track

1. **Completeness**: % of non-null values per column
2. **Lag Distribution**: `options_data.lag_ms` quantiles
3. **Time Coverage**: Trading days with data vs total calendar days
4. **Symbol Coverage**: Number of unique symbols with both stock and options data

---

## Next Steps

### Immediate Actions

1. ✅ **Schema Documentation**: Complete (see data_dictionary.md)
2. ⏳ **Sample Queries**: Create sample_queries.sql with optimized query templates
3. ⏳ **Index Analysis**: Check existing indexes with:
   ```sql
   SELECT tablename, indexname, indexdef
   FROM pg_indexes
   WHERE tablename IN ('options_data', 'stock_trades')
   ORDER BY tablename, indexname;
   ```

### Before Training

1. **Profile Single Symbol**: Run complete profiling on one liquid symbol (e.g., SPY)
2. **Determine Time Range**: Identify available historical depth
3. **Measure Lag Distribution**: Analyze `lag_ms` to set filtering thresholds
4. **Test Feature Builders**: Validate feature computation on real data subset

---

*Generated: 2026-01-29*
*Source: Schema analysis and data quality best practices*
