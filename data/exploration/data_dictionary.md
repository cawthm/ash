# Data Dictionary

## Database Overview

**Database**: `datawarehouse` (PostgreSQL with TimescaleDB)
**Host**: localhost:5432
**Tables**: `options_data`, `stock_trades`

---

## Table: `options_data`

### Description
Contains options market data with implied volatility calculations, Greeks, and pricing information.

### Schema

| Column Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `time` | timestamp with time zone | YES | Timestamp of the options snapshot |
| `raw_ticker` | text | YES | Raw options ticker symbol |
| `underlying_ticker` | text | YES | Underlying stock ticker (e.g., 'AAPL', 'SPY') |
| `exp_date` | date | YES | Option expiration date |
| `option_type` | text | YES | Option type ('call' or 'put') |
| `strike_price` | numeric | YES | Strike price of the option |
| `option_price` | numeric | YES | Market price of the option |
| `option_volume` | numeric | YES | Trading volume for the option |
| `underlying_price` | numeric | YES | Price of the underlying stock at snapshot time |
| `lag_ms` | numeric | YES | Lag in milliseconds between underlying and option prices |
| `min_distance_ms` | numeric | YES | Minimum time distance to underlying trade (ms) |
| `avg_distance_ms` | numeric | YES | Average time distance to underlying trade (ms) |
| `buckets_used` | integer | YES | Number of buckets used in calculation |
| `time_to_expiry` | numeric | YES | Time to expiration (years) |
| `moneyness` | numeric | YES | Strike price / underlying price ratio |
| `days_to_expiry` | numeric | YES | Days until expiration |
| `delta` | numeric | YES | Option delta (sensitivity to underlying price) |
| `gamma` | numeric | YES | Option gamma (rate of change of delta) |
| `theta` | numeric | YES | Option theta (time decay) |
| `vega` | numeric | YES | Option vega (sensitivity to volatility) |
| `rho` | numeric | YES | Option rho (sensitivity to interest rates) |
| `implied_vol` | numeric | YES | Implied volatility |
| `gkyz_vol_1min_annual` | numeric | YES | Garman-Klass-Yang-Zhang volatility (1-min, annualized) |
| `gkyz_vol_5min_annual` | numeric | YES | Garman-Klass-Yang-Zhang volatility (5-min, annualized) |
| `theoretical_price_1m` | numeric | YES | Theoretical option price based on 1-min volatility |
| `theoretical_price_5m` | numeric | YES | Theoretical option price based on 5-min volatility |

### Key Features
- **Greeks**: Full set of option Greeks (delta, gamma, theta, vega, rho) pre-computed
- **Implied Volatility**: Direct IV values for building volatility surfaces
- **Realized Volatility**: GKYZ estimates at multiple time windows
- **Time Alignment**: Lag metrics indicate synchronization quality with underlying

### Usage Notes
- `underlying_ticker` should be used to join with `stock_trades` table (use `ticker` column)
- `moneyness` = strike_price / underlying_price (useful for ATM/OTM classification)
- `time_to_expiry` is in years (fractional)
- `days_to_expiry` is in days (integer-ish)
- All volatility values are annualized

---

## Table: `stock_trades`

### Description
Contains tick-level stock trade data with volume-weighted average prices and realized volatility estimates.

### Schema

| Column Name | Data Type | Nullable | Description |
|------------|-----------|----------|-------------|
| `time` | timestamp with time zone | YES | Trade timestamp |
| `ticker` | text | YES | Stock ticker symbol (e.g., 'AAPL', 'SPY') |
| `dsecs` | numeric | YES | Decimal seconds (sub-second precision) |
| `gkyz_vol_1min_annual` | numeric | YES | Garman-Klass-Yang-Zhang volatility (1-min, annualized) |
| `gkyz_vol_5min_annual` | numeric | YES | Garman-Klass-Yang-Zhang volatility (5-min, annualized) |
| `conditions` | text | YES | Trade conditions/flags |
| `correction` | integer | YES | Correction indicator |
| `exchange` | integer | YES | Exchange code |
| `id` | bigint | YES | Unique trade identifier |
| `participant_timestamp` | bigint | YES | Participant-reported timestamp (nanoseconds) |
| `price` | numeric | YES | Trade price |
| `sequence_number` | integer | YES | Sequence number for ordering |
| `sip_timestamp` | bigint | YES | SIP (Securities Information Processor) timestamp (nanoseconds) |
| `size` | integer | YES | Trade size (shares) |
| `tape` | integer | YES | Tape identifier |
| `trf_id` | integer | YES | Trade Reporting Facility ID |
| `trf_timestamp` | bigint | YES | TRF timestamp (nanoseconds) |
| `vwap_exp_3min` | numeric | YES | Exponential VWAP with 3-minute decay |
| `vwap_exp_10min` | numeric | YES | Exponential VWAP with 10-minute decay |

### Key Features
- **Tick-Level Data**: Individual trade records
- **Multiple Timestamps**: participant, SIP, and TRF timestamps for latency analysis
- **Realized Volatility**: Pre-computed GKYZ volatility estimates
- **VWAP**: Exponential VWAP at multiple time scales
- **Trade Metadata**: Exchange, tape, conditions for filtering

### Usage Notes
- `ticker` should be used to join with `options_data` table (use `underlying_ticker` column)
- Timestamps in nanoseconds: `participant_timestamp`, `sip_timestamp`, `trf_timestamp`
- Primary timestamp column: `time` (timestamp with time zone)
- `size` is in shares (integer)
- `price` is per-share price
- `conditions` may contain trade type indicators (e.g., opening print, closing print, odd lot)

---

## Cross-Table Relationships

### Symbol Mapping
- `stock_trades.ticker` ↔ `options_data.underlying_ticker`

### Time Alignment
- Both tables use `time` column (timestamp with time zone)
- Options data may lag underlying stock trades (see `lag_ms` in options_data)
- For temporal alignment, use `time` column with appropriate tolerance window

### Common Use Cases

**1. Building IV Surface for a Symbol**
```sql
SELECT
    time,
    moneyness,
    days_to_expiry,
    implied_vol,
    option_type
FROM options_data
WHERE underlying_ticker = 'SPY'
  AND time BETWEEN '2024-01-01' AND '2024-01-02'
ORDER BY time, moneyness, days_to_expiry;
```

**2. Joining Options with Underlying Trades**
```sql
SELECT
    o.time,
    o.underlying_ticker,
    o.implied_vol,
    t.price AS underlying_price_trade,
    t.gkyz_vol_1min_annual AS realized_vol
FROM options_data o
JOIN stock_trades t
  ON o.underlying_ticker = t.ticker
  AND ABS(EXTRACT(EPOCH FROM (o.time - t.time))) < 1.0  -- within 1 second
WHERE o.underlying_ticker = 'AAPL'
LIMIT 100;
```

**3. Extracting Recent Price History**
```sql
SELECT
    time,
    ticker,
    price,
    size,
    vwap_exp_3min
FROM stock_trades
WHERE ticker = 'SPY'
  AND time >= NOW() - INTERVAL '1 hour'
ORDER BY time DESC;
```

---

## Data Quality Notes

### Table Size
Both tables are **very large** (likely millions to billions of rows), as evidenced by:
- Approximate row count queries using `pg_class` statistics
- MIN/MAX aggregations on `time` column taking 2+ minutes to execute
- This indicates the data spans significant time periods with high-frequency updates

### Recommendations
1. **Always filter by time range** when querying to avoid full table scans
2. **Check for indexes** on `time`, `ticker`/`underlying_ticker` columns
3. **Use LIMIT** clauses during exploration and development
4. **Consider partitioning** if tables are partitioned by time (TimescaleDB hypertables)

### Known Characteristics
- All columns are nullable (YES)
- No explicit primary keys visible in schema (may exist as constraints)
- Realized volatility columns (`gkyz_vol_*`) are pre-computed
- VWAP columns in stock_trades are pre-computed exponential moving averages

---

*Generated: 2026-01-29*
*Source: Direct PostgreSQL queries via psycopg2*
