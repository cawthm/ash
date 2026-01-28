# SPECS.md — Probabilistic Price Prediction Model

## 1. Project Overview

**Goal**: Build a next-token-prediction-style model that outputs probability distributions over future asset prices at horizons from 1 to 600 seconds.

**Core Design Principles**:
- Speed: Sub-10ms inference latency for real-time trading decisions
- Readability: Clear, maintainable code over clever optimizations
- Probabilistic output: Discrete buckets with calibrated probabilities

**Architecture Summary**: Lightweight transformer with discrete classification heads, outputting probability distributions across price buckets at multiple time horizons simultaneously.

**Language Strategy**: Python for analysis and training (Phases 1-5), Rust for production inference (Phase 6).

| Component | Language | Rationale |
|-----------|----------|-----------|
| Data exploration | Python | Pandas, database connectors, rapid iteration |
| Feature engineering | Python | NumPy/Pandas ecosystem, prototyping speed |
| Model training | Python | PyTorch, no practical alternative |
| Production inference | Rust | Deterministic latency, no GC pauses, memory safety |

**Model Handoff**: Train in PyTorch → Export to ONNX → Load in Rust via `ort` crate (ONNX Runtime bindings).

---

## 2. Phase 1: Data Exploration

### Objective
Understand the schema, quality, and characteristics of available data.

### Tasks
1. Query `options_data` table schema via MCP
2. Query `stock_trades` table schema via MCP
3. Profile each table:
   - Time range (min/max timestamps)
   - Symbol coverage
   - Data granularity (tick-level vs aggregated)
   - Gap analysis (missing periods, market hours)
   - Volume statistics

### Deliverables
- `data/exploration/data_dictionary.md` — Schema documentation
- `data/exploration/quality_report.md` — Statistics and gap analysis
- `data/exploration/sample_queries.sql` — Useful query templates

### Success Criteria
- Complete understanding of available symbols and time coverage
- Identified any data quality issues requiring preprocessing

---

## 3. Phase 2: Discretization Strategy

### Objective
Define how continuous values are mapped to discrete buckets for prediction.

### Price Buckets
Three approaches to evaluate:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| Fixed-width | Equal $ intervals (e.g., $0.01) | Simple, interpretable | Uneven bucket populations |
| Percentile-based | Equal-population buckets | Balanced training signal | Bucket widths vary |
| Log-return | Percentage change buckets | Scale-invariant | Requires reference price |

**Recommended**: Log-return buckets centered on 0%, symmetric around price changes.

### Time Horizons
Predict at 8 horizons simultaneously:
- 1s, 5s, 10s, 30s, 60s, 120s, 300s, 600s

### Additional Discretization Dimensions
- Implied volatility buckets (for options features)
- Realized volatility buckets (rolling calculations)
- Volume buckets (relative to historical average)

### Deliverables
- `data/processors/discretizer.py` — Bucket mapping logic
- Configuration for bucket boundaries (tunable)

### Open Questions
- [ ] How many price buckets? (Start with 101: -50 to +50 basis points)
- [ ] Should bucket width vary by volatility regime?

---

## 4. Phase 3: Feature Engineering

### Objective
Construct input features for the model.

### Input Feature Categories

**Price History**
- Recent prices (last N ticks/seconds)
- Returns at multiple lookback windows
- VWAP calculations

**Order Flow** (from `stock_trades`)
- Trade direction imbalance
- Trade size distribution
- Arrival rate

**Options Data** (from `options_data`)
- Implied volatility surface snapshots
- Greeks (delta, gamma, vega, theta)
- Put/call ratios
- Term structure slopes

**Volatility Features**
- Realized volatility (multiple windows)
- IV/RV spread
- Volatility of volatility

**Cross-Asset Features**
- Correlation with market index
- Sector co-movement
- Lead-lag relationships

### Sequence Construction
- Fixed-length input windows (configurable)
- Temporal alignment across data sources
- Handle asynchronous arrivals (options vs trades)

### Deliverables
- `data/processors/feature_builder.py` — Feature computation
- `data/processors/sequence_builder.py` — Temporal alignment
- Feature configuration schema

### Open Questions
- [ ] Which securities to include as input features?
- [ ] Optimal lookback window length?
- [ ] How to handle overnight gaps?

---

## 5. Phase 4: Model Architecture

### Objective
Design a fast, accurate transformer for multi-horizon price distribution prediction.

### Architecture

```
Input Sequence (features × time steps)
        ↓
   Embedding Layer
        ↓
   Positional Encoding (learned)
        ↓
   Transformer Encoder (N layers)
   - Multi-head self-attention
   - Feed-forward networks
   - Layer normalization
        ↓
   [CLS] Token Representation
        ↓
   Multi-Horizon Classification Heads (parallel)
   - Head_1s  → softmax → P(bucket | t+1s)
   - Head_5s  → softmax → P(bucket | t+5s)
   - ...
   - Head_600s → softmax → P(bucket | t+600s)
```

### Key Design Choices

**Model Size** (for <10ms inference):
- Embedding dim: 128-256
- Layers: 2-4
- Attention heads: 4-8
- Hidden dim: 512-1024

**Loss Function**: Ordinal-aware loss
- Earth Mover's Distance (EMD) to penalize predictions far from true bucket
- Alternative: Soft-label cross-entropy with Gaussian smoothing

**Output**: Probability distribution over K buckets for each horizon

### CUDA Optimization
- ONNX Runtime for production inference (Rust `ort` crate)
- Batched inference for multiple symbols
- FP16 mixed precision training and inference
- Pre-allocated memory buffers

### Model Export
- Export trained PyTorch model to ONNX format
- Validate ONNX model outputs match PyTorch within tolerance
- Optimize ONNX graph (constant folding, operator fusion)
- Target ONNX opset version compatible with `ort` crate

### Deliverables
- `models/architectures/price_transformer.py` — Model definition
- `models/architectures/losses.py` — EMD and calibration losses
- `models/export/onnx_export.py` — PyTorch → ONNX conversion
- `models/export/validate_export.py` — Parity testing between PyTorch and ONNX

### Open Questions
- [ ] Causal attention vs bidirectional?
- [ ] Separate models per underlying vs shared model?

---

## 6. Phase 5: Training Pipeline

### Objective
Train models with proper temporal separation and relevant metrics.

### Data Splits
```
|-------- Train --------|--- Val ---|--- Test ---|
     70%                   15%          15%
```
- **Critical**: Strictly chronological splits (no shuffling)
- Gap between splits to prevent lookahead bias
- Rolling window validation for robustness checks

### Training Loop
- Adam optimizer with learning rate warmup
- Gradient clipping
- Early stopping on validation loss
- Checkpoint best models

### Metrics

**Probabilistic Quality**:
- Log-likelihood (primary)
- Brier score (per bucket)
- Calibration curves (reliability diagrams)

**Trading-Relevant**:
- Directional accuracy (sign of predicted mean)
- Sharpness (concentration of predicted distribution)
- Profit/loss simulation on held-out data

### Deliverables
- `models/training/trainer.py` — Training loop
- `models/training/data_loader.py` — Temporal data loading
- `evaluation/metrics.py` — All evaluation metrics
- `evaluation/calibration.py` — Calibration analysis

### Open Questions
- [ ] Batch size vs sequence length tradeoffs?
- [ ] Pre-training on longer history?

---

## 7. Phase 6: Streaming Integration (Rust)

### Objective
Connect trained model to real-time data feeds for live inference, implemented in Rust for deterministic low-latency performance.

### Interface Design

```rust
pub struct PricePredictor {
    model: ort::Session,
    feature_buffer: FeatureBuffer,
    config: Config,
}

impl PricePredictor {
    /// Load ONNX model and initialize buffers
    pub fn new(model_path: &Path, config: Config) -> Result<Self>;

    /// Ingest new data point, update internal state
    pub fn update(&mut self, tick: &TickData);

    /// Return probability distributions for all horizons
    pub fn predict(&self) -> HashMap<Horizon, Vec<f32>>;
}
```

### Feature Computation Boundary

Two viable approaches for computing features at inference time:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Option 1: Rust reimplements** | Feature logic rewritten in Rust | Lowest latency, single process | Risk of train/serve skew* |
| **Option 2: Python computes** | Python service computes features, Rust runs model | Single source of truth | IPC overhead, more moving parts |

*Train/serve skew: When feature computation differs between training and inference (e.g., edge case handling, floating point behavior), causing the model to see different distributions at serve time than it was trained on.

**Recommendation**: Start with Option 2 for correctness, measure IPC overhead. If latency budget allows, keep it. If not, migrate to Option 1 with rigorous parity testing between Python and Rust implementations.

### Parity Testing (if Option 1)
If reimplementing features in Rust:
- Generate test vectors from Python feature builder
- Assert Rust outputs match within floating-point tolerance
- Run on historical data, compare distributions
- Include edge cases: market open, overnight gaps, extreme moves

### State Management
- Minimal state: rolling feature buffers
- Stateless inference (features computed fresh each call)
- Clear separation: state update vs prediction

### Integration Points
- Input: Streaming tick data (WebSocket, TCP, etc.)
- Output: Probability distributions to decision engine
- Logging: Predictions and latencies for analysis

### Deliverables
- `inference-rs/src/predictor.rs` — Main inference interface
- `inference-rs/src/buffer.rs` — Rolling data buffers
- `inference-rs/src/features.rs` — Feature computation (if Option 1)
- `inference-rs/benches/latency.rs` — Latency benchmarks
- `tests/feature_parity/` — Python↔Rust parity test vectors

---

## 8. Directory Structure

```
ash/
├── README.md                   # Project overview
├── SPECS.md                    # This specification
├── pyproject.toml              # Python dependencies and build config
│
├── data/
│   ├── exploration/            # Data profiling outputs
│   │   ├── data_dictionary.md
│   │   ├── quality_report.md
│   │   └── sample_queries.sql
│   └── processors/             # Data transformation code (Python)
│       ├── discretizer.py
│       ├── feature_builder.py
│       └── sequence_builder.py
│
├── models/
│   ├── architectures/          # Model definitions (Python/PyTorch)
│   │   ├── price_transformer.py
│   │   └── losses.py
│   ├── training/               # Training infrastructure (Python)
│   │   ├── trainer.py
│   │   └── data_loader.py
│   └── export/                 # Model export (Python → ONNX)
│       ├── onnx_export.py
│       └── validate_export.py
│
├── evaluation/                 # Metrics and analysis (Python)
│   ├── metrics.py
│   └── calibration.py
│
├── inference-rs/               # Production inference (Rust)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── predictor.rs        # Main inference interface
│   │   ├── buffer.rs           # Rolling data buffers
│   │   └── features.rs         # Feature computation (if Option 1)
│   └── benches/
│       └── latency.rs          # Latency benchmarks
│
├── tests/
│   ├── python/                 # Python unit tests
│   └── feature_parity/         # Python↔Rust feature parity tests
│       ├── generate_vectors.py
│       └── test_parity.rs
│
└── configs/                    # Configuration files
    ├── model.yaml
    ├── training.yaml
    └── features.yaml
```

---

## 9. Open Questions

### Data
- [ ] Full list of symbols in `stock_trades` and `options_data`?
- [ ] Are timestamps aligned between tables?
- [ ] Historical depth available?

### Model
- [ ] Single underlying (which one to start)?
- [ ] Multi-asset feature set composition?
- [ ] Pre-trained embeddings or from scratch?

### Infrastructure
- [ ] Target GPU (RTX 3090, A100, etc.)?
- [ ] Streaming data format/protocol?
- [ ] Decision engine interface requirements?

### Rust Inference
- [ ] Feature computation boundary: Option 1 (Rust reimplements) or Option 2 (Python computes)?
- [ ] IPC mechanism if Option 2 (Unix socket, shared memory, etc.)?
- [ ] ONNX opset version constraints?
- [ ] Acceptable floating-point tolerance for parity tests?

---

## 10. Implementation Progress

### Phase 0: Project Setup - COMPLETE

| File | Status | Type Coverage | Errors | Notes |
|------|--------|---------------|--------|-------|
| `pyproject.toml` | Complete | N/A | 0 | Dependencies configured |
| `data/__init__.py` | Complete | 100% | 0 | Package initialized |
| `data/processors/__init__.py` | Complete | 100% | 0 | Package initialized |
| `models/__init__.py` | Complete | 100% | 0 | Package initialized |
| `models/architectures/__init__.py` | Complete | 100% | 0 | Package initialized |
| `models/training/__init__.py` | Complete | 100% | 0 | Package initialized |
| `models/export/__init__.py` | Complete | 100% | 0 | Package initialized |
| `evaluation/__init__.py` | Complete | 100% | 0 | Package initialized |
| `tests/python/test_setup.py` | Complete | 100% | 0 | 6 tests passing |
| `configs/model.yaml` | Complete | N/A | 0 | Model hyperparameters |
| `configs/training.yaml` | Complete | N/A | 0 | Training configuration |
| `configs/features.yaml` | Complete | N/A | 0 | Feature configuration |
| `.gitignore` | Complete | N/A | 0 | Python/Rust artifacts |

**Verified**: `uv run mypy` passes, `uv run pytest` passes (6 tests), `uv run ruff check` passes.

### Phase 1: Data Exploration - BLOCKED

Requires MCP database connectivity to query `options_data` and `stock_trades` tables.

### Phase 2: Discretization Strategy - COMPLETE

| File | Status | Type Coverage | Errors | Notes |
|------|--------|---------------|--------|-------|
| `data/processors/discretizer.py` | Complete | 100% | 0 | All discretizers implemented |
| `tests/python/test_discretizer.py` | Complete | 100% | 0 | 74 tests passing |

**Implemented**:
- `BucketConfig` dataclass with validation (odd buckets, range includes 0)
- `LogReturnDiscretizer` class with:
  - Log-return to basis points conversion
  - Discretization with edge clipping
  - Soft label generation with Gaussian smoothing
  - Default: 101 buckets, -50 to +50 bps (±0.5%)
- `VolatilityConfig` and `VolatilityDiscretizer` for IV/RV:
  - Linear buckets from 5% to 150% annualized volatility
  - Discretization with edge clipping
  - Soft label generation
  - Default: 101 buckets
- `VolumeConfig` and `VolumeDiscretizer` for relative volume:
  - Logarithmic spacing (0.5x and 2x equidistant from 1x)
  - Optional linear scale mode
  - `get_average_bucket()` to find bucket containing ratio 1.0
  - Default: 101 buckets, 0.1x to 5.0x relative volume

### Phase 3: Feature Engineering - COMPLETE

| File | Status | Type Coverage | Errors | Notes |
|------|--------|---------------|--------|-------|
| `data/processors/feature_builder.py` | Complete | 100% | 0 | Price, volatility, order flow & options features |
| `tests/python/test_feature_builder.py` | Complete | 100% | 0 | 98 tests passing |
| `data/processors/sequence_builder.py` | Complete | 100% | 0 | Temporal alignment |
| `tests/python/test_sequence_builder.py` | Complete | 100% | 0 | 46 tests passing |

**Implemented**:
- `PriceFeatureConfig` dataclass for price feature configuration
- `PriceFeatureBuilder` class with:
  - `compute_log_returns()` for single-window log returns
  - `compute_returns_multi_window()` for all configured lookback windows
  - `compute_vwap()` for Volume Weighted Average Price
  - `compute_vwap_deviation()` for price deviation from VWAP
  - `compute_features()` to compute all price features
  - Default: returns at 1s, 5s, 10s, 30s, 60s, 120s, 300s windows + VWAP
- `VolatilityFeatureConfig` dataclass for volatility feature configuration
- `VolatilityFeatureBuilder` class with:
  - `compute_realized_volatility()` annualized rolling RV
  - `compute_rv_multi_window()` for all configured RV windows
  - `compute_volatility_of_volatility()` rolling std of RV
  - `compute_iv_rv_spread()` for IV - RV calculation
  - `compute_features()` to compute all volatility features
  - Default: RV at 60s, 300s, 600s + IV/RV spread + vol-of-vol
- `OrderFlowFeatureConfig` dataclass for order flow feature configuration
- `OrderFlowFeatureBuilder` class with:
  - `infer_trade_direction()` using tick rule with zero-tick propagation
  - `compute_trade_imbalance()` volume-weighted buy/sell pressure
  - `compute_size_quantiles()` rolling trade size distribution
  - `compute_arrival_rate()` trades per second calculation
  - `compute_features()` to compute all order flow features
  - Default: 30s imbalance window, quantiles (25/50/75%), 60s arrival rate window
- `OptionsFeatureConfig` dataclass for options feature configuration
- `OptionsFeatureBuilder` class with:
  - `compute_moneyness()` for strike/spot ratio calculation
  - `compute_days_to_expiry()` for DTE calculation
  - `compute_atm_iv_by_expiry()` ATM IV across expiry buckets
  - `compute_iv_surface_features()` IV surface across moneyness x expiry grid
  - `compute_aggregated_greeks()` OI-weighted delta, gamma, vega, theta
  - `compute_put_call_ratios()` volume and OI put/call ratios
  - `compute_term_structure_slope()` IV term structure gradient
  - `compute_features()` to compute all options features
  - Default: 4 expiry buckets (7, 30, 60, 90d) x 5 moneyness buckets (0.95-1.05)
- `SequenceConfig` dataclass for sequence construction configuration
- `OvernightStrategy` enum (RESET, MASK, INTERPOLATE) for gap handling
- `AlignedSequence` dataclass for aligned time series output
- `SequenceBuilder` class with:
  - `create_time_grid()` for regular-interval time grids
  - `forward_fill_to_grid()` for 1D value alignment
  - `forward_fill_2d_to_grid()` for 2D value alignment
  - `detect_overnight_gaps()` for identifying market gaps
  - `apply_overnight_strategy()` for gap handling
  - `align_trade_data()` for stock trade alignment
  - `align_options_data()` for options snapshot alignment
  - `build_sequence()` for complete multi-source alignment
  - `get_valid_range()` for contiguous valid data detection
  - Default: 300s lookback, 1s interval, RESET overnight strategy

**Note**: Cross-asset features are disabled in config until multi-asset decision is made (see Open Questions).

### Phase 4: Model Architecture - NOT STARTED

### Phase 5: Training Pipeline - NOT STARTED

### Phase 6: Rust Inference - NOT STARTED

---

## 11. Next Steps

1. Phase 1 data exploration (requires MCP database access)
2. Phase 4: Model architecture and training
3. Phase 5: Training pipeline
4. Phase 6: Streaming integration and benchmarking

---

*Document version: 1.8*
*Last updated: 2026-01-28*

**Changelog**:
- v1.8: Phase 3 complete - added SequenceBuilder for temporal alignment with forward-fill, overnight gap handling, and multi-source data alignment (224 tests total, 100% type coverage).
- v1.7: Phase 3 continued - added OptionsFeatureBuilder with IV surface, Greeks, put/call ratios, and term structure slope (98 tests total, 100% type coverage).
- v1.6: Phase 3 continued - added OrderFlowFeatureBuilder with trade direction imbalance, size quantiles, and arrival rate (69 tests total, 100% type coverage).
- v1.5: Phase 3 started - implemented feature_builder.py with PriceFeatureBuilder and VolatilityFeatureBuilder (42 tests, 100% type coverage).
- v1.4: Phase 2 complete - all discretizers implemented (LogReturn, Volatility, Volume) with 74 tests, 100% type coverage.
- v1.3: Phase 2 started - implemented discretizer.py with log-return buckets (29 tests, 100% type coverage).
- v1.2: Phase 0 complete - project setup with pyproject.toml, directory structure, configs, and verified type checking.
- v1.1: Added Python/Rust hybrid language strategy; updated Phase 6 for Rust inference; added ONNX export phase; restructured directory for Rust components; added feature parity testing approach.
