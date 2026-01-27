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

### Phase 2: Discretization Strategy - IN PROGRESS

| File | Status | Type Coverage | Errors | Notes |
|------|--------|---------------|--------|-------|
| `data/processors/discretizer.py` | Complete | 100% | 0 | BucketConfig, LogReturnDiscretizer |
| `tests/python/test_discretizer.py` | Complete | 100% | 0 | 29 tests passing |

**Implemented**:
- `BucketConfig` dataclass with validation (odd buckets, range includes 0)
- `LogReturnDiscretizer` class with:
  - Log-return to basis points conversion
  - Discretization with edge clipping
  - Soft label generation with Gaussian smoothing
- Default: 101 buckets, -50 to +50 bps (±0.5%)

**Remaining**:
- [ ] Implied volatility discretizer
- [ ] Realized volatility discretizer
- [ ] Volume discretizer

### Phase 3: Feature Engineering - NOT STARTED

### Phase 4: Model Architecture - NOT STARTED

### Phase 5: Training Pipeline - NOT STARTED

### Phase 6: Rust Inference - NOT STARTED

---

## 11. Next Steps

1. **Immediate**: Complete Phase 2 volatility and volume discretizers
2. Phase 1 data exploration (requires MCP database access)
3. Feature engineering pipeline
4. Model architecture and training
5. Streaming integration and benchmarking

---

*Document version: 1.3*
*Last updated: 2026-01-27*

**Changelog**:
- v1.3: Phase 2 started - implemented discretizer.py with log-return buckets (29 tests, 100% type coverage).
- v1.2: Phase 0 complete - project setup with pyproject.toml, directory structure, configs, and verified type checking.
- v1.1: Added Python/Rust hybrid language strategy; updated Phase 6 for Rust inference; added ONNX export phase; restructured directory for Rust components; added feature parity testing approach.
