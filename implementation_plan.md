# Implementation Plan — Probabilistic Price Prediction Model

## Project Overview

**Goal**: Build a next-token-prediction-style model that outputs probability distributions over future asset prices at horizons from 1 to 600 seconds.

**Core Constraints**:
- Sub-10ms inference latency for real-time trading decisions
- Readable, maintainable code over clever optimizations
- Calibrated probabilistic outputs via discrete buckets

**Language Strategy**: Python for analysis and training (Phases 1-5), Rust for production inference (Phase 6).

**Model Handoff**: PyTorch → ONNX → Rust (`ort` crate)

---

## Phase 0: Project Setup

### Tasks
- [ ] Create `pyproject.toml` with dependencies:
  - [ ] PyTorch (specify version and CUDA requirements)
  - [ ] NumPy, Pandas
  - [ ] ONNX export utilities
  - [ ] Testing framework (pytest)
- [ ] Create directory structure per SPECS.md Section 8:
  - [ ] `data/exploration/`
  - [ ] `data/processors/`
  - [ ] `models/architectures/`
  - [ ] `models/training/`
  - [ ] `models/export/`
  - [ ] `evaluation/`
  - [ ] `inference-rs/`
  - [ ] `tests/python/`
  - [ ] `tests/feature_parity/`
  - [ ] `configs/`
- [ ] Configure development environment:
  - [ ] Python virtual environment setup
  - [ ] Pre-commit hooks (linting, formatting)
  - [ ] Rust toolchain for inference-rs

### Deliverables
- `pyproject.toml`
- Directory structure in place
- `.gitignore` updated for Python/Rust artifacts

---

## Phase 1: Data Exploration

### Tasks
- [ ] Query `options_data` table schema via MCP
- [ ] Query `stock_trades` table schema via MCP
- [ ] Profile `options_data` table:
  - [ ] Time range (min/max timestamps)
  - [ ] Symbol coverage
  - [ ] Data granularity
  - [ ] Gap analysis (missing periods, market hours)
- [ ] Profile `stock_trades` table:
  - [ ] Time range (min/max timestamps)
  - [ ] Symbol coverage
  - [ ] Volume statistics
  - [ ] Data granularity
- [ ] Analyze cross-table alignment:
  - [ ] Timestamp synchronization between tables
  - [ ] Symbol overlap
- [ ] Document data quality issues requiring preprocessing

### Deliverables
- [ ] `data/exploration/data_dictionary.md` — Schema documentation
- [ ] `data/exploration/quality_report.md` — Statistics and gap analysis
- [ ] `data/exploration/sample_queries.sql` — Useful query templates

### Decisions Required
- [ ] **Full list of symbols in `stock_trades` and `options_data`?**
- [ ] **Are timestamps aligned between tables?**
- [ ] **Historical depth available?**

### Success Criteria
- Complete understanding of available symbols and time coverage
- Identified any data quality issues requiring preprocessing

### Critical Path
Phase 2 cannot begin until data availability is confirmed.

---

## Phase 2: Discretization Strategy

### Tasks
- [ ] Evaluate bucket approaches:
  - [ ] Fixed-width (equal $ intervals)
  - [ ] Percentile-based (equal-population)
  - [ ] Log-return (percentage change buckets) — **Recommended**
- [ ] Finalize bucket count decision (starting point: 101 buckets, -50 to +50 bps)
- [ ] Define bucket boundaries configuration
- [ ] Implement discretization for price targets
- [ ] Implement discretization for additional dimensions:
  - [ ] Implied volatility buckets
  - [ ] Realized volatility buckets
  - [ ] Volume buckets (relative to historical average)
- [ ] Test discretizer on sample data

### Deliverables
- [ ] `data/processors/discretizer.py` — Bucket mapping logic
- [ ] `configs/discretization.yaml` — Bucket boundary configuration

### Decisions Required
- [ ] **How many price buckets?** (Recommendation: 101, centered on 0%)
- [ ] **Should bucket width vary by volatility regime?**
- [ ] **Lock discretization strategy** before proceeding to Phase 3

### Success Criteria
- Discretizer correctly maps continuous values to buckets
- Configuration is tunable without code changes
- Strategy documented with rationale

---

## Phase 3: Feature Engineering

### Tasks
- [ ] Implement price history features:
  - [ ] Recent prices (last N ticks/seconds)
  - [ ] Returns at multiple lookback windows
  - [ ] VWAP calculations
- [ ] Implement order flow features (from `stock_trades`):
  - [ ] Trade direction imbalance
  - [ ] Trade size distribution
  - [ ] Arrival rate
- [ ] Implement options features (from `options_data`):
  - [ ] Implied volatility surface snapshots
  - [ ] Greeks (delta, gamma, vega, theta)
  - [ ] Put/call ratios
  - [ ] Term structure slopes
- [ ] Implement volatility features:
  - [ ] Realized volatility (multiple windows)
  - [ ] IV/RV spread
  - [ ] Volatility of volatility
- [ ] Implement cross-asset features:
  - [ ] Correlation with market index
  - [ ] Sector co-movement
  - [ ] Lead-lag relationships
- [ ] Build sequence construction:
  - [ ] Fixed-length input windows (configurable)
  - [ ] Temporal alignment across data sources
  - [ ] Handle asynchronous arrivals (options vs trades)
- [ ] Create feature configuration schema

### Deliverables
- [ ] `data/processors/feature_builder.py` — Feature computation
- [ ] `data/processors/sequence_builder.py` — Temporal alignment
- [ ] `configs/features.yaml` — Feature configuration schema

### Decisions Required
- [ ] **Which securities to include as input features?**
- [ ] **Optimal lookback window length?**
- [ ] **How to handle overnight gaps?**

### Success Criteria
- Features compute correctly on sample data
- Sequence builder aligns multi-source data properly
- Configuration-driven feature selection

---

## Phase 4: Model Architecture

### Tasks
- [ ] Implement transformer architecture:
  - [ ] Embedding layer
  - [ ] Positional encoding (learned)
  - [ ] Transformer encoder (2-4 layers)
  - [ ] Multi-head self-attention (4-8 heads)
  - [ ] Feed-forward networks
  - [ ] Layer normalization
  - [ ] [CLS] token handling
- [ ] Implement multi-horizon classification heads:
  - [ ] Head for 1s horizon
  - [ ] Head for 5s horizon
  - [ ] Head for 10s horizon
  - [ ] Head for 30s horizon
  - [ ] Head for 60s horizon
  - [ ] Head for 120s horizon
  - [ ] Head for 300s horizon
  - [ ] Head for 600s horizon
- [ ] Implement loss functions:
  - [ ] Earth Mover's Distance (EMD) for ordinal awareness
  - [ ] Alternative: Soft-label cross-entropy with Gaussian smoothing
- [ ] Validate model size meets latency target:
  - [ ] Embedding dim: 128-256
  - [ ] Layers: 2-4
  - [ ] Attention heads: 4-8
  - [ ] Hidden dim: 512-1024
- [ ] Implement ONNX export:
  - [ ] Export trained PyTorch model to ONNX
  - [ ] Validate outputs match PyTorch within tolerance
  - [ ] Optimize ONNX graph (constant folding, operator fusion)
  - [ ] Target opset version compatible with `ort` crate

### Deliverables
- [ ] `models/architectures/price_transformer.py` — Model definition
- [ ] `models/architectures/losses.py` — EMD and calibration losses
- [ ] `models/export/onnx_export.py` — PyTorch → ONNX conversion
- [ ] `models/export/validate_export.py` — Parity testing PyTorch/ONNX
- [ ] `configs/model.yaml` — Model hyperparameters

### Decisions Required (Resolve Before Implementation)
- [ ] **Causal attention vs bidirectional?**
  - Causal: prevents future leakage, more natural for time series
  - Bidirectional: potentially better context, but must ensure no lookahead
- [ ] **Separate models per underlying vs shared model?**
  - Separate: asset-specific patterns, simpler training
  - Shared: transfer learning, parameter efficiency
- [ ] **Single underlying to start with (which one)?**
- [ ] **Pre-trained embeddings or from scratch?**
- [ ] **Target GPU (RTX 3090, A100, etc.)?**

### Success Criteria
- Model runs inference in <10ms
- ONNX export validates within tolerance
- All 8 horizon heads produce valid probability distributions

---

## Phase 5: Training Pipeline

### Tasks
- [ ] Implement data loader with temporal splits:
  - [ ] Chronological splits (no shuffling)
  - [ ] 70% train / 15% validation / 15% test
  - [ ] Gap between splits to prevent lookahead
  - [ ] Rolling window validation option
- [ ] Implement training loop:
  - [ ] Adam optimizer with learning rate warmup
  - [ ] Gradient clipping
  - [ ] Early stopping on validation loss
  - [ ] Checkpoint best models
  - [ ] FP16 mixed precision training
- [ ] Implement probabilistic quality metrics:
  - [ ] Log-likelihood (primary)
  - [ ] Brier score (per bucket)
  - [ ] Calibration curves (reliability diagrams)
- [ ] Implement trading-relevant metrics:
  - [ ] Directional accuracy (sign of predicted mean)
  - [ ] Sharpness (concentration of predicted distribution)
  - [ ] Profit/loss simulation on held-out data
- [ ] Implement calibration analysis:
  - [ ] Reliability diagrams
  - [ ] Expected calibration error (ECE)
  - [ ] Temperature scaling if needed

### Deliverables
- [ ] `models/training/trainer.py` — Training loop
- [ ] `models/training/data_loader.py` — Temporal data loading
- [ ] `evaluation/metrics.py` — All evaluation metrics
- [ ] `evaluation/calibration.py` — Calibration analysis
- [ ] `configs/training.yaml` — Training hyperparameters

### Decisions Required
- [ ] **Batch size vs sequence length tradeoffs?**
- [ ] **Pre-training on longer history?**

### Success Criteria
- Model trains without data leakage
- Calibration curves show well-calibrated predictions
- Validation metrics stable across training runs

---

## Phase 6: Rust Inference

### Tasks
- [ ] Set up inference-rs Rust project:
  - [ ] Create `Cargo.toml` with dependencies (`ort`, etc.)
  - [ ] Configure ONNX Runtime bindings
- [ ] Implement `PricePredictor` struct:
  - [ ] `new()` — Load ONNX model and initialize buffers
  - [ ] `update()` — Ingest new tick data, update state
  - [ ] `predict()` — Return probability distributions for all horizons
- [ ] Implement rolling data buffers (`buffer.rs`)
- [ ] Implement feature computation (if Option 1):
  - [ ] Port Python feature logic to Rust
  - [ ] Match floating-point behavior
  - [ ] Handle edge cases: market open, overnight gaps, extreme moves
- [ ] Implement parity testing:
  - [ ] Generate test vectors from Python feature builder
  - [ ] Assert Rust outputs match within tolerance
  - [ ] Run on historical data, compare distributions
  - [ ] Include edge case coverage
- [ ] Implement latency benchmarks:
  - [ ] Measure inference time
  - [ ] Measure feature computation time (if Option 1)
  - [ ] Validate <10ms total latency
- [ ] Set up integration points:
  - [ ] Input: Streaming tick data interface
  - [ ] Output: Probability distributions to decision engine
  - [ ] Logging: Predictions and latencies

### Deliverables
- [ ] `inference-rs/Cargo.toml` — Rust project configuration
- [ ] `inference-rs/src/lib.rs` — Library root
- [ ] `inference-rs/src/predictor.rs` — Main inference interface
- [ ] `inference-rs/src/buffer.rs` — Rolling data buffers
- [ ] `inference-rs/src/features.rs` — Feature computation (if Option 1)
- [ ] `inference-rs/benches/latency.rs` — Latency benchmarks
- [ ] `tests/feature_parity/generate_vectors.py` — Test vector generator
- [ ] `tests/feature_parity/test_parity.rs` — Parity test suite

### Decisions Required (Resolve Before Implementation)
- [ ] **Feature computation boundary: Option 1 (Rust reimplements) or Option 2 (Python computes)?**
  - Recommendation: Start with Option 2 for correctness, measure IPC overhead
  - If latency allows, keep it; otherwise migrate to Option 1 with parity testing
- [ ] **IPC mechanism if Option 2 (Unix socket, shared memory, etc.)?**
- [ ] **ONNX opset version constraints?**
- [ ] **Acceptable floating-point tolerance for parity tests?**
- [ ] **Decision engine interface requirements?**
- [ ] **Streaming data format/protocol?**

### Success Criteria
- End-to-end inference latency <10ms
- Rust feature outputs match Python within tolerance (if Option 1)
- No train/serve skew in production predictions

---

## Open Questions Checklist

Consolidated from SPECS.md Section 9 and architectural assessment.

### Data Questions (Resolve in Phase 1)
- [ ] Full list of symbols in `stock_trades` and `options_data`?
- [ ] Are timestamps aligned between tables?
- [ ] Historical depth available?

### Model Questions (Resolve Before Phase 4)
- [ ] Single underlying to start (which one)?
- [ ] Multi-asset feature set composition?
- [ ] Pre-trained embeddings or from scratch?
- [ ] Causal vs bidirectional attention?
- [ ] Separate models per underlying vs shared?

### Infrastructure Questions (Resolve Before Phase 5-6)
- [ ] Target GPU (RTX 3090, A100, etc.)?
- [ ] Streaming data format/protocol?
- [ ] Decision engine interface requirements?

### Rust Inference Questions (Resolve Before Phase 6)
- [ ] Feature computation boundary: Option 1 or Option 2?
- [ ] IPC mechanism if Option 2?
- [ ] ONNX opset version constraints?
- [ ] Acceptable floating-point tolerance for parity tests?

---

## Critical Path Dependencies

```
Phase 0 (Setup)
    ↓
Phase 1 (Data Exploration)
    ↓
Phase 2 (Discretization) ←── Lock strategy before proceeding
    ↓
Phase 3 (Feature Engineering)
    ↓
Phase 4 (Model Architecture) ←── Resolve attention/multi-asset decisions
    ↓
Phase 5 (Training Pipeline)
    ↓
Phase 6 (Rust Inference) ←── Resolve feature computation boundary
```

**Key Gates**:
1. Phase 1 → Phase 2: Data availability confirmed
2. Phase 2 → Phase 3: Discretization strategy locked
3. Phase 3 → Phase 4: Feature set finalized
4. Phase 4 → Phase 5: Model architecture validated for latency
5. Phase 5 → Phase 6: Trained model with acceptable metrics

---

*Document version: 1.0*
*Created: 2026-01-27*
*Source: SPECS.md v1.1, architectural_decisions assessment*
