# SPECS.md — Probabilistic Price Prediction Model

## 1. Project Overview

**Goal**: Build a next-token-prediction-style model that outputs probability distributions over future asset prices at horizons from 1 to 600 seconds.

**Core Design Principles**:
- Speed: Sub-10ms inference latency for real-time trading decisions
- Readability: Clear, maintainable code over clever optimizations
- Probabilistic output: Discrete buckets with calibrated probabilities

**Architecture Summary**: Lightweight transformer with discrete classification heads, outputting probability distributions across price buckets at multiple time horizons simultaneously.

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
- TensorRT or ONNX Runtime for production inference
- Batched inference for multiple symbols
- FP16 mixed precision training and inference
- Pre-allocated memory buffers

### Deliverables
- `models/architectures/price_transformer.py` — Model definition
- `models/architectures/losses.py` — EMD and calibration losses
- `models/inference/cuda_runner.py` — Optimized inference wrapper

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

## 7. Phase 6: Streaming Integration

### Objective
Connect trained model to real-time data feeds for live inference.

### Interface Design

```python
class PricePredictor:
    def __init__(self, model_path: str, config: Config):
        """Load model and initialize CUDA buffers."""

    def update(self, tick: TickData) -> None:
        """Ingest new data point, update internal state."""

    def predict(self) -> dict[str, np.ndarray]:
        """Return probability distributions for all horizons."""
        # Returns: {"1s": [p0, p1, ...], "5s": [...], ...}
```

### State Management
- Minimal state: rolling feature buffers
- Stateless inference (features computed fresh each call)
- Clear separation: state update vs prediction

### Integration Points
- Input: Streaming tick data (WebSocket, TCP, etc.)
- Output: Probability distributions to decision engine
- Logging: Predictions and latencies for analysis

### Deliverables
- `streaming/predictor.py` — Main inference interface
- `streaming/buffer.py` — Rolling data buffers
- `streaming/benchmark.py` — Latency testing

---

## 8. Directory Structure

```
ash/
├── README.md                   # Project overview
├── SPECS.md                    # This specification
├── pyproject.toml              # Dependencies and build config
│
├── data/
│   ├── exploration/            # Data profiling outputs
│   │   ├── data_dictionary.md
│   │   ├── quality_report.md
│   │   └── sample_queries.sql
│   └── processors/             # Data transformation code
│       ├── discretizer.py
│       ├── feature_builder.py
│       └── sequence_builder.py
│
├── models/
│   ├── architectures/          # Model definitions
│   │   ├── price_transformer.py
│   │   └── losses.py
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py
│   │   └── data_loader.py
│   └── inference/              # Production inference
│       └── cuda_runner.py
│
├── evaluation/                 # Metrics and analysis
│   ├── metrics.py
│   └── calibration.py
│
├── streaming/                  # Real-time integration
│   ├── predictor.py
│   ├── buffer.py
│   └── benchmark.py
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

---

## 10. Next Steps

1. **Immediate**: Run Phase 1 data exploration via MCP queries
2. **Week 1**: Complete discretization strategy with sample data
3. **Week 2**: Feature engineering pipeline
4. **Week 3-4**: Model architecture and training
5. **Week 5**: Streaming integration and benchmarking

---

*Document version: 1.0*
*Last updated: 2026-01-23*
