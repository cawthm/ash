# AGENTS.md — Development Agents

Specialized agents for building the probabilistic price prediction model.

---

## 1. data-explorer

**Purpose**: Query and profile the MCP database to understand available data.

**When to use**: Phase 1 data exploration, or whenever investigating data questions.

**Capabilities**:
- Query `options_data` and `stock_trades` schemas via MCP
- Profile time ranges, symbol coverage, data granularity
- Identify gaps and quality issues
- Generate data dictionary entries

**Example prompts**:
- "What symbols are available in stock_trades?"
- "Profile the options_data table for AAPL"
- "Find gaps in trading data for the last month"

---

## 2. feature-builder

**Purpose**: Design and implement feature engineering pipelines.

**When to use**: Phase 2-3, building discretization and feature computation.

**Capabilities**:
- Implement discretization strategies (log-return buckets, percentiles)
- Build feature computation functions (returns, VWAP, order flow)
- Handle temporal alignment between data sources
- Write to `data/processors/`

**Example prompts**:
- "Implement log-return bucketing with 101 buckets"
- "Build rolling volatility features with 1m, 5m, 15m windows"
- "Align options and trades data by timestamp"

---

## 3. trainer

**Purpose**: Implement and run the model training pipeline.

**When to use**: Phase 4-5, model implementation and training.

**Capabilities**:
- Implement PyTorch model architectures
- Build temporal data loaders (no lookahead bias)
- Run training loops with proper metrics
- Implement EMD loss and calibration metrics
- Write to `models/` and `evaluation/`

**Example prompts**:
- "Implement the price transformer architecture from SPECS.md"
- "Create a temporal train/val/test split for the dataset"
- "Train the model and report calibration curves"

---

## Future Agents

To be added as needed:
- **cuda-optimizer** — Production inference optimization
- **streaming-integrator** — Real-time data connection
- **evaluator** — Detailed backtesting and metrics

---

*Document version: 1.0*
