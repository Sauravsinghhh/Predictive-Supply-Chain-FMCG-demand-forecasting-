# ADR-001: Streamlit and Simple Baseline Forecasting for Week 2 Skinny Pipeline

## Context
We need to quickly build and demonstrate a complete end-to-end forecasting and inventory replenishment pipeline (the "skinny" product) for FreshMind. The goal is to verify component integrations, establish testing, and demonstrate a working UI to stakeholders before investing in complex machine learning models (e.g. LightGBM, PyTorch Temporal Fusion Transformers) or building heavy frontend/backend architectures (e.g. React with FastAPI).

## Decision
We chose to:
1. Build the dashboard interface using **Streamlit** to facilitate rapid, python-only frontend development.
2. Implement **simple statistical baselines** (Naive, Seasonal Naive, Simple Moving Average) to act as benchmarks and allow verification of the downstream inventory logic without model training overhead.
3. Filter series data in-memory at the Store-SKU level to optimize dashboard reactivity.

## Consequences

Positive:
- Established a fully interactive forecasting dashboard in under 200 lines of code.
- Avoided the overhead of a separate web server, API routing, and state synchronization.
- Provided a solid mathematical baseline against which future machine learning models can be evaluated.
- Enabled immediate validation of the inventory replenishment policy formula.

Negative:
- Streamlit's execution model re-runs the entire script on user interaction (mitigated by `@st.cache_data` for data loading).
- Simple baseline models fail to capture complex holiday, event, or promotional (SNAP) dependencies, which will be addressed in future weeks.

## Alternatives Considered

### Option 1: FastAPI Backend with a React/Next.js Frontend
- **Reason rejected:** Building a decoupled client-server architecture adds significant development overhead and distracts from the core machine learning pipeline. It is better suited for later deployment phases.

### Option 2: Launching Immediately with LightGBM or Deep Learning (TFT)
- **Reason rejected:** Advanced machine learning models require complex feature engineering pipelines, hyperparameter tuning, and model registries (MLflow/DVC), introducing high risk and complexity before verifying the basic end-to-end data flow.
