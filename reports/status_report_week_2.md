# Project Status Report - Week 2: Skinny End-to-End Pipeline

**Project:** FreshMind – Predictive Supply Chain: FMCG Demand Forecasting & Replenishment  
**Report Date:** 2026-07-11  
**Author:** Saurav Singh  

---

## 1. What's Done
- **Modular Data Extensions:** Extended `DataLoader` to dynamically expose calendar, price, and sales DataFrames for down-stream processing.
- **Single-Series Preprocessing:** Implemented memory-efficient wide-to-long transformation and rolling average computations in `src/features/build_features.py`.
- **Baseline Forecasting Engine:** Implemented object-oriented `Naive`, `Seasonal Naive` (7-day seasonality), and `Simple Moving Average` models in `src/models/baseline_forecast.py`.
- **Replenishment Logic:** Created modular inventory replenishment calculator in `src/inventory/replenishment.py` enforcing:
  $$\text{Recommended Order} = \max(0, \text{Forecast Sum} - \text{Current Inventory} + \text{Safety Buffer})$$
- **Streamlit Interactive UI:** Developed an interactive dashboard (`dashboard/app.py`) allowing store/SKU filtering, model configuration, parameter adjustments, and rendering interactive Plotly charts.
- **Robust Verification:** Added unit tests to `tests/test_baseline.py`, verifying forecast calculations and replenishment logic, passing all 7 suite tests successfully.
- **Architecture Documentation:** Created Architecture Decision Record (ADR-001) and updated the System Architecture Context Diagram to represent the Week 2 implemented pipeline.

---

## 2. What's Stuck
- **None:** The end-to-end pipeline operates successfully on the sample/sandbox dataset.

---

## 3. Risks
- **In-Memory Melting Scale:** Melting and joining full M5 datasets (~30 million rows) will exceed standard RAM limits.
  - *Mitigation:* We filter datasets to the selected Store/SKU *prior* to melting. Future feature engineering steps will run batch pre-processing off-line or use Chunking.
- **Baseline Inaccuracy:** Baseline statistical models are highly responsive to recent lags but fail to capture holidays (e.g., Thanksgiving) or SNAP promotional calendar events.
  - *Mitigation:* Transitioning to LightGBM in Week 3, including features for holiday flags and SNAP indicators.

---

## 4. Next Week's Three Goals
1. **Comprehensive Feature Engineering:** Build lag, rolling mean/std features across multiple windows (7, 14, 30 days) and integrate event/holiday flags from the calendar.
2. **LightGBM Modeling Engine:** Build training, evaluation, and serialization scripts for a LightGBM model to predict demand at scale.
3. **Hyperparameter Tracking & MLflow:** Integrate MLflow to track training runs, record metrics (RMSE/MAPE), and register model binaries.
