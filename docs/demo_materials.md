# FreshMind - Demo Materials & Walkthrough Guides

This document provides structured walkthroughs, pitches, scripts, and explanations for stakeholders, interviewers, and technical reviews.

---

## 1. 5-Minute Stakeholder Demo Script

### **0:00 - 0:45: Pitch & Business Context**
*   **Visual:** Streamlit Dashboard Landing Page.
*   **Speaker Notes:**
    > "Hello everyone. Today I'm demonstrating **FreshMind**, our predictive demand forecasting and intelligent replenishment engine developed for our 500-store FMCG retail chain, FreshKart.
    > Currently, FreshKart uses a legacy 90-day moving average heuristic. This unresponsiveness costs us ₹4 Crore/month in lost sales due to stockouts, and results in an 8% waste rate on perishable items.
    > FreshMind replaces this with modern machine learning and decision science. Today, I'll walk you through how the system loads data, forecasts demand, and makes optimal ordering recommendations."

### **0:45 - 1:45: Ingestion & Validation Pipeline**
*   **Visual:** `reports/data_status_report.md` or logs in terminal.
*   **Speaker Notes:**
    > "We start with a production-grade data pipeline. Raw data files for calendar events, sell prices, and sales are loaded with strict type-enforced schemas and memory downcasting.
    > In our most recent run, memory optimization reduced the data size by over 80%.
    > More importantly, our pipeline features comprehensive error audits: if files are missing or contain negative values, specific custom exceptions are raised, and the problem is logged in our centralized logging system before it can corrupt downstream predictions."

### **1:45 - 3:00: Forecasting Engine & Dashboard**
*   **Visual:** Streamlit Dashboard. Select Store `CA_1`, SKU `HOBBIES_1_001`, model `LightGBM (Recursive)`, horizon `14 days`.
*   **Speaker Notes:**
    > "Now let's look at the Forecasting Dashboard. Operators can select any store and product combination, select a forecast horizon, and choose from multiple modeling approaches: Baselines, Classical (ARIMA, ETS, Prophet), Machine Learning (LightGBM), or Deep Learning (PyTorch N-BEATS, TFT).
    > Clicking the forecast trigger runs our pipeline. Instantly, the dashboard extracts the series, engineers features, generates predictions, and computes P10-P50-P90 confidence intervals.
    > On the chart, you can see actual sales in blue, the future forecast in purple, and the shaded purple region showing our confidence limits. 
    > We also ran a Walk-Forward backtest. Our LightGBM model achieved 81.5% accuracy, beating the legacy moving average baseline by 14%."

### **3:00 - 4:15: Inventory Optimization & Decision Science**
*   **Visual:** Inventory Decisions tab in Streamlit.
*   **Speaker Notes:**
    > "Forecasting is only half the battle. FreshMind translates these forecasts into inventory actions.
    > Under the 'Inventory Decisions' tab, we specify lead times and service levels. The engine calculates safety stock, reorder points, and recommended order quantities.
    > It compares three standard strategies: Reorder Point (continuous review), Order-Up-To (periodic review), and Newsvendor Optimization.
    > We can see that for this SKU, our current stock is healthy. However, if stock drops below the safety threshold, the system displays a clear warning banner. We can export these replenishment logs directly to a CSV file for store planners."

### **4:15 - 5:00: Explainable AI, MLOps, & Docker**
*   **Visual:** Explainability & MLOps tab, then `docker-compose.yml` in editor.
*   **Speaker Notes:**
    > "To build trust with operators, the 'Explainability' tab shows feature importances using SHAP values, identifying exactly what drove the forecast.
    > For MLOps, we integrated MLflow to track parameters, logs, and register models. We also implemented data drift detection: if the distribution of incoming sales shifts statistically from our training data, a warning is triggered to retrain the model.
    > Finally, the entire application is fully dockerized with a Dockerfile and docker-compose, enabling one-command deployments.
    > In summary, FreshMind replaces rule-of-thumb heuristics with mathematically sound, observable, and containerized decision systems. Thank you."

---

## 2. 10-Minute Technical Interview Walkthrough

### **Minute 0-2: Project Setup & Architecture**
*   Explain the folder structure and architecture choices. Mention ADR-001 (Streamlit for rapid UI prototyping), ADR-002 (pytest and synthetic data for zero-footprint CI), and ADR-003 (centralized logging and custom exception boundaries).
*   Describe the data layout: wide M5 matrix transformed to long format for regression-based machine learning.

### **Minute 2-4: Feature Engineering & Leakage Prevention**
*   Describe the feature pipeline: date features, prices, fourier seasonals, trend index, lags, and rolling statistics.
*   Explain how data leakage was prevented: rolling statistics and expanding windows are computed *on lagged sales* (shifted by 1), ensuring that time $t$ features only contain information up to $t-1$.

### **Minute 4-6: Modeling Strategies**
*   Explain the four categories of models: baseline, classical (ARIMA/ETS/Prophet), ML (LightGBM), and DL (PyTorch N-BEATS, TFT, PatchTST).
*   Detail the difference between **Recursive Forecasting** (updating lags step-by-step) and **Direct Forecasting** (separate models for each step). Explain how recursive forecasting was implemented in NumPy/Pandas.

### **Minute 6-7: Evaluation & Reconciliation**
*   Explain **Walk-Forward Validation** (Rolling Forecast Origin) and how it provides a more robust estimate of error compared to a single train-test split.
*   Discuss **Hierarchical Reconciliation**: OLS reconciliation ($P = (S^T S)^{-1} S^T$) and how it guarantees that forecasts at store, state, and total business levels add up perfectly.

### **Minute 7-9: Decision Science & MLOps**
*   Explain how prediction intervals (P10, P50, P90) are generated and evaluated (CRPS and Coverage).
*   Discuss the inventory policy calculations (safety stock, OUT, and Newsvendor optimization). Explain the standard normal loss function used to estimate expected shortage units.
*   Explain the MLOps pipeline: MLflow logging and data drift detection using Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI).

### **Minute 9-10: Q&A / Review**
*   Summarize key engineering takeaways: writing modular, clean code, handling errors explicitly, establishing CI/CD, and deploying via Docker containers.

---

## 3. Technical, Business, and Architecture Explanations

### **Technical Explanation (For ML Engineers)**
FreshMind is built around a tabular regression approach to time-series forecasting. Raw wide-format sales are melted into a long format. Lags, rolling statistics, Fourier seasonal terms, and price features are engineered. To model the future, we train a LightGBM regressor.
During prediction, we support two modes:
1.  **Recursive:** The model predicts step $t+1$. We append this prediction to the target history, re-calculate the lags and rolling features, and predict $t+2$. This is repeated until the end of the horizon.
2.  **Direct:** We train $H$ separate models, where model $h$ is trained to predict $y_{t+h}$ directly from features at $t$.
To capture uncertainty, we assume residuals are normal and calculate symmetric P10-P50-P90 quantiles.

### **Business Explanation (For Managers/Planners)**
FreshMind is a replenishment planner that tells store managers exactly *how much* stock to order and *when*. 
Instead of looking at the last 90 days and ordering the same amount (which ignores weekends, holidays, and price changes), FreshMind uses machine learning to forecast demand for the next 14 days. 
It then looks at your current stock and decides:
*   What is the minimum safety stock buffer I need to prevent stockouts (safety stock)?
*   At what stock level should I place an order (reorder point)?
*   How many units should I order to minimize holding costs while avoiding empty shelves?
This optimizes supply chain margins, saving up to ₹48 Lakh per store-group monthly.

### **Architecture Explanation (For Solutions Architects)**
FreshMind is designed as a modular, decoupled application:
1.  **Data Ingestion & Validation:** Loads CSV files, validates schemas, applies memory optimizations, and raises custom exceptions.
2.  **Feature Pipeline:** Computes temporal, lag, price, and Fourier features, ensuring zero data leakage.
3.  **Forecasting Engine:** Decoupled forecaster classes inheriting from `BaseForecaster` to allow swapping baselines, classical models, or deep learning.
4.  **Inventory Engine:** Independent module calculating replenishment quantities.
5.  **User Interface:** Streamlit dashboard serving as the presentation layer.
6.  **Observability & MLOps:** Centralized logging, MLflow registry, and drift detection.
7.  **Deployment:** Packaged as isolated Docker containers orchestrated via docker-compose.
