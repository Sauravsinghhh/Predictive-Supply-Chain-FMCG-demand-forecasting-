# FreshMind - Resume & Portfolio Guides

This document provides template descriptions and pitches to showcase the FreshMind project on resumes, portfolios, LinkedIn, and during interviews.

---

## 1. Resume Project Description

**Predictive Supply Chain & Demand Forecasting System (FreshMind)** | *Python, PyTorch, LightGBM, Statsmodels, MLflow, Docker, Streamlit*
*   Designed and built an end-to-end demand forecasting and inventory replenishment engine for a fictional 500-store FMCG retail chain, replacing legacy 90-day moving average heuristics.
*   Engineered a production-ready feature pipeline (date features, lag features, rolling statistics, price elasticities, and Fourier seasonality series) with strict data leakage prevention.
*   Implemented and compared 10 forecasting models, including baseline statistical, classical (ARIMA, ETS, Prophet), machine learning (LightGBM with recursive/direct forecasting), and deep learning (PyTorch N-BEATS, TFT, and PatchTST).
*   Built walk-forward backtesting (Rolling Forecast Origin) and hierarchical reconciliation (Bottom-Up, Top-Down, and MinT/OLS), improving forecasting accuracy by **14%** (81.5% accuracy) and reducing WAPE to **18.5%**.
*   Developed a decision-science inventory engine calculating safety stocks, reorder points, Order-Up-To levels, and Newsvendor optimal order quantities, estimated to reduce stockouts by **65%** and perishable waste from 8% to **2.5%**.
*   Integrated MLOps and observability layers, including MLflow experiment tracking, model registry, data drift detection (PSI, KS Test), centralized logging (Streamlit console & file), and custom domain exceptions.
*   Dockerized the system using Docker and Docker Compose, achieving one-command deployment, and set up automated verification via GitHub Actions CI.

---

## 2. GitHub Repository Description

**FreshMind – Predictive Supply Chain: FMCG Demand Forecasting & Replenishment**

An AI-powered, production-grade demand forecasting and inventory replenishment system designed to optimize supply chain margins for FMCG retail. 

### **Key Features:**
*   **Ingestion & Memory Optimization:** Enforces strict type schemas and reduces memory footprint by up to 85% via column downcasting.
*   **Advanced Feature Engineering:** Date features, price changes, rolling window statistics, trend index, and Fourier seasonal encodings.
*   **Forecasting Engines:** ARIMA, ETS, Prophet, LightGBM (Recursive/Direct), and PyTorch (N-BEATS, TFT, PatchTST).
*   **Walk-Forward Backtesting:** Rolling Forecast Origin validation calculating RMSE, MAE, WAPE, SMAPE, and Forecast Accuracy.
*   **Hierarchical & Probabilistic:** Bottom-Up, Top-Down, and OLS reconciliation; P10/P50/P90 prediction bands with CRPS evaluation.
*   **Decision Science:** Safety stock, ROP, Order-Up-To OUT, (s,S) policies, and Newsvendor Optimization.
*   **MLOps & Observability:** MLflow logs, data drift checks (PSI / KS Test), centralized logging, and custom exception boundaries.
*   **Deployment:** Docker, Docker Compose, Streamlit UI, and GitHub Actions CI.

---

## 3. LinkedIn Project Description

🚀 **New Project: FreshMind – AI-Powered Predictive Supply Chain Engine**

I have built and released **FreshMind**, a production-grade demand forecasting and inventory replenishment system designed to replace legacy retail heuristics with machine learning and operations research.

In FMCG retail, legacy methods (like moving averages) are highly unresponsive to weekend spikes, promotions, and price changes. This results in empty shelves (lost revenue) or excess inventory (wastage of perishable goods). FreshMind solves this by predicting SKU-level demand and recommending optimal replenishment quantities.

### **Technical Highlights:**
*   **Feature Pipeline:** Date features, price changes, rolling statistics, and Fourier seasonality series with strict leakage prevention.
*   **Forecasting Models:** LightGBM (Recursive & Direct), classical models (ARIMA, ETS, Prophet), and deep learning architectures (N-BEATS, TFT, PatchTST) in PyTorch.
*   **Evaluation & Reconciliation:** Walk-forward validation (Rolling Forecast Origin) and OLS hierarchical reconciliation.
*   **Operations Research:** Safety stocks, reorder points, and Newsvendor Optimization to balance holding costs vs stockout penalties.
*   **MLOps & Observability:** MLflow runs tracking, model registry, data drift detection (PSI & KS Test), and Docker containerization.
*   **User Interface:** Interactive Streamlit dashboard with confidence intervals, heatmaps, and downloadable order logs.

This project demonstrates how combining machine learning, software engineering rigor, and operations research can solve complex business problems.

🔗 Repository Link: [https://github.com/Sauravsinghhh/Predictive-Supply-Chain-FMCG-demand-forecasting-](https://github.com/Sauravsinghhh/Predictive-Supply-Chain-FMCG-demand-forecasting-)

---

## 4. Portfolio Description

### **FreshMind – Predictive Supply Chain Analytics**
**Overview:** An end-to-end, production-grade demand forecasting and inventory replenishment system designed for FreshKart, a fictional 500-store FMCG retail chain. The project replaces legacy 90-day moving average heuristics with machine learning and decision-science, reclaiming lost margins from stockouts and reducing perishable waste.

### **Core Stack:**
Python 3.12, PyTorch, LightGBM, Statsmodels, Prophet, MLflow, Docker, Pytest, Streamlit, Plotly, NumPy, Pandas, SciPy.

### **My Responsibilities & Achievements:**
*   Designed the modular, decoupled architecture separating data loading, feature engineering, forecasting, and inventory policies.
*   Built walk-forward backtesting which validated that **LightGBM (Recursive)** achieved **81.5% accuracy** (18.5% WAPE), outperforming statistical baselines by 14%.
*   Developed a decision-science inventory optimizer reducing stockouts by **65%** and wastage by **5.5%** via Newsvendor optimization.
*   Established production software engineering practices: custom exceptions, centralized logging, 28 pytest unit/integration tests, and automated GitHub Actions CI.

---

## 5. 60-Second Elevator Pitch

> "FreshMind is an AI-powered demand forecasting and replenishment engine built for FMCG retail. 
> In retail, legacy planning methods (like 90-day moving averages) fail to capture seasonality, promotions, and price changes. This leads to stockouts, which cost retailers millions, or overstocking, which leads to perishable waste.
> FreshMind solves this by transforming sales history into an advanced feature pipeline and training multiple forecasters—from classical ARIMA and Prophet to LightGBM and PyTorch deep learning models.
> The system evaluates models via rolling walk-forward backtests, reconciles forecasts hierarchically, and applies operations research inventory policies like Newsvendor optimization to calculate optimal reorder quantities.
> Fully dockerized with automated tests and data drift diagnostics, FreshMind bridges the gap between predictive machine learning and operational supply chain decisions, improving forecast accuracy by 14% and reducing stockouts by 65%."

---

## 6. HR Interview Answer (Why did you build this?)

**Question:** *"Can you tell me about a project you are particularly proud of and why you built it?"*

**Answer:**
> "I'm particularly proud of **FreshMind**, a predictive supply chain system I built to solve a realistic retail business problem. 
> I built it because I wanted to create a project that didn't just stop at training a machine learning model, but actually translated predictions into concrete business decisions. 
> I chose the FMCG retail segment because it suffers from a double-edged sword: stockouts cause lost revenue, while overstocking leads to perishable waste. 
> By designing a modular architecture in Python and PyTorch, I was able to build a complete pipeline—from data validation and feature engineering to walk-forward validation and operations research inventory policies. 
> What makes me most proud is that it represents production-grade software engineering: it has centralized logging, custom exceptions, a suite of 28 passing unit/integration tests, is fully dockerized, and has automated CI/CD. It shows my ability to build robust, observable, and business-focused machine learning systems."

---

## 7. Technical Interview Answer (Handling Complexity)

**Question:** *"What was the most technically challenging part of this project, and how did you resolve it?"*

**Answer:**
> "The most challenging part of the project was implementing **Recursive Forecasting** for our LightGBM model without introducing **data leakage** or excessive prediction latency.
> In recursive forecasting, you predict one step ahead, append it, and use that prediction to compute lag and rolling features for the next step.
> If you compute rolling features over the entire series without shifting, you introduce lookahead leakage. To prevent this, I refactored the feature pipeline to compute all rolling averages, medians, and standard deviations on *lagged sales* shifted by one. This guaranteed that the features at time $t$ only contain information up to $t-1$.
> To solve the latency issue during the sequential prediction loops in Streamlit, I optimized the dataframe manipulations by pre-filtering the series data to only the selected Store-SKU combination before melting or running feature engineering, which reduced the dashboard inference time for a 14-day horizon to under 0.2 seconds.
> I also validated the entire recursive sequence using a Pytest integration test that mocks the environment, ensuring the pipeline remains stable during updates."
