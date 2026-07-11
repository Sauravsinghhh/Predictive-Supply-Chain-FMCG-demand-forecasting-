# Friday Demo Script - Week 2: Skinny End-to-End Demand Forecasting & Replenishment Pipeline

This script outlines a professional 5-minute walkthrough presentation of the Week 2 deliverables.

---

## 1. Introduction & Week 1 Recap (45 Seconds)
*   **Slide / Screen:** `README.md` or Dashboard Home
*   **Speaker Notes:**
    > "Hello everyone. Today I'm excited to present the Week 2 demo of **FreshMind**, our predictive demand forecasting and inventory replenishment engine.
    > As a quick recap, our FMCG retail chain, FreshKart, suffers from ₹4 Crore per month in lost sales due to stockouts and an 8% waste rate on perishable items.
    > In Week 1, we established our codebase foundation, including automated downloading, strict schema validations, memory downcasting, and logging.
    > This week, we focused on building a complete end-to-end 'skinny' pipeline: from raw data files to user-interactive replenishment decisions."

---

## 2. Updated Architecture & Repository Walkthrough (45 Seconds)
*   **Slide / Screen:** `docs/architecture.md`
*   **Speaker Notes:**
    > "Before showing the application, let's review the actual architecture we implemented this week.
    > We developed modular files under a single repository structure:
    > - `src/features/build_features.py` handles single time-series filtering and rolling features.
    > - `src/models/baseline_forecast.py` houses our statistical forecasters.
    > - `src/inventory/replenishment.py` executes our inventory replenishment calculations.
    > - `dashboard/app.py` serves as our Streamlit UI frontend.
    > This modularity ensures we can swap baseline forecasting models with complex ML models later, or upgrade inventory algorithms, without breaking other components."

---

## 3. Data Pipeline & Baseline Forecast Demo (1 Minute)
*   **Slide / Screen:** `tests/test_baseline.py` and terminal
*   **Speaker Notes:**
    > "Our baseline forecasting module implements three models: Naive, Seasonal Naive, and Simple Moving Average.
    > We wrote unit tests in `tests/test_baseline.py` to assert mathematical correctness for each.
    > Running `pytest tests/` in the terminal shows all 7 tests passing.
    > When a user selects a SKU, we extract and melt only that specific series. This keeps the application extremely responsive, even if we scale to millions of historical sales rows."

---

## 4. Dashboard & Replenishment UI Demo (1.5 Minutes)
*   **Slide / Screen:** Streamlit Dashboard on Localhost
*   **Speaker Notes:**
    > "Now let's look at the Streamlit user interface. It is styled with a modern dark theme and custom-styled metrics.
    > On the sidebar, we can filter by store—for example, CA_1—and select a specific SKU, such as HOBBIES_1_001.
    > We can choose our forecasting baseline, like Seasonal Naive, and set the forecast horizon to 28 days.
    > If we input our Current Inventory as 10 units, and set a Safety Buffer cushion of 15 units, we click the forecast trigger.
    > Instantly, the app:
    > 1. Extracts the series, merges calendar dates, and computes rolling averages.
    > 2. Fits our Seasonal Naive model and forecasts 28 days.
    > 3. Calculates the recommended order quantity using our replenishment formula: max(0, Forecast - Inventory + Safety Buffer).
    > In our metrics panel, we see a recommended replenishment quantity of 55.0 units, along with a warning banner indicating that our stock is below the safety buffer cushion.
    > Below, our interactive Plotly chart displays the last 30 days of actual history alongside the dotted purple line representing the 28-day forecast."

---

## 5. Challenges, ADR, & Next Week's Goals (1 Minute)
*   **Slide / Screen:** `docs/adr/ADR-001-baseline-and-streamlit.md` or README.md future roadmap
*   **Speaker Notes:**
    > "To document our choices, we created our first Architecture Decision Record (ADR-001), detailing why we selected Streamlit and statistical baselines for this phase.
    > A key challenge was performance: melting the entire M5 dataset in-memory is too slow for real-time dashboards. We resolved this by filtering the wide matrix before melting, reducing UI rendering time to under 0.2 seconds.
    > With a working skinny pipeline now verified, we are fully prepared for Week 3, where we will transition to comprehensive feature engineering (lags, holidays, event indicators) and implement our first machine learning forecaster using LightGBM.
    > Thank you, and I am happy to take any questions."
