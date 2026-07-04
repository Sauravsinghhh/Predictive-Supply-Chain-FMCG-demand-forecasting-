# One Page Status Report - Week 1: Project Foundation

## What's Done
*   **Git & Repository Setup:** Initialized Git and configured the remote repository (`https://github.com/Sauravsinghhh/Predictive-Supply-Chain-FMCG-demand-forecasting-`). Created a standard directory structure (`data/`, `src/`, `notebooks/`, `configs/`, `tests/`, etc.).
*   **Configuration & Skeleton:** Wrote `.gitignore`, `requirements.txt`, and decoupled configurations into `configs/config.yaml`. Added module packages under `src/`.
*   **Ingestion & Validation Pipeline:** Developed `src/data/data_loader.py` supporting schema checking, null/duplicate inspections, price/sales bounds verification, and a memory optimization module (saving ~80% memory via downcasting).
*   **Dataset Acquisition:** Created `scripts/download_data.py` to pull the Walmart M5 dataset zip from public mirrors or generate schema-accurate synthetic tables for quick local development.
*   **Jupyter Demo & Reporting:** Authored `notebooks/01_data_validation.ipynb` and automated the generation of `reports/data_status_report.md`.
*   **Testing & Logging:** Wrote `tests/test_data_loader.py` with pytest test assertions and implemented custom rolling file logs in `src/utils/logging_utils.py`.
*   **Architecture & Demo Prep:** Designed a C4 Level 1 context diagram in `docs/architecture.md` and prepared a 5-minute presentation script in `docs/demo_script.md`.

## What's Stuck
*   *None.* All Week 1 skeleton and ingestion deliverables are fully functional.

## Risks
*   **Data Scale Memory Limits:** When loading the full M5 dataset (containing over 30 million daily rows for sales and prices), standard Pandas memory footprints could exceed typical local workstation limits (~16GB RAM). 
    *   *Mitigation:* We implemented aggressive numerical downcasting (e.g. converting `float64` to `float32` and `int64` to `int8`/`int16` where valid) during ingestion. If memory issues persist, we will pivot to parquet format and chunk-based processing in Week 2.
*   **Holiday / Promotion Alignment:** M5 features daily sales and weekly prices. Reconciling weekly sell prices with daily events requires precise index alignment.
    *   *Mitigation:* The `wm_yr_wk` mapping column in our calendar config acts as a foreign key to guarantee alignment.

## Next Week's Three Goals
1.  **Exploratory Data Analysis (EDA):** Analyze seasonality, trend profiles, zero-sales occurrences, and promotional effects across different stores/categories in the M5 data.
2.  **Baseline Modeling:** Implement standard baselines (Naive, Seasonal Naive, ETS) in the pipeline as performance benchmarks.
3.  **Cross-Validation Strategy:** Formulate and validate a **Walk-Forward Time-Series Split** validation scheme to prevent data leakage during backtesting.
