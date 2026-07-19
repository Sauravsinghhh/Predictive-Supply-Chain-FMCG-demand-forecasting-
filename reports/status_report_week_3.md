# FreshMind - Weekly Status Report (Week 3: Project Hardening)

*Date: July 19, 2026*  
*Project:* FreshMind – Predictive Supply Chain: FMCG Demand Forecasting & Replenishment  
*Author:* Saurav Singh  

---

### What's Done
1.  **Centralized Logging System:** 
    *   Created `configs/logging_config.py` to parse configurations from `configs/config.yaml`.
    *   Refactored `src/utils/logging_utils.py` to support dual console/file logs and keep the API backward-compatible.
2.  **Custom Exception Layer:** 
    *   Created `src/utils/errors.py` housing typed domain exceptions: `MissingDatasetError`, `InvalidStoreError`, `InvalidSKUError`, `EmptyDataFrameError`, and `DataValidationError`.
3.  **Strict Error Audits:** 
    *   Refactored `src/data/data_loader.py` and `src/features/build_features.py` to remove silent catch blocks, log failure context, and throw structured exceptions.
4.  **Test Suite Expansion:** 
    *   Developed `tests/test_features.py` and `tests/test_utils.py` expanding unit coverage to feature engineering and utility modules.
    *   Developed `tests/test_integration.py` which programmatically generates synthetic data and verifies the entire end-to-end MVP pipeline.
    *   Achieved a passing suite of 16 tests in under 1 second.
5.  **Continuous Integration:** 
    *   Created `.github/workflows/ci.yml` setting up clean environments, caching dependencies, downloading synthetic datasets, and running all tests automatically.
6.  **Architecture & Documentation Hardening:**
    *   Created `docs/adr/ADR-002-testing-strategy.md` and `docs/adr/ADR-003-logging-and-error-handling.md`.
    *   Polished the master `README.md` and prepared a Friday Demo Script.
    *   Wrote the first Technical Blog Post in `reports/technical_blog_01.md`.

---

### What's Improved
*   **Observability:** Diagnostic logs are now standardized, severities are filterable, and execution histories are persisted in `logs/app.log`.
*   **Reliability:** Eliminated all silent failures; downstream modules like the Streamlit UI can intercept specific exception classes (e.g. `InvalidSKUError`) and display clean banners instead of crashing.
*   **Developer Loop:** New developers can download the sandbox data, configure packages, and run the test suite in under 15 minutes.
*   **Build Confidence:** The automated GitHub Actions CI ensures no broken schemas or feature calculation regressions can merge into the code repository.

---

### Risks
*   **Data Drift:** Synthetic datasets utilized in local integration testing do not capture the high cardinality, promotions schedules, and anomalies of real-world M5 datasets, which might cause unanticipated feature failures when training on full production data.
*   **Model Latency:** Implementing complex ML models (such as LightGBM or PyTorch TFT) in subsequent weeks could increase dashboard inference latency if series extraction and prediction pipelines are not optimized.

---

### Challenges
*   **API Compatibility:** Updating the logging signature required keeping backwards compatibility with earlier files and custom developer scripts (e.g., `download_data.py`). We resolved this by making `log_dir`, `log_file`, and `level` parameters optional with configuration fallback logic in `setup_logger`.
*   **In-Memory Pandas Melting:** Converting wide matrices to long formats for M5 datasets is computationally expensive. We addressed this by pre-filtering row series before melting.

---

### Next Week's Three Goals
1.  **Feature Engineering Expansion:** Engineer advanced temporal features including SNAP benefit windows, holiday indicators, weekly pricing ratios, and rolling historical demand lag variables.
2.  **Machine Learning Forecaster:** Train and deploy a high-performance **LightGBM** regressor, logging parameters and artifacts in the model directory.
3.  **Model Evaluation:** Implement cross-validation backtesting and calculate key evaluation metrics (RMSE, MAE, WAPE) compared against baseline forecasters.
