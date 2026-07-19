# Friday Demo Script - Week 3: Hardening and Productionizing the Pipeline

This script outlines a professional 5-minute walkthrough presentation of the Week 3 project hardening deliverables.

---

## 1. Introduction & Week 2 Recap (45 Seconds)
*   **Slide / Screen:** `README.md` or Streamlit Dashboard Home
*   **Speaker Notes:**
    > "Hello everyone. Today I'm presenting our progress for Week 3 of **FreshMind**, our FMCG demand forecasting and replenishment engine.
    > In Week 2, we created an end-to-end MVP. We had a functional UI dashboard and baseline forecasters. 
    > However, a production system must be reliable, testable, and observable. This week, we focused entirely on **hardening** the project. 
    > We transitioned the repository from a working prototype into a robust, merge-ready codebase with comprehensive testing, centralized logging, strict error handling, and a complete GitHub Actions CI pipeline."

---

## 2. Testing Framework & CI Pipeline (1.5 Minutes)
*   **Slide / Screen:** VS Code with `tests/` folder open, then run pytest in terminal, then show `.github/workflows/ci.yml`.
*   **Speaker Notes:**
    > "First, we significantly expanded our automated test suite. We now have 16 passing unit and integration tests under the `tests/` directory.
    > In addition to validating loading and baseline models, we added:
    > - `tests/test_features.py` to assert that our feature engineering functions compute correct rolling averages and standard deviations.
    > - `tests/test_utils.py` to verify our custom logging and error structures.
    > - `tests/test_integration.py`, which mocks a sandbox environment and verifies the entire end-to-end pipeline (Data Ingestion -> Validation -> Feature Prep -> Forecast -> Replenishment Decision) in under 1 second.
    > To automate this, we configured GitHub Actions in `.github/workflows/ci.yml`. Whenever a developer pushes code or submits a pull request, the runner automatically spins up Python 3.12, installs dependencies, generates a lightweight synthetic dataset, and runs our entire test suite to prevent regressions."

---

## 3. Observability: Centralized Logging (1 Minute)
*   **Slide / Screen:** `configs/logging_config.py` and `logs/app.log`.
*   **Speaker Notes:**
    > "For observability, we replaced all ad-hoc print statements with a centralized logging configuration.
    > In `configs/logging_config.py`, we created a standard setup that reads logging formats, files, and levels directly from `configs/config.yaml`.
    > In `src/utils/logging_utils.py`, we implemented a backward-compatible `setup_logger` that routes logs simultaneously to standard output (crucial for container environments) and a persistent `logs/app.log` file.
    > Now, every step in our pipeline—from configurations loading and memory optimization downcasts to forecasting execution and replenishment decisions—writes clear, structured timestamps and log levels, giving us full operational visibility."

---

## 4. Reliability: Specific Error Handling (1 Minute)
*   **Slide / Screen:** `src/utils/errors.py` and `src/data/data_loader.py`.
*   **Speaker Notes:**
    > "To guarantee there are no silent failures in our critical execution paths, we performed a codebase audit and eliminated all generic try-except catch blocks.
    > In `src/utils/errors.py`, we defined domain-specific exception classes: `MissingDatasetError`, `InvalidStoreError`, `InvalidSKUError`, `EmptyDataFrameError`, and `DataValidationError`.
    > In `src/data/data_loader.py` and `src/features/build_features.py`, the code now explicitly checks inputs and raises these typed errors with logging statements before failures occur. For example, if a store manager searches for a SKU that does not exist in the dataset, the dashboard intercepts `InvalidSKUError` and displays a clean alert instead of crashing."

---

## 5. Architectural Decisions, Updated README, & Next Steps (45 Seconds)
*   **Slide / Screen:** `docs/adr/` directory containing ADR-002 and ADR-003, then `README.md`.
*   **Speaker Notes:**
    > "We documented our engineering choices by creating two new Architecture Decision Records:
    > - ADR-002, detailing our automated testing strategy using pytest and synthetic data in CI.
    > - ADR-003, explaining our structured logging and custom exception schema.
    > We also polished the `README.md` to make it recruiter- and developer-friendly, including a 15-minute quick-start setup and execution guide.
    > With this production-grade software foundation established, we are fully ready for next week's scope: implementing machine learning models (LightGBM and Prophet) and comprehensive feature engineering.
    > Thank you, and I'd be happy to take any questions."
