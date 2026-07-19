# Git Commit History Plan (15 Meaningful Commits)

This plan maps out 15 clean, structured Git commits representing the realistic progression of the FreshMind project from initial scaffold to Week 3 project hardening. 

---

### Commit 1: Initialize repository and folder structure
*   **Commit Message:** `feat: initialize project directory structure and basic configuration`
*   **Files Changed:** `.gitignore`, `configs/config.yaml`, `README.md` (initial stub)
*   **Purpose:** Setup the repository layout, external config mappings, and git ignores.

### Commit 2: Add dataset downloader script
*   **Commit Message:** `feat: create M5 dataset download and synthetic data generator script`
*   **Files Changed:** `scripts/download_data.py`
*   **Purpose:** Enable automated downloading of M5 csvs or generation of schema-compliant sandbox data.

### Commit 3: Implement data loader and schema validations
*   **Commit Message:** `feat: add DataLoader pipeline with schema validation and memory optimization`
*   **Files Changed:** `src/data/data_loader.py`
*   **Purpose:** Parse CSV files, perform memory downcasting, validate types, and output status reports.

### Commit 4: Add basic ingestion unit tests
*   **Commit Message:** `test: add unit tests for config parsing and memory downcasting`
*   **Files Changed:** `tests/test_data_loader.py`
*   **Purpose:** Verify that memory downcasting reduces float/integer sizes and configs read correctly.

### Commit 5: Define system architecture documentation
*   **Commit Message:** `docs: add system C4 architecture and ADR-001 documentation`
*   **Files Changed:** `docs/architecture.md`, `docs/adr/ADR-001-baseline-and-streamlit.md`
*   **Purpose:** Record design choices for the skinny pipeline (Streamlit and statistical baselines).

### Commit 6: Create feature engineering module
*   **Commit Message:** `feat: implement prepare_single_series and rolling features transformations`
*   **Files Changed:** `src/features/build_features.py`
*   **Purpose:** Extract specific store-SKU time-series, melt to long format, merge calendar, and add rolling averages.

### Commit 7: Build baseline forecasters
*   **Commit Message:** `feat: implement Naive, Seasonal Naive, and Moving Average models`
*   **Files Changed:** `src/models/baseline_forecast.py`
*   **Purpose:** Add baseline models to acts as benchmarks for future machine learning runs.

### Commit 8: Add baseline and replenishment unit tests
*   **Commit Message:** `test: add unit tests for baseline forecasters and replenishment logic`
*   **Files Changed:** `tests/test_baseline.py`
*   **Purpose:** Verify that predictions and order quantities match exact mathematical formulas.

### Commit 9: Implement Streamlit dashboard UI
*   **Commit Message:** `feat: build Streamlit dashboard with interactive filters and Plotly charts`
*   **Files Changed:** `dashboard/app.py`
*   **Purpose:** Provide store planners with an interactive frontend to run forecasts and view recommendations.

### Commit 10: Centralize logging configurations
*   **Commit Message:** `refactor: centralize logging configurations loaded from config.yaml`
*   **Files Changed:** `configs/logging_config.py`, `src/utils/logging_utils.py`
*   **Purpose:** Standardize log formats, files, and levels across the repository, replacing plain prints.

### Commit 11: Implement domain-specific custom exceptions
*   **Commit Message:** `feat: define specific exceptions in errors.py for pipeline robustness`
*   **Files Changed:** `src/utils/errors.py`
*   **Purpose:** Create typed custom errors (e.g. MissingDatasetError, InvalidSKUError) to eliminate silent crashes.

### Commit 12: Add error audits to DataLoader and Features
*   **Commit Message:** `refactor: audit data loader and feature prep modules to throw custom exceptions`
*   **Files Changed:** `src/data/data_loader.py`, `src/features/build_features.py`
*   **Purpose:** Ensure errors are logged and explicitly thrown when data violations or missing inputs occur.

### Commit 13: Add unit tests for features and utility modules
*   **Commit Message:** `test: add unit tests for feature calculations and logger behavior`
*   **Files Changed:** `tests/test_features.py`, `tests/test_utils.py`
*   **Purpose:** Increase test coverage to ensure feature pipelines and error modules are reliable.

### Commit 14: Add end-to-end integration test
*   **Commit Message:** `test: create end-to-end pipeline integration test with synthetic data`
*   **Files Changed:** `tests/test_integration.py`
*   **Purpose:** Validate the complete ingestion -> features -> model -> replenishment lifecycle.

### Commit 15: Configure GitHub Actions CI and final README polishing
*   **Commit Message:** `ci: configure GitHub Actions workflow and update master README.md`
*   **Files Changed:** `.github/workflows/ci.yml`, `README.md`
*   **Purpose:** Enable automated build and test runner on pushes and PRs, and document a 15-minute quick start guide.
