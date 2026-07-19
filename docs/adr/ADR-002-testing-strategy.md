# ADR-002: Automated Testing Strategy with Pytest & GitHub Actions

## Context
As the FreshMind forecasting and replenishment codebase grows, we need to ensure its reliability and correctness. Specifically, we must prevent data ingestion regressions, verify that custom feature engineering transformations behave as expected, confirm forecasting mathematical operations, and ensure replenishment formulas are correct. We need a testing framework that integrates with CI/CD and supports both rapid local runs and automated verification.

## Decision
We chose to:
1. Standardize on **pytest** as the testing runner and framework for both unit testing and end-to-end integration tests.
2. Structure tests clearly inside a `tests/` directory with file-level prefix `test_` to separate unit and integration domains.
3. Design the integration tests to run on a mock sandbox environment, utilizing programmatically generated synthetic datasets via our `scripts/download_data.py --sample` utility, avoiding the need to download large raw data files in CI environments.
4. Establish a Continuous Integration (CI) pipeline using **GitHub Actions** to execute all tests automatically upon code push or pull request.

## Consequences

### Positive
- **Pytest Ecosystem:** Access to rich testing assertions, fixtures (`pytest.fixture`), test parameters, and coverage reporting plugins (`pytest-cov`).
- **Low CI Footprint:** Generating synthetic dataset files in memory/temp folder during integration tests avoids transferring and parsing massive datasets (~300MB), resulting in execution times under 15 seconds.
- **Immediate Feedback:** GitHub Actions runs automated workflows for every commit, ensuring that any code change that breaks the interface, schema, or formulas is caught immediately.

### Negative
- Mocking/synthetic data might not capture all extreme anomalies or data distribution shifts present in the real Walmart M5 dataset.

## Alternatives Considered

### Option 1: Python's Built-in unittest Framework
- **Reason rejected:** `unittest` is more verbose and requires boilerplate class setups. `pytest` supports simple function-based tests and clean dependency injection via fixtures, which leads to more readable and maintainable test code.

### Option 2: Running Integration Tests Against the Full Dataset in CI
- **Reason rejected:** Downloading, unzipping, and parsing a 300MB CSV dataset on each CI run consumes significant runner resources, prolongs build feedback loops to minutes, and introduces network dependency risks during builds.
