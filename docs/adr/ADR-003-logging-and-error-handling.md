# ADR-003: Centralized Logging and Custom Error Handling Strategy

## Context
In early prototypes, exceptions were caught genericly using standard python `try/except` clauses without logging context, or silent failures occurred. For a production-ready supply chain system, this is unacceptable. System failures like missing data files, corrupted item schemas, invalid store IDs, or empty datasets must be diagnosed immediately with structured error details, explicit exception classes, and clear log messages in stdout and persistent log files.

## Decision
We chose to:
1. Implement a centralized logging configuration system in [logging_config.py](file:///C:/Users/Saurav/Desktop/Predictive%20Supply%20Chain/configs/logging_config.py) that reads configurations from `configs/config.yaml` and sets up standard formatters and handlers.
2. Define a suite of domain-specific custom exceptions in [errors.py](file:///C:/Users/Saurav/Desktop/Predictive%20Supply%20Chain/src/utils/errors.py) inheriting from a base `FreshMindError`.
3. Force modules (e.g. `DataLoader`, `build_features`) to log meaningful context (using `logger.info`, `logger.warning`, and `logger.error`) immediately prior to raising custom exceptions.
4. Eliminate all generic `except: pass` or silent catch blocks across the critical path execution files.

## Consequences

### Positive
- **Observability:** Centralized logging writes identically formatted logs to stdout (for container tracking) and `logs/app.log` (for persistent analysis).
- **Graceful Debugging:** Raising custom exceptions such as `MissingDatasetError`, `InvalidSKUError`, or `EmptyDataFrameError` enables caller code (like the Streamlit UI or subsequent API routers) to handle failures selectively and report specific user-facing errors rather than crashing with unhandled generic tracebacks.
- **Auditing:** Complete pipeline trace logging makes it simple to reconstruct pipeline execution paths.

### Negative
- Introducing explicit raise and log clauses adds some code complexity and increases the codebase size.

## Alternatives Considered

### Option 1: Print Statements and Standard Exceptions
- **Reason rejected:** Plain print statements do not write to log files, cannot be filtered by severity (DEBUG, INFO, ERROR), lack timestamp metadata, and are difficult to ingest in production monitoring solutions (e.g., Splunk, Datadog). Throwing generic `ValueError` or `FileNotFoundError` makes it difficult for upstream callers to programmatically identify and recover from specific business errors.
