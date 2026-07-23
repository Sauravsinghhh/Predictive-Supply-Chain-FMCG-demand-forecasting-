# Data Ingestion & Schema Validation Report

**Report Generated At:** 2026-07-24 02:05:56

> [!NOTE]
> **Status:** :white_check_mark: **ALL INGESTION & SCHEMA VALIDATION CHECKS PASSED**

## 1. Calendar Dataset Summary
*   **Shape:** (100, 14)
*   **Schema Matches Config:** True
*   **Null Row Count:** 398
*   **Duplicate Rows:** 0

## 2. Sell Prices Dataset Summary
*   **Shape:** (3000, 4)
*   **Schema Matches Config:** True
*   **Null Row Count:** 0
*   **Duplicate Rows:** 0
*   **Invalid Price Values (<= 0):** 0

## 3. Sales Historical Ingestion Summary
*   **Shape:** (200, 106)
*   **Total Columns:** 106
*   **Day Time-series columns (d_1 to d_N):** 100
*   **Schema Matches Config:** True
*   **Null Values Count:** 0
*   **Duplicate Series:** 0
*   **Negative Sales Volumes:** 0

## 4. Sample Query Execution
**Query:** Filter sales for item 'HOBBIES_1_001' in store 'CA_1'
*   **Rows Returned:** 1

### Sample Loaded Row Data (First 12 columns):
| id | item_id | dept_id | cat_id | store_id | state_id | d_1 | d_2 | d_3 | d_4 | d_5 | d_6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HOBBIES_1_001_CA_1_validation | HOBBIES_1_001 | HOBBIES_1 | HOBBIES | CA_1 | CA | 1 | 2 | 2 | 3 | 2 | 1 |
