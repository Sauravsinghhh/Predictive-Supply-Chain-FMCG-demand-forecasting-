# Data Layer Documentation – M5 Forecasting

This directory stores the raw and processed files for the **FreshMind** demand forecasting model.

## Directory Structure

```text
data/
│
├── raw/                # Immutable raw CSV files (ignored by git, fetch via scripts/download_data.py)
│   ├── calendar.csv
│   ├── sell_prices.csv
│   ├── sales_train_validation.csv
│   └── sample_submission.csv
│
├── processed/          # Empty initially; reserved for cleaned, aggregated, and feature-engineered datasets
│
└── archive/            # Staging folder for legacy or raw non-M5 datasets (e.g. stores.csv, train.csv, features.csv)
```

---

## File Schemas & Mappings

### 1. `calendar.csv`
Defines the historical calendar days, calendar events, holidays, and SNAP benefits schedules.
*   **Primary Key:** `d` (Day ID matching sales columns)
*   **Foreign Key:** `wm_yr_wk` (Links to pricing data)

| Column | Type | Description |
| :--- | :--- | :--- |
| `date` | Object (Date) | Date formatted as `YYYY-MM-DD`. |
| `wm_yr_wk` | Int64 | Week ID. Unique integer assigned to each week (starts Saturdays). |
| `weekday` | Object | Name of the day of the week (e.g., Saturday, Sunday). |
| `wday` | Int64 | Day index of the week (Starts Saturday = 1, Sunday = 2, ..., Friday = 7). |
| `month` | Int64 | Month index (1-12). |
| `year` | Int64 | Year (e.g., 2011, 2012). |
| `d` | Object | Day identifier matching the sales target column names (e.g., `d_1`, `d_2`). |
| `event_name_1` | Object | Name of the primary event/holiday on that date (e.g. SuperBowl, Thanksgiving). |
| `event_type_1` | Object | Type of the primary event (e.g. Sporting, Cultural, National, Religious). |
| `event_name_2` | Object | Name of a secondary event/holiday (if overlapping). |
| `event_type_2` | Object | Type of the secondary event. |
| `snap_CA` | Int64 (Binary) | 1 if SNAP benefits are active in California on this date, 0 otherwise. |
| `snap_TX` | Int64 (Binary) | 1 if SNAP benefits are active in Texas on this date, 0 otherwise. |
| `snap_WI` | Int64 (Binary) | 1 if SNAP benefits are active in Wisconsin on this date, 0 otherwise. |

---

### 2. `sell_prices.csv`
Contains the historical selling price for each item, in each store, per week.
*   **Composite Primary Key:** `(store_id, item_id, wm_yr_wk)`

| Column | Type | Description |
| :--- | :--- | :--- |
| `store_id` | Object | Store code identifier (e.g., `CA_1`, `TX_2`). Represents 10 total stores. |
| `item_id` | Object | Item/SKU identifier (e.g., `HOBBIES_1_001`). Represents 3,049 items. |
| `wm_yr_wk` | Int64 | Week ID (matches `calendar.csv`). |
| `sell_price` | Float64 | Product selling price for the week. |

---

### 3. `sales_train_validation.csv`
Contains the daily sales volumes (unit count) for each item in each store.
*   **Primary Key:** `id` (Composite of `item_id` and `store_id`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | Object | Concatenated item, store, and validation label (e.g. `HOBBIES_1_001_CA_1_validation`). |
| `item_id` | Object | Item/SKU identifier. |
| `dept_id` | Object | Department identifier (e.g., HOBBIES_1, HOUSEHOLD_2, FOODS_3). |
| `cat_id` | Object | Category identifier (Hobbies, Household, Foods). |
| `store_id` | Object | Store code identifier. |
| `state_id` | Object | State code identifier (CA, TX, WI). |
| `d_1` to `d_1913` | Int64 | Ingestion time-series columns. Daily unit sales for day 1 through day 1913. |

---

### 4. `sample_submission.csv`
Template submission format defining output specifications.
*   **Primary Key:** `id`
*   **Forecast Columns:** `F1` to `F28` representing daily unit sales forecast for the next 28-day horizon.

---

## Data Relationships

```
               [ sales_train_validation ]
                           |
                           | (links via store_id + item_id)
                           v
    [ calendar ] <---> [ sell_prices ]
          |
          | (links via wm_yr_wk)
          v
  (provides event_name, snap_benefits, weekdays per date)
```
