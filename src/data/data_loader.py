"""
Data Loading and Validation Pipeline for FreshMind.
Ingests M5 Forecasting CSV files, performs data type casting for memory optimization,
checks integrity (duplicates, nulls, shapes), and outputs validation status.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple

# Fix path to import utils and errors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.logging_utils import setup_logger
from src.utils.errors import (
    ConfigurationError,
    MissingDatasetError,
    EmptyDataFrameError,
    DataValidationError
)

logger = setup_logger("data_loader")

class DataLoader:
    """
    Handles loading, memory optimization, and validation of M5 forecasting datasets.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.data_raw_dir = self.config["paths"]["data_raw_dir"]
        self.data_processed_dir = self.config["paths"]["data_processed_dir"]
        self.reports_dir = self.config["paths"]["reports_dir"]
        
    def load_config(self) -> Dict[str, Any]:
        """
        Loads YAML configuration file.
        
        Raises:
            ConfigurationError: If the config file cannot be found or parsed.
        """
        if not os.path.exists(self.config_path):
            msg = f"Configuration file not found at {self.config_path}"
            logger.error(msg)
            raise ConfigurationError(msg)
            
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
        except Exception as e:
            msg = f"Error parsing config file {self.config_path}: {e}"
            logger.error(msg)
            raise ConfigurationError(msg) from e

    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimizes memory footprint of a Pandas DataFrame by downcasting numeric columns.
        """
        if df.empty:
            return df
            
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object and not isinstance(col_type, pd.CategoricalDtype):
                c_min = df[col].min()
                c_max = df[col].max()
                
                # Check for null values in float columns to prevent issues
                has_nans = df[col].isnull().any()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if not has_nans and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                # Limit categories to object columns with low cardinality (<= 20 unique values)
                if df[col].nunique() <= 20:
                    df[col] = df[col].astype('category')
                    
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory optimization reduced usage from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
        return df

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """
        Loads a CSV file from the raw data directory.
        
        Raises:
            MissingDatasetError: If the file does not exist.
            EmptyDataFrameError: If the file is empty.
            DataValidationError: If the file is corrupted or cannot be read.
        """
        filepath = os.path.join(self.data_raw_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"Required file {filepath} not found. Run download_data.py first.")
            raise MissingDatasetError(filepath)
            
        logger.info(f"Loading {filename} from {self.data_raw_dir}...")
        try:
            df = pd.read_csv(filepath)
        except pd.errors.EmptyDataError as e:
            logger.error(f"File {filename} is empty or corrupted: {e}")
            raise EmptyDataFrameError(filename, f"Empty CSV file loaded: {filename}") from e
        except Exception as e:
            logger.error(f"Failed to read CSV file {filename}: {e}")
            raise DataValidationError(f"Could not read CSV file {filename}: {e}") from e
            
        if df.empty:
            logger.error(f"Loaded DataFrame from {filename} is empty.")
            raise EmptyDataFrameError(filename, f"DataFrame loaded from {filename} is empty")
            
        return self.optimize_memory(df)

    def load_raw_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads and returns the three primary dataframes: calendar, sell_prices, sales.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (calendar_df, prices_df, sales_df)
            
        Raises:
            MissingDatasetError: If any of the three datasets are missing.
            EmptyDataFrameError: If any of the datasets are empty.
        """
        logger.info("Starting loading of all raw datasets...")
        calendar_df = self.load_dataset("calendar.csv")
        prices_df = self.load_dataset("sell_prices.csv")
        sales_df = self.load_dataset("sales_train_validation.csv")
        logger.info("All raw datasets loaded successfully.")
        return calendar_df, prices_df, sales_df

    def run_validation_checks(self) -> Dict[str, Any]:
        """
        Runs complete shape, schema, duplicate, and null value validations on M5 datasets.
        
        Raises:
            DataValidationError: If critical schemas are violated.
        """
        results = {}
        logger.info("Running validation checks on loaded datasets...")
        
        # Load datasets
        calendar_df = self.load_dataset("calendar.csv")
        prices_df = self.load_dataset("sell_prices.csv")
        sales_df = self.load_dataset("sales_train_validation.csv")
        
        # Validate Calendar
        cal_schema = self.config["schemas"]["calendar"]
        cal_cols_exist = all(col in calendar_df.columns for col in cal_schema["columns"])
        if not cal_cols_exist:
            missing = [col for col in cal_schema["columns"] if col not in calendar_df.columns]
            msg = f"Calendar schema mismatch. Missing columns: {missing}"
            logger.error(msg)
            raise DataValidationError(msg)
            
        cal_nulls = calendar_df.isnull().sum().to_dict()
        cal_duplicates = int(calendar_df.duplicated().sum())
        
        results["calendar"] = {
            "shape": calendar_df.shape,
            "columns": list(calendar_df.columns),
            "schema_check_pass": cal_cols_exist,
            "null_count": sum(cal_nulls.values()),
            "null_summary": {k: int(v) for k, v in cal_nulls.items() if v > 0},
            "duplicate_count": cal_duplicates,
            "sample_head": calendar_df.head(3).to_dict(orient="records"),
            "describe": calendar_df.describe(include='all').fillna("N/A").to_dict()
        }
        
        # Validate Sell Prices
        price_schema = self.config["schemas"]["sell_prices"]
        price_cols_exist = all(col in prices_df.columns for col in price_schema["columns"])
        if not price_cols_exist:
            missing = [col for col in price_schema["columns"] if col not in prices_df.columns]
            msg = f"Sell prices schema mismatch. Missing columns: {missing}"
            logger.error(msg)
            raise DataValidationError(msg)
            
        price_nulls = prices_df.isnull().sum().to_dict()
        price_duplicates = int(prices_df.duplicated().sum())
        # Price validation rules (non-negative, non-zero prices)
        invalid_prices = int((prices_df["sell_price"] <= 0).sum())
        
        results["sell_prices"] = {
            "shape": prices_df.shape,
            "columns": list(prices_df.columns),
            "schema_check_pass": price_cols_exist,
            "null_count": sum(price_nulls.values()),
            "null_summary": {k: int(v) for k, v in price_nulls.items() if v > 0},
            "duplicate_count": price_duplicates,
            "invalid_prices_count": invalid_prices,
            "sample_head": prices_df.head(3).to_dict(orient="records"),
            "describe": prices_df.describe().fillna("N/A").to_dict()
        }
        
        # Validate Sales Train Validation
        sales_schema = self.config["schemas"]["sales_train_validation"]
        sales_cols_exist = all(col in sales_df.columns for col in sales_schema["required_columns"])
        if not sales_cols_exist:
            missing = [col for col in sales_schema["required_columns"] if col not in sales_df.columns]
            msg = f"Sales schema mismatch. Missing columns: {missing}"
            logger.error(msg)
            raise DataValidationError(msg)
            
        sales_nulls = sales_df.isnull().sum().to_dict()
        sales_duplicates = int(sales_df.duplicated().sum())
        # Sales quantity validation (negative sales counts are invalid)
        d_cols = [col for col in sales_df.columns if col.startswith("d_")]
        negative_sales = int((sales_df[d_cols] < 0).sum().sum())
        
        results["sales"] = {
            "shape": sales_df.shape,
            "columns_count": len(sales_df.columns),
            "d_columns_count": len(d_cols),
            "schema_check_pass": sales_cols_exist,
            "null_count": sum(sales_nulls.values()),
            "null_summary": {k: int(v) for k, v in sales_nulls.items() if v > 0},
            "duplicate_count": sales_duplicates,
            "negative_sales_count": negative_sales,
            "sample_head": sales_df.iloc[:3, :10].to_dict(orient="records"),
            "describe_metadata": sales_df[sales_schema["required_columns"]].describe().fillna("N/A").to_dict()
        }

        # Run Sample Query: Filter sales for item HOBBIES_1_001 in store CA_1
        sample_item = sales_df[(sales_df["item_id"] == "HOBBIES_1_001") & (sales_df["store_id"] == "CA_1")]
        results["sample_query"] = {
            "query_description": "Filter sales for item 'HOBBIES_1_001' in store 'CA_1'",
            "rows_returned": len(sample_item),
            "sample_data": sample_item.iloc[:, :12].to_dict(orient="records") if len(sample_item) > 0 else []
        }

        logger.info("All data validations passed validation checks successfully.")
        self.generate_report(results)
        return results
 
    def generate_report(self, results: Dict[str, Any]):
        """Generates a markdown validation report in reports/ directory."""
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
            
        report_path = os.path.join(self.reports_dir, "data_status_report.md")
        logger.info(f"Generating data validation status report at {report_path}")
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# Data Ingestion & Schema Validation Report\n\n")
                f.write(f"**Report Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Overall Status
                all_passed = (
                    results["calendar"]["schema_check_pass"] and 
                    results["sell_prices"]["schema_check_pass"] and 
                    results["sales"]["schema_check_pass"] and
                    results["sell_prices"]["invalid_prices_count"] == 0 and
                    results["sales"]["negative_sales_count"] == 0
                )
                
                if all_passed:
                    f.write("> [!NOTE]\n> **Status:** :white_check_mark: **ALL INGESTION & SCHEMA VALIDATION CHECKS PASSED**\n\n")
                else:
                    f.write("> [!WARNING]\n> **Status:** :warning: **SOME CHECKS RETURNED WARNINGS OR FAILS**\n\n")
                    
                # Calendar Section
                f.write("## 1. Calendar Dataset Summary\n")
                f.write(f"*   **Shape:** {results['calendar']['shape']}\n")
                f.write(f"*   **Schema Matches Config:** {results['calendar']['schema_check_pass']}\n")
                f.write(f"*   **Null Row Count:** {results['calendar']['null_count']}\n")
                f.write(f"*   **Duplicate Rows:** {results['calendar']['duplicate_count']}\n\n")
                
                # Sell Prices Section
                f.write("## 2. Sell Prices Dataset Summary\n")
                f.write(f"*   **Shape:** {results['sell_prices']['shape']}\n")
                f.write(f"*   **Schema Matches Config:** {results['sell_prices']['schema_check_pass']}\n")
                f.write(f"*   **Null Row Count:** {results['sell_prices']['null_count']}\n")
                f.write(f"*   **Duplicate Rows:** {results['sell_prices']['duplicate_count']}\n")
                f.write(f"*   **Invalid Price Values (<= 0):** {results['sell_prices']['invalid_prices_count']}\n\n")
                
                # Sales Train Validation Section
                f.write("## 3. Sales Historical Ingestion Summary\n")
                f.write(f"*   **Shape:** {results['sales']['shape']}\n")
                f.write(f"*   **Total Columns:** {results['sales']['columns_count']}\n")
                f.write(f"*   **Day Time-series columns (d_1 to d_N):** {results['sales']['d_columns_count']}\n")
                f.write(f"*   **Schema Matches Config:** {results['sales']['schema_check_pass']}\n")
                f.write(f"*   **Null Values Count:** {results['sales']['null_count']}\n")
                f.write(f"*   **Duplicate Series:** {results['sales']['duplicate_count']}\n")
                f.write(f"*   **Negative Sales Volumes:** {results['sales']['negative_sales_count']}\n\n")
                
                # Sample Query Section
                f.write("## 4. Sample Query Execution\n")
                f.write(f"**Query:** {results['sample_query']['query_description']}\n")
                f.write(f"*   **Rows Returned:** {results['sample_query']['rows_returned']}\n\n")
                
                if results['sample_query']['rows_returned'] > 0:
                    f.write("### Sample Loaded Row Data (First 12 columns):\n")
                    sample_data = results['sample_query']['sample_data']
                    cols = list(sample_data[0].keys())
                    f.write("| " + " | ".join(cols) + " |\n")
                    f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
                    for row in sample_data:
                        f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
                else:
                    f.write("*No rows returned in sample query check.*\n")
            logger.info("Successfully generated data status markdown report.")
        except Exception as e:
            logger.error(f"Failed to generate status report file: {e}")
            raise

if __name__ == "__main__":
    loader = DataLoader()
    try:
        loader.run_validation_checks()
        print("Data Ingestion & Validation Pipeline completed successfully. See logs/app.log and reports/data_status_report.md for details.")
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
