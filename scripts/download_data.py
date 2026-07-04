"""
Script to download or synthetically generate the Walmart M5 Forecasting Dataset.
Provides command-line arguments to select between downloading the full dataset
or creating a small-scale, schema-compliant sample dataset for testing.
"""

import argparse
import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import sys

# Import custom logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logging_utils import setup_logger

logger = setup_logger("download_data", log_dir="logs", log_file="app.log")

# Configuration
DOWNLOAD_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
RAW_DATA_DIR = os.path.join("data", "raw")
ZIP_FILE_PATH = os.path.join(RAW_DATA_DIR, "m5.zip")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    logger.info(f"Downloading from {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    logger.info("Download completed successfully.")

def generate_synthetic_data(output_dir: str, num_items: int = 50, num_days: int = 100):
    """
    Generates small, schema-compliant synthetic CSV files for local development and testing.
    """
    logger.info(f"Generating synthetic M5 dataset with {num_items} items over {num_days} days...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Generate Calendar
    start_date = datetime(2011, 1, 29) # M5 Start Date
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    calendar_rows = []
    for i, dt in enumerate(dates):
        d_id = f"d_{i+1}"
        wm_yr_wk = 11101 + (i // 7)
        wday = ((dt.weekday() + 2) % 7) + 1  # M5 weekday starts Saturday=1, Sunday=2... Friday=7
        weekday = dt.strftime("%A")
        month = dt.month
        year = dt.year
        
        # Add basic event simulation
        event_name = "SuperBowl" if dt.month == 2 and dt.day == 6 else "None"
        event_type = "Sporting" if event_name != "None" else "None"
        
        calendar_rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "wm_yr_wk": wm_yr_wk,
            "weekday": weekday,
            "wday": wday,
            "month": month,
            "year": year,
            "d": d_id,
            "event_name_1": event_name if event_name != "None" else np.nan,
            "event_type_1": event_type if event_type != "None" else np.nan,
            "event_name_2": np.nan,
            "event_type_2": np.nan,
            "snap_CA": 1 if dt.day <= 10 else 0,
            "snap_TX": 1 if dt.day in [1, 3, 5, 6, 7, 9, 11, 12, 13, 15] else 0,
            "snap_WI": 1 if dt.day in [2, 3, 5, 8, 9, 10, 11, 14, 15, 18] else 0
        })
    calendar_df = pd.DataFrame(calendar_rows)
    calendar_df.to_csv(os.path.join(output_dir, "calendar.csv"), index=False)
    logger.info("Saved calendar.csv")

    # 2. Generate Sales Train Validation
    items = [f"HOBBIES_1_{i:03d}" for i in range(1, num_items + 1)]
    stores = ["CA_1", "CA_2", "TX_1", "WI_1"]
    
    sales_rows = []
    for store in stores:
        state = store.split("_")[0]
        for item in items:
            row = {
                "id": f"{item}_{store}_validation",
                "item_id": item,
                "dept_id": "HOBBIES_1",
                "cat_id": "HOBBIES",
                "store_id": store,
                "state_id": state
            }
            # Generate random sales history
            for day_idx in range(1, num_days + 1):
                row[f"d_{day_idx}"] = int(np.random.poisson(lam=1.5))
            sales_rows.append(row)
            
    sales_df = pd.DataFrame(sales_rows)
    sales_df.to_csv(os.path.join(output_dir, "sales_train_validation.csv"), index=False)
    logger.info("Saved sales_train_validation.csv")

    # 3. Generate Sell Prices
    price_rows = []
    for store in stores:
        for item in items:
            base_price = round(np.random.uniform(1.99, 14.99), 2)
            # Find unique weeks
            weeks = calendar_df["wm_yr_wk"].unique()
            for week in weeks:
                price_rows.append({
                    "store_id": store,
                    "item_id": item,
                    "wm_yr_wk": week,
                    "sell_price": base_price
                })
    prices_df = pd.DataFrame(price_rows)
    prices_df.to_csv(os.path.join(output_dir, "sell_prices.csv"), index=False)
    logger.info("Saved sell_prices.csv")

    # 4. Generate Sample Submission
    submission_rows = []
    for r in sales_df["id"].values:
        sub_row = {"id": r}
        for d in range(1, 29):
            sub_row[f"F{d}"] = 0
        submission_rows.append(sub_row)
        
    # Also add evaluation rows for evaluation set (standard M5 format)
    for r in sales_df["id"].values:
        sub_row = {"id": r.replace("validation", "evaluation")}
        for d in range(1, 29):
            sub_row[f"F{d}"] = 0
        submission_rows.append(sub_row)
        
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(os.path.join(output_dir, "sample_submission.csv"), index=False)
    logger.info("Saved sample_submission.csv")
    
    logger.info("Synthetic dataset creation complete.")

def main():
    parser = argparse.ArgumentParser(description="Acquire raw M5 dataset files.")
    parser.add_argument("--sample", action="store_true", help="Generate a small, schema-compliant synthetic M5 dataset instead of downloading.")
    parser.add_argument("--force", action="store_true", help="Force download even if files already exist.")
    args = parser.parse_args()

    # Ensure raw directory exists
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

    # Check if files already exist
    required_files = ["calendar.csv", "sell_prices.csv", "sales_train_validation.csv", "sample_submission.csv"]
    files_exist = all(os.path.exists(os.path.join(RAW_DATA_DIR, f)) for f in required_files)

    if files_exist and not args.force:
        logger.info("M5 dataset files already exist in data/raw/. Skipping generation/download.")
        return

    if args.sample:
        generate_synthetic_data(RAW_DATA_DIR)
        return

    try:
        # Attempt to download the full dataset
        download_url(DOWNLOAD_URL, ZIP_FILE_PATH)
        
        logger.info(f"Extracting {ZIP_FILE_PATH} into {RAW_DATA_DIR}...")
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
            
        logger.info("Removing temporary ZIP archive...")
        os.remove(ZIP_FILE_PATH)
        
        logger.info("M5 Dataset successfully downloaded and extracted.")
    except Exception as e:
        logger.error(f"Failed to download/extract full dataset: {e}")
        logger.warning("Falling back to generating synthetic sample dataset so project setup remains runnable.")
        generate_synthetic_data(RAW_DATA_DIR)

if __name__ == "__main__":
    main()
