"""
Feature engineering and preparation pipeline for FreshMind.
Transforms raw wide-format time series into long-format datasets,
merges calendar metadata, and computes rolling statistical features.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.logging_utils import setup_logger
from src.utils.errors import (
    InvalidStoreError,
    InvalidSKUError,
    EmptyDataFrameError
)

logger = setup_logger("build_features")

def prepare_single_series(
    sales_df: pd.DataFrame, 
    calendar_df: pd.DataFrame, 
    store_id: str, 
    item_id: str
) -> pd.DataFrame:
    """
    Extracts, melts, and merges a single store-SKU time-series with calendar data.
    Ensures memory efficiency by filtering before melting.
    
    Args:
        sales_df: DataFrame containing daily sales in wide format.
        calendar_df: DataFrame containing calendar mapping (dates, events, SNAP).
        store_id: Identifier of the target store.
        item_id: Identifier of the target SKU.
        
    Returns:
        pd.DataFrame: Long-format DataFrame for the single time-series sorted by date.
        
    Raises:
        InvalidStoreError: If store_id is not in the dataset.
        InvalidSKUError: If item_id is not in the dataset.
        EmptyDataFrameError: If the combination results in no data or if inputs are empty.
    """
    if sales_df.empty:
        raise EmptyDataFrameError("sales_df", "Sales DataFrame is empty during feature preparation.")
    if calendar_df.empty:
        raise EmptyDataFrameError("calendar_df", "Calendar DataFrame is empty during feature preparation.")
        
    # Check if store_id exists
    valid_stores = sales_df["store_id"].unique()
    if store_id not in valid_stores:
        msg = f"Store ID {store_id} not found in sales dataset."
        logger.error(msg)
        raise InvalidStoreError(store_id, msg)
        
    # Check if item_id exists
    valid_items = sales_df["item_id"].unique()
    if item_id not in valid_items:
        msg = f"SKU ID {item_id} not found in sales dataset."
        logger.error(msg)
        raise InvalidSKUError(item_id, msg)

    logger.info(f"Extracting series for Store: {store_id}, SKU: {item_id}...")
    
    # Filter for the specific store and SKU
    series_row = sales_df[(sales_df["store_id"] == store_id) & (sales_df["item_id"] == item_id)]
    
    if series_row.empty:
        msg = f"No series found for combination of store_id: {store_id} and item_id: {item_id}"
        logger.error(msg)
        raise EmptyDataFrameError(f"{store_id}_{item_id}", msg)
        
    # Get metadata columns
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    # Melt the single row
    d_cols = [col for col in sales_df.columns if col.startswith("d_")]
    if not d_cols:
        msg = "Sales DataFrame does not contain any daily time-series columns (d_...)"
        logger.error(msg)
        raise EmptyDataFrameError("sales_df", msg)
        
    melted = series_row.melt(
        id_vars=meta_cols,
        value_vars=d_cols,
        var_name="d",
        value_name="sales"
    )
    
    # Merge with calendar
    if "d" not in calendar_df.columns:
        msg = "Calendar DataFrame does not contain column 'd' for merging."
        logger.error(msg)
        raise EmptyDataFrameError("calendar_df", msg)
        
    merged = pd.merge(melted, calendar_df, on="d", how="inner")
    
    if merged.empty:
        msg = f"Merging with calendar returned empty DataFrame for Store: {store_id}, SKU: {item_id}."
        logger.error(msg)
        raise EmptyDataFrameError("merged_df", msg)
        
    # Parse date and sort
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").reset_index(drop=True)
    
    logger.info(f"Successfully prepared series with shape {merged.shape} for Store: {store_id}, SKU: {item_id}")
    return merged

def add_rolling_features(df: pd.DataFrame, window_sizes: Tuple[int, ...] = (7, 14)) -> pd.DataFrame:
    """
    Computes rolling averages and standard deviations of historical sales.
    
    Args:
        df: Long-format DataFrame containing 'sales'.
        window_sizes: Tuple of rolling window sizes.
        
    Returns:
        pd.DataFrame: DataFrame with rolling features appended.
        
    Raises:
        EmptyDataFrameError: If the input DataFrame is empty or doesn't contain 'sales'.
    """
    if df.empty:
        raise EmptyDataFrameError("df", "Cannot add rolling features to empty DataFrame.")
    if "sales" not in df.columns:
        raise EmptyDataFrameError("df", "Input DataFrame must contain 'sales' column to calculate rolling features.")
        
    logger.info(f"Adding rolling features for windows: {window_sizes}...")
    df = df.copy()
    for window in window_sizes:
        df[f"sales_roll_mean_{window}"] = df["sales"].rolling(window=window, min_periods=1).mean()
        df[f"sales_roll_std_{window}"] = df["sales"].rolling(window=window, min_periods=1).std().fillna(0)
    return df
