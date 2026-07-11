"""
Feature engineering and preparation pipeline for FreshMind.
Transforms raw wide-format time series into long-format datasets,
merges calendar metadata, and computes rolling statistical features.
"""

import pandas as pd
import numpy as np
from typing import Tuple

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
    """
    # Filter for the specific store and SKU
    series_row = sales_df[(sales_df["store_id"] == store_id) & (sales_df["item_id"] == item_id)]
    
    if series_row.empty:
        raise ValueError(f"No series found for store_id: {store_id} and item_id: {item_id}")
        
    # Get metadata columns
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    # Melt the single row
    d_cols = [col for col in sales_df.columns if col.startswith("d_")]
    melted = series_row.melt(
        id_vars=meta_cols,
        value_vars=d_cols,
        var_name="d",
        value_name="sales"
    )
    
    # Merge with calendar
    merged = pd.merge(melted, calendar_df, on="d", how="inner")
    
    # Parse date and sort
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").reset_index(drop=True)
    
    return merged

def add_rolling_features(df: pd.DataFrame, window_sizes: Tuple[int, ...] = (7, 14)) -> pd.DataFrame:
    """
    Computes rolling averages and standard deviations of historical sales.
    
    Args:
        df: Long-format DataFrame containing 'sales'.
        window_sizes: Tuple of rolling window sizes.
        
    Returns:
        pd.DataFrame: DataFrame with rolling features appended.
    """
    df = df.copy()
    for window in window_sizes:
        df[f"sales_roll_mean_{window}"] = df["sales"].rolling(window=window, min_periods=1).mean()
        df[f"sales_roll_std_{window}"] = df["sales"].rolling(window=window, min_periods=1).std().fillna(0)
    return df
