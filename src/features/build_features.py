"""
Feature engineering and preparation pipeline for FreshMind.
Transforms raw wide-format time series into long-format datasets,
merges calendar metadata, and computes advanced rolling statistical, fourier, price, and seasonal features.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List

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
    item_id: str,
    prices_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Extracts, melts, and merges a single store-SKU time-series with calendar and price data.
    Ensures memory efficiency by filtering before melting.
    
    Args:
        sales_df: DataFrame containing daily sales in wide format.
        calendar_df: DataFrame containing calendar mapping (dates, events, SNAP).
        store_id: Identifier of the target store.
        item_id: Identifier of the target SKU.
        prices_df: Optional DataFrame containing pricing details.
        
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
    
    # Optionally merge with prices
    if prices_df is not None and not prices_df.empty:
        logger.info("Merging with sell price records...")
        merged = pd.merge(
            merged, 
            prices_df, 
            on=["store_id", "item_id", "wm_yr_wk"], 
            how="left"
        )
        # Forward fill missing prices if any, then backfill if needed
        if "sell_price" in merged.columns:
            merged["sell_price"] = merged["sell_price"].ffill().bfill()
        else:
            logger.warning("Prices DataFrame was provided, but 'sell_price' was not found.")
            
    logger.info(f"Successfully prepared series with shape {merged.shape} for Store: {store_id}, SKU: {item_id}")
    return merged

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends date-based temporal characteristics to the series DataFrame.
    """
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' column for temporal extraction.")
        
    # Standard datetime extractions
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    
    # Holiday Indicators (from calendar metadata)
    if "event_name_1" in df.columns:
        df["is_holiday"] = df["event_name_1"].notnull().astype(int)
    else:
        df["is_holiday"] = 0
    
    # Promotion Indicators (SNAP events matching state)
    state = df["state_id"].iloc[0] if "state_id" in df.columns else "CA"
    snap_col = f"snap_{state}"
    if snap_col in df.columns:
        df["is_promo"] = df[snap_col].astype(int)
    else:
        df["is_promo"] = 0
        
    return df

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates weather variables (temperature and precipitation/rainfall)
    to act as external regressors.
    """
    df = df.copy()
    # Simulate seasonal temperature based on the month (hot in July, cold in January)
    if "date" in df.columns:
        months = df["date"].dt.month
        df["temperature"] = 17.5 + 12.5 * np.cos(2 * np.pi * (months - 7) / 12.0)
    else:
        df["temperature"] = 20.0
        
    # Simulate precipitation/rainfall
    np.random.seed(42)
    df["precipitation"] = np.random.exponential(scale=1.5, size=len(df))
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes price changes and relative statistics.
    """
    df = df.copy()
    if "sell_price" not in df.columns:
        # Default price if missing
        df["sell_price"] = 1.99
        
    # Lagged price differences
    df["price_lag_1"] = df["sell_price"].shift(1).ffill().bfill()
    df["price_diff_1"] = df["sell_price"] - df["price_lag_1"]
    df["price_pct_change"] = (df["price_diff_1"] / df["price_lag_1"]).fillna(0.0)
    
    # Rolling price relative standard deviation (7 days)
    df["price_roll_std_7"] = df["sell_price"].rolling(window=7, min_periods=1).std().fillna(0.0)
    
    return df

def add_lag_and_rolling_features(
    df: pd.DataFrame, 
    lags: List[int] = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28],
    windows: List[int] = [7, 14, 28]
) -> pd.DataFrame:
    """
    Calculates lags and rolling/expanding statistics for demand forecasting.
    Includes precautions to prevent looking forward in time (data leakage).
    """
    df = df.copy()
    if "sales" not in df.columns:
        raise EmptyDataFrameError("df", "Input DataFrame must contain 'sales' column to calculate lag features.")
        
    # 1. Sales Lags
    for lag in lags:
        df[f"sales_lag_{lag}"] = df["sales"].shift(lag).fillna(0.0)
        
    # 2. Rolling Statistics (computed on lagged sales to prevent leakage during recursive steps)
    # The statistics at time t should only reflect values up to time t-1
    for window in windows:
        # Shift sales by 1 first to prevent any leakage
        lagged_sales = df["sales"].shift(1).fillna(0.0)
        df[f"sales_roll_mean_{window}"] = lagged_sales.rolling(window=window, min_periods=1).mean().fillna(0.0)
        df[f"sales_roll_median_{window}"] = lagged_sales.rolling(window=window, min_periods=1).median().fillna(0.0)
        df[f"sales_roll_std_{window}"] = lagged_sales.rolling(window=window, min_periods=1).std().fillna(0.0)
        
    # 3. Expanding Window Statistics
    lagged_sales = df["sales"].shift(1).fillna(0.0)
    df["sales_expanding_mean"] = lagged_sales.expanding(min_periods=1).mean().fillna(0.0)
    df["sales_expanding_std"] = lagged_sales.expanding(min_periods=1).std().fillna(0.0)
    
    return df

def add_seasonal_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends Fourier sine/cosine transforms for weekly and annual seasonality.
    """
    df = df.copy()
    if "date" not in df.columns:
        # Use simple integer index if date not present
        t = np.arange(len(df))
    else:
        # Convert date to day of year and day of week
        t = df["date"].dt.dayofyear.values
        
    # Weekly seasonality (period = 7)
    df["sin_weekly"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7.0)
    df["cos_weekly"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7.0)
    
    # Annual seasonality (period = 365.25)
    df["sin_annual"] = np.sin(2 * np.pi * t / 365.25)
    df["cos_annual"] = np.cos(2 * np.pi * t / 365.25)
    
    return df

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a linear time trend feature.
    """
    df = df.copy()
    df["trend_idx"] = np.arange(len(df)) + 1.0
    return df

def build_advanced_features(
    sales_df: pd.DataFrame, 
    calendar_df: pd.DataFrame, 
    store_id: str, 
    item_id: str,
    prices_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Combines all feature engineering steps to return a fully featured DataFrame.
    """
    df = prepare_single_series(sales_df, calendar_df, store_id, item_id, prices_df)
    df = add_temporal_features(df)
    df = add_weather_features(df)
    df = add_price_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_seasonal_fourier_features(df)
    df = add_trend_features(df)
    return df

def add_rolling_features(df: pd.DataFrame, window_sizes: Tuple[int, ...] = (7, 14)) -> pd.DataFrame:
    """
    Appends rolling average and standard deviation to preserve backward compatibility.
    Computes directly on the sales column (unshifted) to match original unit tests.
    """
    if df.empty:
        raise EmptyDataFrameError("df", "Cannot add rolling features to empty DataFrame.")
    if "sales" not in df.columns:
        raise EmptyDataFrameError("df", "Input DataFrame must contain 'sales' column to calculate rolling features.")
        
    df = df.copy()
    for window in window_sizes:
        df[f"sales_roll_mean_{window}"] = df["sales"].rolling(window=window, min_periods=1).mean()
        df[f"sales_roll_std_{window}"] = df["sales"].rolling(window=window, min_periods=1).std().fillna(0.0)
    return df
