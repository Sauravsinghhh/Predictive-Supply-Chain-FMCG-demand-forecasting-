"""
Unit Tests for Feature Engineering Pipeline.
Verifies preparation of single series, schema validation checks,
and rolling feature calculations.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import prepare_single_series, add_rolling_features
from src.utils.errors import InvalidStoreError, InvalidSKUError, EmptyDataFrameError

@pytest.fixture
def sample_sales_df():
    """Generates a sample wide-format sales DataFrame."""
    return pd.DataFrame({
        "id": ["HOBBIES_1_001_CA_1_validation", "HOBBIES_1_002_CA_1_validation"],
        "item_id": ["HOBBIES_1_001", "HOBBIES_1_002"],
        "dept_id": ["HOBBIES_1", "HOBBIES_1"],
        "cat_id": ["HOBBIES", "HOBBIES"],
        "store_id": ["CA_1", "CA_1"],
        "state_id": ["CA", "CA"],
        "d_1": [1, 2],
        "d_2": [0, 4],
        "d_3": [3, 1]
    })

@pytest.fixture
def sample_calendar_df():
    """Generates a sample calendar DataFrame."""
    return pd.DataFrame({
        "date": ["2011-01-29", "2011-01-29", "2011-01-31"],
        "wm_yr_wk": [11101, 11101, 11101],
        "weekday": ["Saturday", "Sunday", "Monday"],
        "wday": [1, 2, 3],
        "month": [1, 1, 1],
        "year": [2011, 2011, 2011],
        "d": ["d_1", "d_2", "d_3"]
    })

def test_prepare_single_series_success(sample_sales_df, sample_calendar_df):
    """Tests successful extraction and preparation of a single time series."""
    merged = prepare_single_series(sample_sales_df, sample_calendar_df, store_id="CA_1", item_id="HOBBIES_1_001")
    
    assert isinstance(merged, pd.DataFrame)
    assert not merged.empty
    assert len(merged) == 3
    assert "sales" in merged.columns
    assert "date" in merged.columns
    # Check values
    assert list(merged["sales"]) == [1, 0, 3]

def test_prepare_single_series_invalid_store(sample_sales_df, sample_calendar_df):
    """Tests if prepare_single_series raises InvalidStoreError for non-existent store."""
    with pytest.raises(InvalidStoreError):
        prepare_single_series(sample_sales_df, sample_calendar_df, store_id="TX_1", item_id="HOBBIES_1_001")

def test_prepare_single_series_invalid_sku(sample_sales_df, sample_calendar_df):
    """Tests if prepare_single_series raises InvalidSKUError for non-existent SKU."""
    with pytest.raises(InvalidSKUError):
        prepare_single_series(sample_sales_df, sample_calendar_df, store_id="CA_1", item_id="HOBBIES_1_999")

def test_prepare_single_series_empty_inputs():
    """Tests if prepare_single_series raises EmptyDataFrameError for empty inputs."""
    empty_df = pd.DataFrame()
    with pytest.raises(EmptyDataFrameError):
        prepare_single_series(empty_df, empty_df, store_id="CA_1", item_id="HOBBIES_1_001")

def test_add_rolling_features_success():
    """Tests successful calculation of rolling mean and standard deviation."""
    df = pd.DataFrame({
        "sales": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    
    res = add_rolling_features(df, window_sizes=(2, 3))
    
    assert "sales_roll_mean_2" in res.columns
    assert "sales_roll_std_2" in res.columns
    assert "sales_roll_mean_3" in res.columns
    assert "sales_roll_std_3" in res.columns
    
    # Check values for window 2
    # rolling mean of [1.0, 2.0]: first row mean is 1.0 (min_periods=1), second is 1.5
    assert res.loc[0, "sales_roll_mean_2"] == 1.0
    assert res.loc[1, "sales_roll_mean_2"] == 1.5
    assert res.loc[2, "sales_roll_mean_2"] == 2.5
    
    # Check rolling standard deviation of [1, 2] -> 0.7071
    assert np.allclose(res.loc[1, "sales_roll_std_2"], np.std([1, 2], ddof=1))

def test_add_rolling_features_empty():
    """Tests if add_rolling_features raises EmptyDataFrameError for empty inputs."""
    empty_df = pd.DataFrame()
    with pytest.raises(EmptyDataFrameError):
        add_rolling_features(empty_df)
