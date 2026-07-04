"""
Unit Tests for Data Loader and Ingestion Pipeline.
Verifies config loading, memory optimization downcasting, and validation functions.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import DataLoader

def test_load_config():
    """Tests if YAML config loads correctly and contains necessary sections."""
    loader = DataLoader(config_path="configs/config.yaml")
    config = loader.config
    assert isinstance(config, dict)
    assert "paths" in config
    assert "schemas" in config
    assert config["paths"]["data_raw_dir"] == "data/raw"

def test_optimize_memory():
    """Tests if memory downcasting works as expected for different datatypes."""
    loader = DataLoader(config_path="configs/config.yaml")
    
    # Create dummy dataframe with large data types
    df = pd.DataFrame({
        "large_int": np.array([1, 2, 10], dtype=np.int64),
        "large_float": np.array([1.5, 2.7, 3.9], dtype=np.float64),
        "low_cardinality_str": ["category_A", "category_B", "category_A"]
    })
    
    optimized_df = loader.optimize_memory(df)
    
    # Check types have been downcasted properly
    assert optimized_df["large_int"].dtype == np.int8
    assert optimized_df["large_float"].dtype == np.float32
    assert isinstance(optimized_df["low_cardinality_str"].dtype, pd.CategoricalDtype)

def test_run_validation_checks():
    """Tests if schema validations and duplicate/missing count checks run successfully."""
    loader = DataLoader(config_path="configs/config.yaml")
    results = loader.run_validation_checks()
    
    # Assert validation results exist
    assert "calendar" in results
    assert "sell_prices" in results
    assert "sales" in results
    assert "sample_query" in results
    
    # Assert schemas match config definitions
    assert results["calendar"]["schema_check_pass"] is True
    assert results["sell_prices"]["schema_check_pass"] is True
    assert results["sales"]["schema_check_pass"] is True
    
    # Assert shapes are valid
    assert results["calendar"]["shape"][0] > 0
    assert results["sell_prices"]["shape"][0] > 0
    assert results["sales"]["shape"][0] > 0
    
    # Assert duplicates, invalid values are calculated
    assert "duplicate_count" in results["calendar"]
    assert "invalid_prices_count" in results["sell_prices"]
    assert "negative_sales_count" in results["sales"]
