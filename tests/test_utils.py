"""
Unit Tests for Project Utilities.
Verifies custom exceptions and logging setup functionalities.
"""

import pytest
import os
import sys
import logging

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.errors import (
    FreshMindError,
    MissingDatasetError,
    InvalidStoreError,
    InvalidSKUError,
    EmptyDataFrameError,
    DataValidationError
)
from src.utils.logging_utils import setup_logger

def test_custom_exceptions():
    """Tests if custom exceptions raise with the correct message and retain attributes."""
    
    # 1. MissingDatasetError
    ex_missing = MissingDatasetError("path/to/missing_file.csv")
    assert ex_missing.file_path == "path/to/missing_file.csv"
    assert "path/to/missing_file.csv" in str(ex_missing)
    assert isinstance(ex_missing, FreshMindError)
    
    # 2. InvalidStoreError
    ex_store = InvalidStoreError("STORE_99")
    assert ex_store.store_id == "STORE_99"
    assert "STORE_99" in str(ex_store)
    
    # 3. InvalidSKUError
    ex_sku = InvalidSKUError("SKU_ABC")
    assert ex_sku.item_id == "SKU_ABC"
    assert "SKU_ABC" in str(ex_sku)
    
    # 4. EmptyDataFrameError
    ex_empty = EmptyDataFrameError("sales_table", "df is empty")
    assert ex_empty.dataset_name == "sales_table"
    assert "sales_table" in str(ex_empty)
    
    # 5. DataValidationError
    ex_val = DataValidationError("missing calendar column X")
    assert ex_val.detail == "missing calendar column X"
    assert "missing calendar column X" in str(ex_val)

def test_setup_logger_behavior():
    """Tests setup_logger to ensure it returns a valid configured logger."""
    test_logger_name = "test_custom_logger"
    logger = setup_logger(test_logger_name)
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == test_logger_name
    assert logger.hasHandlers()
    
    # Verify console and file handlers exist
    handler_types = [type(h) for h in logger.handlers]
    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types
