"""
Unit Tests for Hierarchical Reconciliation Forecasting.
"""

import pytest
import numpy as np
import os
import sys

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.hierarchical import HierarchicalReconciler

@pytest.fixture
def base_forecasts():
    """Generates simple incoherent base forecasts across levels."""
    # Total, States (CA, TX, WI), Stores (CA_1, CA_2, TX_1, WI_1)
    return {
        "Total": np.array([45.0, 50.0]),
        "CA": np.array([25.0, 28.0]),
        "TX": np.array([12.0, 14.0]),
        "WI": np.array([11.0, 10.0]),
        "CA_1": np.array([13.0, 15.0]),
        "CA_2": np.array([11.0, 12.0]),
        "TX_1": np.array([13.0, 13.0]),
        "WI_1": np.array([10.0, 9.0])
    }

def test_bottom_up_reconciliation(base_forecasts):
    """Verifies that bottom-up reconciliation sums perfectly."""
    reconciler = HierarchicalReconciler()
    reconciled = reconciler.bottom_up_reconciliation(base_forecasts)
    
    # Bottom levels: CA_1 (13), CA_2 (11), TX_1 (13), WI_1 (10)
    # Reconciled CA should equal CA_1 + CA_2 = 13 + 11 = 24
    assert reconciled["CA"][0] == 24.0
    # Reconciled Total should equal sum of all bottom: 13 + 11 + 13 + 10 = 47
    assert reconciled["Total"][0] == 47.0

def test_top_down_reconciliation(base_forecasts):
    """Verifies that top-down reconciliation allocates and sums correctly."""
    reconciler = HierarchicalReconciler()
    # Simple historical store proportions
    proportions = {
        "CA_1": 0.3,
        "CA_2": 0.3,
        "TX_1": 0.2,
        "WI_1": 0.2
    }
    reconciled = reconciler.top_down_reconciliation(base_forecasts, proportions)
    
    # Total forecast is 45.0
    # Reconciled Total must equal 45.0
    assert reconciled["Total"][0] == 45.0
    # CA_1 should be 45.0 * 0.3 = 13.5
    assert reconciled["CA_1"][0] == 13.5
    # CA should equal CA_1 + CA_2 = 13.5 + 13.5 = 27.0
    assert reconciled["CA"][0] == 27.0

def test_ols_reconciliation(base_forecasts):
    """Verifies that OLS projection generates coherent forecasts."""
    reconciler = HierarchicalReconciler()
    reconciled = reconciler.mint_ols_reconciliation(base_forecasts)
    
    # Reconciled CA should equal CA_1 + CA_2
    assert np.allclose(reconciled["CA"], reconciled["CA_1"] + reconciled["CA_2"])
    # Reconciled Total should equal CA + TX + WI
    assert np.allclose(reconciled["Total"], reconciled["CA"] + reconciled["TX"] + reconciled["WI"])
