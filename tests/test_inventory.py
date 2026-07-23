"""
Unit Tests for Inventory Optimization and Decision Science Engine.
"""

import pytest
import os
import sys

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inventory.optimization import calculate_inventory_policies, normal_loss_function

def test_normal_loss_function():
    """Asserts mathematical correctness of normal loss function."""
    # L(0) should equal phi(0) = 0.3989
    val_zero = normal_loss_function(0.0)
    assert abs(val_zero - 0.3989) < 1e-3
    
    # L(z) should decrease as z increases
    val_high = normal_loss_function(2.0)
    assert val_high < val_zero

def test_calculate_inventory_policies():
    """Asserts policy recommendation bounds and health classifications."""
    # Understock condition
    res_under = calculate_inventory_policies(
        forecast_demand_mean=10.0,
        forecast_demand_std=2.0,
        current_on_hand=2.0, # Below lead time safety stock
        lead_time_days=3,
        service_level=0.95
    )
    assert res_under["inventory_health"] == "Understock Alert"
    assert res_under["under_stock_alert"] is True
    assert res_under["recommended_order_out"] > 0.0
    
    # Healthy condition
    res_healthy = calculate_inventory_policies(
        forecast_demand_mean=10.0,
        forecast_demand_std=2.0,
        current_on_hand=35.0, # Safe stock level
        lead_time_days=3,
        service_level=0.95
    )
    assert res_healthy["inventory_health"] == "Healthy"
    assert res_healthy["under_stock_alert"] is False
    
    # Expected shortage cost check
    assert res_under["expected_stockout_cost"] > res_healthy["expected_stockout_cost"]
