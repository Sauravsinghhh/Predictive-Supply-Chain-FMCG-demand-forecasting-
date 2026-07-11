"""
Unit Tests for Baseline Forecasting Models and Replenishment Logic.
"""

import pytest
import numpy as np
import sys
import os

# Fix path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baseline_forecast import NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster
from src.inventory.replenishment import calculate_replenishment_order

def test_naive_forecaster():
    y = np.array([1.0, 2.0, 3.0, 5.0])
    model = NaiveForecaster()
    model.fit(y)
    preds = model.predict(horizon=3)
    assert len(preds) == 3
    assert np.all(preds == 5.0)

def test_seasonal_naive_forecaster():
    # 7-day seasonality
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    model = SeasonalNaiveForecaster(period=7)
    model.fit(y)
    preds = model.predict(horizon=5)
    assert len(preds) == 5
    assert np.allclose(preds, [3.0, 4.0, 5.0, 6.0, 7.0])

def test_moving_average_forecaster():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = MovingAverageForecaster(window=3)
    model.fit(y)
    preds = model.predict(horizon=4)
    # mean of last 3: (3.0 + 4.0 + 5.0) / 3 = 4.0
    assert len(preds) == 4
    assert np.all(preds == 4.0)

def test_replenishment_logic():
    res = calculate_replenishment_order(forecast_demand_sum=15.0, current_inventory=5.0, safety_buffer=10.0)
    assert res["recommended_order"] == 20.0
    assert res["under_stock_alert"] is True
    
    res_neg = calculate_replenishment_order(forecast_demand_sum=5.0, current_inventory=15.0, safety_buffer=5.0)
    assert res_neg["recommended_order"] == 0.0
    assert res_neg["under_stock_alert"] is False
