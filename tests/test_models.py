"""
Unit Tests for Classical, Machine Learning, and Deep Learning Forecasting Models.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.classical import ARIMAForecaster, ETSForecaster, ProphetForecaster
from src.models.ml_models import LightGBMForecaster
from src.models.deep_learning import PyTorchForecaster

@pytest.fixture
def dummy_history():
    """Generates simple dummy demand sequence."""
    return np.array([10.0, 12.0, 11.0, 15.0, 14.0, 16.0, 18.0, 20.0, 19.0, 22.0], dtype=np.float32)

@pytest.fixture
def dummy_dataframe(dummy_history):
    """Generates dummy long-format dataframe for ML models."""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(dummy_history))
    return pd.DataFrame({
        "date": dates,
        "sales": dummy_history,
        "store_id": ["CA_1"] * len(dummy_history),
        "item_id": ["HOBBIES_1_001"] * len(dummy_history),
        "state_id": ["CA"] * len(dummy_history),
        "wm_yr_wk": [11101] * len(dummy_history),
        "snap_CA": [0] * len(dummy_history)
    })

def test_arima_forecaster(dummy_history):
    """Verifies ARIMA fits and predicts correct shape."""
    forecaster = ARIMAForecaster(order=(1, 0, 0))
    forecaster.fit(dummy_history)
    preds = forecaster.predict(horizon=4)
    assert len(preds) == 4
    assert not np.isnan(preds).any()

def test_ets_forecaster(dummy_history):
    """Verifies ETS fits and predicts correct shape."""
    forecaster = ETSForecaster(trend="add")
    forecaster.fit(dummy_history)
    preds = forecaster.predict(horizon=3)
    assert len(preds) == 3
    assert not np.isnan(preds).any()

def test_prophet_forecaster(dummy_history):
    """Verifies Prophet fits and predicts correct shape."""
    forecaster = ProphetForecaster()
    # Dummy dates
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(dummy_history))
    forecaster.fit(dummy_history, dates=dates)
    preds = forecaster.predict(horizon=5)
    assert len(preds) == 5
    assert not np.isnan(preds).any()

def test_lgb_forecaster_recursive(dummy_dataframe):
    """Verifies LightGBM in recursive mode runs and predicts."""
    # Use short lags to fit on tiny dataframe
    forecaster = LightGBMForecaster(mode="recursive", lags=[1, 2], windows=[2])
    forecaster.fit(dummy_dataframe, target_horizon=3)
    preds = forecaster.predict(dummy_dataframe, horizon=3)
    assert len(preds) == 3
    assert not np.isnan(preds).any()

def test_lgb_forecaster_direct(dummy_dataframe):
    """Verifies LightGBM in direct mode runs and predicts."""
    forecaster = LightGBMForecaster(mode="direct", lags=[1, 2], windows=[2])
    forecaster.fit(dummy_dataframe, target_horizon=2)
    preds = forecaster.predict(dummy_dataframe, horizon=2)
    assert len(preds) == 2
    assert not np.isnan(preds).any()

def test_pytorch_nbeats(dummy_history):
    """Verifies PyTorch NBEATS fits and predicts."""
    forecaster = PyTorchForecaster(model_type="nbeats", seq_len=4, epochs=2)
    forecaster.fit(dummy_history, horizon=2)
    preds = forecaster.predict(horizon=2)
    assert len(preds) == 2
    assert not np.isnan(preds).any()

def test_pytorch_tft(dummy_history):
    """Verifies PyTorch TFT fits and predicts."""
    forecaster = PyTorchForecaster(model_type="tft", seq_len=4, epochs=2)
    forecaster.fit(dummy_history, horizon=2)
    preds = forecaster.predict(horizon=2)
    assert len(preds) == 2
    assert not np.isnan(preds).any()
