"""
Baseline Time-Series Forecasting Models for FreshMind.
Provides Naive, Seasonal Naive, and Moving Average forecasters.
"""

import numpy as np
import pandas as pd
from typing import Union, List

class BaseForecaster:
    """Base class for all baseline forecasters."""
    def fit(self, y: Union[np.ndarray, pd.Series]) -> 'BaseForecaster':
        """
        Fits the model on training target values.
        
        Args:
            y: Time-series of target values (historical demand).
            
        Returns:
            self: The fitted forecaster instance.
        """
        self.y_ = np.asarray(y)
        if len(self.y_) == 0:
            raise ValueError("Training series cannot be empty.")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """
        Generates point forecasts for the specified horizon.
        
        Args:
            horizon: Number of time steps to forecast.
            
        Returns:
            np.ndarray: Forecasted demand values.
        """
        raise NotImplementedError("Subclasses must implement predict()")

class NaiveForecaster(BaseForecaster):
    """
    Naive Forecasting Model.
    Predicts all future values to be equal to the last observed value.
    """
    def predict(self, horizon: int) -> np.ndarray:
        if horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")
        last_val = self.y_[-1]
        return np.full(shape=(horizon,), fill_value=last_val, dtype=np.float32)

class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal Naive Forecasting Model.
    Predicts future values by repeating the last season's observations.
    For daily data, standard period is 7 (weekly seasonality).
    """
    def __init__(self, period: int = 7):
        self.period = period

    def predict(self, horizon: int) -> np.ndarray:
        if horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")
        if len(self.y_) < self.period:
            # Fallback to naive if series is shorter than period
            last_val = self.y_[-1]
            return np.full(shape=(horizon,), fill_value=last_val, dtype=np.float32)
            
        forecast = []
        for h in range(1, horizon + 1):
            # Index back into the series
            back_idx = len(self.y_) - self.period + ((h - 1) % self.period)
            forecast.append(self.y_[back_idx])
        return np.array(forecast, dtype=np.float32)

class MovingAverageForecaster(BaseForecaster):
    """
    Simple Moving Average Forecaster.
    Predicts future values as the mean of the last W observed values.
    """
    def __init__(self, window: int = 7):
        self.window = window

    def predict(self, horizon: int) -> np.ndarray:
        if horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")
        actual_window = min(self.window, len(self.y_))
        mean_val = np.mean(self.y_[-actual_window:])
        return np.full(shape=(horizon,), fill_value=mean_val, dtype=np.float32)
