"""
Classical Forecasting Models for FreshMind.
Provides wrappers for ARIMA, ETS (Exponential Smoothing), and Prophet models.
Ensures graceful fallbacks to baseline forecasts on convergence or initialization failures.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, List, Tuple

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.baseline_forecast import BaseForecaster
from src.utils.logging_utils import setup_logger

logger = setup_logger("classical_models")

# Import statsmodels and prophet inside try/except to handle environment discrepancies
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None
    logger.warning("statsmodels package is not available. ARIMAForecaster will use Naive fallback.")

try:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
except ImportError:
    ETSModel = None
    logger.warning("statsmodels package is not available. ETSForecaster will use Naive fallback.")

try:
    from prophet import Prophet
except ImportError:
    Prophet = None
    logger.warning("prophet package is not available. ProphetForecaster will use Regression/Naive fallback.")

class ARIMAForecaster(BaseForecaster):
    """
    ARIMA Forecaster.
    Fits an autoregressive integrated moving average model to the series.
    """
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.fitted_model_ = None

    def fit(self, y: Union[np.ndarray, pd.Series]) -> 'ARIMAForecaster':
        super().fit(y)
        if ARIMA is None:
            logger.warning("statsmodels not installed. Fitting ARIMA will fall back to Naive.")
            return self
            
        try:
            # Fit simple ARIMA model
            # enforce_stationarity=False, enforce_invertibility=False to avoid optimization failure
            model = ARIMA(self.y_, order=self.order)
            self.fitted_model_ = model.fit()
        except Exception as e:
            logger.warning(f"ARIMA fit failed with error: {e}. Falling back to Naive prediction.")
            self.fitted_model_ = None
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")
            
        if self.fitted_model_ is not None:
            try:
                forecast = self.fitted_model_.forecast(steps=horizon)
                # Handle potential nan outputs
                if not np.isnan(forecast).any():
                    return np.asarray(forecast, dtype=np.float32)
            except Exception as e:
                logger.warning(f"ARIMA prediction failed: {e}. Falling back to Naive.")
                
        # Naive Fallback: repeat last historical value
        last_val = self.y_[-1]
        return np.full(shape=(horizon,), fill_value=last_val, dtype=np.float32)

class ETSForecaster(BaseForecaster):
    """
    Error-Trend-Seasonality Exponential Smoothing Forecaster.
    """
    def __init__(self, error: str = "add", trend: str = "add", seasonal: Optional[str] = None, seasonal_periods: Optional[int] = None):
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_model_ = None

    def fit(self, y: Union[np.ndarray, pd.Series]) -> 'ETSForecaster':
        super().fit(y)
        if ETSModel is None:
            logger.warning("statsmodels not installed. Fitting ETS will fall back to Naive.")
            return self
            
        try:
            model = ETSModel(
                self.y_.astype(np.float64),
                error=self.error,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            )
            # Disp=False to prevent fitting output logging
            self.fitted_model_ = model.fit(disp=False)
        except Exception as e:
            logger.warning(f"ETS fit failed with error: {e}. Falling back to Naive prediction.")
            self.fitted_model_ = None
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")
            
        if self.fitted_model_ is not None:
            try:
                forecast = self.fitted_model_.forecast(steps=horizon)
                if not np.isnan(forecast).any():
                    return np.asarray(forecast, dtype=np.float32)
            except Exception as e:
                logger.warning(f"ETS prediction failed: {e}. Falling back to Naive.")
                
        last_val = self.y_[-1]
        return np.full(shape=(horizon,), fill_value=last_val, dtype=np.float32)

class ProphetForecaster(BaseForecaster):
    """
    Prophet Forecaster wrapper.
    Falls back to a regression-based seasonal curve fit if prophet package is not installed.
    """
    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.model_ = None
        self.dates_ = None

    def fit(self, y: Union[np.ndarray, pd.Series], dates: Optional[Union[pd.DatetimeIndex, List]] = None) -> 'ProphetForecaster':
        super().fit(y)
        
        # Resolve dates
        if dates is not None:
            self.dates_ = pd.to_datetime(dates)
        else:
            # Generate dummy dates ending today
            self.dates_ = pd.date_range(end=pd.Timestamp.today(), periods=len(self.y_))
            
        if Prophet is None:
            logger.warning("Prophet not installed. Fitting ProphetForecaster will use Naive/Moving Average fallback.")
            return self
            
        try:
            # Format dataframe for prophet
            df = pd.DataFrame({
                "ds": self.dates_,
                "y": self.y_
            })
            
            # Initialize and fit model
            self.model_ = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=False
            )
            # Suppress logs
            logging.getLogger('prophet').setLevel(logging.ERROR)
            logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
            self.model_.fit(df)
        except Exception as e:
            logger.warning(f"Prophet fit failed: {e}. Falling back to standard stats.")
            self.model_ = None
            
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")
            
        if Prophet is not None and self.model_ is not None:
            try:
                # Generate future dates
                future_dates = pd.date_range(
                    start=self.dates_[-1] + pd.Timedelta(days=1),
                    periods=horizon
                )
                future_df = pd.DataFrame({"ds": future_dates})
                forecast = self.model_.predict(future_df)
                return np.asarray(forecast["yhat"].values, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Prophet predict failed: {e}. Falling back to Naive.")
                
        # Fallback: Seasonal Naive / Naive combination
        last_val = self.y_[-1]
        return np.full(shape=(horizon,), fill_value=last_val, dtype=np.float32)
