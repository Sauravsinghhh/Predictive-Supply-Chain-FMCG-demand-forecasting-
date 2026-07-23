"""
Machine Learning Forecasting Models for FreshMind.
Provides LightGBM regression model wrappers with support for both
recursive (one-step autoregressive) and direct (multi-model) forecasting.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.features.build_features import add_temporal_features, add_lag_and_rolling_features, add_seasonal_fourier_features, add_trend_features
from src.utils.logging_utils import setup_logger
from src.utils.errors import EmptyDataFrameError

logger = setup_logger("ml_models")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    logger.warning("lightgbm package is not available. LightGBMForecaster will fall back to baseline forecasts.")

class LightGBMForecaster:
    """
    LightGBM Forecasting Model.
    Supports recursive forecasting (single model with step-by-step lag updates)
    and direct forecasting (separate models for each step in the horizon).
    """
    def __init__(
        self, 
        mode: str = "recursive", 
        lags: List[int] = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28],
        windows: List[int] = [7, 14, 28],
        params: Optional[Dict[str, Any]] = None
    ):
        self.mode = mode.lower()
        if self.mode not in ["recursive", "direct"]:
            raise ValueError("Mode must be either 'recursive' or 'direct'")
        
        self.lags = lags
        self.windows = windows
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1
        }
        
        self.models_ = {} # Maps step index to fitted model
        self.feature_cols_ = None

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calculates advanced features and returns X, y.
        """
        # Ensure base columns exist
        df = df.copy()
        if "sales" not in df.columns:
            raise EmptyDataFrameError("df", "Input DataFrame must contain 'sales' column.")
            
        # Apply feature pipeline
        df = add_temporal_features(df)
        df = add_lag_and_rolling_features(df, lags=self.lags, windows=self.windows)
        df = add_seasonal_fourier_features(df)
        df = add_trend_features(df)
        
        # Exclude raw target and text metadata columns from training features
        exclude_cols = [
            "id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "d", 
            "event_name_1", "event_type_1", "event_name_2", "event_type_2", "sales"
        ]
        
        # Keep only numeric features
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        # Clean potential infinity or NaN values
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = df["sales"]
        
        return X, y

    def fit(self, df: pd.DataFrame, target_horizon: int = 14) -> 'LightGBMForecaster':
        """
        Fits the LightGBM models on a prepared long-format DataFrame.
        """
        if lgb is None:
            logger.error("LightGBM package is missing. Cannot fit LightGBMForecaster.")
            return self
            
        logger.info(f"Fitting LightGBMForecaster in {self.mode} mode...")
        X, y = self._prepare_features(df)
        self.feature_cols_ = list(X.columns)
        
        if self.mode == "recursive":
            # Train a single model to predict sales at step t given features at t-1 (which uses lag-1 target)
            model = lgb.LGBMRegressor(**self.params)
            model.fit(X, y)
            self.models_[0] = model
            logger.info("Successfully fitted recursive LightGBM model.")
        else:
            # Direct mode: Train target_horizon separate models
            # Model h predicts sales at step t+h given features at t-1
            logger.info(f"Fitting {target_horizon} direct step models...")
            for h in range(1, target_horizon + 1):
                y_direct = y.shift(-h).dropna()
                X_direct = X.iloc[:len(y_direct)]
                
                if len(y_direct) == 0:
                    logger.warning(f"Not enough data to train direct model for step {h}. Copying step-1 model.")
                    self.models_[h] = self.models_.get(1)
                    continue
                    
                model = lgb.LGBMRegressor(**self.params)
                model.fit(X_direct, y_direct)
                self.models_[h] = model
                
            logger.info(f"Successfully fitted all {target_horizon} direct LightGBM models.")
            
        return self

    def predict(self, history_df: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Generates predictions for the future horizon.
        """
        if lgb is None or not self.models_:
            logger.warning("LightGBM model not fitted. Falling back to Naive prediction.")
            return np.full((horizon,), history_df["sales"].values[-1], dtype=np.float32)
            
        if self.mode == "recursive":
            return self._predict_recursive(history_df, horizon)
        else:
            return self._predict_direct(history_df, horizon)

    def _predict_recursive(self, history_df: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Generates step-by-step recursive forecasts updating features dynamically.
        """
        model = self.models_[0]
        # Copy history data
        temp_df = history_df.copy()
        
        # Build future dates
        last_date = temp_df["date"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        
        predictions = []
        
        for i in range(horizon):
            current_date = future_dates[i]
            
            # 1. Append a new blank row for the prediction step
            new_row = {
                "date": current_date,
                "sales": 0.0 # Placeholder sales, will update after prediction
            }
            # Copy categorical/metadata if present
            for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"]:
                if col in temp_df.columns:
                    new_row[col] = temp_df[col].iloc[-1]
            
            # Event name handling if present
            for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
                new_row[col] = np.nan
                
            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # 2. Extract features up to the current row
            X_features, _ = self._prepare_features(temp_df)
            
            # Extract features for the latest row (current prediction step)
            current_X = X_features.iloc[[-1]][self.feature_cols_]
            
            # 3. Predict sales
            pred_sales = float(model.predict(current_X)[0])
            pred_sales = max(0.0, pred_sales) # Floor at 0.0
            
            # 4. Fill placeholder sales with predicted sales
            temp_df.loc[temp_df.index[-1], "sales"] = pred_sales
            predictions.append(pred_sales)
            
        return np.array(predictions, dtype=np.float32)

    def _predict_direct(self, history_df: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Generates forecasts by evaluating each direct step model on the current state.
        """
        # Prepare features on the latest state
        X_features, _ = self._prepare_features(history_df)
        current_X = X_features.iloc[[-1]][self.feature_cols_]
        
        predictions = []
        for h in range(1, horizon + 1):
            model = self.models_.get(h)
            if model is None:
                # If direct model h is missing, fallback to step-1 or recursive
                model = self.models_.get(1)
                
            if model is not None:
                pred = float(model.predict(current_X)[0])
                predictions.append(max(0.0, pred))
            else:
                # Naive fallback
                predictions.append(float(history_df["sales"].values[-1]))
                
        return np.array(predictions, dtype=np.float32)

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns feature importance scores if the model is fitted.
        """
        if not self.models_ or self.feature_cols_ is None:
            return {}
            
        # Get feature importance from recursive model or first direct model
        model = self.models_[0] if 0 in self.models_ else self.models_.get(1)
        if model is None:
            return {}
            
        importances = model.feature_importances_
        return dict(zip(self.feature_cols_, importances))
