"""
Evaluation and Walk-Forward Backtesting for FreshMind Forecasting.
Implements Rolling Forecast Origin / Walk-Forward Validation, leakage prevention,
and computes WAPE, MAE, RMSE, SMAPE, MAPE, Bias, and Forecast Accuracy.
Generates metrics comparison reports and automatically determines the best performing model.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.logging_utils import setup_logger

logger = setup_logger("evaluation")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard forecasting performance metrics and operational supply chain metrics.
    """
    # Clean possible inf or nan
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Avoid zero division
    sum_true = np.sum(y_true)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # WAPE: sum(|y_true - y_pred|) / sum(y_true)
    wape = np.sum(np.abs(y_true - y_pred)) / sum_true if sum_true > 0 else 0.0
    
    # MAPE: mean(|y_true - y_pred| / y_true)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs(y_true - y_pred) / np.clip(y_true, 1e-5, None))
        
    # SMAPE: 200 * mean(|y_true - y_pred| / (|y_true| + |y_pred|))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        smape = np.mean(np.abs(y_true - y_pred) / np.clip(denom, 1e-5, None)) * 100.0
        
    # Forecast Bias: sum(y_pred - y_true) / N
    bias = np.mean(y_pred - y_true)
    
    # Forecast Accuracy: 1.0 - WAPE
    accuracy = max(0.0, 1.0 - wape)
    
    # Simulate supply chain inventory loop to calculate Stockout Days and Wastage
    on_hand = 20.0
    stockout_days = 0
    wastage_units = 0.0
    lead_time = 3
    
    orders_placed = list(y_pred)
    for t in range(len(y_true)):
        received = orders_placed[t - lead_time] if t >= lead_time else y_pred[0]
        on_hand += received
        demand = y_true[t]
        
        if on_hand < demand:
            stockout_days += 1
            on_hand = 0.0
        else:
            on_hand -= demand
            
        # Overstock threshold
        if on_hand > 50.0:
            wastage_units += (on_hand - 50.0)
            on_hand = 50.0
            
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE": float(wape),
        "SMAPE": float(smape),
        "MAPE": float(mape),
        "Bias": float(bias),
        "Accuracy": float(accuracy),
        "Stockout_Days": float(stockout_days),
        "Wastage_Units": float(wastage_units)
    }

def walk_forward_validation(
    df: pd.DataFrame,
    models_dict: Dict[str, Any],
    initial_train_days: int = 14,
    step_size: int = 7,
    horizon: int = 7,
    prices_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Performs Rolling Forecast Origin (Walk-Forward validation) over a single series.
    
    Args:
        df: DataFrame representing a single long-format series.
        models_dict: Dictionary mapping model names to forecaster instances.
        initial_train_days: Initial size of the training window.
        step_size: Number of steps to roll the origin forward.
        horizon: Forecasting horizon for each evaluation step.
        prices_df: Optional pricing metadata dataframe.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, np.ndarray]]: (Comparison table DataFrame, actual/predictions dictionary)
    """
    total_len = len(df)
    if total_len < initial_train_days + horizon:
        raise ValueError(f"Series length ({total_len}) is too short for walk-forward validation with train={initial_train_days} and horizon={horizon}")
        
    logger.info(f"Starting walk-forward validation. Length={total_len}, train={initial_train_days}, step={step_size}, horizon={horizon}")
    
    # Store predictions for each model
    model_predictions = {name: [] for name in models_dict.keys()}
    actual_values = []
    
    # Walk forward
    origins = list(range(initial_train_days, total_len - horizon + 1, step_size))
    if not origins:
        origins = [initial_train_days]
        
    for origin in origins:
        # Split train and test
        train_df = df.iloc[:origin].copy()
        test_df = df.iloc[origin : origin + horizon].copy()
        
        actual_values.extend(test_df["sales"].values)
        
        # Fit and predict for each model
        for name, model in models_dict.items():
            try:
                # Format variables depending on model type
                if hasattr(model, "fit") and "LightGBM" in str(type(model)):
                    # LightGBM needs dataframe training
                    model.fit(train_df, target_horizon=horizon)
                    pred = model.predict(train_df, horizon=horizon)
                elif hasattr(model, "fit") and "PyTorch" in str(type(model)):
                    model.fit(train_df["sales"].values, horizon=horizon)
                    pred = model.predict(horizon=horizon)
                elif hasattr(model, "fit") and "Prophet" in str(type(model)):
                    model.fit(train_df["sales"].values, dates=train_df["date"].values)
                    pred = model.predict(horizon=horizon)
                else:
                    # Classical/Baselines
                    model.fit(train_df["sales"].values)
                    pred = model.predict(horizon=horizon)
                    
                model_predictions[name].extend(pred)
            except Exception as e:
                logger.warning(f"Model {name} failed at origin {origin}: {e}. Falling back to Naive values.")
                # Fallback to Naive prediction
                naive_val = train_df["sales"].values[-1]
                model_predictions[name].extend(np.full((horizon,), naive_val))
                
    # Evaluate metrics
    actuals = np.array(actual_values)
    results_list = []
    
    # Ensure correct array lengths matching actuals
    clean_predictions = {}
    for name, preds_list in model_predictions.items():
        preds = np.array(preds_list)[:len(actuals)]
        clean_predictions[name] = preds
        metrics = calculate_metrics(actuals, preds)
        metrics["Model"] = name
        results_list.append(metrics)
        
    results_df = pd.DataFrame(results_list)
    
    # Set Model column as index and reorder columns
    results_df = results_df.set_index("Model")
    results_df = results_df[["Accuracy", "WAPE", "MAE", "RMSE", "SMAPE", "MAPE", "Bias", "Stockout_Days", "Wastage_Units"]]
    
    return results_df, {"actual": actuals, **clean_predictions}

def get_best_model(comparison_df: pd.DataFrame) -> Tuple[str, float]:
    """
    Identifies the best performing model based on WAPE (Weighted Absolute Percentage Error).
    """
    if comparison_df.empty:
        raise ValueError("Comparison DataFrame cannot be empty.")
    # Find model with minimum WAPE
    best_model_name = comparison_df["WAPE"].idxmin()
    best_wape = float(comparison_df.loc[best_model_name, "WAPE"])
    return str(best_model_name), best_wape
