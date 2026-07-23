"""
Probabilistic Forecasting and Interval Estimation for FreshMind.
Generates P10, P50, and P90 prediction intervals using standard deviation
of forecast residuals, and evaluates quality using:
1. CRPS (Continuous Ranked Probability Score)
2. Coverage (percentage of actuals in interval)
3. Average Interval Width
"""

import os
import sys
import numpy as np
import scipy.stats as stats
from typing import Dict, Tuple, Any

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.logging_utils import setup_logger

logger = setup_logger("probabilistic")

def generate_prediction_intervals(
    point_forecasts: np.ndarray,
    historical_residuals: np.ndarray,
    confidence_level: float = 0.8
) -> Dict[str, np.ndarray]:
    """
    Generates P10, P50, and P90 prediction intervals based on historical forecast residuals.
    Assumes normal distribution of residuals.
    
    Args:
        point_forecasts: Array of point predictions (assumed as P50).
        historical_residuals: Historical errors (y_true - y_pred) to calculate variance.
        confidence_level: Confidence level (e.g. 0.8 for an 80% interval between P10 and P90).
        
    Returns:
        Dict[str, np.ndarray]: Quantile forecast arrays mapping keys "P10", "P50", "P90".
    """
    logger.info(f"Generating probabilistic forecast intervals (confidence={confidence_level})...")
    point_forecasts = np.asarray(point_forecasts, dtype=np.float32)
    
    # Calculate residual standard deviation
    if len(historical_residuals) > 1:
        std_error = np.std(historical_residuals)
    else:
        # Fallback if residuals are not available
        std_error = np.clip(np.mean(point_forecasts) * 0.15, 1.0, None)
        
    # Z-score for symmetric interval
    # For confidence_level = 0.8, alpha = 0.1, z = 1.28
    z_score = stats.norm.ppf((1.0 + confidence_level) / 2.0)
    
    p10 = point_forecasts - z_score * std_error
    p90 = point_forecasts + z_score * std_error
    
    # Non-negative constraint for demand
    p10 = np.clip(p10, a_min=0.0, a_max=None)
    p90 = np.clip(p90, a_min=0.0, a_max=None)
    
    return {
        "P10": p10,
        "P50": point_forecasts,
        "P90": p90,
        "std_error": std_error
    }

def calculate_crps_normal(y_true: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """
    Computes exact Continuous Ranked Probability Score (CRPS) under normal distribution:
    CRPS = sigma * {z * [2*Phi(z) - 1] + 2*phi(z) - 1/sqrt(pi)}
    where z = (y - mu)/sigma
    """
    sigma = max(1e-5, sigma)
    z = (y_true - mu) / sigma
    
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return crps

def evaluate_probabilistic_forecast(
    y_true: np.ndarray,
    intervals: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluates probabilistic forecasts based on actual observations.
    
    Args:
        y_true: Array of actual observed values.
        intervals: Dictionary containing "P10", "P50", "P90" and optionally "std_error".
        
    Returns:
        Dict[str, float]: Metrics dictionary containing CRPS, Coverage, and Interval Width.
    """
    y_true = np.asarray(y_true)
    p10 = intervals["P10"]
    p50 = intervals["P50"]
    p90 = intervals["P90"]
    
    # 1. Coverage: fraction of actual values falling inside [P10, P90]
    covered = (y_true >= p10) & (y_true <= p90)
    coverage = np.mean(covered)
    
    # 2. Average Interval Width: mean(P90 - P10)
    width = np.mean(p90 - p10)
    
    # 3. CRPS calculation
    # If std_error is available, use exact normal CRPS, else use quantile approximation
    std_error = intervals.get("std_error", np.std(y_true - p50))
    crps_vals = calculate_crps_normal(y_true, p50, std_error)
    mean_crps = np.mean(crps_vals)
    
    return {
        "CRPS": float(mean_crps),
        "Coverage": float(coverage),
        "IntervalWidth": float(width)
    }
