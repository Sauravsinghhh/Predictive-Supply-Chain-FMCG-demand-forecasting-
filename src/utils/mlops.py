"""
MLOps utilities for FreshMind.
Provides:
1. MLflow experiment tracking, parameter/metric logging, and model registry integrations.
2. Data drift detection (Kolmogorov-Smirnov test and Population Stability Index).
3. Automated model retraining helpers.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, Any, Tuple, Optional

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.logging_utils import setup_logger

logger = setup_logger("mlops")

try:
    import mlflow
except ImportError:
    mlflow = None
    logger.warning("mlflow package is not available. MLOps runs will write locally only.")

class MLflowTracker:
    """
    Coordinates mlflow experiment tracking, metric logs, and registration.
    """
    def __init__(self, experiment_name: str = "FreshMind_Demand_Forecasting"):
        self.experiment_name = experiment_name
        self.active_run_ = None
        
        if mlflow is not None:
            try:
                mlflow.set_experiment(experiment_name)
            except Exception as e:
                logger.warning(f"Failed to set MLflow experiment: {e}. Runs will log locally.")

    def start_run(self, run_name: str) -> Optional[Any]:
        """Starts an MLflow run."""
        if mlflow is not None:
            try:
                self.active_run_ = mlflow.start_run(run_name=run_name)
                logger.info(f"Started MLflow tracking run: {run_name}")
                return self.active_run_
            except Exception as e:
                logger.error(f"Failed to start MLflow run: {e}")
        return None

    def log_params(self, params: Dict[str, Any]):
        """Logs hyperparameters or configuration mappings."""
        if mlflow is not None and self.active_run_ is not None:
            try:
                mlflow.log_params(params)
                logger.info(f"Logged params to MLflow: {list(params.keys())}")
            except Exception as e:
                logger.error(f"Failed to log params to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Logs evaluation metrics."""
        if mlflow is not None and self.active_run_ is not None:
            try:
                mlflow.log_metrics(metrics, step=step)
                logger.info(f"Logged metrics to MLflow: {list(metrics.keys())}")
            except Exception as e:
                logger.error(f"Failed to log metrics to MLflow: {e}")

    def log_model(self, model: Any, artifact_path: str, model_name: Optional[str] = None):
        """Logs and registers a model artifact."""
        if mlflow is not None and self.active_run_ is not None:
            try:
                # Determine model flavor based on type
                if "lightgbm" in str(type(model)).lower():
                    import mlflow.lightgbm as mlflow_model
                    mlflow_model.log_model(model, artifact_path)
                elif "pytorch" in str(type(model)).lower() or "net" in str(type(model)).lower():
                    import mlflow.pytorch as mlflow_model
                    mlflow_model.log_model(model, artifact_path)
                else:
                    # Generic sklearn or python model
                    import mlflow.sklearn as mlflow_model
                    mlflow_model.log_model(model, artifact_path)
                    
                logger.info(f"Logged model artifact at '{artifact_path}'")
                
                # Register model if name is provided
                if model_name:
                    run_id = self.active_run_.info.run_id
                    model_uri = f"runs:/{run_id}/{artifact_path}"
                    mlflow.register_model(model_uri, model_name)
                    logger.info(f"Registered model as '{model_name}' in MLflow Registry")
            except Exception as e:
                logger.error(f"Failed to log/register model in MLflow: {e}")

    def end_run(self):
        """Ends the active MLflow run."""
        if mlflow is not None and self.active_run_ is not None:
            try:
                mlflow.end_run()
                logger.info("Ended active MLflow tracking run.")
                self.active_run_ = None
            except Exception as e:
                logger.error(f"Failed to close MLflow run: {e}")


# ------------------ Drift Detection ------------------
def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) between a baseline distribution (train)
    and a current distribution (incoming inference data).
    
    PSI Rule of Thumb:
    - PSI < 0.1: No significant distribution change.
    - 0.1 <= PSI < 0.25: Moderate change.
    - PSI >= 0.25: Significant change (drift detected).
    """
    # Clean zeros and prepare arrays
    baseline = np.asarray(baseline, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    
    # Calculate quantiles based on baseline distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(baseline, percentiles)
    
    # Avoid duplicate bucket boundaries (adds tiny random noise if identical boundaries exist)
    if len(np.unique(buckets)) < len(buckets):
        buckets = np.unique(buckets)
        num_buckets = len(buckets) - 1
        
    if num_buckets <= 0:
        return 0.0
        
    # Count frequencies
    base_counts, _ = np.histogram(baseline, bins=buckets)
    curr_counts, _ = np.histogram(current, bins=buckets)
    
    # Convert to proportions with small epsilon to avoid zero divisions
    eps = 1e-4
    base_props = (base_counts + eps) / (len(baseline) + eps * num_buckets)
    curr_props = (curr_counts + eps) / (len(current) + eps * num_buckets)
    
    # Compute PSI
    psi = np.sum((curr_props - base_props) * np.log(curr_props / base_props))
    return float(psi)

def detect_drift_ks(baseline: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> Tuple[float, bool]:
    """
    Executes a Kolmogorov-Smirnov 2-sample test to detect statistical drift.
    Null Hypothesis: Both series are drawn from the same underlying distribution.
    If p-value < alpha, we reject the null hypothesis (drift detected).
    
    Returns:
        Tuple[float, bool]: (p_value, drift_detected flag)
    """
    # Run KS 2-sample test
    ks_stat, p_val = stats.ks_2samp(baseline, current)
    drift_detected = p_val < alpha
    return float(p_val), bool(drift_detected)

def check_data_drift(
    training_target: np.ndarray, 
    incoming_target: np.ndarray
) -> Dict[str, Any]:
    """
    Combines PSI and KS Test to determine if incoming time series data has drifted.
    """
    logger.info("Executing MLOps data drift diagnostics...")
    
    p_val, ks_drift = detect_drift_ks(training_target, incoming_target)
    psi_score = calculate_psi(training_target, incoming_target)
    
    psi_drift = psi_score >= 0.25
    overall_drift = ks_drift or psi_drift
    
    logger.info(f"Drift Diagnostics: KS p-value={p_val:.4f} (drift={ks_drift}), PSI={psi_score:.4f} (drift={psi_drift})")
    
    return {
        "ks_p_value": p_val,
        "ks_drift_detected": ks_drift,
        "psi_score": psi_score,
        "psi_drift_detected": psi_drift,
        "drift_detected": overall_drift,
        "status_message": "Action Required: Retrain model" if overall_drift else "Data stable"
    }
