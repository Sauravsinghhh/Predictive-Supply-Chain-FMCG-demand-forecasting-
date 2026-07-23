"""
Explainable AI (XAI) Utilities for FreshMind.
Provides integrations with SHAP (Shapley Additive exPlanations) and feature importance
computations to interpret machine learning forecasts.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.logging_utils import setup_logger

logger = setup_logger("explainability")

try:
    import shap
except ImportError:
    shap = None
    logger.warning("shap package is not available. Explainable AI features will use baseline importances.")

def get_lgb_feature_importance(model_forecaster: Any) -> pd.DataFrame:
    """
    Extracts feature importances from a trained LightGBMForecaster.
    """
    logger.info("Extracting feature importances...")
    importances = model_forecaster.get_feature_importances()
    if not importances:
        return pd.DataFrame(columns=["Feature", "Importance"])
        
    df = pd.DataFrame([
        {"Feature": k, "Importance": float(v)} 
        for k, v in importances.items()
    ])
    df = df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df

def generate_shap_explanations(
    model: Any, 
    X_train: pd.DataFrame
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """
    Initializes a SHAP TreeExplainer and computes SHAP values for the training features.
    
    Returns:
        Tuple[Optional[np.ndarray], Optional[Any]]: (shap_values array, explainer object)
    """
    if shap is None:
        logger.warning("SHAP package not installed. Skipping SHAP generation.")
        return None, None
        
    logger.info("Generating SHAP explanations using TreeExplainer...")
    try:
        explainer = shap.TreeExplainer(model)
        # Compute shap values
        shap_values = explainer.shap_values(X_train)
        return shap_values, explainer
    except Exception as e:
        logger.error(f"Failed to generate SHAP values: {e}")
        return None, None

def save_shap_summary_plot(
    shap_values: np.ndarray, 
    X_train: pd.DataFrame, 
    output_path: str
) -> bool:
    """
    Generates a SHAP summary plot and saves it as a PNG image for rendering in Streamlit.
    """
    if shap is None or shap_values is None:
        return False
        
    logger.info(f"Saving SHAP summary plot to {output_path}...")
    try:
        plt.figure(figsize=(10, 6))
        # Plot SHAP summary
        shap.summary_plot(shap_values, X_train, show=False)
        plt.tight_layout()
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save SHAP summary plot: {e}")
        return False

def get_shap_feature_importance(
    shap_values: np.ndarray, 
    X_train: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates mean absolute SHAP values per feature to rank overall impact.
    """
    if shap_values is None:
        return pd.DataFrame(columns=["Feature", "Mean_SHAP"])
        
    # Mean absolute SHAP value for each feature
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame({
        "Feature": X_train.columns,
        "Mean_SHAP": mean_shap
    })
    df = df.sort_values(by="Mean_SHAP", ascending=False).reset_index(drop=True)
    return df
