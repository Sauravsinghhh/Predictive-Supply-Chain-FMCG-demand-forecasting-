"""
Hierarchical Forecasting and Reconciliation for FreshMind.
Provides Bottom-Up, Top-Down, and MinT/OLS (Ordinary Least Squares) linear reconciliation
projection methods to ensure time-series coherency across multiple aggregation levels
(SKU-Store -> Store -> State/Region -> Total Business).
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.logging_utils import setup_logger

logger = setup_logger("hierarchical")

class HierarchicalReconciler:
    """
    Coordinates and reconciles forecasts across a three-level hierarchy:
    Level 0: Total Business (Total sales)
    Level 1: State/Region Level (CA, TX, WI)
    Level 2: Store Level (CA_1, CA_2, TX_1, WI_1)
    """
    def __init__(self, store_to_state_map: Optional[Dict[str, str]] = None):
        self.store_to_state_map = store_to_state_map or {
            "CA_1": "CA",
            "CA_2": "CA",
            "TX_1": "TX",
            "WI_1": "WI"
        }
        self.stores = list(self.store_to_state_map.keys())
        self.states = list(set(self.store_to_state_map.values()))
        
        # Build summing matrix S
        # Total series = 1 (Total) + n_states + n_stores
        # Let's order the series: [Total, State_CA, State_TX, State_WI, CA_1, CA_2, TX_1, WI_1]
        # S maps bottom series (stores) to all series.
        # S size: [n_total_series, n_bottom_series]
        self.num_bottom = len(self.stores)
        self.series_names = ["Total"] + self.states + self.stores
        self.num_series = len(self.series_names)
        
        self.S = np.zeros((self.num_series, self.num_bottom))
        
        # Bottom level (stores) map to themselves (Identity block)
        for i, store in enumerate(self.stores):
            self.S[1 + len(self.states) + i, i] = 1.0
            
            # Map store to state
            state = self.store_to_state_map[store]
            state_idx = self.states.index(state)
            self.S[1 + state_idx, i] = 1.0
            
            # Map store to Total
            self.S[0, i] = 1.0

    def bottom_up_reconciliation(self, base_forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Reconciles forecasts using the Bottom-Up approach:
        sums up store-level forecasts to compute state and total forecasts.
        """
        logger.info("Performing Bottom-Up reconciliation...")
        horizon = len(next(iter(base_forecasts.values())))
        reconciled = {}
        
        # Initialize bottom forecasts (stores)
        bottom_matrix = np.zeros((self.num_bottom, horizon))
        for i, store in enumerate(self.stores):
            if store in base_forecasts:
                bottom_matrix[i] = base_forecasts[store]
            else:
                logger.warning(f"Missing base forecast for bottom series {store}. Filling with zeros.")
                
        # Project using summing matrix S
        # reconciled_matrix = S * bottom_matrix
        reconciled_matrix = np.dot(self.S, bottom_matrix)
        
        # Populate output dictionary
        for idx, name in enumerate(self.series_names):
            reconciled[name] = reconciled_matrix[idx]
            
        return reconciled

    def top_down_reconciliation(self, base_forecasts: Dict[str, np.ndarray], historical_proportions: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Reconciles forecasts using the Top-Down approach:
        takes the Total forecast and distributes it to stores based on historical proportions,
        then sums back up to states.
        """
        logger.info("Performing Top-Down reconciliation...")
        horizon = len(next(iter(base_forecasts.values())))
        reconciled = {}
        
        # Get total forecast
        total_forecast = base_forecasts.get("Total", np.zeros(horizon))
        
        # Distribute to bottom level (stores)
        bottom_matrix = np.zeros((self.num_bottom, horizon))
        for i, store in enumerate(self.stores):
            prop = historical_proportions.get(store, 1.0 / self.num_bottom)
            bottom_matrix[i] = total_forecast * prop
            
        # Re-sum bottom levels up the tree using summing matrix S
        reconciled_matrix = np.dot(self.S, bottom_matrix)
        
        for idx, name in enumerate(self.series_names):
            reconciled[name] = reconciled_matrix[idx]
            
        return reconciled

    def mint_ols_reconciliation(self, base_forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Performs OLS (Ordinary Least Squares) reconciliation (MinT under identity covariance assumptions):
        tilde_y = S * (S^T * S)^-1 * S^T * hat_y
        """
        logger.info("Performing MinT/OLS reconciliation...")
        horizon = len(next(iter(base_forecasts.values())))
        reconciled = {}
        
        # Stack base forecasts into a matrix of shape [num_series, horizon]
        base_matrix = np.zeros((self.num_series, horizon))
        for idx, name in enumerate(self.series_names):
            if name in base_forecasts:
                base_matrix[idx] = base_forecasts[name]
            else:
                logger.warning(f"Base forecast missing for series {name} in OLS reconciliation.")
                
        # Calculate OLS projection matrix P = (S^T * S)^-1 * S^T
        StS_inv = np.linalg.inv(np.dot(self.S.T, self.S))
        P = np.dot(StS_inv, self.S.T)
        
        # Project: tilde_y = S * P * base_matrix
        reconciled_matrix = np.dot(self.S, np.dot(P, base_matrix))
        
        # Floor reconciled forecasts at 0.0 (non-negative constraint)
        reconciled_matrix = np.clip(reconciled_matrix, a_min=0.0, a_max=None)
        
        for idx, name in enumerate(self.series_names):
            reconciled[name] = reconciled_matrix[idx]
            
        return reconciled
