"""
Inventory Optimization and Decision Science Engine for FreshMind.
Implements:
1. Safety stock and reorder point calculations under demand uncertainty.
2. Continuous review (s,S) and (R,Q) replenishment policies.
3. Periodic review Order-Up-To (OUT) policy.
4. Newsvendor optimization balancing underage (stockout) and overage (holding/waste) costs.
5. Operational metrics (expected shortage, holding costs, expected stockout costs, inventory health status).
"""

import os
import sys
import numpy as np
import scipy.stats as stats
from typing import Dict, Any, Tuple

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.logging_utils import setup_logger

logger = setup_logger("inventory_optimization")

def normal_loss_function(z: float) -> float:
    """
    Computes the standard normal loss function L(z) = phi(z) - z * (1 - Phi(z))
    used to calculate expected unit shortages.
    """
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    return float(phi - z * (1.0 - Phi))

def calculate_inventory_policies(
    forecast_demand_mean: float,
    forecast_demand_std: float,
    current_on_hand: float,
    current_on_order: float = 0.0,
    lead_time_days: int = 3,
    review_period_days: int = 7,
    service_level: float = 0.95,
    unit_holding_cost: float = 0.1,    # h (holding cost per unit per day)
    unit_stockout_cost: float = 1.5,   # p (lost profit margin/penalty per unit)
) -> Dict[str, Any]:
    """
    Performs comprehensive inventory optimization calculations.
    
    Returns:
        Dict[str, Any]: Optimization metrics and policy recommendations.
    """
    logger.info("Executing inventory policy calculations...")
    
    # 1. Lead Time Demand Statistics
    # For daily demand, lead time demand mean is daily_mean * L
    # Standard deviation scales as daily_std * sqrt(L)
    ltd_mean = forecast_demand_mean * lead_time_days
    ltd_std = forecast_demand_std * np.sqrt(lead_time_days)
    ltd_std = max(0.5, ltd_std) # Floor to avoid zero division
    
    # 2. Safety Stock and Reorder Point (ROP)
    z_service = stats.norm.ppf(service_level)
    safety_stock = z_service * ltd_std
    reorder_point = ltd_mean + safety_stock
    
    # 3. Inventory Position
    inventory_position = current_on_hand + current_on_order
    
    # 4. Periodic Review Order-Up-To (s, S) Policy
    # Review period + Lead time demand statistics
    protection_days = review_period_days + lead_time_days
    prot_mean = forecast_demand_mean * protection_days
    prot_std = forecast_demand_std * np.sqrt(protection_days)
    prot_std = max(0.5, prot_std)
    
    order_up_to_level = prot_mean + z_service * prot_std
    
    # Recommended Order Qty under Periodic OUT
    recommended_out_order = max(0.0, order_up_to_level - inventory_position)
    
    # 5. Continuous Review (s, S) Policy
    # Reorders up to S when position drops below s
    s = reorder_point
    S = order_up_to_level
    if inventory_position <= s:
        recommended_sS_order = S - inventory_position
    else:
        recommended_sS_order = 0.0
        
    # 6. Newsvendor Optimal Quantities
    # Critical Ratio = Underage / (Underage + Overage)
    # Underage cost (stockout penalty) = profit margin
    # Overage cost (holding cost over lead time)
    c_u = unit_stockout_cost
    c_o = unit_holding_cost * lead_time_days
    critical_ratio = c_u / (c_u + c_o)
    
    z_newsvendor = stats.norm.ppf(critical_ratio)
    newsvendor_order_up_to = ltd_mean + z_newsvendor * ltd_std
    recommended_newsvendor_order = max(0.0, newsvendor_order_up_to - inventory_position)
    
    # 7. Financial & Risk Metrics (Expected Shortage & Costs)
    # Z-safety factor of current stock
    current_z = (current_on_hand - ltd_mean) / ltd_std
    expected_shortage = ltd_std * normal_loss_function(current_z)
    
    expected_holding_cost = current_on_hand * unit_holding_cost * lead_time_days
    expected_stockout_cost = expected_shortage * unit_stockout_cost
    
    # 8. Inventory Health Classification
    if current_on_hand < safety_stock:
        health_status = "Understock Alert"
    elif current_on_hand > order_up_to_level:
        health_status = "Overstock Alert"
    else:
        health_status = "Healthy"
        
    return {
        "ltd_mean": float(ltd_mean),
        "ltd_std": float(ltd_std),
        "safety_stock": float(safety_stock),
        "reorder_point": float(reorder_point),
        "inventory_position": float(inventory_position),
        "order_up_to_level": float(order_up_to_level),
        "recommended_order_out": float(recommended_out_order),
        "recommended_order_sS": float(recommended_sS_order),
        "newsvendor_order_up_to": float(newsvendor_order_up_to),
        "recommended_order_newsvendor": float(recommended_newsvendor_order),
        "expected_shortage_units": float(expected_shortage),
        "expected_holding_cost": float(expected_holding_cost),
        "expected_stockout_cost": float(expected_stockout_cost),
        "inventory_health": health_status,
        "under_stock_alert": bool(current_on_hand < safety_stock)
    }
