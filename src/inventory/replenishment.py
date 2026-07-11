"""
Inventory Replenishment and Decision Logic for FreshMind.
Implements simple replenishment rules using safety buffers and demand forecasts.
"""

from typing import Dict, Any

def calculate_replenishment_order(
    forecast_demand_sum: float,
    current_inventory: float,
    safety_buffer: float
) -> Dict[str, Any]:
    """
    Calculates the replenishment order quantity using a simple inventory policy:
    Recommended Order = max(0, Forecast Demand - Current Inventory + Safety Buffer)
    
    Args:
        forecast_demand_sum: Total forecasted demand over the replenishment lead time/horizon.
        current_inventory: Current on-hand stock quantity.
        safety_buffer: Configurable safety buffer/stock level.
        
        Recommended Order = max(0.0, Forecast Demand - Current Inventory + Safety Buffer)
        
    Returns:
        Dict[str, Any]: Dictionary containing calculation inputs and recommended order quantity.
    """
    recommended = forecast_demand_sum - current_inventory + safety_buffer
    recommended_order = max(0.0, float(recommended))
    
    # Simple alert if current inventory is below safety buffer
    under_stock_alert = current_inventory < safety_buffer
    
    return {
        "forecast_demand_sum": float(forecast_demand_sum),
        "current_inventory": float(current_inventory),
        "safety_buffer": float(safety_buffer),
        "recommended_order": recommended_order,
        "under_stock_alert": under_stock_alert
    }
