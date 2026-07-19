"""
Integration Tests for FreshMind Forecasting & Replenishment Pipeline.
Runs the end-to-end flow from synthetic data generation, ingestion, schema validation,
feature engineering, baseline forecasting, to replenishment recommendations.
"""

import os
import sys
import pytest
import shutil
import yaml
import pandas as pd
import numpy as np

# Adjust path to import src modules and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.download_data import generate_synthetic_data
from src.data.data_loader import DataLoader
from src.features.build_features import prepare_single_series, add_rolling_features
from src.models.baseline_forecast import SeasonalNaiveForecaster
from src.inventory.replenishment import calculate_replenishment_order

@pytest.fixture(scope="module")
def integration_environment(tmp_path_factory):
    """
    Sets up a temporary sandbox environment with synthetic data files
    and a custom configuration file for integration testing.
    """
    # Create temp directory
    temp_dir = tmp_path_factory.mktemp("freshmind_integration")
    
    raw_dir = temp_dir / "data" / "raw"
    processed_dir = temp_dir / "data" / "processed"
    reports_dir = temp_dir / "reports"
    models_dir = temp_dir / "models"
    logs_dir = temp_dir / "logs"
    
    os.makedirs(raw_dir)
    os.makedirs(processed_dir)
    os.makedirs(reports_dir)
    os.makedirs(models_dir)
    os.makedirs(logs_dir)
    
    # Generate synthetic M5 data (5 items, 30 days history)
    generate_synthetic_data(str(raw_dir), num_items=5, num_days=30)
    
    # Create custom config.yaml pointing to temp directories
    test_config = {
        "paths": {
            "data_raw_dir": str(raw_dir),
            "data_processed_dir": str(processed_dir),
            "reports_dir": str(reports_dir),
            "models_dir": str(models_dir),
            "logs_dir": str(logs_dir),
            "log_file": str(logs_dir / "integration_test.log")
        },
        "ingestion": {
            "download_url": "mock_url",
            "zip_name": "mock.zip"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "schemas": {
            "calendar": {
                "columns": [
                    "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                    "snap_CA", "snap_TX", "snap_WI"
                ]
            },
            "sell_prices": {
                "columns": ["store_id", "item_id", "wm_yr_wk", "sell_price"]
            },
            "sales_train_validation": {
                "required_columns": ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
            }
        }
    }
    
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
        
    yield {
        "config_path": str(config_path),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "reports_dir": str(reports_dir)
    }
    
    # Teardown: Clean up temp directory
    shutil.rmtree(temp_dir)

def test_end_to_end_pipeline(integration_environment):
    """
    Executes and validates the end-to-end MVP pipeline:
    1. Ingestion & Validation
    2. Feature Preparation
    3. Baseline Forecasting
    4. Replenishment Calculation
    """
    config_path = integration_environment["config_path"]
    
    # Step 1: Ingestion & Validation
    loader = DataLoader(config_path=config_path)
    
    # Assert validation check succeeds
    validation_results = loader.run_validation_checks()
    assert validation_results["calendar"]["schema_check_pass"] is True
    assert validation_results["sell_prices"]["schema_check_pass"] is True
    assert validation_results["sales"]["schema_check_pass"] is True
    
    # Load dataframes
    calendar_df, prices_df, sales_df = loader.load_raw_dataframes()
    
    # Step 2: Feature Engineering & Preparation
    # Select store and item from sales dataset
    store_id = sales_df["store_id"].iloc[0]
    item_id = sales_df["item_id"].iloc[0]
    
    single_series_df = prepare_single_series(
        sales_df=sales_df, 
        calendar_df=calendar_df, 
        store_id=store_id, 
        item_id=item_id
    )
    
    assert isinstance(single_series_df, pd.DataFrame)
    assert not single_series_df.empty
    
    # Add rolling features
    featured_df = add_rolling_features(single_series_df, window_sizes=(7,))
    assert "sales_roll_mean_7" in featured_df.columns
    
    # Step 3: Baseline Forecasting
    historical_sales = featured_df["sales"].values
    
    # Fit seasonal naive forecaster (weekly period = 7)
    forecaster = SeasonalNaiveForecaster(period=7)
    forecaster.fit(historical_sales)
    
    horizon = 14
    forecast_predictions = forecaster.predict(horizon=horizon)
    
    assert len(forecast_predictions) == horizon
    assert not np.isnan(forecast_predictions).any()
    
    # Step 4: Replenishment Calculation
    forecast_sum = float(np.sum(forecast_predictions))
    current_inventory = 10.0
    safety_buffer = 15.0
    
    replenishment = calculate_replenishment_order(
        forecast_demand_sum=forecast_sum,
        current_inventory=current_inventory,
        safety_buffer=safety_buffer
    )
    
    # Verify replenishment recommendation outputs
    assert "recommended_order" in replenishment
    assert "under_stock_alert" in replenishment
    assert isinstance(replenishment["recommended_order"], float)
    assert replenishment["recommended_order"] >= 0.0
    
    # Validate expected replenishment formula:
    # order = max(0, forecast_sum - current_inventory + safety_buffer)
    expected_order = max(0.0, forecast_sum - current_inventory + safety_buffer)
    assert replenishment["recommended_order"] == expected_order
    
    # Verify generated report file exists
    report_file = os.path.join(integration_environment["reports_dir"], "data_status_report.md")
    assert os.path.exists(report_file)
