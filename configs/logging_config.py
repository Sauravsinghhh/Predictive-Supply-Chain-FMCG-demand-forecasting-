"""
Centralized logging configuration for FreshMind.
Loads logging settings from configs/config.yaml and provides standard configuration helpers.
"""

import os
import logging
import yaml
from typing import Dict, Any

def get_logging_settings(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads logging settings from the YAML configuration file.
    Falls back to sensible defaults if the file is missing or invalid.
    
    Args:
        config_path (str): Path to the config file.
        
    Returns:
        Dict[str, Any]: Logging configuration dictionary.
    """
    defaults = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_dir": "logs",
        "log_file": "app.log"
    }
    
    if not os.path.exists(config_path):
        return defaults
        
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        settings = {}
        # Parse paths
        paths_cfg = config.get("paths", {})
        settings["log_dir"] = paths_cfg.get("logs_dir", defaults["log_dir"])
        settings["log_file"] = os.path.basename(paths_cfg.get("log_file", defaults["log_file"]))
        
        # Parse logging level and format
        log_cfg = config.get("logging", {})
        settings["level"] = log_cfg.get("level", defaults["level"])
        settings["format"] = log_cfg.get("format", defaults["format"])
        
        return settings
    except Exception:
        # Silently fall back to defaults if parsing fails during early initialization
        return defaults

def configure_logging(config_path: str = "configs/config.yaml") -> None:
    """
    Performs root-level logging configuration based on external settings.
    
    Args:
        config_path (str): Path to the config file.
    """
    settings = get_logging_settings(config_path)
    
    # Resolve numeric level
    level_name = settings["level"].upper()
    numeric_level = getattr(logging, level_name, logging.INFO)
    
    log_dir = settings["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_path = os.path.join(log_dir, settings["log_file"])
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=settings["format"],
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8")
        ],
        force=True  # Overwrites any prior basicConfig setup (Python 3.8+)
    )
