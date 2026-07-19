"""
Logging Utilities for FreshMind Project.
Provides a standardized logging interface that writes to both stdout and logs/app.log
using settings loaded from the centralized configuration, with fallback / override parameters.
"""

import logging
import os
import sys
from typing import Optional

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from configs.logging_config import get_logging_settings

def setup_logger(
    name: str, 
    log_dir: Optional[str] = None, 
    log_file: Optional[str] = None, 
    level: Optional[int] = None
) -> logging.Logger:
    """
    Sets up a logger that outputs to both standard output and a file.
    Uses centralized configurations from configs/config.yaml, with optional parameter overrides
    to maintain backward compatibility with previous calls.
    
    Args:
        name (str): Name of the logger (typically __name__).
        log_dir (str, optional): Directory for the log file. Overrides config if specified.
        log_file (str, optional): Filename of the log. Overrides config if specified.
        level (int, optional): Numeric logging level. Overrides config if specified.
        
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if the logger is already initialized
    if logger.handlers:
        return logger
        
    # Get centralized settings
    settings = get_logging_settings()
    
    # Resolve logging level
    if level is not None:
        resolved_level = level
    else:
        level_name = settings["level"].upper()
        resolved_level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(resolved_level)
    
    # Resolve directory and file
    final_log_dir = log_dir if log_dir is not None else settings["log_dir"]
    final_log_file = log_file if log_file is not None else settings["log_file"]
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(final_log_dir):
        os.makedirs(final_log_dir)
        
    log_path = os.path.join(final_log_dir, final_log_file)
    
    # Define log format
    formatter = logging.Formatter(settings["format"])
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
