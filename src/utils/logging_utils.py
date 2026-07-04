"""
Logging Utilities for FreshMind Project.
Provides a standardized logging interface that writes to both stdout and logs/app.log.
"""

import logging
import os
from typing import Optional

def setup_logger(name: str, log_dir: str = "logs", log_file: str = "app.log", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that outputs to both standard output and a file.
    
    Args:
        name (str): Name of the logger (typically __name__).
        log_dir (str): Directory where the log file will be stored.
        log_file (str): Filename of the log.
        level (int): Logging level (e.g. logging.INFO).
        
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if the logger is already initialized
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_path = os.path.join(log_dir, log_file)
    
    # Define log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
