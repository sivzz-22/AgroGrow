"""
AgroGrow Logging Utility Module.
Provides centralized logging configuration for all system modules.
"""

import logging
import sys
from pathlib import Path
from AgroGrow.config import global_config

def setup_logger(name: str = "AgroGrow") -> logging.Logger:
    """
    Sets up and configures the logger.
    Logs are printed to stdout and saved to results/agrogrow.log.
    
    Args:
        name (str): The name of the logger.
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Formatter for log messages
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = global_config.results_dir / "agrogrow.log"
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to create file handler for logging: {e}")
        
    return logger

# Create global logger instance
logger = setup_logger()
