"""
Structured logging configuration for the robot application.

Features:
- Consistent log format across all modules
- File and console output
- Log level configuration
- Context tracking (robot state, operation)
"""

import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    app_name: str = "robot",
) -> logging.Logger:
    """
    Configure structured logging for robot application.
    
    Args:
        log_level: LOG level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        app_name: Application name for log file
        
    Returns:
        Configured root logger
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Define format with context
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    if root_logger.handlers:
        return root_logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = os.path.join(log_dir, f"{app_name}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,  # Keep 5 backups
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.
    
    Usage in modules:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
