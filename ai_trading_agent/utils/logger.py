"""
Logging utility for the AI Trading Agent
"""
import logging
import os
from datetime import datetime
from config.config import Config

def setup_logger(name: str, log_file: str = None, level: str = None):
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (optional)
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    log_level = getattr(logging, (level or Config.LOG_LEVEL).upper())
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file or Config.LOG_FILE:
        log_path = log_file or Config.LOG_FILE
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str):
    """Get or create a logger"""
    return setup_logger(name)

