# src/utils.py
import logging
import os
import sys
import pandas as pd
from typing import Optional

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configures the logging format for the entire application.
    Ensures consistent logs across training, inference, and dashboard.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw maternity register dataset from an Excel file.
    Includes basic validation to ensure the file exists and is not empty.
    
    Args:
        filepath (str): Path to the .xlsx file.
        
    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(filepath):
        logger.error(f"Data file not found at {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
        
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath)
    
    if df.empty:
        logger.error("The loaded dataframe is empty.")
        raise ValueError("Data file is empty.")
        
    logger.info(f"Successfully loaded data with shape: {df.shape}")
    return df

def ensure_directory_exists(dir_path: str) -> None:
    """
    Creates a directory if it does not already exist.
    Useful for saving models, figures, and reports.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.getLogger(__name__).info(f"Created directory: {dir_path}")