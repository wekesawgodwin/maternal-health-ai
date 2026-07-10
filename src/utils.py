# src/utils.py
import logging
import os
import sys
import json
from pathlib import Path
from typing import Optional, Any
import pandas as pd

# ==========================================
# CANONICAL PROJECT PATHS
# ==========================================
# Every other module imports these instead of re-deriving paths itself.
# Previously each module computed its own PROJECT_ROOT differently
# (train.py used bare relative paths assuming cwd == repo root, while
# evaluate.py/recommend.py derived an absolute path from __file__). That
# meant train.py would silently write to the wrong place if run from a
# different working directory. Fixing that here, once, for everyone.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

DEFAULT_DATA_PATH = DATA_DIR / "KUfacility_register_data_for_uploadWAISWAetalPLOSONE82020.xlsx"


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configures the logging format for the entire application.
    Ensures consistent logs across training, inference, and dashboard.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw maternity register dataset from an Excel file.
    Includes basic validation to ensure the file exists and is not empty.
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


def ensure_directory_exists(dir_path) -> None:
    """Creates a directory if it does not already exist."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.getLogger(__name__).info(f"Created directory: {dir_path}")


def ensure_all_project_dirs() -> None:
    """Creates data/models/figures directories if missing."""
    for d in (DATA_DIR, MODELS_DIR, FIGURES_DIR):
        ensure_directory_exists(d)


def save_json(obj: Any, filepath) -> None:
    """Writes a dict/list to disk as pretty-printed JSON."""
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=4, default=str)
    logging.getLogger(__name__).info(f"Saved JSON to {filepath}")


def load_json(filepath) -> Any:
    with open(filepath, "r") as f:
        return json.load(f)


def load_model_metadata(models_dir: Path = MODELS_DIR) -> dict:
    """
    Loads {model_name, threshold, min_recall_target} saved by train.py.
    Every module that scores patients (evaluate.py, predict.py) MUST use
    this threshold instead of the sklearn default of 0.5 - the whole point
    of the GridSearch + threshold-tuning step in train.py is that 0.5
    under-catches adverse outcomes on this imbalanced dataset.
    """
    import joblib
    meta_path = Path(models_dir) / "model_metadata.pkl"
    if not meta_path.exists():
        logging.getLogger(__name__).warning(
            f"No model_metadata.pkl found at {meta_path} - falling back to threshold=0.5. "
            "Re-run train.py to generate a tuned threshold."
        )
        return {"model_name": "unknown", "threshold": 0.5, "min_recall_target": None}
    return joblib.load(meta_path)
