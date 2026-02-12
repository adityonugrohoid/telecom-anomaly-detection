"""
Configuration management for Telecom Anomaly Detection.

This module centralizes all configuration parameters, making it easy to
adjust settings without modifying core logic.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# ============================================================================
# DATA GENERATION CONFIG
# ============================================================================

DATA_GEN_CONFIG = {
    "random_seed": 42,
    "n_samples": 36_000,  # 50 cells x 30 days x 24 hours
    "test_size": 0.2,
    "validation_size": 0.1,
    "use_case_params": {
        "n_cells": 50,
        "n_days": 30,
        "hours_per_day": 24,
        "anomaly_rate": 0.05,
        "anomaly_types": ["traffic_spike", "sinr_drop", "latency_surge", "throughput_collapse"],
    },
}


# ============================================================================
# FEATURE ENGINEERING CONFIG
# ============================================================================

FEATURE_CONFIG = {
    "categorical_features": [
        "cell_type",
        "area_type",
    ],
    "numerical_features": [
        "traffic_load_gb",
        "avg_sinr_db",
        "avg_throughput_mbps",
        "avg_latency_ms",
        "packet_loss_pct",
        "connected_users",
        "prb_utilization",
    ],
    "datetime_features": ["timestamp"],
    "rolling_windows": [3, 6, 24],  # hours
    "create_features": True,
}


# ============================================================================
# MODEL TRAINING CONFIG
# ============================================================================

MODEL_CONFIG = {
    "algorithm": "isolation_forest",
    "cv_folds": 5,
    "cv_strategy": "kfold",
    "hyperparameters": {
        "n_estimators": 200,
        "max_samples": "auto",
        "contamination": 0.05,
        "max_features": 1.0,
        "random_state": 42,
    },
    "early_stopping_rounds": None,
    "verbose": True,
}


# ============================================================================
# EVALUATION CONFIG
# ============================================================================

EVAL_CONFIG = {
    "primary_metric": "f1",
    "threshold": 0.5,
    "compute_metrics": [
        "precision",
        "recall",
        "f1",
        "accuracy",
    ],
}


# ============================================================================
# VISUALIZATION CONFIG
# ============================================================================

VIZ_CONFIG = {
    "style": "whitegrid",
    "palette": "husl",
    "context": "notebook",
    "figure_size": (12, 6),
    "dpi": 100,
}


# ============================================================================
# UTILITIES
# ============================================================================


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_custom_config(config_path: Path) -> Dict[str, Any]:
    """Load custom configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "data_gen": DATA_GEN_CONFIG,
        "features": FEATURE_CONFIG,
        "model": MODEL_CONFIG,
        "eval": EVAL_CONFIG,
        "viz": VIZ_CONFIG,
        "paths": {
            "root": PROJECT_ROOT,
            "data": DATA_DIR,
            "raw": RAW_DATA_DIR,
            "processed": PROCESSED_DATA_DIR,
            "notebooks": NOTEBOOKS_DIR,
        },
    }


if __name__ == "__main__":
    ensure_directories()
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Random seed: {DATA_GEN_CONFIG['random_seed']}")
    print(f"Algorithm: {MODEL_CONFIG['algorithm']}")
