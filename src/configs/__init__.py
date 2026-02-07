"""
Configuration module for cross-dimensional transfer learning.
"""

from .default import (
    Config,
    DEFAULT_CONFIG,
    MODEL_CONFIGS,
    DATASET_CONFIGS,
    TRAINING_CONFIGS,
    get_config
)

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "MODEL_CONFIGS",
    "DATASET_CONFIGS",
    "TRAINING_CONFIGS",
    "get_config"
]
