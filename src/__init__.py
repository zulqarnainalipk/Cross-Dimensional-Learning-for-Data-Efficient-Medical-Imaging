"""
Cross-Dimensional Knowledge Transfer for Data-Efficient Medical Image Analysis

A novel framework for cross-dimensional knowledge transfer from 3D CT volumes
to 2D pathology images, enabling effective few-shot cancer classification.
"""

__version__ = "1.0.0"
__author__ = ""

from .main import run_experiment
from .configs.default import cfg

__all__ = [
    "run_experiment",
    "cfg",
    "__version__",
]
