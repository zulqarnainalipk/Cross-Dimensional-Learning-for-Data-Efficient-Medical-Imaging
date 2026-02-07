"""
Training components for cross-dimensional knowledge transfer.
"""

from .trainer import Trainer, create_trainer
from .losses import (
    ContrastiveLoss,
    PrototypicalLoss,
    DomainAlignmentLoss,
    SemanticAlignmentLoss,
    TotalLoss
)

__all__ = [
    "Trainer",
    "create_trainer",
    "ContrastiveLoss",
    "PrototypicalLoss",
    "DomainAlignmentLoss",
    "SemanticAlignmentLoss",
    "TotalLoss",
]
