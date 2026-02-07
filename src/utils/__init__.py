"""
Utility modules for cross-dimensional knowledge transfer.
"""

from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_feature_space,
    plot_attention_maps,
    plot_domain_alignment,
    plot_proto_visualization,
    save_figure
)

from .metrics import (
    calculate_accuracy,
    calculate_auc_roc,
    calculate_f1_score,
    calculate_sensitivity_specificity,
    calculate_mean_iou,
    evaluate_episode,
    compute_embedding_similarity,
    compute_domain_alignment_metrics
)

__all__ = [
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_feature_space",
    "plot_attention_maps",
    "plot_domain_alignment",
    "plot_proto_visualization",
    "save_figure",
    "calculate_accuracy",
    "calculate_auc_roc",
    "calculate_f1_score",
    "calculate_sensitivity_specificity",
    "calculate_mean_iou",
    "evaluate_episode",
    "compute_embedding_similarity",
    "compute_domain_alignment_metrics"
]
