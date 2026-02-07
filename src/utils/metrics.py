"""
Metrics and evaluation utilities for cross-dimensional transfer learning.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)

    Returns:
        Accuracy score (0-1)
    """
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total


def calculate_auc_roc(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2
) -> float:
    """
    Calculate Area Under ROC Curve.

    Args:
        probabilities: Predicted probabilities (N, C) or (N,) for binary
        targets: Ground truth labels (N,)
        num_classes: Number of classes

    Returns:
        AUC-ROC score (0-1)
    """
    probs = probabilities.detach().cpu().numpy()
    tgt = targets.detach().cpu().numpy()

    if num_classes == 2:
        # Binary classification
        if probs.ndim == 1:
            auc = roc_auc_score(tgt, probs)
        else:
            auc = roc_auc_score(tgt, probs[:, 1])
    else:
        # Multi-class (macro-averaged)
        try:
            auc = roc_auc_score(tgt, probs, multi_class='ovr', average='macro')
        except ValueError:
            # Fallback if classes are not balanced
            auc = 0.0

    return auc


def calculate_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'binary'
) -> float:
    """
    Calculate F1 score.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)
        average: Averaging strategy ('binary', 'macro', 'micro')

    Returns:
        F1 score (0-1)
    """
    preds = predictions.detach().cpu().numpy()
    tgt = targets.detach().cpu().numpy()

    return f1_score(tgt, preds, average=average)


def calculate_sensitivity_specificity(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Tuple[float, float]:
    """
    Calculate sensitivity (recall) and specificity.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)

    Returns:
        Tuple of (sensitivity, specificity)
    """
    preds = predictions.detach().cpu().numpy()
    tgt = targets.detach().cpu().numpy()

    cm = confusion_matrix(tgt, preds)

    if cm.shape[0] < 2 or cm.shape[1] < 2:
        return 0.0, 0.0

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return sensitivity, specificity


def calculate_mean_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2
) -> float:
    """
    Calculate mean Intersection over Union.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)
        num_classes: Number of classes

    Returns:
        Mean IoU score (0-1)
    """
    preds = predictions.detach().cpu().numpy()
    tgt = targets.detach().cpu().numpy()

    cm = confusion_matrix(tgt, preds, labels=range(num_classes))

    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection

    # Avoid division by zero
    union = np.maximum(union, 1e-10)

    iou = intersection / union
    mean_iou = np.mean(iou)

    return mean_iou


def evaluate_episode(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes: int
) -> Dict[str, float]:
    """
    Evaluate a few-shot learning episode.

    Args:
        query_embeddings: Query set embeddings (Q, D)
        query_labels: Query set labels (Q,)
        support_embeddings: Support set embeddings (S, D)
        support_labels: Support set labels (S,)
        num_classes: Number of classes in episode

    Returns:
        Dictionary of episode metrics
    """
    # Compute prototypes
    prototypes = {}
    for c in range(num_classes):
        mask = support_labels == c
        if mask.sum() > 0:
            prototypes[c] = support_embeddings[mask].mean(dim=0)
        else:
            # Handle missing class in support set
            prototypes[c] = torch.zeros_like(support_embeddings[0])

    prototypes = torch.stack([prototypes[c] for c in range(num_classes)])

    # Compute distances and predictions
    distances = torch.cdist(query_embeddings.unsqueeze(0),
                           prototypes.unsqueeze(0)).squeeze(0)
    predictions = distances.argmin(dim=1)

    # Calculate metrics
    accuracy = calculate_accuracy(predictions, query_labels)

    # Compute embedding similarity
    query_norm = query_embeddings / (query_embeddings.norm(dim=1, keepdim=True) + 1e-8)
    proto_norm = prototypes / (prototypes.norm(dim=1, keepdim=True) + 1e-8)
    intra_class_sim = []
    for c in range(num_classes):
        mask = query_labels == c
        if mask.sum() > 0:
            sim = (query_norm[mask] @ proto_norm[c:c+1].t()).mean()
            intra_class_sim.append(sim.item())

    avg_intra_class_sim = np.mean(intra_class_sim) if intra_class_sim else 0.0

    return {
        'accuracy': accuracy,
        'num_queries': len(query_labels),
        'num_support': len(support_labels),
        'avg_intra_class_similarity': avg_intra_class_sim
    }


def compute_embedding_similarity(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    method: str = 'cosine'
) -> float:
    """
    Compute similarity between two embedding sets.

    Args:
        embeddings_a: First embedding set (N, D)
        embeddings_b: Second embedding set (M, D)
        method: Similarity method ('cosine', 'euclidean', ' Manhattan')

    Returns:
        Similarity score
    """
    if method == 'cosine':
        # Cosine similarity between means
        mean_a = embeddings_a.mean(dim=0)
        mean_b = embeddings_b.mean(dim=0)
        cos_sim = torch.nn.functional.cosine_similarity(
            mean_a.unsqueeze(0), mean_b.unsqueeze(0)
        )
        return cos_sim.item()
    elif method == 'euclidean':
        # Negative mean Euclidean distance
        mean_a = embeddings_a.mean(dim=0)
        mean_b = embeddings_b.mean(dim=0)
        return -torch.nn.functional.pairwise_distance(
            mean_a.unsqueeze(0), mean_b.unsqueeze(0)
        ).item()
    elif method == ' Manhattan':
        # Negative mean Manhattan distance
        mean_a = embeddings_a.mean(dim=0)
        mean_b = embeddings_b.mean(dim=0)
        return -torch.abs(mean_a - mean_b).sum().item()
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def compute_domain_alignment_metrics(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    source_labels: Optional[torch.Tensor] = None,
    target_labels: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute domain alignment quality metrics.

    Args:
        source_embeddings: Source domain embeddings (N, D)
        target_embeddings: Target domain embeddings (M, D)
        source_labels: Source domain labels (optional)
        target_labels: Target domain labels (optional)

    Returns:
        Dictionary of domain alignment metrics
    """
    metrics = {}

    # Normalize embeddings
    source_norm = source_embeddings / (
        source_embeddings.norm(dim=1, keepdim=True) + 1e-8
    )
    target_norm = target_embeddings / (
        target_embeddings.norm(dim=1, keepdim=True) + 1e-8
    )

    # Domain-level alignment (global mean similarity)
    domain_cos_sim = torch.mm(source_norm, target_norm.t()).mean().item()
    metrics['domain_cos_similarity'] = domain_cos_sim

    # Compute class-conditional alignment if labels available
    if source_labels is not None and target_labels is not None:
        unique_classes = torch.cat([source_labels, target_labels]).unique()
        class_alignment_scores = []

        for c in unique_classes:
            s_mask = source_labels == c
            t_mask = target_labels == c

            if s_mask.sum() > 0 and t_mask.sum() > 0:
                s_class = source_norm[s_mask].mean(dim=0)
                t_class = target_norm[t_mask].mean(dim=0)

                class_sim = torch.nn.functional.cosine_similarity(
                    s_class.unsqueeze(0), t_class.unsqueeze(0)
                ).item()
                class_alignment_scores.append(class_sim)

        if class_alignment_scores:
            metrics['class_conditional_alignment'] = np.mean(class_alignment_scores)
            metrics['class_alignment_std'] = np.std(class_alignment_scores)
        else:
            metrics['class_conditional_alignment'] = 0.0
            metrics['class_alignment_std'] = 0.0
    else:
        metrics['class_conditional_alignment'] = None
        metrics['class_alignment_std'] = None

    # Embedding distribution statistics
    source_var = source_embeddings.var(dim=0).mean().item()
    target_var = target_embeddings.var(dim=0).mean().item()
    metrics['source_embedding_variance'] = source_var
    metrics['target_embedding_variance'] = target_var

    # Distribution similarity (variance ratio)
    if target_var > 0:
        metrics['variance_ratio'] = source_var / target_var
    else:
        metrics['variance_ratio'] = 0.0

    return metrics
