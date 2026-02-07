"""
Loss functions for cross-dimensional knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised pre-training.

    Uses temperature-scaled cosine similarity for contrastive learning.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature scaling factor
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            features1: Features from first view
            features2: Features from second view

        Returns:
            Contrastive loss value
        """
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(features1, features2.t()) / self.temperature

        # Create labels (diagonal is positive)
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size).to(similarity.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity, labels)

        return loss


class PrototypicalLoss(nn.Module):
    """
    Prototypical loss for few-shot classification.

    Computes class prototypes and classifies based on distance.
    """

    def __init__(self, distance_metric: str = 'euclidean'):
        """
        Initialize prototypical loss.

        Args:
            distance_metric: Distance metric ('euclidean' or 'cosine')
        """
        super().__init__()
        self.distance_metric = distance_metric

    def forward(
        self,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prototypical loss.

        Args:
            query_features: Features from query set
            query_labels: Labels for query set
            support_features: Features from support set
            support_labels: Labels for support set

        Returns:
            Prototypical loss value
        """
        # Compute prototypes
        num_classes = support_labels.max().item() + 1
        prototypes = torch.zeros(num_classes, query_features.size(1)).to(query_features.device)

        for c in range(num_classes):
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                prototypes[c] = support_features[class_mask].mean(dim=0)

        # Compute distances
        if self.distance_metric == 'euclidean':
            squared_distances = torch.cdist(query_features, prototypes, p=2) ** 2
        else:  # cosine
            query_normalized = F.normalize(query_features, p=2, dim=1)
            prototype_normalized = F.normalize(prototypes, p=2, dim=1)
            similarity = torch.mm(query_normalized, prototype_normalized.t())
            squared_distances = 1 - similarity

        # Compute loss
        logits = -squared_distances
        loss = F.cross_entropy(logits, query_labels)

        return loss


class DomainAlignmentLoss(nn.Module):
    """
    Domain alignment loss using adversarial training.

    Encourages learning of domain-invariant features.
    """

    def __init__(self):
        """Initialize domain alignment loss."""
        super().__init__()

    def forward(
        self,
        features_3d: torch.Tensor,
        features_2d: torch.Tensor,
        discriminator: nn.Module
    ) -> torch.Tensor:
        """
        Compute domain alignment loss.

        Args:
            features_3d: Features from 3D encoder
            features_2d: Features from 2D encoder
            discriminator: Domain discriminator network

        Returns:
            Domain alignment loss value
        """
        # Combine features
        all_features = torch.cat([features_3d, features_2d], dim=0)

        # Create domain labels
        batch_3d = features_3d.size(0)
        batch_2d = features_2d.size(0)
        domain_labels = torch.cat([
            torch.zeros(batch_3d, dtype=torch.long).to(features_3d.device),
            torch.ones(batch_2d, dtype=torch.long).to(features_3d.device)
        ])

        # Compute domain predictions
        domain_logits = discriminator(all_features)

        # Compute loss
        loss = F.cross_entropy(domain_logits, domain_labels)

        return loss


class SemanticAlignmentLoss(nn.Module):
    """
    Semantic alignment loss using MedSigLIP.

    Aligns visual features with textual class descriptions.
    """

    def __init__(self, class_descriptions: Optional[List[str]] = None):
        """
        Initialize semantic alignment loss.

        Args:
            class_descriptions: List of class descriptions
        """
        super().__init__()

        if class_descriptions is None:
            class_descriptions = [
                "Tumor stroma region showing organized connective tissue",
                "Invasion front region showing irregular tumor glands"
            ]

        self.class_descriptions = class_descriptions

    def forward(
        self,
        visual_features: torch.Tensor,
        medsiglip_interface,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute semantic alignment loss.

        Args:
            visual_features: Visual features from encoder
            medsiglip_interface: MedSigLIP interface for alignment
            target_labels: Ground truth labels

        Returns:
            Semantic alignment loss value
        """
        if medsiglip_interface is None:
            return torch.tensor(0.0, device=visual_features.device)

        if target_labels is None:
            return torch.tensor(0.0, device=visual_features.device)

        # This is a simplified version - full implementation
        # would compute actual alignment scores using MedSigLIP
        return torch.tensor(0.0, device=visual_features.device)


class TotalLoss(nn.Module):
    """
    Combined loss for end-to-end training.

    Combines classification, contrastive, and alignment losses.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        proto_weight: float = 0.5,
        adv_weight: float = 0.3,
        contrastive_weight: float = 0.1,
        semantic_weight: float = 0.15
    ):
        """
        Initialize total loss.

        Args:
            cls_weight: Weight for classification loss
            proto_weight: Weight for prototypical loss
            adv_weight: Weight for adversarial loss
            contrastive_weight: Weight for contrastive loss
            semantic_weight: Weight for semantic alignment loss
        """
        super().__init__()

        self.cls_weight = cls_weight
        self.proto_weight = proto_weight
        self.adv_weight = adv_weight
        self.contrastive_weight = contrastive_weight
        self.semantic_weight = semantic_weight

        self.cls_loss = nn.CrossEntropyLoss()
        self.proto_loss = PrototypicalLoss()
        self.contrastive_loss = ContrastiveLoss()

    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor
    ) -> dict:
        """
        Compute total loss.

        Args:
            outputs: Dictionary of model outputs
            targets: Ground truth labels

        Returns:
            Dictionary with total loss and individual loss components
        """
        losses = {}

        # Classification loss
        if 'logits' in outputs:
            losses['cls_loss'] = self.cls_loss(outputs['logits'], targets)
        else:
            losses['cls_loss'] = torch.tensor(0.0, device=targets.device)

        # Prototypical loss
        if 'query_features' in outputs and 'support_features' in outputs:
            if 'support_labels' in outputs:
                losses['proto_loss'] = self.proto_loss(
                    outputs['query_features'],
                    targets,
                    outputs['support_features'],
                    outputs['support_labels']
                )
            else:
                losses['proto_loss'] = torch.tensor(0.0, device=targets.device)
        else:
            losses['proto_loss'] = torch.tensor(0.0, device=targets.device)

        # Contrastive loss
        if 'view1' in outputs and 'view2' in outputs:
            losses['contrastive_loss'] = self.contrastive_loss(
                outputs['view1'],
                outputs['view2']
            )
        else:
            losses['contrastive_loss'] = torch.tensor(0.0, device=targets.device)

        # Total loss
        total_loss = (
            self.cls_weight * losses['cls_loss'] +
            self.proto_weight * losses['proto_loss'] +
            self.contrastive_weight * losses['contrastive_loss']
        )

        losses['total_loss'] = total_loss

        return losses
