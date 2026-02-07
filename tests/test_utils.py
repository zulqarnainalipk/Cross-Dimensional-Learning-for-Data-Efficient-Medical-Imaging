"""
Unit tests for metrics and utilities.
"""

import pytest
import torch
import numpy as np
from cross_dim_transfer.src.utils.metrics import (
    calculate_accuracy,
    calculate_auc_roc,
    calculate_f1_score,
    calculate_sensitivity_specificity,
    evaluate_episode,
    compute_embedding_similarity,
    compute_domain_alignment_metrics
)


class TestCalculateAccuracy:
    """Tests for calculate_accuracy."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        predictions = torch.tensor([0, 1, 1, 0])
        targets = torch.tensor([0, 1, 1, 0])
        assert calculate_accuracy(predictions, targets) == 1.0

    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        predictions = torch.tensor([0, 0, 0, 0])
        targets = torch.tensor([1, 1, 1, 1])
        assert calculate_accuracy(predictions, targets) == 0.0

    def test_partial_accuracy(self):
        """Test partial accuracy."""
        predictions = torch.tensor([0, 1, 0, targets = torch.tensor 1])
       ([0, 0, 1, 1])
        assert calculate_accuracy(predictions, targets) == 0.5


class TestCalculateAucRoc:
    """Tests for calculate_auc_roc."""

    def test_binary_classification(self):
        """Test binary classification AUC."""
        probabilities = torch.tensor([0.1, 0.4, 0.6, 0.9])
        targets = torch.tensor([0, 0, 1, 1])

        auc = calculate_auc_roc(probabilities, targets, num_classes=2)
        assert 0.0 <= auc <= 1.0

    def test_perfect_auc(self):
        """Test perfect AUC (1.0)."""
        probabilities = torch.tensor([0.0, 0.0, 1.0, 1.0])
        targets = torch.tensor([0, 0, 1, 1])

        auc = calculate_auc_roc(probabilities, targets, num_classes=2)
        assert auc == 1.0

    def test_random_auc(self):
        """Test random AUC (~0.5)."""
        np.random.seed(42)
        probabilities = torch.tensor(np.random.rand(100))
        targets = torch.randint(0, 2, (100,))

        auc = calculate_auc_roc(probabilities, targets, num_classes=2)
        assert 0.4 <= auc <= 0.6  # Should be close to 0.5


class TestCalculateF1Score:
    """Tests for calculate_f1_score."""

    def test_perfect_f1(self):
        """Test perfect F1 score."""
        predictions = torch.tensor([0, 1, 1, 0])
        targets = torch.tensor([0, 1, 1, 0])

        f1 = calculate_f1_score(predictions, targets, average='binary')
        assert f1 == 1.0


class TestEvaluateEpisode:
    """Tests for evaluate_episode."""

    def test_few_shot_evaluation(self):
        """Test few-shot episode evaluation."""
        # Support set: 2 classes, 3 samples each
        support_embeddings = torch.cat([
            torch.randn(3, 512) + 1,  # Class 0
            torch.randn(3, 512) - 1   # Class 1
        ])
        support_labels = torch.tensor([0, 0, 0, 1, 1, 1])

        # Query set: 6 samples
        query_embeddings = torch.cat([
            torch.randn(3, 512) + 1,  # Class 0
            torch.randn(3, 512) - 1   # Class 1
        ])
        query_labels = torch.tensor([0, 0, 0, 1, 1, 1])

        result = evaluate_episode(
            query_embeddings, query_labels,
            support_embeddings, support_labels,
            num_classes=2
        )

        assert 'accuracy' in result
        assert 'num_queries' in result
        assert 'num_support' in result
        assert result['accuracy'] >= 0.0
        assert result['accuracy'] <= 1.0


class TestComputeEmbeddingSimilarity:
    """Tests for compute_embedding_similarity."""

    def test_cosine_similarity_same(self):
        """Test cosine similarity for identical embeddings."""
        emb = torch.randn(10, 512)
        similarity = compute_embedding_similarity(emb, emb, method='cosine')
        assert 0.99 <= similarity <= 1.0

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity for opposite embeddings."""
        emb1 = torch.randn(10, 512)
        emb2 = -emb1
        similarity = compute_embedding_similarity(emb1, emb2, method='cosine')
        assert -1.0 <= similarity <= -0.99


class TestComputeDomainAlignmentMetrics:
    """Tests for compute_domain_alignment_metrics."""

    def test_domain_alignment(self):
        """Test domain alignment metric computation."""
        source = torch.randn(50, 512)
        target = torch.randn(50, 512)

        metrics = compute_domain_alignment_metrics(source, target)

        assert 'domain_cos_similarity' in metrics
        assert 'source_embedding_variance' in metrics
        assert 'target_embedding_variance' in metrics
        assert -1.0 <= metrics['domain_cos_similarity'] <= 1.0

    def test_with_labels(self):
        """Test with class labels."""
        source = torch.randn(50, 512)
        target = torch.randn(50, 512)
        source_labels = torch.randint(0, 2, (50,))
        target_labels = torch.randint(0, 2, (50,))

        metrics = compute_domain_alignment_metrics(
            source, target, source_labels, target_labels
        )

        assert 'class_conditional_alignment' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
