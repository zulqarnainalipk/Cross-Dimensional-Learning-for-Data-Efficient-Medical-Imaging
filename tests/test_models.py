"""
Unit tests for model components.
"""

import pytest
import torch
import torch.nn as nn
from cross_dim_transfer.src.models import (
    CrossDimensionalAttentionBridge,
    GradientReversalLayer,
    DomainAdversarialNetwork,
    PrototypicalNetwork
)


class TestGradientReversalLayer:
    """Tests for GradientReversalLayer."""

    def test_forward_identity(self):
        """Test that forward pass is identity."""
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 128)
        assert torch.allclose(grl(x), x)

    def test_backward_reverses_gradient(self):
        """Test that backward pass reverses gradient."""
        grl = GradientReversalLayer(lambda_=1.0)

        # Create computation graph
        x = torch.randn(4, 128, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be reversed
        assert x.grad is not None
        # Check that gradient direction is reversed
        # (not exactly -x because of batch norm etc., but direction should flip)
        manual_grad = torch.ones_like(x.grad)
        assert torch.allclose(x.grad, -manual_grad, atol=1e-6)


class TestCrossDimensionalAttentionBridge:
    """Tests for CrossDimensionalAttentionBridge."""

    @pytest.fixture
    def config(self):
        class MockConfig:
            ct_embedding_dim = 768
            pathology_embedding_dim = 768
            shared_embedding_dim = 512
            num_attention_heads = 8
            attention_dropout = 0.1
            dropout_rate = 0.1
        return MockConfig()

    @pytest.fixture
    def bridge(self, config):
        return CrossDimensionalAttentionBridge(config)

    @pytest.fixture
    def sample_features(self):
        ct_features = torch.randn(768)
        path_features = torch.randn(768)
        return ct_features, path_features

    def test_output_shape(self, bridge, sample_features):
        """Test that output has correct shape."""
        ct_features, path_features = sample_features
        fused, attention = bridge(ct_features, path_features)

        assert fused.shape == (512,)  # shared_embedding_dim
        assert attention.shape[0] == 8  # num_attention_heads

    def test_retain_gradients(self, bridge, sample_features):
        """Test that gradients can flow through bridge."""
        ct_features, path_features = sample_features
        ct_features.requires_grad = True

        fused, attention = bridge(ct_features, path_features)
        loss = fused.sum()
        loss.backward()

        assert ct_features.grad is not None


class TestDomainAdversarialNetwork:
    """Tests for DomainAdversarialNetwork."""

    @pytest.fixture
    def config(self):
        class MockConfig:
            shared_embedding_dim = 512
            dropout_rate = 0.1
        return MockConfig()

    @pytest.fixture
    def aligner(self, config):
        return DomainAdversarialNetwork(config)

    @pytest.fixture
    def sample_features(self):
        return torch.randn(32, 512)

    def test_output_shape(self, aligner, sample_features):
        """Test that output has correct shape."""
        features = sample_features
        loss, logits = aligner(features)

        assert logits.shape == (32, 2)  # 2 domains

    def test_with_labels(self, aligner, sample_features):
        """Test with domain labels."""
        features = sample_features
        domain_labels = torch.randint(0, 2, (32,))

        loss, logits = aligner(features, domain_labels)

        assert loss is not None
        assert loss.item() >= 0

    def test_without_labels(self, aligner, sample_features):
        """Test without domain labels."""
        features = sample_features

        loss, logits = aligner(features, None)

        assert loss is None


class TestPrototypicalNetwork:
    """Tests for PrototypicalNetwork."""

    @pytest.fixture
    def config(self):
        class MockConfig:
            shared_embedding_dim = 512
        return MockConfig()

    @pytest.fixture
    def proto_net(self, config):
        return PrototypicalNetwork(config)

    @pytest.fixture
    def support_set(self):
        embeddings = torch.randn(10, 512)
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        return embeddings, labels

    @pytest.fixture
    def query_set(self):
        return torch.randn(10, 512)

    def test_compute_prototypes(self, proto_net, support_set):
        """Test prototype computation."""
        embeddings, labels = support_set
        prototypes = proto_net.compute_prototypes(embeddings, labels)

        # Should have 2 prototypes (2 classes)
        assert prototypes.shape == (2, 512)

    def test_classification(self, proto_net, support_set, query_set):
        """Test few-shot classification."""
        support_embeddings, support_labels = support_set

        logits, prototypes = proto_net(
            query_set, support_embeddings, support_labels
        )

        # Should have logits for all queries against 2 classes
        assert logits.shape == (10, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
