# Architecture Documentation

This document provides detailed technical documentation for the Cross-Dimensional Knowledge Transfer architecture.

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Feature Extraction](#feature-extraction)
4. [Cross-Dimensional Attention Bridge](#cross-dimensional-attention-bridge)
5. [Domain Alignment](#domain-alignment)
6. [Few-Shot Learning](#few-shot-learning)
7. [Training Pipeline](#training-pipeline)

---

## Overview

The Cross-Dimensional Knowledge Transfer model enables knowledge transfer from 3D CT volumes to 2D pathology images for few-shot HNSCC cancer classification. The architecture consists of:

- **Feature Encoders**: Separate encoders for CT and pathology modalities
- **Cross-Dimensional Attention Bridge**: Transfers knowledge across modalities
- **Domain Alignment Module**: Aligns source and target distributions
- **Prototypical Network**: Performs few-shot classification

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cross-Dimensional Transfer                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐         ┌─────────────────┐                   │
│  │   3D CT     │         │   2D Pathology  │                   │
│  │   Volume    │         │     Image       │                   │
│  └──────┬──────┘         └────────┬────────┘                   │
│         │                          │                            │
│         ▼                          ▼                            │
│  ┌─────────────┐         ┌─────────────────┐                   │
│  │    CT       │         │  Pathology      │                   │
│  │  Encoder    │         │   Encoder       │                   │
│  │ (MedGemma)  │         │(Path Foundation)│                   │
│  └──────┬──────┘         └────────┬────────┘                   │
│         │                          │                            │
│         └──────────┬───────────────┘                            │
│                    ▼                                            │
│         ┌─────────────────────┐                                 │
│         │  Cross-Dimensional  │                                 │
│         │   Attention Bridge  │                                 │
│         └──────────┬──────────┘                                 │
│                    │                                            │
│         ┌──────────┴──────────┐                                 │
│         ▼                     ▼                                 │
│  ┌─────────────┐      ┌─────────────────┐                      │
│  │    Domain   │      │   Prototypical  │                      │
│  │  Alignment  │      │     Network     │                      │
│  │   Module    │      │  (Few-Shot)     │                      │
│  └─────────────┘      └─────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

### Configuration Parameters

```python
@dataclass
class ModelConfig:
    # Feature dimensions
    ct_embedding_dim: int = 768      # MedGemma output dimension
    pathology_embedding_dim: int = 768  # Path Foundation output dimension
    shared_embedding_dim: int = 512     # Common representation space

    # Attention parameters
    num_attention_heads: int = 8
    num_cross_attention_layers: int = 2
    attention_dropout: float = 0.1

    # Regularization
    dropout_rate: float = 0.1
    projection_layers: int = 2
```

### Main Model Class

```python
class CrossDimensionalTransferModel(nn.Module):
    """
    Main model for cross-dimensional knowledge transfer.

    Architecture:
        1. CT Encoder (MedGemma-based)
        2. Pathology Encoder (Path Foundation-based)
        3. Cross-Dimensional Attention Bridge
        4. Domain Alignment Module
        5. Prototypical Classification Head
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Feature encoders
        self.ct_encoder = CTEncoder(config)
        self.pathology_encoder = PathologyEncoder(config)

        # Cross-dimensional attention
        self.cross_attention = CrossDimensionalAttentionBridge(config)

        # Domain alignment
        self.domain_aligner = DomainAdversarialNetwork(config)

        # Few-shot classification
        self.proto_network = PrototypicalNetwork(config)

    def forward(self, ct_volume, pathology_image, domain_labels=None):
        # Extract features
        ct_features = self.ct_encoder(ct_volume)
        path_features = self.pathology_encoder(pathology_image)

        # Cross-dimensional transfer
        fused_features = self.cross_attention(ct_features, path_features)

        # Domain alignment
        if domain_labels is not None:
            domain_loss = self.domain_aligner(fused_features, domain_labels)
        else:
            domain_loss = None

        # Few-shot classification
        logits = self.proto_network(fused_features)

        return logits, domain_loss
```

---

## Feature Extraction

### CT Encoder (MedGemma)

Uses MedGemma 1.5 4B-it for CT interpretation and feature extraction.

```python
class CTEncoder(nn.Module):
    """
    Encoder for 3D CT volumes using MedGemma.

    Process:
        1. Sample key slices from CT volume
        2. Generate clinical descriptions using MedGemma
        3. Extract text embeddings
        4. Project to feature space
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.medgemma = MedGemmaInterface(config)
        self.feature_proj = nn.Linear(
            config.medgemma_feature_dim,
            config.ct_embedding_dim
        )

    def forward(self, ct_volume):
        # Extract semantic features from CT
        features = self.medgemma.extract_semantic_features(ct_volume)
        features = self.feature_proj(features)
        return features
```

### Pathology Encoder (Path Foundation)

Uses Path Foundation for digital pathology feature extraction.

```python
class PathologyEncoder(nn.Module):
    """
    Encoder for 2D pathology images using Path Foundation.

    Process:
        1. Preprocess pathology image (resize to 224x224)
        2. Extract patch-level features
        3. Aggregate to image-level embedding
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.path_foundation = PathFoundationInterface(config)
        self.feature_proj = nn.Linear(
            config.path_foundation_feature_dim,
            config.pathology_embedding_dim
        )

    def forward(self, pathology_image):
        # Extract features from pathology
        features = self.path_foundation.extract_features(pathology_image)
        features = self.feature_proj(features)
        return features
```

---

## Cross-Dimensional Attention Bridge

The Cross-Dimensional Attention Bridge (CDAB) enables knowledge transfer from 3D CT to 2D pathology representations.

### Architecture

```
CT Features (CT_H)          Pathology Features (PATH_H)
      │                            │
      │                            │
      ▼                            ▼
  Query Proj                  Key/Value Proj
      │                            │
      └──────────┬─────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │ Multi-Head     │
        │ Cross-Attention│
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │   Feature      │
        │   Fusion       │
        └────────────────┘
                 │
                 ▼
         Fused Features
```

### Implementation

```python
class CrossDimensionalAttentionBridge(nn.Module):
    """
    Bridge for cross-dimensional attention between CT and pathology.

    Transfers knowledge from CT feature space to pathology representation
    using cross-attention mechanism.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # CT feature projection (as query)
        self.ct_query_proj = nn.Linear(
            config.ct_embedding_dim,
            config.shared_embedding_dim
        )

        # Pathology feature projection (as key/value)
        self.path_key_proj = nn.Linear(
            config.pathology_embedding_dim,
            config.shared_embedding_dim
        )
        self.path_value_proj = nn.Linear(
            config.pathology_embedding_dim,
            config.shared_embedding_dim
        )

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.shared_embedding_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # Feature fusion
        self.fusion_layer = FeatureFusionModule(config)

    def forward(self, ct_features, pathology_features):
        # Project features
        ct_query = self.ct_query_proj(ct_features)
        path_key = self.path_key_proj(pathology_features)
        path_value = self.path_value_proj(pathology_features)

        # Cross-attention: CT queries attend to pathology
        attended_features, attention_weights = self.cross_attention(
            query=ct_query.unsqueeze(0),
            key=path_key.unsqueeze(0),
            value=path_value.unsqueeze(0)
        )

        # Fuse with original CT features
        fused = self.fusion_layer(ct_features, attended_features.squeeze(0))

        return fused, attention_weights
```

### Attention Visualization

The attention weights can be visualized to understand which pathology regions are attended to when processing CT features.

```python
def visualize_attention(attention_weights, pathology_image):
    """
    Visualize cross-attention weights overlaid on pathology image.

    Args:
        attention_weights: (H, W) attention map
        pathology_image: (C, H, W) pathology image
    """
    # Resize attention to match image size
    attention_map = F.interpolate(
        attention_weights.unsqueeze(0).unsqueeze(0),
        size=pathology_image.shape[1:],
        mode='bilinear'
    ).squeeze()

    # Overlay on image
    fig, ax = plt.subplots()
    ax.imshow(pathology_image.permute(1, 2, 0))
    ax.imshow(attention_map, cmap='viridis', alpha=0.5)
    ax.set_title('Cross-Attention Visualization')
```

---

## Domain Alignment

The Domain Alignment module aligns CT (source) and pathology (target) feature distributions using adversarial training.

### Architecture

```
                 ┌─────────────────────┐
                 │   Feature Encoder   │
                 │   (Shared Backbone) │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
         ┌─────────────┐      ┌─────────────────┐
         │  Gradient   │      │  Domain         │
         │  Reversal   │─────▶│  Discriminator  │
         │  Layer      │      │                 │
         └─────────────┘      └─────────────────┘
                 │                     │
                 │                     ▼
                 │            ┌─────────────────┐
                 │            │   Domain Loss   │
                 │            │   (Minimax)     │
                 │            └─────────────────┘
                 ▼
         ┌─────────────┐
         │   Feature   │
         │   Space     │
         └─────────────┘
```

### Implementation

```python
class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for domain adversarial training.

    Forward: Identity transformation
    Backward: Multiplies gradient by -lambda (reverses gradient)
    """

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return -self.lambda_ * grad_output


class DomainAdversarialNetwork(nn.Module):
    """
    Domain discriminator with gradient reversal for adversarial alignment.
    """

    def __init__(self, config):
        super().__init__()
        self.grl = GradientReversalLayer()

        self.discriminator = nn.Sequential(
            nn.Linear(config.shared_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 2)  # Source vs Target
        )

    def forward(self, features, domain_labels=None):
        # Apply gradient reversal
        reversed_features = self.grl(features)

        # Classify domain
        domain_logits = self.discriminator(reversed_features)

        if domain_labels is not None:
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
        else:
            domain_loss = None

        return domain_loss, domain_logits
```

### Training Objective

```python
def compute_domain_alignment_loss(features, domain_labels, lambda_=1.0):
    """
    Compute domain alignment loss with gradient reversal.

    Total Loss = L_classification + lambda_ * L_domain

    Where L_domain encourages features to be domain-invariant.
    """
    # Classification loss (standard)
    class_loss = F.cross_entropy(class_logits, class_labels)

    # Domain loss (adversarial)
    domain_loss, domain_logits = domain_discriminator(features, domain_labels)

    # Combined loss
    total_loss = class_loss + lambda_ * domain_loss

    return total_loss, domain_loss
```

---

## Few-Shot Learning

The model uses a Prototypical Network for few-shot classification.

### Concept

```
Feature Space Visualization:

    Class 1 Prototype ──────●────────────────── Class 2 Prototype
        (P1)               / \                     (P2)
                         /   \
         Query ○────────/     \\────────── Query ○
                   \     \   /
                    \     ○ Query (classified as Class 1)
                     \   /
                      ○
              Support Points (Class 1)
```

### Implementation

```python
class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot classification.

    For each episode:
        1. Compute prototype for each class from support set
        2. Classify query samples by distance to prototypes
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.distance_metric = 'euclidean'

    def compute_prototypes(self, support_embeddings, support_labels):
        """
        Compute class prototypes as mean of support embeddings.
        """
        prototypes = {}
        for class_id in support_labels.unique():
            mask = support_labels == class_id
            prototypes[class_id] = support_embeddings[mask].mean(dim=0)

        return torch.stack([prototypes[c] for c in sorted(prototypes.keys())])

    def forward(self, embeddings, support_embeddings=None, support_labels=None):
        """
        Classify embeddings using prototypical representation.

        If support set provided: Compute prototypes, classify queries
        If no support: Return raw logits (for training features)
        """
        if support_embeddings is not None and support_labels is not None:
            # Compute prototypes
            prototypes = self.compute_prototypes(
                support_embeddings, support_labels
            )

            # Compute distances to prototypes
            distances = torch.cdist(
                embeddings.unsqueeze(0),
                prototypes.unsqueeze(0)
            ).squeeze(0)

            # Convert to logits (negative distances)
            logits = -distances
            return logits, prototypes
        else:
            # Training mode: return features
            return embeddings, None
```

### Episode Sampling

```python
class FewShotEpisodeSampler:
    """
    Sampler for few-shot learning episodes.
    """

    def __init__(self, dataset, num_way=2, num_shots=5, num_queries=15):
        self.dataset = dataset
        self.num_way = num_way
        self.num_shots = num_shots
        self.num_queries = num_queries

    def sample_episode(self):
        """
        Sample a single few-shot episode.
        """
        # Sample classes
        classes = np.random.choice(
            self.dataset.classes,
            size=self.num_way,
            replace=False
        )

        # Sample support set
        support_embeddings = []
        support_labels = []
        for i, class_name in enumerate(classes):
            samples = self.dataset.get_class_samples(class_name, self.num_shots)
            for sample in samples:
                support_embeddings.append(sample['embedding'])
                support_labels.append(i)

        # Sample query set
        query_embeddings = []
        query_labels = []
        for i, class_name in enumerate(classes):
            samples = self.dataset.get_class_samples(class_name, self.num_queries)
            for sample in samples:
                query_embeddings.append(sample['embedding'])
                query_labels.append(i)

        return {
            'support_embeddings': torch.stack(support_embeddings),
            'support_labels': torch.tensor(support_labels),
            'query_embeddings': torch.stack(query_embeddings),
            'query_labels': torch.tensor(query_labels),
            'classes': classes
        }
```

---

## Training Pipeline

### Overall Training Loop

```python
def train_epoch(model, dataloader, optimizer, config):
    """
    Single epoch of training.
    """
    model.train()
    total_loss = 0
    total_acc = 0

    for batch in dataloader:
        # Unpack batch
        ct_volume = batch['ct_volume']
        pathology_image = batch['pathology_image']
        labels = batch['label']
        domain_labels = batch['domain_label']

        # Forward pass
        logits, domain_loss = model(
            ct_volume,
            pathology_image,
            domain_labels
        )

        # Compute losses
        class_loss = F.cross_entropy(logits, labels)
        total_batch_loss = class_loss + config.domain_alignment_weight * domain_loss

        # Backward pass
        total_batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Metrics
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        total_loss += class_loss.item()
        total_acc += acc.item()

    return total_loss / len(dataloader), total_acc / len(dataloader)
```

### Evaluation Protocol

```python
def evaluate_few_shot(model, dataset, num_episodes=100, num_way=2, num_shots=5):
    """
    Evaluate model on few-shot classification tasks.
    """
    model.eval()
    sampler = FewShotEpisodeSampler(dataset, num_way, num_shots)

    results = []
    for _ in range(num_episodes):
        episode = sampler.sample_episode()

        with torch.no_grad():
            # Get query predictions
            query_logits, _ = model(
                episode['query_embeddings'],
                episode['support_embeddings'],
                episode['support_labels']
            )

            preds = query_logits.argmax(dim=1)
            correct = (preds == episode['query_labels']).sum().item()
            accuracy = correct / len(episode['query_labels'])

            results.append(accuracy)

    return {
        'mean_accuracy': np.mean(results),
        'std_accuracy': np.std(results),
        '95_ci': 1.96 * np.std(results) / np.sqrt(len(results))
    }
```

---

## Summary

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| CT Encoder | 3D CT Volume | 768-dim vector | Extract CT semantic features |
| Pathology Encoder | 2D Pathology Image | 768-dim vector | Extract pathology features |
| Cross-Attention | CT + Pathology features | 512-dim fused | Transfer CT knowledge to pathology |
| Domain Discriminator | Fused features | Domain logits | Align domain distributions |
| Proto Network | Features | Class logits | Few-shot classification |

### Key Hyperparameters

| Parameter | Value | Impact |
|-----------|-------|--------|
| shared_embedding_dim | 512 | Model capacity |
| num_attention_heads | 8 | Attention granularity |
| num_cross_attention_layers | 2 | Transfer depth |
| domain_alignment_weight | 0.3 | Domain invariance strength |
| num_shots | 5 | Few-shot learning performance |
