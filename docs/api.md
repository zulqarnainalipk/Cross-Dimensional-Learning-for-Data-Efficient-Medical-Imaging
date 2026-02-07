# API Documentation

This document provides comprehensive API documentation for the Cross-Dimensional Knowledge Transfer package.

## Table of Contents

1. [Main Model](#main-model)
2. [Feature Encoders](#feature-encoders)
3. [Attention Bridge](#attention-bridge)
4. [Domain Alignment](#domain-alignment)
5. [Few-Shot Learning](#few-shot-learning)
6. [Training](#training)
7. [Datasets](#datasets)
8. [Metrics](#metrics)

---

## Main Model

### CrossDimensionalTransferModel

```python
class CrossDimensionalTransferModel(nn.Module):
    """
    Main model for cross-dimensional knowledge transfer from 3D CT to 2D pathology.

    Args:
        config: Configuration object containing model hyperparameters

    Attributes:
        ct_encoder: Encoder for 3D CT volumes
        pathology_encoder: Encoder for 2D pathology images
        cross_attention: Cross-dimensional attention bridge
        domain_aligner: Domain adversarial alignment module
        proto_network: Prototypical network for few-shot classification

    Example:
        >>> config = Config()
        >>> model = CrossDimensionalTransferModel(config)
        >>> logits, domain_loss = model(ct_volume, pathology_image, domain_labels)
    """

    def __init__(self, config: Config):
        """
        Initialize the cross-dimensional transfer model.

        Args:
            config: Configuration object with model settings
        """
        super().__init__()
        self.config = config
        self._build_model()

    def _build_model(self):
        """Construct model architecture from config."""
        self.ct_encoder = CTEncoder(self.config)
        self.pathology_encoder = PathologyEncoder(self.config)
        self.cross_attention = CrossDimensionalAttentionBridge(self.config)
        self.domain_aligner = DomainAdversarialNetwork(self.config)
        self.proto_network = PrototypicalNetwork(self.config)

    def forward(
        self,
        ct_volume: torch.Tensor,
        pathology_image: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            ct_volume: 3D CT volume tensor (B, D, H, W)
            pathology_image: 2D pathology image tensor (B, C, H, W)
            domain_labels: Optional domain labels for adversarial training

        Returns:
            Tuple of (classification logits, domain loss)
        """
        # Extract features
        ct_features = self.ct_encoder(ct_volume)
        path_features = self.pathology_encoder(pathology_image)

        # Cross-dimensional attention
        fused_features, attention_weights = self.cross_attention(
            ct_features, path_features
        )

        # Domain alignment
        if domain_labels is not None:
            domain_loss, domain_logits = self.domain_aligner(
                fused_features, domain_labels
            )
        else:
            domain_loss = None
            domain_logits = None

        # Few-shot classification
        logits = self.proto_network(fused_features)

        return logits, domain_loss

    def get_attention_weights(
        self,
        ct_volume: torch.Tensor,
        pathology_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Get cross-attention weights for visualization.

        Args:
            ct_volume: 3D CT volume
            pathology_image: 2D pathology image

        Returns:
            Attention weight tensor
        """
        self.eval()
        with torch.no_grad():
            ct_features = self.ct_encoder(ct_volume)
            path_features = self.pathology_encoder(pathology_image)
            _, attention_weights = self.cross_attention(
                ct_features, path_features
            )
        return attention_weights
```

---

## Feature Encoders

### CTEncoder

```python
class CTEncoder(nn.Module):
    """
    Encoder for 3D CT volumes using MedGemma for semantic interpretation.

    Args:
        config: Configuration object

    Attributes:
        medgemma: MedGemmaInterface for CT interpretation
        feature_proj: Linear projection to target dimension

    Example:
        >>> encoder = CTEncoder(config)
        >>> features = encoder(ct_volume)  # (B, 768)
    """

    def __init__(self, config: Config):
        """Initialize CT encoder with MedGemma."""
        super().__init__()
        self.config = config
        self.medgemma = MedGemmaInterface(config)
        self.feature_proj = nn.Linear(
            config.medgemma_feature_dim,
            config.ct_embedding_dim
        )

    def forward(self, ct_volume: torch.Tensor) -> torch.Tensor:
        """
        Extract features from CT volume.

        Args:
            ct_volume: 3D CT volume tensor (D, H, W)

        Returns:
            Feature tensor of shape (embedding_dim,)
        """
        features = self.medgemma.extract_semantic_features(ct_volume)
        features = self.feature_proj(features)
        return features

    def generate_description(self, ct_slice: np.ndarray) -> str:
        """
        Generate clinical description for CT slice.

        Args:
            ct_slice: 2D CT slice numpy array

        Returns:
            Clinical description string
        """
        return self.medgemma.generate_ct_description(ct_slice)
```

### PathologyEncoder

```python
class PathologyEncoder(nn.Module):
    """
    Encoder for 2D pathology images using Path Foundation.

    Args:
        config: Configuration object

    Attributes:
        path_foundation: PathFoundationInterface for feature extraction
        feature_proj: Linear projection to target dimension

    Example:
        >>> encoder = PathologyEncoder(config)
        >>> features = encoder(pathology_image)  # (B, 768)
    """

    def __init__(self, config: Config):
        """Initialize pathology encoder with Path Foundation."""
        super().__init__()
        self.config = config
        self.path_foundation = PathFoundationInterface(config)
        self.feature_proj = nn.Linear(
            config.path_foundation_feature_dim,
            config.pathology_embedding_dim
        )

    def forward(self, pathology_image: torch.Tensor) -> torch.Tensor:
        """
        Extract features from pathology image.

        Args:
            pathology_image: 2D pathology image tensor (C, H, W)

        Returns:
            Feature tensor of shape (embedding_dim,)
        """
        features = self.path_foundation.extract_features(pathology_image)
        features = self.feature_proj(features)
        return features
```

---

## Attention Bridge

### CrossDimensionalAttentionBridge

```python
class CrossDimensionalAttentionBridge(nn.Module):
    """
    Cross-dimensional attention bridge for CT to pathology knowledge transfer.

    Args:
        config: Configuration object

    Example:
        >>> bridge = CrossDimensionalAttentionBridge(config)
        >>> fused, weights = bridge(ct_features, path_features)
    """

    def __init__(self, config: Config):
        """Initialize cross-attention bridge."""
        super().__init__()
        self.config = config

        # Feature projections
        self.ct_query_proj = nn.Linear(
            config.ct_embedding_dim,
            config.shared_embedding_dim
        )
        self.path_key_proj = nn.Linear(
            config.pathology_embedding_dim,
            config.shared_embedding_dim
        )
        self.path_value_proj = nn.Linear(
            config.pathology_embedding_dim,
            config.shared_embedding_dim
        )

        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.shared_embedding_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.shared_embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.shared_embedding_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.shared_embedding_dim, config.shared_embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.shared_embedding_dim * 4, config.shared_embedding_dim),
            nn.Dropout(config.dropout_rate)
        )

    def forward(
        self,
        ct_features: torch.Tensor,
        pathology_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-attention between CT and pathology features.

        Args:
            ct_features: CT feature tensor (B, ct_dim)
            pathology_features: Pathology feature tensor (B, path_dim)

        Returns:
            Tuple of (fused features, attention weights)
        """
        # Project features
        ct_query = self.ct_query_proj(ct_features)
        path_key = self.path_key_proj(pathology_features)
        path_value = self.path_value_proj(pathology_features)

        # Cross-attention
        attended, attention_weights = self.cross_attention(
            query=ct_query.unsqueeze(0),
            key=path_key.unsqueeze(0),
            value=path_value.unsqueeze(0)
        )

        # Residual connection and layer norm
        attended = self.layer_norm1(attended + ct_query.unsqueeze(0))

        # Feed-forward
        output = self.ffn(attended)
        output = self.layer_norm2(output + attended)

        return output.squeeze(0), attention_weights
```

---

## Domain Alignment

### GradientReversalLayer

```python
class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for domain adversarial training.

    Forward: Identity transformation (x -> x)
    Backward: Gradient reversal (dy/dx -> -lambda * dy/dx)

    Args:
        lambda_: Gradient reversal strength

    Example:
        >>> grl = GradientReversalLayer(lambda_=1.0)
        >>> output = grl(input)
    """

    def __init__(self, lambda_: float = 1.0):
        """Initialize gradient reversal layer."""
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (identity transformation).

        Args:
            x: Input tensor

        Returns:
            Same tensor as input
        """
        return x

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass (gradient reversal).

        Args:
            grad_output: Gradient from downstream layer

        Returns:
            Reversed gradient
        """
        return -self.lambda_ * grad_output
```

### DomainAdversarialNetwork

```python
class DomainAdversarialNetwork(nn.Module):
    """
    Domain discriminator with gradient reversal for adversarial alignment.

    Args:
        config: Configuration object

    Example:
        >>> aligner = DomainAdversarialNetwork(config)
        >>> loss, logits = aligner(features, domain_labels)
    """

    def __init__(self, config: Config):
        """Initialize domain adversarial network."""
        super().__init__()
        self.grl = GradientReversalLayer()

        self.discriminator = nn.Sequential(
            nn.Linear(config.shared_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 2)  # Source (CT) vs Target (Pathology)
        )

    def forward(
        self,
        features: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute domain classification with adversarial loss.

        Args:
            features: Feature tensor
            domain_labels: Optional domain labels (0=source, 1=target)

        Returns:
            Tuple of (domain loss, domain logits)
        """
        reversed_features = self.grl(features)
        domain_logits = self.discriminator(reversed_features)

        if domain_labels is not None:
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
        else:
            domain_loss = None

        return domain_loss, domain_logits
```

---

## Few-Shot Learning

### PrototypicalNetwork

```python
class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot classification.

    Computes class prototypes as mean of support set embeddings,
    then classifies queries by distance to prototypes.

    Args:
        config: Configuration object
        distance_metric: Distance metric ('euclidean', 'cosine')

    Example:
        >>> proto_net = PrototypicalNetwork(config)
        >>> logits, prototypes = proto_net(queries, support, support_labels)
    """

    def __init__(
        self,
        config: Config,
        distance_metric: str = 'euclidean'
    ):
        """Initialize prototypical network."""
        super().__init__()
        self.config = config
        self.distance_metric = distance_metric

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class prototypes from support set.

        Args:
            support_embeddings: Support set embeddings (N, D)
            support_labels: Support set labels (N,)

        Returns:
            Class prototypes tensor (C, D)
        """
        prototypes = {}
        for class_id in support_labels.unique():
            mask = support_labels == class_id
            prototypes[class_id.item()] = support_embeddings[mask].mean(dim=0)

        # Stack in order of class IDs
        sorted_ids = sorted(prototypes.keys())
        return torch.stack([prototypes[i] for i in sorted_ids])

    def forward(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Classify queries using prototypical representation.

        Args:
            query_embeddings: Query embeddings (Q, D)
            support_embeddings: Support embeddings (S, D) if episodic
            support_labels: Support labels (S,) if episodic

        Returns:
            Tuple of (logits, prototypes)
        """
        if support_embeddings is not None and support_labels is not None:
            # Compute prototypes
            prototypes = self.compute_prototypes(
                support_embeddings, support_labels
            )

            # Compute distances
            if self.distance_metric == 'euclidean':
                distances = torch.cdist(
                    query_embeddings.unsqueeze(0),
                    prototypes.unsqueeze(0)
                ).squeeze(0)
            elif self.distance_metric == 'cosine':
                queries_norm = F.normalize(query_embeddings, p=2, dim=1)
                proto_norm = F.normalize(prototypes, p=2, dim=1)
                distances = 1 - torch.mm(queries_norm, proto_norm.t())

            # Convert to logits
            logits = -distances
            return logits, prototypes
        else:
            return query_embeddings, None
```

---

## Training

### Trainer

```python
class Trainer:
    """
    Training manager for cross-dimensional transfer learning.

    Args:
        model: CrossDimensionalTransferModel
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        config: Configuration object

    Example:
        >>> trainer = Trainer(model, optimizer, scheduler, config)
        >>> trainer.train(train_loader, val_loader)
        >>> trainer.save_checkpoint('checkpoint.pth')
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Config
    ):
        """Initialize trainer."""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.history = defaultdict(list)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for batch in dataloader:
            loss, acc = self._train_step(batch)
            total_loss += loss
            total_acc += acc
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'acc': total_acc / num_batches
        }

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Single training step.

        Args:
            batch: Data batch dictionary

        Returns:
            Tuple of (loss, accuracy)
        """
        # Unpack batch
        ct_volume = batch['ct_volume'].to(self.config.device)
        pathology_image = batch['pathology_image'].to(self.config.device)
        labels = batch['label'].to(self.config.device)
        domain_labels = batch['domain_label'].to(self.config.device)

        # Forward pass
        logits, domain_loss = self.model(
            ct_volume, pathology_image, domain_labels
        )

        # Compute loss
        class_loss = F.cross_entropy(logits, labels)
        total_loss = class_loss + (
            self.config.domain_alignment_weight * domain_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Metrics
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        return total_loss.item(), acc.item()

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate model.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_acc = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                ct_volume = batch['ct_volume'].to(self.config.device)
                pathology_image = batch['pathology_image'].to(self.config.device)
                labels = batch['label'].to(self.config.device)

                # Forward pass
                logits, _ = self.model(ct_volume, pathology_image)

                # Loss
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()

                # Metrics
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                acc = (preds == labels).float().mean()
                total_acc += acc.item()

        # Compute additional metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return {
            'val_loss': total_loss / len(dataloader),
            'val_acc': total_acc / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='binary')
        }

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Checkpoint save path
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': dict(self.history),
            'config': self.config
        }

        torch.save(checkpoint, path)

        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load model checkpoint.

        Args:
            path: Checkpoint load path

        Returns:
            Epoch number to resume from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = defaultdict(list, checkpoint['history'])
        return checkpoint['epoch']
```

---

## Datasets

### CT3DVolumeDataset

```python
class CT3DVolumeDataset(Dataset):
    """
    Dataset for 3D CT volumes.

    Args:
        data_dir: Directory containing CT volumes
        config: Configuration object
        transform: Optional transform to apply

    Example:
        >>> dataset = CT3DVolumeDataset('./data/ct', config)
        >>> volume, label = dataset[0]
    """

    def __init__(
        self,
        data_dir: str,
        config: Config,
        transform: Optional[Callable] = None
    ):
        """Initialize CT dataset."""
        self.data_dir = Path(data_dir)
        self.config = config
        self.transform = transform

        # Load metadata
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """Load dataset samples from directory."""
        samples = []
        for case_dir in self.data_dir.iterdir():
            if case_dir.is_dir():
                ct_file = case_dir / "CT.nii.gz"
                if ct_file.exists():
                    samples.append({
                        'path': str(ct_file),
                        'case_id': case_dir.name
                    })
        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get dataset item.

        Args:
            idx: Sample index

        Returns:
            Tuple of (CT volume tensor, label)
        """
        sample = self.samples[idx]

        # Load and preprocess
        ct = sitk.ReadImage(sample['path'])
        volume = sitk.GetArrayFromImage(ct)

        # Convert to tensor
        volume = torch.from_numpy(volume).float()

        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Apply transform
        if self.transform:
            volume = self.transform(volume)

        # Get label (from filename or external metadata)
        label = self._get_label(sample['case_id'])

        return volume, label
```

### ct_dataset (Convenience Function)

```python
def ct_dataset(
    data_root: str = "./data",
    split: str = "train",
    num_shots: int = 5,
    transform: Optional[Callable] = None,
    **kwargs
) -> CT3DVolumeDataset:
    """
    Convenience function to create CT dataset for few-shot learning.

    Args:
        data_root: Root data directory
        split: Dataset split ('train', 'val', 'test')
        num_shots: Number of support samples per class
        transform: Optional transform
        **kwargs: Additional arguments for CT3DVolumeDataset

    Returns:
        CT3DVolumeDataset instance
    """
    data_dir = Path(data_root) / "ct_scans" / split
    config = Config()
    return CT3DVolumeDataset(str(data_dir), config, transform)
```

---

## Metrics

### calculate_accuracy

```python
def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)

    Returns:
        Accuracy score (0-1)
    """
```

### calculate_auc_roc

```python
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
```

### evaluate_episode

```python
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
        Dictionary with 'accuracy' and other metrics
    """
```

---

## Configuration

### Config

```python
@dataclass
class Config:
    """
    Central configuration class for cross-dimensional transfer.

    Attributes:
        # Model Architecture
        ct_embedding_dim: int = 768
        pathology_embedding_dim: int = 768
        shared_embedding_dim: int = 512
        num_attention_heads: int = 8
        num_cross_attention_layers: int = 2
        dropout_rate: float = 0.1

        # HAI-DEF Models
        medgemma_model_name: str = "google/medgemma-1.5-4b-it"
        medgemma_load_in_4bit: bool = True
        path_foundation_model_name: str = "google/path-foundation-v2"
        medsiglip_model_name: str = "google/medsiglip-448"

        # Training
        batch_size: int = 2
        num_epochs: int = 100
        learning_rate: float = 1e-4
        weight_decay: float = 0.01

        # Few-Shot Learning
        num_shots: int = 5
        num_query_samples: int = 15
        num_way: int = 2

        # Loss Weights
        contrastive_weight: float = 0.5
        prototypical_weight: float = 1.0
        domain_alignment_weight: float = 0.3
        semantic_alignment_weight: float = 0.2
    """
```

### get_config

```python
def get_config(
    model_size: str = "medium",
    dataset_size: str = "full",
    training_mode: str = "standard",
    **kwargs
) -> Config:
    """
    Get configuration with preset values and custom overrides.

    Args:
        model_size: Size preset ('small', 'medium', 'large')
        dataset_size: Dataset size ('full', 'lite', 'debug')
        training_mode: Training mode ('fast', 'standard', 'thorough')
        **kwargs: Additional custom parameters

    Returns:
        Config object with applied settings
    """
```

---

## Imports

```python
# Core imports
from cross_dim_transfer import CrossDimensionalTransferModel
from cross_dim_transfer.configs import Config, get_config

# HAI-DEF interfaces
from cross_dim_transfer.haidef import (
    MedGemmaInterface,
    PathFoundationInterface,
    MedSigLIPInterface
)

# Training
from cross_dim_transfer.training import Trainer
from cross_dim_transfer.training.losses import (
    ContrastiveLoss,
    PrototypicalLoss,
    DomainAlignmentLoss,
    TotalLoss
)

# Datasets
from cross_dim_transfer.data import CT3DVolumeDataset, ct_dataset

# Utilities
from cross_dim_transfer.utils import (
    calculate_accuracy,
    calculate_auc_roc,
    evaluate_episode,
    plot_training_history
)
```
