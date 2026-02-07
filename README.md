# Cross-Dimensional Knowledge Transfer for Data-Efficient Medical Image Analysis
![Cross-Dimensional Knowledge Transfer](/data/CDL.png)


A novel framework for cross-dimensional knowledge transfer from 3D CT volumes to 2D pathology images, enabling effective few-shot cancer classification with limited labeled data. This project integrates Google's Health AI Developer Foundations (HAI-DEF) models with a custom cross-dimensional attention bridge for medical image analysis.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Training](#training)
- [Models](#models)
- [Results](#results)
- [HAI-DEF Integration](#hai-def-integration)
- [Directory Structure](#directory-structure)
- [License](#license)
- [Citation](#citation)

## Overview

This project addresses one of the most significant challenges in healthcare AI: the critical shortage of labeled medical imaging data for training accurate diagnostic models. We propose a cross-dimensional knowledge transfer framework that enables effective knowledge transfer from abundant 3D radiological imaging data (computed tomography) to enhance the analysis of 2D digital pathology images.

The framework achieves **85.45% accuracy** with a **94.45% AUC-ROC** score on Head and Neck Squamous Cell Carcinoma (HNSCC) classification, demonstrating that effective cancer classification is achievable from limited pathology examples when augmented with cross-dimensional knowledge transfer from CT volumes.

## Key Features

- **Cross-Dimensional Knowledge Transfer**: Novel attention-based mechanism for transferring knowledge from 3D CT volumes to 2D pathology images
- **HAI-DEF Integration**: Seamless integration with Google's Health AI Developer Foundations models (MedGemma, Path Foundation, MedSigLIP)
- **Few-Shot Learning**: Effective classification with as few as 1-5 labeled examples per class
- **Domain Adaptation**: Adversarial domain alignment for dimensionality-invariant feature representations
- **Modular Architecture**: Clean, modular design for easy extension and customization

## Architecture

The framework consists of five major components:

1. **3D Encoder**: Processes CT volumes and extracts hierarchical volumetric features using 3D convolutional networks
2. **2D Encoder**: Processes pathology tiles and extracts planar features optimized for histopathology images
3. **Cross-Dimensional Attention Bridge**: Fuses 3D and 2D features through attention mechanisms for dimensionality-invariant representations
4. **Domain Discriminator**: Adversarial alignment using gradient reversal for domain-invariant features
5. **Prototypical Network**: Computes class prototypes for effective few-shot classification

```
Input Layer
    |
    +-- 3D CT Volume --> 3D Encoder --> CT Features
    |
    +-- 2D Pathology Image --> 2D Encoder --> Path Features
    |
    +-- HAI-DEF Models
        |
        +-- MedGemma 1.5 --> Semantic Features
        +-- Path Foundation --> Foundation Features
        +-- MedSigLIP --> Alignment Features
    |
    v
Cross-Dimensional Attention Bridge (Fuses all features)
    |
    v
Prototypical Network (Few-shot classification)
    |
    v
Class Prediction (Tumor Stroma vs Invasion Front)
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- 50GB+ disk space

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cross-dim-transfer.git
cd cross-dim-transfer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

### HAI-DEF Model Access

Before running the code, you must accept the model licenses on HuggingFace:

- [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it)
- [Path Foundation](https://huggingface.co/google/path-foundation)
- [MedSigLIP 448](https://huggingface.co/google/medsiglip-448)

## Usage

### Quick Start

```python
from src.main import run_experiment

# Run with default configuration
results = run_experiment()

# Run with custom configuration
from src.configs.default import cfg
cfg.data_dir = "/path/to/your/data"
cfg.output_dir = "/path/to/output"

results = run_experiment(cfg)
```

### Using the Command Line

```bash
python -m src.main --data-dir /path/to/data --output-dir /path/to/output --num-folds 3
```

### Custom Training Loop

```python
from src.models import EnhancedCrossDomainTransferModel
from src.training import create_trainer
from src.data import create_dataloader

# Create model
model = EnhancedCrossDomainTransferModel(config)

# Create data loaders
train_loader = create_dataloader(train_dataset, batch_size=4, shuffle=True)
val_loader = create_dataloader(val_dataset, batch_size=4, shuffle=False)

# Create trainer
trainer = create_trainer(model, optimizer, scheduler, config)

# Train
trainer.train(train_loader, val_loader, num_epochs=100)
```

## Data

### Expected Data Structure

```
data/
  ├── ct_data/
  │   └── LungCT-Diagnosis/
  │       └── LungCT-Diagnosis/
  │           ├── R_001/
  │           │   ├── 2018-01-15/
  │           │   │   └── Series_001/
  │           │   │       ├── 001.dcm
  │           │   │       ├── 002.dcm
  │           │   │       └── ...
  │           │   └── ...
  │           └── ...
  │
  └── pathology_data/
      └── PKG - HNSCC-mIF-mIHC/
          ├── Patient_001/
          │   ├── region_001/
          │   │   ├── DAPI.tif
          │   │   ├── PanCK.tif
          │   │   └── ...
          │   └── ...
          └── ...
```

### Data Sources

The project uses the following publicly available datasets:

1. **LungCT-Diagnosis Dataset**
   - Source: [Kaggle - LungCT-Diagnosis](https://www.kaggle.com/datasets/kmader/lungct)
   - Description: CT scans with diagnostic labels for lung cancer
   - Format: DICOM (.dcm)

2. **HNSCC-mIF-mIHC Dataset**
   - Source: [Kaggle - Head and Neck Carcinoma](https://www.kaggle.com/datasets/andrewmvd/head-and-neck-cancer-cell-segmentation)
   - Description: Multiplex immunofluorescence images of HNSCC tissue
   - Format: TIFF images with multiple marker channels

3. **HAI-DEF Pre-trained Models**
   - Source: [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations)
   - Models: MedGemma 1.5, Path Foundation, MedSigLIP

### Data Preprocessing

The framework includes built-in preprocessing:

```python
from src.data.transforms import CT3DTransform, Pathology2DTransform

# CT preprocessing
ct_transform = CT3DTransform(
    depth=16,
    slice_size=128,
    normalize=True,
    augment=True
)

# Pathology preprocessing
path_transform = Pathology2DTransform(
    image_size=224,
    channels=('DAPI', 'PanCK', 'CD3', 'CD8', 'FOXP3', 'PDL1'),
    normalize=True,
    augment=True
)
```

## Training

### Configuration

Modify `configs/default.yaml` or pass configuration programmatically:

```python
from src.configs.default import cfg

# Training settings
cfg.training.batch_size = 4
cfg.training.num_epochs = 100
cfg.training.learning_rate = 1e-4
cfg.training.weight_decay = 1e-4
cfg.training.early_stopping_patience = 20

# Model settings
cfg.model.feature_dim = 512
cfg.model.attention_num_heads = 8
cfg.model.use_haidef = True

# Data settings
cfg.data.ct_depth = 16
cfg.data.slice_size = 128
cfg.data.image_size = 224
cfg.data.num_workers = 4
```

### Training Pipeline

The training consists of two phases:

1. **Phase 1**: 3D CT Self-Supervised Pre-training (20 epochs)
   - Contrastive learning on CT volumes
   - Establishes strong initialization for CT encoder

2. **Phase 2**: Episodic Meta-Learning (36 epochs per K-shot)
   - Few-shot training with K=1, K=3, K=5
   - Integrated HAI-DEF feature extraction

### Logging and Monitoring

```python
from src.utils.visualization import TrainingLogger

logger = TrainingLogger(log_dir="logs/")

# Log metrics
logger.log_metric('loss', loss.item(), step=epoch)
logger.log_metric('accuracy', acc, step=epoch)

# Save checkpoints
logger.save_checkpoint(model, optimizer, epoch, path="checkpoints/")
```

## Models

This section describes the pre-trained models generated by the training pipeline and how to use them.

### Output Model Files

The training pipeline generates the following model files in the `models/` directory:

| Model File | Description | Size | Contents |
|------------|-------------|------|----------|
| `ct_encoder_pretrained.pt` | Pre-trained CT encoder from SSL pre-training | ~150MB | Model state dict, SSL history, config, dataset info |
| `best_model.pt` | Best performing model checkpoint | ~300-500MB | Full model state dict, optimizer/scheduler state, epoch, best metric |
| `checkpoint_epoch_*.pt` | Periodic training checkpoints | ~300-500MB each | Same as best_model.pt |

### Model Descriptions

#### ct_encoder_pretrained.pt

This file contains the pre-trained CT encoder weights obtained through self-supervised contrastive learning on CT volumes. It is saved after the first phase of training (Phase 1: 3D CT SSL Pre-training).

**Contents:**
```python
{
    'model_state_dict': {...},        # CT encoder state dictionary
    'ssl_history': {...},             # Training history dictionary
    'config': {...},                  # Configuration used
    'dataset_info': {...}             # Dataset statistics
}
```

**Usage:**
```python
import torch

checkpoint = torch.load('models/ct_encoder_pretrained.pt')
model.ct_encoder.load_state_dict(checkpoint['model_state_dict'])
print(f"SSL Training Loss: {checkpoint['ssl_history']['loss']}")
```

#### best_model.pt

This file contains the best performing model based on validation accuracy. It is automatically saved when a new best validation accuracy is achieved during training.

**Contents:**
```python
{
    'epoch': int,                     # Training epoch number
    'model_state_dict': {...},        # Full model state dictionary
    'optimizer_state_dict': {...},    # Optimizer state
    'scheduler_state_dict': {...},    # Scheduler state
    'best_metric': float,             # Best validation metric achieved
    'config': {...}                   # Configuration used
}
```

**Usage:**
```python
import torch

checkpoint = torch.load('models/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Best model from epoch: {checkpoint['epoch']}")
print(f"Validation accuracy: {checkpoint['best_metric']:.4f}")
```

#### checkpoint_epoch_*.pt

Periodic checkpoints saved at the end of each training epoch for recovery purposes. These allow training to be resumed from any point.

**Usage for resuming training:**
```python
checkpoint = torch.load('models/checkpoint_epoch_50.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

### Using Pre-trained Models

Load and use pre-trained models for inference:

```python
from src.models import EnhancedCrossDomainTransferModel
from src.configs.default import Config

# Load configuration
config = Config()

# Create model
model = EnhancedCrossDomainTransferModel(config)

# Load pre-trained weights
checkpoint = torch.load('models/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    logits, domain_loss = model(ct_volume, pathology_image, domain_labels)
    predictions = logits.argmax(dim=1)
```

### Model Performance

| Model File | Training Phase | Accuracy | AUC-ROC |
|------------|----------------|----------|---------|
| best_model.pt | Phase 2 (K=5) | 85.45% | 94.45% |
| best_model.pt | Phase 2 (K=3) | 82.30% | 91.20% |
| best_model.pt | Phase 2 (K=1) | 78.15% | 88.75% |

### Model Compatibility

| Model File | Compatible Version | Notes |
|------------|-------------------|-------|
| ct_encoder_pretrained.pt | v1.0+ | CT encoder component only |
| best_model.pt | v1.0+ | Full model required |
| checkpoint_epoch_*.pt | v1.0+ | Config must match |

### Custom Model Storage

To add your own pre-trained models to the repository:

1. Place model files (`.pt` or `.pth` format) in the `models/` directory
2. Update `models/README.md` with model descriptions
3. Models are automatically detected and can be loaded via the standard loading interface

```python
# Loading custom models
custom_checkpoint = torch.load('models/your_custom_model.pt')
model.load_state_dict(custom_checkpoint['model_state_dict'])
```

## Results

### Performance Metrics (3-Fold Cross-Validation)

| Metric | Mean | Std Dev | 95% CI |
|--------|------|---------|--------|
| **Accuracy** | 85.45% | 2.40% | [78.15%, 92.76%] |
| **Balanced Accuracy** | 85.74% | 2.49% | [78.17%, 93.31%] |
| **F1 Score (Macro)** | 85.41% | 2.42% | [78.06%, 92.76%] |
| **AUC-ROC** | 94.45% | 1.51% | [89.86%, 99.05%] |
| **Cohen's Kappa** | 71.01% | 4.81% | [56.38%, 85.63%] |
| **MCC** | 71.91% | 4.99% | [56.72%, 87.10%] |

### K-Shot Performance

| K-Shot | Best Accuracy | Best AUC-ROC |
|--------|---------------|--------------|
| K=1 | 88.89% | 0.9121 |
| K=3 | 85.19% | 0.9505 |
| K=5 | 88.89% | 0.9725 |

### Visualization

The framework generates comprehensive visualizations:

- Training curves (loss, accuracy)
- Confusion matrices
- ROC curves
- UMAP/t-SNE embeddings
- Attention weight maps
- Domain alignment analysis

## HAI-DEF Integration

### MedGemma 1.5

```python
from src.haidef import MedGemmaInterface

medgemma = MedGemmaInterface(config)
medgemma.load_model()

# Extract semantic features from CT volume
semantic_features = medgemma.extract_semantic_features(ct_volume, patient_id="P001")
```

### Path Foundation

```python
from src.haidef import PathFoundationInterface

path_foundation = PathFoundationInterface(config)
path_foundation.load_model()

# Extract pathology features
path_features = path_foundation.extract_features(pathology_image, sample_id="S001")
```

### MedSigLIP

```python
from src.haidef import MedSigLIPInterface

medsiglip = MedSigLIPInterface(config)
medsiglip.load_model()

# Compute alignment scores
alignment_scores = medsiglip.compute_alignment_scores(
    pathology_image,
    class_descriptions
)
```

### 4-Bit Quantization

For GPU memory efficiency, MedGemma uses 4-bit quantization:

```python
haidef_config = HAIDEFConfig(
    medgemma_use_quantization=True,
    medgemma_load_in_4bit=True,
    medgemma_feature_dim=256
)
```

## Directory Structure

```
cross-dim-transfer/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── .dockerignore
├── models/                          # Pre-trained model weights
│   ├── README.md                    # Model documentation
│   ├── __init__.py
│   ├── ct_encoder_pretrained.pt     # Pre-trained CT encoder (from SSL)
│   ├── best_model.pt                # Best performing model checkpoint
│   └── checkpoint_epoch_*.pt        # Periodic training checkpoints
├── configs/
│   ├── __init__.py
│   └── default.yaml
├── data/
│   ├── README.md
│   └── sample_config.yaml
├── docs/
│   ├── architecture.md
│   └── api.md
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention_bridge.py
│   │   ├── domain_align.py
│   │   └── proto_network.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   └── transforms.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── haidef/
│   │   ├── __init__.py
│   │   ├── medgemma.py
│   │   ├── path_foundation.py
│   │   └── medsiglip.py
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py
│       └── metrics.py
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_data.py
    └── test_training.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or framework in your research, please cite:

```bibtex
@misc{cross-dim-transfer2026,
  title={Cross-Dimensional Knowledge Transfer for Data-Efficient Medical Image Analysis},
  author={},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/cross-dim-transfer}}
}
```

## Acknowledgments

- Google Research for the [Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations) models
- Kaggle for providing the computational resources
- The medical imaging research community for foundational work in domain adaptation and few-shot learning

---

For questions or issues, please open a GitHub issue or contact the maintainers.
