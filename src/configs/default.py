"""
Default configuration for cross-dimensional knowledge transfer.

This module provides configuration classes and preset configurations
for training cross-dimensional knowledge transfer models.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class Config:
    """
    Configuration class for cross-dimensional knowledge transfer.

    This class encapsulates all hyperparameters and settings for:
    - Model architecture (feature dimensions, attention heads, etc.)
    - Training settings (batch size, learning rate, epochs)
    - Data settings (paths, transforms, augmentations)
    - HAI-DEF model settings (paths, quantization, etc.)
    """
    # Model Architecture
    # ==================
    ct_embedding_dim: int = 768
    pathology_embedding_dim: int = 768
    shared_embedding_dim: int = 512
    num_attention_heads: int = 8
    num_cross_attention_layers: int = 2
    dropout_rate: float = 0.1
    projection_layers: int = 2

    # HAI-DEF Model Settings
    # ======================
    medgemma_model_name: str = "google/medgemma-1.5-4b-it"
    medgemma_feature_dim: int = 768
    medgemma_max_new_tokens: int = 128
    medgemma_temperature: float = 0.7
    medgemma_use_quantization: bool = True
    medgemma_load_in_4bit: bool = True

    path_foundation_model_name: str = "google/path-foundation-v2"
    path_foundation_feature_dim: int = 768

    medsiglip_model_name: str = "google/medsiglip-448"

    # Dataset Settings
    # ================
    data_root: str = "./data"
    ct_data_path: str = "./data/ct_scans"
    pathology_data_path: str = "./data/pathology"
    num_ct_slices: int = 64
    ct_slice_spacing: float = 1.0
    target_ct_size: tuple = (64, 64, 64)
    target_pathology_size: tuple = (224, 224)
    num_pathology_channels: int = 3

    # Training Settings
    # =================
    batch_size: int = 2
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Few-Shot Learning Settings
    # ==========================
    num_shots: int = 5
    num_query_samples: int = 15
    num_way: int = 2
    episode_batch_size: int = 4

    # Loss Weights
    # ============
    contrastive_weight: float = 0.5
    prototypical_weight: float = 1.0
    domain_alignment_weight: float = 0.3
    semantic_alignment_weight: float = 0.2
    reconstruction_weight: float = 0.1

    # Optimization
    # ============
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1

    # Evaluation
    # ==========
    eval_frequency: int = 1
    num_eval_episodes: int = 100
    k_shot_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    cross_validation_folds: int = 3

    # Checkpointing
    # =============
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 10
    best_model_metric: str = "val_acc"

    # Logging
    # =======
    log_frequency: int = 10
    use_wandb: bool = True
    wandb_project: str = "cross-dim-transfer"

    # Hardware
    # ========
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True

    # Experiment
    # ==========
    experiment_name: str = "cross_dim_transfer"
    seed: int = 42
    debug: bool = False


# Preset Configurations
# =====================

MODEL_CONFIGS: Dict[str, Dict] = {
    "small": {
        "ct_embedding_dim": 512,
        "pathology_embedding_dim": 512,
        "shared_embedding_dim": 256,
        "num_attention_heads": 4,
        "num_cross_attention_layers": 1,
    },
    "medium": {
        "ct_embedding_dim": 768,
        "pathology_embedding_dim": 768,
        "shared_embedding_dim": 512,
        "num_attention_heads": 8,
        "num_cross_attention_layers": 2,
    },
    "large": {
        "ct_embedding_dim": 1024,
        "pathology_embedding_dim": 1024,
        "shared_embedding_dim": 768,
        "num_attention_heads": 12,
        "num_cross_attention_layers": 3,
    }
}

DATASET_CONFIGS: Dict[str, Dict] = {
    "full": {
        "num_ct_slices": 64,
        "target_ct_size": (64, 64, 64),
        "target_pathology_size": (224, 224),
    },
    "lite": {
        "num_ct_slices": 32,
        "target_ct_size": (32, 32, 32),
        "target_pathology_size": (128, 128),
    },
    "debug": {
        "num_ct_slices": 16,
        "target_ct_size": (16, 16, 16),
        "target_pathology_size": (64, 64),
    }
}

TRAINING_CONFIGS: Dict[str, Dict] = {
    "fast": {
        "num_epochs": 20,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "num_eval_episodes": 50,
    },
    "standard": {
        "num_epochs": 100,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "num_eval_episodes": 100,
    },
    "thorough": {
        "num_epochs": 200,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "num_eval_episodes": 200,
    }
}


def get_config(
    model_size: str = "medium",
    dataset_size: str = "full",
    training_mode: str = "standard",
    **kwargs
) -> Config:
    """
    Get a configuration with preset values and custom overrides.

    Args:
        model_size: Size preset ('small', 'medium', 'large')
        dataset_size: Dataset size preset ('full', 'lite', 'debug')
        training_mode: Training mode preset ('fast', 'standard', 'thorough')
        **kwargs: Additional custom parameters

    Returns:
        Config object with applied settings
    """
    # Start with default config
    config = Config()

    # Apply model preset
    if model_size in MODEL_CONFIGS:
        for key, value in MODEL_CONFIGS[model_size].items():
            setattr(config, key, value)

    # Apply dataset preset
    if dataset_size in DATASET_CONFIGS:
        for key, value in DATASET_CONFIGS[dataset_size].items():
            setattr(config, key, value)

    # Apply training preset
    if training_mode in TRAINING_CONFIGS:
        for key, value in TRAINING_CONFIGS[training_mode].items():
            setattr(config, key, value)

    # Apply custom overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# Default configuration instance
DEFAULT_CONFIG = Config()


def print_config(config: Config, indent: int = 0) -> None:
    """
    Print configuration in a formatted way.

    Args:
        config: Configuration object to print
        indent: Number of spaces for indentation
    """
    prefix = " " * indent
    print(f"{prefix}Cross-Dimensional Transfer Configuration:")
    print(f"{prefix}{'='*50}")

    categories = {
        "Model Architecture": [
            "ct_embedding_dim", "pathology_embedding_dim", "shared_embedding_dim",
            "num_attention_heads", "num_cross_attention_layers", "dropout_rate"
        ],
        "HAI-DEF Models": [
            "medgemma_model_name", "path_foundation_model_name", "medsiglip_model_name",
            "medgemma_use_quantization", "medgemma_load_in_4bit"
        ],
        "Dataset": [
            "data_root", "num_ct_slices", "target_ct_size",
            "target_pathology_size", "num_pathology_channels"
        ],
        "Training": [
            "num_epochs", "batch_size", "learning_rate", "weight_decay",
            "gradient_accumulation_steps"
        ],
        "Few-Shot Learning": [
            "num_shots", "num_query_samples", "num_way", "episode_batch_size"
        ],
        "Loss Weights": [
            "contrastive_weight", "prototypical_weight", "domain_alignment_weight",
            "semantic_alignment_weight"
        ],
        "Evaluation": [
            "eval_frequency", "num_eval_episodes", "k_shot_values",
            "cross_validation_folds"
        ],
        "Hardware": [
            "device", "num_workers", "pin_memory"
        ]
    }

    for category, keys in categories.items():
        print(f"\n{prefix}{category}:")
        for key in keys:
            value = getattr(config, key)
            print(f"{prefix}  {key}: {value}")

    print(f"\n{prefix}{'='*50}")
