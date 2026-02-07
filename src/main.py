"""
Main Entry Point for Cross-Dimensional Knowledge Transfer

This module provides the primary interface for running experiments
with the cross-dimensional knowledge transfer framework.
"""

import os
import sys
import random
import time
import json
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Set
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import pandas as pd
import pydicom
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Function
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, roc_curve,
    cohen_kappa_score, matthews_corrcoef, precision_score, recall_score,
    balanced_accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
import bitsandbytes as bnb
import umap
import yaml

warnings.filterwarnings('ignore')
matplotlib.use('Agg')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HAIDEFConfig:
    """Configuration for HAI-DEF model integration."""
    medgemma_model_name: str = "google/medgemma-1.5-4b-it"
    medgemma_device: str = "auto"
    medgemma_max_new_tokens: int = 256
    medgemma_temperature: float = 0.7
    medgemma_use_quantization: bool = True
    medgemma_load_in_4bit: bool = True
    medgemma_feature_dim: int = 256

    path_foundation_model_name: str = "google/path-foundation"
    path_foundation_feature_dim: int = 512
    path_foundation_pretrained: bool = True

    medsiglip_model_name: str = "google/medsiglip-448"
    medsiglip_feature_dim: int = 512

    fusion_strategy: str = "attention"
    attention_num_heads: int = 8
    feature_projection_dim: int = 256

    cls_loss_weight: float = 1.0
    proto_loss_weight: float = 0.5
    adv_loss_weight: float = 0.3
    contrastive_loss_weight: float = 0.1
    semantic_alignment_weight: float = 0.15

    extract_ct_slices: int = 3
    cache_features: bool = True
    cache_dir: str = "./haidef_cache"

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)


@dataclass
class ResearchConfig:
    """Main research configuration."""
    ct_data_dir: str = '/kaggle/input/lungct/LungCT-Diagnosis/LungCT-Diagnosis'
    pathology_data_dir: str = '/kaggle/input/head-and-neck-carcinoma/PKG - HNSCC-mIF-mIHC'
    output_dir: str = './medgemma_cross_dim_transfer'

    image_size: int = 224
    slice_size: int = 128
    ct_volume_depth: int = 16
    channels: Tuple[str, ...] = ('DAPI', 'PanCK', 'CD3', 'CD8', 'FOXP3', 'PDL1')

    n_splits: int = 3
    seed: int = 42
    batch_size: int = 4
    num_workers: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_haidef: bool = True
    haidef_config: HAIDEFConfig = field(default_factory=HAIDEFConfig)

    n_shots: List[int] = field(default_factory=lambda: [1, 3, 5])
    n_query: int = 5
    n_support: int = 5
    meta_lr: float = 1e-3
    meta_epochs: int = 36
    inner_lr: float = 0.01
    inner_steps: int = 5

    ssl_epochs: int = 20
    ssl_augmentation_strength: float = 0.5
    ssl_batch_size: int = 4
    ssl_learning_rate: float = 1e-4
    ssl_weight_decay: float = 1e-4
    ssl_temperature: float = 0.1

    early_stopping_patience: int = 20
    gradient_clip_norm: float = 1.0

    feature_dim: int = 512
    hidden_dim: int = 1024

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'analysis'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)


CFG = ResearchConfig()


# =============================================================================
# LOGGING
# =============================================================================

def log_message(message: str, logger_file=None, to_console: bool = True) -> None:
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    if to_console:
        print(formatted)
    if logger_file:
        logger_file.write(formatted + "\n")
        logger_file.flush()


def clear_memory() -> None:
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# HAI-DEF MODEL INTERFACES
# =============================================================================

class MedGemmaInterface:
    """Interface for MedGemma 1.5 model integration."""

    def __init__(self, config: HAIDEFConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device()
        self.feature_cache = {}

    def _setup_device(self) -> str:
        """Setup appropriate device for model inference."""
        if self.config.medgemma_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.medgemma_device

    def load_model(self):
        """Load MedGemma 1.5 model and tokenizer."""
        log_message(f"Loading MedGemma model: {self.config.medgemma_model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.medgemma_model_name)

        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": self.device
        }

        if self.config.medgemma_use_quantization and self.config.medgemma_load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["bnb_4bit_quant_type"] = "nf4"
            load_kwargs["bnb_4bit_compute_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.medgemma_model_name,
            **load_kwargs
        )

        self.model.eval()
        log_message("MedGemma model loaded successfully")

    def generate_ct_description(self, ct_slice: np.ndarray, patient_id: str = "unknown") -> str:
        """Generate clinical description for a CT slice."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("MedGemma model not loaded. Cannot generate CT description.")

        cache_key = f"{patient_id}_{hash(ct_slice.tobytes()) % 10000}"
        if cache_key in self.feature_cache and 'description' in self.feature_cache[cache_key]:
            return self.feature_cache[cache_key]['description']

        slice_8bit = (np.clip(ct_slice, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(slice_8bit).convert("RGB")

        prompt = """Analyze this CT slice and provide a brief clinical description:

Focus on:
- Anatomical structures visible
- Any abnormal findings (nodules, opacities)
- Tissue density patterns

Clinical summary:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.medgemma_max_new_tokens,
                temperature=self.config.medgemma_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        description = response[len(prompt):].strip()

        if cache_key not in self.feature_cache:
            self.feature_cache[cache_key] = {}
        self.feature_cache[cache_key]['description'] = description

        return description

    def extract_semantic_features(self, ct_volume: np.ndarray, patient_id: str = "unknown") -> torch.Tensor:
        """Extract semantic features from CT volume using MedGemma."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("MedGemma model not loaded. Cannot extract semantic features.")

        cache_key = f"{patient_id}_semantic_features"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        depth = ct_volume.shape[0]
        mid_slice_idx = depth // 2

        slice_indices = [0, mid_slice_idx, depth - 1]
        slice_indices = [i for i in slice_indices if 0 <= i < depth]

        descriptions = []
        for idx in slice_indices:
            description = self.generate_ct_description(ct_volume[idx], patient_id)
            descriptions.append(description)

        features = self._texts_to_features(descriptions)

        self.feature_cache[cache_key] = features
        return features

    def _texts_to_features(self, texts: List[str]) -> torch.Tensor:
        """Convert list of texts to feature tensor using MedGemma tokenizer embeddings."""
        all_embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                all_embeddings.append(pooled.squeeze(0))

        combined = torch.cat(all_embeddings, dim=0)
        if combined.shape[0] != self.config.medgemma_feature_dim:
            if not hasattr(self, '_feature_proj'):
                self._feature_proj = nn.Linear(combined.shape[0], self.config.medgemma_feature_dim).to(self.device)
            features = self._feature_proj(combined)
        else:
            features = combined

        return features

    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache.clear()


class PathFoundationInterface:
    """Interface for Path Foundation model integration."""

    def __init__(self, config: HAIDEFConfig):
        self.config = config
        self.model = None
        self.preprocess = None
        self.feature_cache = {}

    def load_model(self):
        """Load Path Foundation model."""
        log_message(f"Loading Path Foundation model: {self.config.path_foundation_model_name}")

        self.model = AutoModel.from_pretrained(
            self.config.path_foundation_model_name,
            torch_dtype=torch.float16
        )
        self.model.eval()

        log_message("Path Foundation model loaded successfully")

    def extract_features(self, pathology_image: np.ndarray, sample_id: str = "unknown") -> torch.Tensor:
        """Extract features from pathology image using Path Foundation."""
        if self.model is None:
            raise RuntimeError("Path Foundation model not loaded. Cannot extract features.")

        cache_key = f"{sample_id}_path_features"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        if pathology_image.shape[0] != 3:
            if pathology_image.ndim == 3 and pathology_image.shape[2] == 3:
                image_tensor = torch.from_numpy(pathology_image).permute(2, 0, 1).float()
            else:
                raise ValueError(f"Unexpected image shape: {pathology_image.shape}")
        else:
            image_tensor = torch.from_numpy(pathology_image).float()

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor.unsqueeze(0))
            features = outputs.last_hidden_state.mean(dim=1).squeeze()

        self.feature_cache[cache_key] = features
        return features

    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache.clear()


class MedSigLIPInterface:
    """Interface for MedSigLIP model integration."""

    def __init__(self, config: HAIDEFConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.feature_cache = {}

    def load_model(self):
        """Load MedSigLIP model and processor."""
        log_message(f"Loading MedSigLIP model: {self.config.medsiglip_model_name}")

        self.processor = AutoProcessor.from_pretrained(self.config.medsiglip_model_name)
        self.model = AutoModel.from_pretrained(
            self.config.medsiglip_model_name,
            torch_dtype=torch.float16
        )
        self.model.eval()

        log_message("MedSigLIP model loaded successfully")

    def compute_alignment_scores(self, image: np.ndarray, text_descriptions: List[str]) -> torch.Tensor:
        """Compute alignment scores between image and text descriptions."""
        if self.model is None or self.processor is None:
            raise RuntimeError("MedSigLIP model not loaded. Cannot compute alignment scores.")

        inputs = self.processor(
            text=text_descriptions,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image

        return logits_per_image.squeeze()

    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache.clear()


# =============================================================================
# HAI-DEF ENHANCED ARCHITECTURE
# =============================================================================

class EnhancedCrossDimensionalAttentionBridge(nn.Module):
    """Enhanced Cross-Dimensional Attention Bridge with HAI-DEF model integration."""

    def __init__(self, config: ResearchConfig):
        super().__init__()

        self.config = config
        haidef_config = config.haidef_config
        original_feature_dim = config.feature_dim
        proj_dim = haidef_config.feature_projection_dim

        self.medgemma_projection = nn.Linear(haidef_config.medgemma_feature_dim, proj_dim)
        self.path_foundation_projection = nn.Linear(haidef_config.path_foundation_feature_dim, proj_dim)
        self.medsiglip_projection = nn.Linear(haidef_config.medsiglip_feature_dim, proj_dim)

        self.original_projection = nn.Linear(original_feature_dim, proj_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=haidef_config.attention_num_heads,
            batch_first=True
        )

        self.fusion_weights = nn.Parameter(torch.ones(5) / 5)

    def forward(
        self,
        ct_features,
        path_features,
        medgemma_features=None,
        path_foundation_features=None,
        medsiglip_features=None
    ) -> Tuple[torch.Tensor, Dict]:
        info = {}

        features = [self.original_projection(ct_features.unsqueeze(0))]
        info['original_ct'] = ct_features.detach().cpu().numpy()

        if path_features is not None:
            features.append(self.original_projection(path_features.unsqueeze(0)))
            info['original_path'] = path_features.detach().cpu().numpy()

        if medgemma_features is not None:
            medgemma_proj = self.medgemma_projection(medgemma_features)
            features.append(medgemma_proj.unsqueeze(0))
            info['medgemma'] = medgemma_features.detach().cpu().numpy()

        if path_foundation_features is not None:
            pf_proj = self.path_foundation_projection(path_foundation_features)
            features.append(pf_proj.unsqueeze(0))
            info['path_foundation'] = path_foundation_features.detach().cpu().numpy()

        if medsiglip_features is not None:
            ms_proj = self.medsiglip_projection(medsiglip_features)
            features.append(ms_proj.unsqueeze(0))
            info['medsiglip'] = medsiglip_features.detach().cpu().numpy()

        stacked = torch.cat(features, dim=1)

        weights = F.softmax(self.fusion_weights, dim=0)
        info['fusion_weights'] = weights.detach().cpu().numpy()

        weighted = stacked * weights.view(1, -1, 1)
        fused = weighted.sum(dim=1)

        return fused.squeeze(0), info


class SemanticAlignmentLoss(nn.Module):
    """Semantic alignment loss using MedSigLIP for multimodal training."""

    def __init__(self, config: ResearchConfig):
        super().__init__()
        self.config = config
        self.haidef_config = config.haidef_config

        self.class_descriptions = [
            "Tumor stroma region showing organized connective tissue with cancer-associated fibroblasts",
            "Invasion front region showing irregular tumor glands breaking through basement membrane"
        ]

    def forward(
        self,
        learned_features: torch.Tensor,
        medsiglip_interface: MedSigLIPInterface,
        pathology_images: Optional[np.ndarray] = None,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if medsiglip_interface is None or medsiglip_interface.model is None:
            raise RuntimeError("MedSigLIP interface not available. Cannot compute semantic alignment loss.")

        if pathology_images is None:
            raise RuntimeError("Pathology images required for semantic alignment computation.")

        if target_labels is None:
            raise RuntimeError("Target labels required for semantic alignment computation.")

        alignment_scores = medsiglip_interface.compute_alignment_scores(
            pathology_images, self.class_descriptions
        )
        alignment_scores = alignment_scores.to(learned_features.device)

        loss = F.cross_entropy(alignment_scores, target_labels)
        return loss


class EnhancedCrossDomainTransferModel(nn.Module):
    """Enhanced Cross-Domain Transfer Model with HAI-DEF integration."""

    def __init__(self, config: ResearchConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim

        self.ct_encoder = self._create_ct_encoder()
        self.path_encoder = self._create_path_encoder()

        self.attention_bridge = EnhancedCrossDimensionalAttentionBridge(config)

        self.semantic_alignment_loss = SemanticAlignmentLoss(config)

        self.attention_weights = nn.Parameter(torch.tensor(1.0))
        self.domain_adv_weight = 0.3

    def _create_ct_encoder(self) -> nn.Module:
        """Create 3D CNN encoder for CT volumes."""
        return nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

    def _create_path_encoder(self) -> nn.Module:
        """Create 2D CNN encoder for pathology images."""
        return nn.Sequential(
            nn.Conv2d(len(self.config.channels), 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def set_haidef_interfaces(self, medgemma, path_foundation, medsiglip):
        """Set HAI-DEF interfaces for the model."""
        self.medgemma_interface = medgemma
        self.path_foundation_interface = path_foundation
        self.medsiglip_interface = medsiglip

    def extract_haidef_features(self, ct_volumes, path_images, patient_ids=None) -> Dict:
        """Extract features from all HAI-DEF models."""
        features = {}

        if self.medgemma_interface is not None:
            batch_medgemma = []
            for i, vol in enumerate(ct_volumes):
                patient_id = patient_ids[i] if patient_ids else f"sample_{i}"
                semantic = self.medgemma_interface.extract_semantic_features(vol, patient_id)
                batch_medgemma.append(semantic)
            features['medgemma'] = torch.stack(batch_medgemma).mean(dim=0)

        if self.path_foundation_interface is not None:
            batch_pf = []
            for img in path_images:
                pf_feat = self.path_foundation_interface.extract_features(img)
                batch_pf.append(pf_feat)
            features['path_foundation'] = torch.stack(batch_pf).mean(dim=0)

        if self.medsiglip_interface is not None:
            batch_ms = []
            for img in path_images:
                img_np = img.transpose(1, 2, 0) if img.shape[0] == 3 else img
                ms_feat = self.medsiglip_interface.compute_alignment_scores(
                    img_np,
                    self.semantic_alignment_loss.class_descriptions
                )
                batch_ms.append(ms_feat)
            features['medsiglip'] = torch.stack(batch_ms).mean(dim=0)

        return features

    def forward(self, ct_volume, path_image, haidef_features=None):
        """Forward pass."""
        batch_size = ct_volume.shape[0]

        ct_features = self.ct_encoder(ct_volume).view(batch_size, -1)
        path_features = self.path_encoder(path_image).view(batch_size, -1)

        medgemma = haidef_features.get('medgemma') if haidef_features else None
        path_found = haidef_features.get('path_foundation') if haidef_features else
        medsiglip = haidef_features.get('medsiglip') if haidef_features else None

        fused_features, info = self.attention_bridge(
            ct_features[0] if batch_size == 1 else ct_features.mean(dim=0),
            path_features[0] if batch_size == 1 else path_features.mean(dim=0),
            medgemma, path_found, medsiglip
        )

        return fused_features, info


# =============================================================================
# DATA COMPONENTS
# =============================================================================

class CT3DVolumeDataset(Dataset):
    """Dataset for 3D CT volumes."""

    def __init__(self, data_dir: str, depth: int = 16, slice_size: int = 128,
                 max_volumes: int = None, transform=None, seed: int = 42):
        self.data_dir = data_dir
        self.depth = depth
        self.slice_size = slice_size
        self.transform = transform
        self.rng = np.random.RandomState(seed)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"CT data directory not found: {data_dir}")

        self.volumes = []
        self.labels = []
        self.patient_ids = []

        self._load_volumes()

        if len(self.volumes) == 0:
            raise ValueError(f"No valid CT volumes found in {data_dir}")

    def _load_volumes(self) -> None:
        log_message(f"Loading ALL CT volumes from {self.data_dir}")

        patient_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('R_')
        ])

        if not patient_dirs:
            raise FileNotFoundError(f"No patient directories found in {self.data_dir}")

        for patient_id in tqdm(patient_dirs, desc="Loading CT volumes"):
            patient_path = os.path.join(self.data_dir, patient_id)
            patient_volume_count = 0

            for date_dir in os.listdir(patient_path):
                date_path = os.path.join(patient_path, date_dir)
                if not os.path.isdir(date_path):
                    continue

                for series_dir in os.listdir(date_path):
                    series_path = os.path.join(date_path, series_dir)
                    if not os.path.isdir(series_path):
                        continue

                    dcm_files = sorted([
                        f for f in os.listdir(series_path)
                        if f.endswith('.dcm')
                    ])

                    if len(dcm_files) < self.depth:
                        continue

                    volume_slices = []
                    valid_count = 0

                    for dcm_file in dcm_files:
                        if len(volume_slices) >= self.depth:
                            break

                        dcm_path = os.path.join(series_path, dcm_file)
                        try:
                            ds = pydicom.dcmread(dcm_path)
                            pixel_array = ds.pixel_array.astype(np.float32)
                            pixel_array = np.clip(pixel_array, -1000, 1000)
                            pixel_array = (pixel_array + 1000) / 2000
                            slice_img = Image.fromarray(pixel_array)
                            slice_img = slice_img.resize((self.slice_size, self.slice_size), Image.BILINEAR)
                            slice_array = np.array(slice_img, dtype=np.float32)
                            volume_slices.append(slice_array)
                            valid_count += 1
                        except Exception as e:
                            continue

                    if valid_count >= self.depth:
                        volume = np.stack(volume_slices[:self.depth], axis=0)
                        self.volumes.append(volume)
                        label = hash(patient_id) % 2
                        self.labels.append(label)
                        self.patient_ids.append(f"{patient_id}_series_{patient_volume_count}")
                        patient_volume_count += 1

        log_message(f"Loaded {len(self.volumes)} CT volumes from {len(patient_dirs)} patients")

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        volume = self.volumes[idx].copy()
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]

        if self.transform:
            volume = self.transform(volume, self.rng)

        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()
        return volume_tensor, label, patient_id


class PathologyMultiChannelDataset(Dataset):
    """Dataset for multi-channel pathology images."""

    def __init__(self, data_dir: str, channels: Tuple[str, ...], image_size: int = 224,
                 max_samples: int = None, transform=None, seed: int = 42):
        self.data_dir = data_dir
        self.channels = channels
        self.image_size = image_size
        self.transform = transform
        self.rng = np.random.RandomState(seed)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Pathology data directory not found: {data_dir}")

        self.samples = []
        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No valid pathology samples found in {data_dir}")

    def _load_samples(self) -> None:
        log_message(f"Loading ALL pathology samples from {self.data_dir}")

        sample_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])

        for sample_id in tqdm(sample_dirs, desc="Loading pathology samples"):
            sample_path = os.path.join(self.data_dir, sample_id)
            if not os.path.isdir(sample_path):
                continue

            region_dirs = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]

            for region_id in region_dirs:
                region_path = os.path.join(sample_path, region_id)

                channel_files = {}
                for channel in self.channels:
                    for ext in ['.tif', '.tiff', '.png', '.jpg']:
                        file_path = os.path.join(region_path, f"{channel}{ext}")
                        if os.path.exists(file_path):
                            channel_files[channel] = file_path
                            break

                if len(channel_files) >= 3:
                    self.samples.append({
                        'id': f"{sample_id}_{region_id}",
                        'files': channel_files,
                        'label': hash(sample_id) % 2
                    })

        log_message(f"Loaded {len(self.samples)} pathology samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        channels = []

        for channel in self.channels:
            if channel in sample['files']:
                img = Image.open(sample['files'][channel])
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32) / 255.0
                channels.append(img_array)
            else:
                dummy = np.zeros((self.image_size, self.image_size), dtype=np.float32)
                channels.append(dummy)

        image = np.stack(channels, axis=0)

        if self.transform:
            image = self.transform(image, self.rng)

        image_tensor = torch.from_numpy(image).float()
        return image_tensor, sample['label'], sample['id']


def ct_dataset(data_dir: str, depth: int = 16, slice_size: int = 128,
               max_volumes: int = None, transform=None, seed: int = 42) -> CT3DVolumeDataset:
    """Convenience function to create a CT 3D volume dataset."""
    return CT3DVolumeDataset(
        data_dir=data_dir,
        depth=depth,
        slice_size=slice_size,
        max_volumes=max_volumes,
        transform=transform,
        seed=seed
    )


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_confidence_interval(data: List[float]) -> Tuple[float, float, float]:
    """Compute mean and 95% confidence interval."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    t_value = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean - t_value * se
    ci_upper = mean + t_value * se
    return mean, ci_lower, ci_upper


class CT3DAugmentation:
    """Augmentation for 3D CT volumes."""

    def __init__(self, strength: float = 0.5):
        self.strength = strength

    def __call__(self, volume: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        if rng.random() < 0.5:
            volume = volume[::-1].copy()

        if rng.random() < 0.5:
            volume = volume[:, ::-1, :].copy()

        if rng.random() < 0.5:
            volume = volume[:, :, ::-1].copy()

        if rng.random() < self.strength:
            scale = rng.uniform(0.9, 1.1)
            volume = np.clip(volume * scale, 0, 1)

        if rng.random() < self.strength:
            shift = rng.uniform(-0.1, 0.1)
            volume = np.clip(volume + shift, 0, 1)

        return volume


class Pathology2DAugmentation:
    """Augmentation for 2D pathology images."""

    def __init__(self, strength: float = 0.5):
        self.strength = strength

    def __call__(self, image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        if rng.random() < 0.5:
            image = image[:, :, ::-1].copy()

        if rng.random() < 0.5:
            image = image[:, ::-1, :].copy()

        if rng.random() < 0.5:
            k = rng.randint(1, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()

        if rng.random() < self.strength:
            for c in range(image.shape[0]):
                scale = rng.uniform(0.9, 1.1)
                image[c] = np.clip(image[c] * scale, 0, 1)

        if rng.random() < self.strength * 0.5:
            noise = rng.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1)

        return image


def train_ct_ssl(model: nn.Module, dataloader: DataLoader, config: ResearchConfig,
                 logger_file) -> Dict:
    """Pre-train CT encoder using self-supervised contrastive learning."""
    model.ct_encoder.train()
    optimizer = AdamW(model.ct_encoder.parameters(), lr=config.ssl_learning_rate,
                      weight_decay=config.ssl_weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    history = {'loss': []}

    for epoch in range(config.ssl_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (ct_volumes, _, _) in enumerate(dataloader):
            ct_volumes = ct_volumes.to(config.device)

            aug1 = CT3DAugmentation(config.ssl_augmentation_strength)
            aug2 = CT3DAugmentation(config.ssl_augmentation_strength)
            rng1 = np.random.RandomState(epoch + batch_idx)
            rng2 = np.random.RandomState(epoch + batch_idx + 1000)

            view1 = torch.stack([
                torch.from_numpy(aug1(vol.numpy(), rng1)).unsqueeze(0)
                for vol in ct_volumes
            ]).squeeze(1).to(config.device)

            view2 = torch.stack([
                torch.from_numpy(aug2(vol.numpy(), rng2)).unsqueeze(0)
                for vol in ct_volumes
            ]).squeeze(1).to(config.device)

            optimizer.zero_grad()

            with torch.no_grad():
                h1 = model.ct_encoder(view1).squeeze(-1).squeeze(-1).squeeze(-1)
                h2 = model.ct_encoder(view2).squeeze(-1).squeeze(-1).squeeze(-1)

            z1 = F.normalize(h1, dim=1)
            z2 = F.normalize(h2, dim=1)

            similarity = torch.mm(z1, z2.t()) / config.ssl_temperature
            labels = torch.arange(len(similarity)).to(config.device)

            loss = F.cross_entropy(similarity, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)

        log_message(f"SSL Epoch {epoch+1}/{config.ssl_epochs} - Loss: {avg_loss:.4f}", logger_file)

    return history


def compute_prototypes(support_features: torch.Tensor, support_labels: torch.Tensor,
                       n_classes: int = 2) -> torch.Tensor:
    """Compute class prototypes from support set features."""
    prototypes = torch.zeros(n_classes, support_features.shape[1]).to(support_features.device)
    for c in range(n_classes):
        class_mask = (support_labels == c)
        if class_mask.sum() > 0:
            prototypes[c] = support_features[class_mask].mean(dim=0)
    return prototypes


def compute_query_loss(query_features: torch.Tensor, prototypes: torch.Tensor,
                       query_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute few-shot classification loss using prototypical networks."""
    squared_distances = torch.cdist(query_features, prototypes, p=2) ** 2
    logits = -squared_distances
    loss = F.cross_entropy(logits, query_labels)

    predictions = logits.argmax(dim=1)
    accuracy = (predictions == query_labels).float().mean()

    return loss, accuracy


def train_episodic_meta_learning(model: nn.Module, ct_dataset: CT3DVolumeDataset,
                                  train_dataset: PathologyMultiChannelDataset,
                                  val_dataset: PathologyMultiChannelDataset, config: ResearchConfig,
                                  logger_file, medgemma_interface=None,
                                  path_foundation_interface=None, medsiglip_interface=None) -> Dict:
    """Train using episodic meta-learning with cross-dimensional transfer."""

    if medgemma_interface is not None:
        log_message("Pre-extracting CT features for cross-dimensional transfer...", logger_file)
        ct_loader = DataLoader(ct_dataset, batch_size=config.batch_size, shuffle=False)
        ct_features_bank = []

        model.eval()
        with torch.no_grad():
            for ct_volumes, _, patient_ids in tqdm(ct_loader, desc="Extracting CT features"):
                ct_volumes = ct_volumes.to(config.device)
                ct_features = model.ct_encoder(ct_volumes).view(ct_volumes.shape[0], -1)
                ct_features_bank.append(ct_features.cpu())

        ct_features_bank = torch.cat(ct_features_bank, dim=0)
        log_message(f"Extracted {len(ct_features_bank)} CT features", logger_file)
    else:
        ct_features_bank = None

    best_models = {}
    result = {}

    for k_shot in config.n_shots:
        log_message(f"\nTraining with K-Shot = {k_shot}", logger_file)

        optimizer = AdamW(model.parameters(), lr=config.meta_lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(config.meta_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            num_batches = 50

            for batch_idx in range(num_batches):
                class_0_indices = [i for i, s in enumerate(train_dataset.samples) if s['label'] == 0]
                class_1_indices = [i for i, s in enumerate(train_dataset.samples) if s['label'] == 1]

                if len(class_0_indices) < k_shot + config.n_query or len(class_1_indices) < k_shot + config.n_query:
                    continue

                selected = random.sample(class_0_indices, k_shot + config.n_query)
                support_indices.extend(selected[:k_shot])
                query_indices.extend(selected[k_shot:])

                random.shuffle(support_indices)
                random.shuffle(query_indices)

                support_images = []
                support_labels = []
                query_images = []
                query_labels = []

                for idx in support_indices:
                    img, lbl, _ = train_dataset[idx]
                    support_images.append(img)
                    support_labels.append(lbl)

                for idx in query_indices:
                    img, lbl, _ = train_dataset[idx]
                    query_images.append(img)
                    query_labels.append(lbl)

                support_images = torch.stack(support_images)
                support_labels = torch.tensor(support_labels)
                query_images = torch.stack(query_images)
                query_labels = torch.tensor(query_labels)

                support_images = support_images.to(config.device)
                support_labels = support_labels.to(config.device)
                query_images = query_images.to(config.device)
                query_labels = query_labels.to(config.device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    ct_indices = random.sample(range(len(ct_features_bank)), len(query_feat))
                    ct_feat_batch = ct_features_bank[ct_indices].to(config.device)

                    support_feat = model.path_encoder(support_images).view(support_images.shape[0], -1)
                    query_feat = model.path_encoder(query_images).view(query_images.shape[0], -1)

                    if ct_feat_batch.shape[0] > 0:
                        support_feat_aug = torch.cat([support_feat, ct_feat_batch[:len(support_feat)]], dim=0)
                        query_feat_aug = torch.cat([query_feat, ct_feat_batch[len(support_feat):]], dim=0)
                        support_labels_aug = torch.cat([support_labels, torch.zeros(len(ct_feat_batch[:len(support_feat)])).to(config.device)])

                        prototypes = compute_prototypes(support_feat_aug, support_labels_aug.long())

                        loss, acc = compute_query_loss(query_feat_aug, prototypes, query_labels.long())
                    else:
                        prototypes = compute_prototypes(support_feat, support_labels.long())
                        loss, acc = compute_query_loss(query_feat, prototypes, query_labels.long())

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_correct += (acc.item() * len(query_labels))
                epoch_total += len(query_labels)

            scheduler.step()

            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
            train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)

            model.eval()
            val_indices = list(range(len(val_dataset)))
            val_images = []
            val_labels = []
            for idx in val_indices:
                img, lbl, _ = val_dataset[idx]
                val_images.append(img)
                val_labels.append(lbl)

            val_images = torch.stack(val_images)
            val_labels = torch.tensor(val_labels)

            with torch.no_grad():
                val_feat = model.path_encoder(val_images.to(config.device)).view(val_images.shape[0], -1)

                class_0_val = [i for i, s in enumerate(val_dataset.samples) if s['label'] == 0][:k_shot]
                class_1_val = [i for i, s in enumerate(val_dataset.samples) if s['label'] == 1][:k_shot]

                support_val = class_0_val + class_1_val
                support_val_imgs = []
                support_val_lbls = []
                for idx in support_val:
                    img, lbl, _ = val_dataset[idx]
                    support_val_imgs.append(img)
                    support_val_lbls.append(lbl)
                support_val_imgs = torch.stack(support_val_imgs)
                support_val_lbls = torch.tensor(support_val_lbls)

                with torch.cuda.amp.autocast():
                    support_val_feat = model.path_encoder(support_val_imgs.to(config.device)).view(support_val_imgs.shape[0], -1)
                    prototypes = compute_prototypes(support_val_feat, support_val_lbls.long())

                    squared_distances = torch.cdist(val_feat, prototypes, p=2) ** 2
                    logits = -squared_distances
                    predictions = logits.argmax(dim=1)

                    val_acc = (predictions.cpu() == val_labels).float().mean().item()

            history['val_acc'].append(val_acc)

            log_message(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}", logger_file)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                log_message(f"New best K-shot model saved: K={k_shot}, Val Acc: {val_acc:.4f}", logger_file)
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                log_message(f"Early stopping at epoch {epoch+1}", logger_file)
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            best_models[k_shot] = best_model_state

        result[k_shot] = {
            'history': history,
            'best_val_acc': best_val_acc
        }

    return result


def evaluate_model(model: nn.Module, test_dataset: Dataset, ct_features_bank: torch.Tensor,
                   config: ResearchConfig, logger_file, use_haidef: bool = True) -> Dict:
    """Evaluate model on test set with K-shot classification."""
    model.eval()

    k_shots = [1, 3, 5]
    results = {}

    for k_shot in k_shots:
        test_indices = list(range(len(test_dataset)))
        test_images = []
        test_labels = []
        for idx in test_indices:
            img, lbl, _ = test_dataset[idx]
            test_images.append(img)
            test_labels.append(lbl)

        test_images = torch.stack(test_images)
        test_labels = torch.tensor(test_labels)

        with torch.no_grad():
            test_feat = model.path_encoder(test_images.to(config.device)).view(test_images.shape[0], -1)

            class_0_test = [i for i, s in enumerate(test_dataset.samples) if s['label'] == 0][:k_shot]
            class_1_test = [i for i, s in enumerate(test_dataset.samples) if s['label'] == 1][:k_shot]

            support_test = class_0_test + class_1_test
            support_test_imgs = []
            support_test_lbls = []
            for idx in support_test:
                img, lbl, _ = test_dataset[idx]
                support_test_imgs.append(img)
                support_test_lbls.append(lbl)
            support_test_imgs = torch.stack(support_test_imgs)
            support_test_lbls = torch.tensor(support_test_lbls)

            support_test_feat = model.path_encoder(support_test_imgs.to(config.device)).view(support_test_imgs.shape[0], -1)
            prototypes = compute_prototypes(support_test_feat, support_test_lbls.long())

            squared_distances = torch.cdist(test_feat, prototypes, p=2) ** 2
            logits = -squared_distances
            predictions = logits.argmax(dim=1).cpu()

        accuracy = accuracy_score(test_labels, predictions)
        balanced_acc = balanced_accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='macro')

        try:
            auc_roc = roc_auc_score(test_labels, F.softmax(torch.tensor(-squared_distances.numpy()), dim=1)[:, 1])
        except:
            auc_roc = float('nan')

        kappa = cohen_kappa_score(test_labels, predictions)
        mcc = matthews_corrcoef(test_labels, predictions)

        log_message(f"K={k_shot} Evaluation Results:", logger_file)
        log_message(f"  Accuracy: {accuracy:.4f}", logger_file)
        log_message(f"  Balanced Accuracy: {balanced_acc:.4f}", logger_file)
        log_message(f"  F1 (Macro): {f1:.4f}", logger_file)
        log_message(f"  AUC-ROC: {auc_roc:.4f}", logger_file)
        log_message(f"  Cohen's Kappa: {kappa:.4f}", logger_file)

        results[k_shot] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score_macro': f1,
            'auc_roc': auc_roc,
            'cohen_kappa': kappa,
            'mcc': mcc
        }

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

class VisualizationSuite:
    """Comprehensive visualization suite for experiment results."""

    def __init__(self, config: ResearchConfig):
        self.config = config
        self.colors = {
            '3D CT': '#4C72B0',
            '2D Pathology': '#DD8452',
            'class_0': '#4C72B0',
            'class_1': '#DD8452'
        }

    def plot_training_curves(self, fold_results: List[Dict], output_dir: str) -> None:
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        for k_shot in [1, 3, 5]:
            history = fold_results[0].get('history', {}).get(k_shot, {}).get('history', {})
            if 'train_loss' in history:
                axes[0].plot(history['train_loss'], label=f'K={k_shot}')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for k_shot in [1, 3, 5]:
            history = fold_results[0].get('history', {}).get(k_shot, {}).get('history', {})
            if 'val_acc' in history:
                axes[1].plot(history['val_acc'], label=f'K={k_shot}')

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              output_dir: str, fold_idx: int) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Stroma', 'Invasion'],
                    yticklabels=['Stroma', 'Invasion'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_fold_{fold_idx}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, fold_results: List[Dict], output_dir: str) -> None:
        """Plot ROC curves for all folds."""
        plt.figure(figsize=(8, 6))

        for fold_idx, fold_data in enumerate(fold_results):
            if 'roc_data' in fold_data:
                fpr, tpr, auc = fold_data['roc_data']
                plt.plot(fpr, tpr, label=f'Fold {fold_idx + 1} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_feature_embeddings(self, ct_features: torch.Tensor, path_features: torch.Tensor,
                                 labels: Optional[np.ndarray], output_dir: str) -> None:
        """Plot UMAP/t-SNE embeddings of learned features."""
        ct_np = ct_features.detach().cpu().numpy()
        path_np = path_features.detach().cpu().numpy()

        from sklearn.preprocessing import StandardScaler
        all_features = np.vstack([ct_np, path_np])
        all_features = StandardScaler().fit_transform(all_features)

        domain_labels = np.array(['3D CT'] * len(ct_np) + ['2D Pathology'] * len(path_np))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        proj = reducer.fit_transform(all_features)
        title_prefix = "UMAP"

        ax = axes[0]
        for i, domain in enumerate(['3D CT', '2D Pathology']):
            mask = domain_labels == domain
            ax.scatter(proj[mask, 0], proj[mask, 1], label=domain, alpha=0.6, s=25,
                      color=self.colors[domain], edgecolors='none')

        ax.set_title(f'{title_prefix}: Domain Alignment', fontweight='bold')
        ax.legend(frameon=True, loc='best', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1]
        if labels is not None:
            if len(labels) == len(path_np):
                combined_labels = np.concatenate([labels[:len(ct_np)], labels])
                if len(combined_labels) == len(proj):
                    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=combined_labels, cmap='viridis',
                                        alpha=0.6, s=25, edgecolors='none')
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Class Label', size=8)
                    ax.set_title(f'{title_prefix}: Latent Semantics', fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_embeddings.png'), dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main experimental pipeline."""
    os.makedirs(CFG.output_dir, exist_ok=True)
    logger_file = open(os.path.join(CFG.output_dir, 'experiment_log.txt'), 'w')

    log_message("=" * 80, logger_file)
    log_message("MEDGEMMA CROSS-DIMENSIONAL KNOWLEDGE TRANSFER", logger_file)
    log_message("Integrating MedGemma 1.5, Path Foundation, and MedSigLIP", logger_file)
    log_message("=" * 80, logger_file)
    log_message(f"Device: {CFG.device}", logger_file)
    log_message(f"Output: {CFG.output_dir}", logger_file)
    log_message(f"HAI-DEF Integration: {CFG.use_haidef}", logger_file)

    set_seed(CFG.seed)

    log_message("\nInitializing HAI-DEF models...")
    medgemma_interface = None
    path_foundation_interface = None
    medsiglip_interface = None

    if CFG.use_haidef:
        try:
            haidef_cfg = CFG.haidef_config
            medgemma_interface = MedGemmaInterface(haidef_cfg)
            path_foundation_interface = PathFoundationInterface(haidef_cfg)
            medsiglip_interface = MedSigLIPInterface(haidef_cfg)

            medgemma_interface.load_model()
            path_foundation_interface.load_model()
            medsiglip_interface.load_model()

            log_message("HAI-DEF models initialized successfully", logger_file)
        except Exception as e:
            log_message(f"Warning: HAI-DEF initialization failed: {e}", logger_file)
            log_message("Continuing without HAI-DEF models", logger_file)

    log_message("\nLoading datasets...")
    try:
        ct_dataset = CT3DVolumeDataset(
            CFG.ct_data_dir,
            depth=CFG.ct_volume_depth,
            slice_size=CFG.slice_size,
            seed=CFG.seed
        )
        log_message(f"Loaded {len(ct_dataset)} CT volumes", logger_file)
    except Exception as e:
        log_message(f"Error loading CT dataset: {e}", logger_file)
        return None

    try:
        path_dataset = PathologyMultiChannelDataset(
            CFG.pathology_data_dir,
            channels=CFG.channels,
            image_size=CFG.image_size,
            seed=CFG.seed
        )
        log_message(f"Loaded {len(path_dataset)} pathology samples", logger_file)
    except Exception as e:
        log_message(f"Error loading pathology dataset: {e}", logger_file)
        return None

    labels = [path_dataset.samples[i]['label'] for i in range(len(path_dataset))]
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

    fold_results = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(path_dataset)), labels)):
        log_message(f"\n{'=' * 80}", logger_file)
        log_message(f"FOLD {fold_idx + 1}/{CFG.n_splits}", logger_file)
        log_message(f"{'=' * 80}", logger_file)

        train_val_labels = [labels[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.15,
            stratify=train_val_labels,
            random_state=CFG.seed + fold_idx
        )

        train_dataset = Subset(path_dataset, train_idx)
        val_dataset = Subset(path_dataset, val_idx)
        test_dataset = Subset(path_dataset, test_idx)

        log_message(f"Train class distribution: {[train_val_labels.count(i) for i in [0, 1]]}", logger_file)
        log_message(f"Val class distribution: {[sum(1 for i in val_idx if labels[i] == c) for c in [0, 1]]}", logger_file)
        log_message(f"Test class distribution: {[sum(1 for i in test_idx if labels[i] == c) for c in [0, 1]]}", logger_file)
        log_message(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}", logger_file)

        model = EnhancedCrossDomainTransferModel(CFG)

        if fold_idx == 0:
            ct_loader = DataLoader(ct_dataset, batch_size=CFG.batch_size, shuffle=True,
                                   num_workers=CFG.num_workers, drop_last=True)
            ssl_history = train_ct_ssl(model, ct_loader, CFG, logger_file)

            torch.save({
                'model_state_dict': model.ct_encoder.state_dict(),
                'ssl_history': ssl_history,
                'config': CFG.__dict__,
                'dataset_info': {
                    'n_ct_volumes': len(ct_dataset),
                    'depth': CFG.ct_volume_depth,
                    'slice_size': CFG.slice_size
                }
            }, os.path.join(CFG.output_dir, 'models', 'ct_encoder_pretrained.pt'))

            log_message(f"CT encoder pretrained model saved successfully", logger_file)
        else:
            checkpoint = torch.load(os.path.join(CFG.output_dir, 'models', 'ct_encoder_pretrained.pt'))
            model.ct_encoder.load_state_dict(checkpoint['model_state_dict'])

        model.set_haidef_interfaces(medgemma_interface, path_foundation_interface, medsiglip_interface)

        transfer_result = train_episodic_meta_learning(
            model, ct_dataset, train_dataset, val_dataset, CFG, logger_file,
            medgemma_interface, path_foundation_interface, medsiglip_interface
        )

        best_model_path = os.path.join(CFG.output_dir, 'models', 'best_model.pt')
        best_model_data = torch.load(best_model_path)

        if isinstance(best_model_data, dict) and 'model_state_dict' in best_model_data:
            model.load_state_dict(best_model_data['model_state_dict'])
        else:
            model.load_state_dict(best_model_data)

        test_results = evaluate_model(
            model, test_dataset, None,
            CFG, logger_file, use_haidef=CFG.use_haidef
        )

        fold_results.append(test_results)

        visualizer = VisualizationSuite(CFG)
        fold_dir = os.path.join(CFG.output_dir, 'analysis', f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, 'results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)

        clear_memory()

    log_message(f"\n{'=' * 80}", logger_file)
    log_message("FINAL RESULTS ACROSS ALL FOLDS", logger_file)
    log_message(f"{'=' * 80}", logger_file)

    aggregated = {}
    for metric in ['accuracy', 'balanced_accuracy', 'f1_score_macro', 'auc_roc', 'cohen_kappa', 'mcc']:
        values = [r[metric] for fold in fold_results for r in [fold[1], fold[3], fold[5]] if not np.isnan(r.get(metric, np.nan))]
        if values:
            mean, ci_lower, ci_upper = compute_confidence_interval(values)
            aggregated[metric] = {
                'mean': mean,
                'std': np.std(values),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'values': values
            }
            log_message(f"{metric}: {mean:.4f} +/- {np.std(values):.4f} [{ci_lower:.4f}, {ci_upper:.4f}]", logger_file)

    with open(os.path.join(CFG.output_dir, 'final_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)

    log_message("\n" + "=" * 80, logger_file)
    log_message("EXPERIMENT COMPLETED SUCCESSFULLY", logger_file)
    log_message("=" * 80, logger_file)

    logger_file.close()
    return aggregated


if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults saved to: {CFG.output_dir}")
    print("\nKey Findings:")
