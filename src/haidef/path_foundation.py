"""
Path Foundation Interface

Interface to Path Foundation model for digital pathology feature extraction.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModel
from typing import Optional


class PathFoundationInterface:
    """
    Interface for Path Foundation model integration.

    Provides enhanced digital pathology feature extraction.
    """

    def __init__(self, config):
        """
        Initialize Path Foundation interface.

        Args:
            config: HAIDEFConfig object
        """
        self.config = config
        self.model = None
        self.preprocess = None
        self.feature_cache = {}

    def load_model(self) -> None:
        """
        Load Path Foundation model.
        """
        model_name = self.config.path_foundation_model_name
        print(f"Loading Path Foundation model: {model_name}")

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        self.model.eval()
        print("Path Foundation model loaded successfully")

    def extract_features(
        self,
        pathology_image: np.ndarray,
        sample_id: str = "unknown"
    ) -> torch.Tensor:
        """
        Extract features from pathology image using Path Foundation.

        Args:
            pathology_image: Multi-channel pathology image (C, H, W) in 0-1 range
            sample_id: Sample identifier for caching

        Returns:
            Feature tensor
        """
        if self.model is None:
            raise RuntimeError("Path Foundation model not loaded. Cannot extract features.")

        cache_key = f"{sample_id}_path_features"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Convert image to expected format
        if pathology_image.shape[0] != 3:
            # Convert (H, W, C) to (C, H, W) if needed
            if pathology_image.ndim == 3 and pathology_image.shape[2] == 3:
                image_tensor = torch.from_numpy(pathology_image).permute(2, 0, 1).float()
            else:
                raise ValueError(
                    f"Unexpected image shape: {pathology_image.shape}. "
                    "Expected (H, W, 3) or (3, H, W)"
                )
        else:
            image_tensor = torch.from_numpy(pathology_image).float()

        # Resize to expected input size (224x224)
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Extract features
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor.unsqueeze(0))
            features = outputs.last_hidden_state.mean(dim=1).squeeze()

        self.feature_cache[cache_key] = features
        return features

    def clear_cache(self) -> None:
        """Clear feature cache."""
        self.feature_cache.clear()

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
