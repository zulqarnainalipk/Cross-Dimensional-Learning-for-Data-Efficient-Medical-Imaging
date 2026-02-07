"""
MedGemma Interface

Interface to MedGemma 1.5 model for CT interpretation and semantic features.
"""

import os
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


class MedGemmaInterface:
    """
    Interface for MedGemma 1.5 model integration.

    Provides CT interpretation and semantic feature extraction capabilities.
    """

    def __init__(
        self,
        config,
        device: str = "auto"
    ):
        """
        Initialize MedGemma interface.

        Args:
            config: HAIDEFConfig object
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.config = config
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.feature_cache = {}

    def _setup_device(self, device: str) -> str:
        """Setup appropriate device for model inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self) -> None:
        """
        Load MedGemma 1.5 model and tokenizer.

        Uses 4-bit quantization for memory efficiency on T4 GPUs.
        """
        model_name = self.config.medgemma_model_name
        print(f"Loading MedGemma model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": self.device
        }

        if self.config.medgemma_use_quantization and self.config.medgemma_load_in_4bit:
            try:
                import bitsandbytes as bnb
                load_kwargs["load_in_4bit"] = True
                load_kwargs["bnb_4bit_quant_type"] = "nf4"
                load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                print("Using 4-bit quantization for MedGemma")
            except ImportError:
                print("bitsandbytes not available, loading in full precision")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        self.model.eval()
        print("MedGemma model loaded successfully")

    def generate_ct_description(
        self,
        ct_slice: np.ndarray,
        patient_id: str = "unknown"
    ) -> str:
        """
        Generate clinical description for a CT slice.

        Args:
            ct_slice: Hounsfield Unit normalized CT slice (0-1 range)
            patient_id: Patient identifier for caching

        Returns:
            Clinical text description
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("MedGemma model not loaded. Cannot generate CT description.")

        cache_key = f"{patient_id}_{hash(ct_slice.tobytes()) % 10000}"
        if cache_key in self.feature_cache and 'description' in self.feature_cache[cache_key]:
            return self.feature_cache[cache_key]['description']

        # Convert to PIL Image
        slice_8bit = (np.clip(ct_slice, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(slice_8bit).convert("RGB")

        # Create prompt
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

        # Cache the description
        if cache_key not in self.feature_cache:
            self.feature_cache[cache_key] = {}
        self.feature_cache[cache_key]['description'] = description

        return description

    def extract_semantic_features(
        self,
        ct_volume: np.ndarray,
        patient_id: str = "unknown"
    ) -> torch.Tensor:
        """
        Extract semantic features from CT volume using MedGemma.

        Args:
            ct_volume: 3D CT volume (D, H, W) normalized to 0-1 range
            patient_id: Patient identifier for caching

        Returns:
            Semantic feature tensor
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("MedGemma model not loaded. Cannot extract semantic features.")

        cache_key = f"{patient_id}_semantic_features"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        depth = ct_volume.shape[0]
        mid_slice_idx = depth // 2

        # Sample key slices
        slice_indices = [0, mid_slice_idx, depth - 1]
        slice_indices = [i for i in slice_indices if 0 <= i < depth]

        descriptions = []
        for idx in slice_indices:
            description = self.generate_ct_description(ct_volume[idx], patient_id)
            descriptions.append(description)

        # Generate feature representation from descriptions
        features = self._texts_to_features(descriptions)

        self.feature_cache[cache_key] = features
        return features

    def _texts_to_features(self, texts: List[str]) -> torch.Tensor:
        """
        Convert list of texts to feature tensor using MedGemma embeddings.

        Args:
            texts: List of clinical descriptions

        Returns:
            Feature tensor
        """
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
                # Get embeddings from the model's base transformer
                outputs = self.model.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                # Use mean pooling over non-padding tokens
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                all_embeddings.append(pooled.squeeze(0))

        # Concatenate and project to target dimension
        combined = torch.cat(all_embeddings, dim=0)

        # Project to config feature dimension
        if combined.shape[0] != self.config.medgemma_feature_dim:
            if not hasattr(self, '_feature_proj'):
                self._feature_proj = nn.Linear(
                    combined.shape[0],
                    self.config.medgemma_feature_dim
                ).to(self.device)
            features = self._feature_proj(combined)
        else:
            features = combined

        return features

    def clear_cache(self) -> None:
        """Clear feature cache."""
        self.feature_cache.clear()

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
