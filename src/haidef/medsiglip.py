"""
MedSigLIP Interface

Interface to MedSigLIP model for semantic alignment and zero-shot classification.
"""

import torch
import numpy as np
from typing import List
from transformers import AutoModel, AutoProcessor


class MedSigLIPInterface:
    """
    Interface for MedSigLIP model integration.

    Provides zero-shot image classification and semantic alignment.
    """

    def __init__(self, config):
        """
        Initialize MedSigLIP interface.

        Args:
            config: HAIDEFConfig object
        """
        self.config = config
        self.model = None
        self.processor = None
        self.feature_cache = {}

    def load_model(self) -> None:
        """
        Load MedSigLIP model and processor.
        """
        model_name = self.config.medsiglip_model_name
        print(f"Loading MedSigLIP model: {model_name}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        self.model.eval()
        print("MedSigLIP model loaded successfully")

    def compute_alignment_scores(
        self,
        image: np.ndarray,
        text_descriptions: List[str]
    ) -> torch.Tensor:
        """
        Compute alignment scores between image and text descriptions.

        Args:
            image: Input image (H, W, C) in 0-1 range
            text_descriptions: List of clinical text descriptions

        Returns:
            Alignment scores tensor
        """
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "MedSigLIP model not loaded. Cannot compute alignment scores."
            )

        # Process inputs
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

    def zero_shot_classify(
        self,
        image: np.ndarray,
        class_descriptions: List[str]
    ) -> tuple:
        """
        Perform zero-shot classification using class descriptions.

        Args:
            image: Input image (H, W, C)
            class_descriptions: Text descriptions for each class

        Returns:
            Tuple of (probabilities, predictions)
        """
        alignment_scores = self.compute_alignment_scores(image, class_descriptions)
        probs = torch.softmax(alignment_scores, dim=0)
        predictions = probs.argmax(dim=0)

        return probs, predictions

    def clear_cache(self) -> None:
        """Clear feature cache."""
        self.feature_cache.clear()

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
