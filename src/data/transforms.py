"""
Transform classes for data augmentation.
"""

import random
from typing import Optional
import numpy as np


class CT3DTransform:
    """
    Transform for 3D CT volumes.

    Combines augmentation and normalization for CT data.
    """

    def __init__(
        self,
        depth: int = 16,
        slice_size: int = 128,
        normalize: bool = True,
        augment: bool = True,
        augmentation_strength: float = 0.5
    ):
        """
        Initialize CT transform.

        Args:
            depth: Target depth for volumes
            slice_size: Target size for slices
            normalize: Whether to normalize data
            augment: Whether to apply augmentation
            augmentation_strength: Strength of augmentation
        """
        self.depth = depth
        self.slice_size = slice_size
        self.normalize = normalize
        self.augment = augment
        self.augmentation_strength = augmentation_strength

    def __call__(
        self,
        volume: np.ndarray,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Apply transform to CT volume.

        Args:
            volume: Input volume (D, H, W)
            rng: Random number generator

        Returns:
            Transformed volume
        """
        if rng is None:
            rng = np.random.RandomState()

        if self.augment:
            volume = CT3DAugmentation(self.augmentation_strength)(volume, rng)

        if self.normalize:
            volume = np.clip(volume, 0, 1)

        return volume


class Pathology2DTransform:
    """
    Transform for 2D pathology images.

    Combines augmentation and normalization for pathology data.
    """

    def __init__(
        self,
        image_size: int = 224,
        channels: tuple = ('DAPI', 'PanCK', 'CD3', 'CD8', 'FOXP3', 'PDL1'),
        normalize: bool = True,
        augment: bool = True,
        augmentation_strength: float = 0.5
    ):
        """
        Initialize pathology transform.

        Args:
            image_size: Target size for images
            channels: Channel names
            normalize: Whether to normalize data
            augment: Whether to apply augmentation
            augmentation_strength: Strength of augmentation
        """
        self.image_size = image_size
        self.channels = channels
        self.normalize = normalize
        self.augment = augment
        self.augmentation_strength = augmentation_strength

    def __call__(
        self,
        image: np.ndarray,
        rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Apply transform to pathology image.

        Args:
            image: Input image (C, H, W)
            rng: Random number generator

        Returns:
            Transformed image
        """
        if rng is None:
            rng = np.random.RandomState()

        if self.augment:
            image = Pathology2DAugmentation(self.augmentation_strength)(image, rng)

        if self.normalize:
            image = np.clip(image, 0, 1)

        return image


class CT3DAugmentation:
    """
    Augmentation for 3D CT volumes.

    Applies random flips, scaling, and additive noise.
    """

    def __init__(self, strength: float = 0.5):
        """
        Initialize augmentation.

        Args:
            strength: Augmentation strength (0.0 to 1.0)
        """
        self.strength = strength

    def __call__(
        self,
        volume: np.ndarray,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Apply augmentation to volume.

        Args:
            volume: Input volume (D, H, W)
            rng: Random number generator

        Returns:
            Augmented volume
        """
        # Random flips
        if rng.random() < 0.5:
            volume = volume[::-1].copy()

        if rng.random() < 0.5:
            volume = volume[:, ::-1, :].copy()

        if rng.random() < 0.5:
            volume = volume[:, :, ::-1].copy()

        # Random scaling
        if rng.random() < self.strength:
            scale = rng.uniform(0.9, 1.1)
            volume = np.clip(volume * scale, 0, 1)

        # Random additive shift
        if rng.random() < self.strength:
            shift = rng.uniform(-0.1, 0.1)
            volume = np.clip(volume + shift, 0, 1)

        # Random additive noise
        if rng.random() < self.strength * 0.5:
            noise = rng.normal(0, 0.02, volume.shape)
            volume = np.clip(volume + noise, 0, 1)

        return volume


class Pathology2DAugmentation:
    """
    Augmentation for 2D pathology images.

    Applies random flips, rotations, scaling, and additive noise.
    """

    def __init__(self, strength: float = 0.5):
        """
        Initialize augmentation.

        Args:
            strength: Augmentation strength (0.0 to 1.0)
        """
        self.strength = strength

    def __call__(
        self,
        image: np.ndarray,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Apply augmentation to image.

        Args:
            image: Input image (C, H, W)
            rng: Random number generator

        Returns:
            Augmented image
        """
        # Random horizontal flip
        if rng.random() < 0.5:
            image = image[:, :, ::-1].copy()

        # Random vertical flip
        if rng.random() < 0.5:
            image = image[:, ::-1, :].copy()

        # Random rotation
        if rng.random() < 0.5:
            k = rng.randint(1, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()

        # Random scaling per channel
        if rng.random() < self.strength:
            for c in range(image.shape[0]):
                scale = rng.uniform(0.9, 1.1)
                image[c] = np.clip(image[c] * scale, 0, 1)

        # Random additive noise
        if rng.random() < self.strength * 0.5:
            noise = rng.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1)

        return image


class Compose:
    """
    Compose multiple transforms together.
    """

    def __init__(self, transforms: list):
        """
        Initialize compose.

        Args:
            transforms: List of transforms to apply
        """
        self.transforms = transforms

    def __call__(
        self,
        data,
        rng: Optional[np.random.RandomState] = None
    ):
        """
        Apply all transforms in sequence.

        Args:
            data: Input data
            rng: Random number generator

        Returns:
            Transformed data
        """
        for transform in self.transforms:
            data = transform(data, rng)
        return data
