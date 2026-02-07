"""
Dataset classes for medical imaging data.
"""

import os
import random
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pydicom
from tqdm import tqdm


class CT3DVolumeDataset(Dataset):
    """
    Dataset for 3D CT volumes.

    This dataset loads DICOM CT scans and creates 3D volume representations
    suitable for training 3D convolutional neural networks.
    """

    def __init__(
        self,
        data_dir: str,
        depth: int = 16,
        slice_size: int = 128,
        max_volumes: Optional[int] = None,
        transform=None,
        seed: int = 42
    ):
        """
        Initialize CT dataset.

        Args:
            data_dir: Directory containing CT DICOM files
            depth: Number of slices per volume
            slice_size: Size to resize each slice
            max_volumes: Maximum number of volumes to load
            transform: Optional transform to apply
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.depth = depth
        self.slice_size = slice_size
        self.transform = transform
        self.rng = np.random.RandomState(seed)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"CT data directory not found: {data_dir}")

        self.volumes: List[np.ndarray] = []
        self.labels: List[int] = []
        self.patient_ids: List[str] = []

        self._load_volumes()

        if len(self.volumes) == 0:
            raise ValueError(f"No valid CT volumes found in {data_dir}")

        if max_volumes is not None and len(self.volumes) > max_volumes:
            indices = random.sample(range(len(self.volumes)), max_volumes)
            self.volumes = [self.volumes[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.patient_ids = [self.patient_ids[i] for i in indices]

    def _load_volumes(self) -> None:
        """Load all CT volumes from the data directory."""
        print(f"Loading ALL CT volumes from {self.data_dir}")

        patient_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('R_')
        ])

        if not patient_dirs:
            raise FileNotFoundError(
                f"No patient directories found in {self.data_dir}. "
                "Expected directories starting with 'R_'"
            )

        for patient_id in tqdm(patient_dirs, desc="Loading CT volumes"):
            patient_path = os.path.join(self.data_dir, patient_id)
            if not os.path.isdir(patient_path):
                continue

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

                            # Normalize Hounsfield units
                            pixel_array = np.clip(pixel_array, -1000, 1000)
                            pixel_array = (pixel_array + 1000) / 2000

                            # Resize to target size
                            slice_img = Image.fromarray(pixel_array)
                            slice_img = slice_img.resize(
                                (self.slice_size, self.slice_size),
                                Image.BILINEAR
                            )
                            slice_array = np.array(slice_img, dtype=np.float32)

                            volume_slices.append(slice_array)
                            valid_count += 1

                        except Exception as e:
                            continue

                    if valid_count >= self.depth:
                        volume = np.stack(volume_slices[:self.depth], axis=0)
                        self.volumes.append(volume)

                        # Generate label from patient ID hash
                        label = hash(patient_id) % 2
                        self.labels.append(label)

                        self.patient_ids.append(
                            f"{patient_id}_series_{patient_volume_count}"
                        )
                        patient_volume_count += 1

        print(f"Loaded {len(self.volumes)} CT volumes from {len(patient_dirs)} patients")

    def __len__(self) -> int:
        """Return number of volumes in dataset."""
        return len(self.volumes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single volume from the dataset.

        Args:
            idx: Index of volume to get

        Returns:
            Tuple of (volume_tensor, label, patient_id)
        """
        volume = self.volumes[idx].copy()
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]

        if self.transform:
            volume = self.transform(volume, self.rng)

        # Convert to tensor with channel dimension
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()

        return volume_tensor, label, patient_id


class PathologyMultiChannelDataset(Dataset):
    """
    Dataset for multi-channel pathology images.

    This dataset loads multiplex immunofluorescence images with multiple
    marker channels and creates normalized tensor representations.
    """

    def __init__(
        self,
        data_dir: str,
        channels: Tuple[str, ...],
        image_size: int = 224,
        max_samples: Optional[int] = None,
        transform=None,
        seed: int = 42
    ):
        """
        Initialize pathology dataset.

        Args:
            data_dir: Directory containing pathology images
            channels: Tuple of channel names to load
            image_size: Target size for images
            max_samples: Maximum number of samples to load
            transform: Optional transform to apply
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.channels = channels
        self.image_size = image_size
        self.transform = transform
        self.rng = np.random.RandomState(seed)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Pathology data directory not found: {data_dir}")

        self.samples: List[Dict] = []
        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No valid pathology samples found in {data_dir}")

        if max_samples is not None and len(self.samples) > max_samples:
            indices = random.sample(range(len(self.samples)), max_samples)
            self.samples = [self.samples[i] for i in indices]

    def _load_samples(self) -> None:
        """Load all pathology samples from the data directory."""
        print(f"Loading ALL pathology samples from {self.data_dir}")

        sample_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])

        for sample_id in tqdm(sample_dirs, desc="Loading pathology samples"):
            sample_path = os.path.join(self.data_dir, sample_id)
            if not os.path.isdir(sample_path):
                continue

            # Find all region directories
            region_dirs = [
                d for d in os.listdir(sample_path)
                if os.path.isdir(os.path.join(sample_path, d))
            ]

            for region_id in region_dirs:
                region_path = os.path.join(sample_path, region_id)

                channel_files = {}
                for channel in self.channels:
                    for ext in ['.tif', '.tiff', '.png', '.jpg']:
                        file_path = os.path.join(region_path, f"{channel}{ext}")
                        if os.path.exists(file_path):
                            channel_files[channel] = file_path
                            break

                # Require at least 3 channels
                if len(channel_files) >= 3:
                    self.samples.append({
                        'id': f"{sample_id}_{region_id}",
                        'files': channel_files,
                        'label': hash(sample_id) % 2
                    })

        print(f"Loaded {len(self.samples)} pathology samples")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of sample to get

        Returns:
            Tuple of (image_tensor, label, sample_id)
        """
        sample = self.samples[idx]
        channels = []

        for channel in self.channels:
            if channel in sample['files']:
                try:
                    img = Image.open(sample['files'][channel])
                    img = img.resize(
                        (self.image_size, self.image_size),
                        Image.BILINEAR
                    )
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    channels.append(img_array)
                except Exception as e:
                    # Create zero array if loading fails
                    dummy = np.zeros(
                        (self.image_size, self.image_size),
                        dtype=np.float32
                    )
                    channels.append(dummy)
            else:
                # Create zero array if channel missing
                dummy = np.zeros(
                    (self.image_size, self.image_size),
                    dtype=np.float32
                )
                channels.append(dummy)

        image = np.stack(channels, axis=0)

        if self.transform:
            image = self.transform(image, self.rng)

        image_tensor = torch.from_numpy(image).float()
        return image_tensor, sample['label'], sample['id']


def ct_dataset(
    data_dir: str,
    depth: int = 16,
    slice_size: int = 128,
    max_volumes: Optional[int] = None,
    transform=None,
    seed: int = 42
) -> CT3DVolumeDataset:
    """
    Convenience function to create a CT 3D volume dataset.

    Args:
        data_dir: Directory containing CT DICOM files
        depth: Number of slices per volume
        slice_size: Size to resize each slice
        max_volumes: Maximum number of volumes to load
        transform: Optional transform to apply
        seed: Random seed for reproducibility

    Returns:
        CT3DVolumeDataset instance
    """
    return CT3DVolumeDataset(
        data_dir=data_dir,
        depth=depth,
        slice_size=slice_size,
        max_volumes=max_volumes,
        transform=transform,
        seed=seed
    )


def pathology_dataset(
    data_dir: str,
    channels: Tuple[str, ...],
    image_size: int = 224,
    max_samples: Optional[int] = None,
    transform=None,
    seed: int = 42
) -> PathologyMultiChannelDataset:
    """
    Convenience function to create a pathology multi-channel dataset.

    Args:
        data_dir: Directory containing pathology images
        channels: Tuple of channel names to load
        image_size: Target size for images
        max_samples: Maximum number of samples to load
        transform: Optional transform to apply
        seed: Random seed for reproducibility

    Returns:
        PathologyMultiChannelDataset instance
    """
    return PathologyMultiChannelDataset(
        data_dir=data_dir,
        channels=channels,
        image_size=image_size,
        max_samples=max_samples,
        transform=transform,
        seed=seed
    )
