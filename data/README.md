# Data for Cross-Dimensional Knowledge Transfer

This document provides information about the datasets used in the Cross-Dimensional Knowledge Transfer project for few-shot HNSCC cancer classification.

## Overview

The project requires two primary data modalities:

1. **3D CT Volumes** - From Head and Neck Cancer patients
2. **2D Pathology Images** - Whole Slide Images (WSI) of tissue samples

These datasets are linked through patient identifiers when available, enabling cross-modal learning.

---

## CT Scan Data Sources

### 1. TCIA Head-Neck-PET-CT Dataset

**Source**: The Cancer Imaging Archive (TCIA)
- **Dataset Link**: https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT
- **Citation**: Vallières M, et al. (2017) Data from Head-Neck-PET-CT. The Cancer Imaging Archive.

**Description**:
- Contains FDG-PET and CT scans from 298 patients with head and neck cancer
- Includes radiotherapy planning data
- Ground truth: Gross Tumor Volumes (GTV) and clinical outcomes

**Access Instructions**:
1. Visit the dataset page on TCIA
2. Accept the data usage agreement
3. Download the dataset (approximately 50GB)
4. Organize into patient directories with NIfTI format

**Data Structure**:
```
data/
  ct_scans/
    HN_PETCT_0001/
      CT.nii.gz
      PET.nii.gz
      GTV.nii.gz (optional)
    HN_PETCT_0002/
      ...
```

### 2. AAPM DeepLesion Dataset (Alternative)

**Source**: https://aapm.org/GrandChallenge/DeepLesion
- **Description**: Contains CT images with annotated lesions
- **Note**: Less specific to HNSCC but useful for pre-training

### 3. NSCLC Radiogenomics Dataset

**Source**: https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics
- **Description**: Non-small cell lung cancer CT with genomic data
- **Note**: Can be used for pre-training features

---

## Pathology Image Data Sources

### 1. HNSCC-mIF-mIHC Dataset (Primary)

**Source**: The Cancer Imaging Archive (TCIA)
- **Dataset Link**: https://wiki.cancerimagingarchive.net/display/Public/HNSCC-mIF-mIHC
- **Citation**: Saltz J, et al. (2018) Data from Head and Neck Squamous Cell Carcinoma (HNSCC) with multiplexed IHC and histology. The Cancer Imaging Archive.

**Description**:
- Contains 39 HNSCC cases with multiplexed immunofluorescence images
- Includes diagnostic H&E WSIs
- Spatial proteomics data from multiple markers
- Clinical annotations including survival data

**Access Instructions**:
1. Visit the dataset page on TCIA
2. Download the multi-channel IHC images
3. Extract diagnostic WSIs (SVS format)

**Data Structure**:
```
data/
  pathology/
    case_001/
      region_001/
        component_001.png  # DAPI
        component_002.png  # Cytokeratin
        component_003.png  # CD8
        ...
      region_002/
        ...
      diagnostic/
        slide_001.svs
        slide_002.svi
    case_002/
      ...
```

### 2. CAMELYON16/17 (Alternative)

**Source**: https://camelyon16.grand-challenge.org/
- **Description**: Breast cancer lymph node metastases (WSI)
- **Note**: Can be used for pre-training pathology features

### 3. Pan-Cancer TCGA Pathology

**Source**: https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga
- **Description**: TCGA provides pathology images across 30+ cancer types
- **Note**: Can supplement HNSCC training data

**Access**: Through NIH GDC (Genomic Data Commons)

---

## Linkage Between Modalities

When patient identifiers are available in both datasets, cross-modal learning can be performed. Some datasets provide explicit linkages:

### 1. TCIA Linked Collections

Several TCIA collections provide linked PET/CT and pathology:
- Search for "linked" collections on TCIA
- Example: Head-Neck-PET-CT linked with pathology when available

### 2. Manual Linking

For datasets without automatic linkage:
1. Match by patient ID in metadata
2. Use DICOM tags (PatientID, StudyDate)
3. Verify by checking slice positions

---

## Data Download Instructions

### Step 1: Download TCIA Data

```bash
# Install NBIA data loader (optional)
pip install pynbia

# Or use wget/curl for direct downloads
# Note: Requires registration and acceptance of usage agreement
```

### Step 2: Organize Directory Structure

```python
from pathlib import Path
import shutil

def organize_ct_data(source_dir: Path, dest_dir: Path):
    """Organize CT data into standardized structure."""
    for patient_dir in source_dir.iterdir():
        if patient_dir.is_dir():
            patient_id = patient_dir.name
            target_dir = dest_dir / patient_id
            target_dir.mkdir(parents=True, exist_ok=True)

            # Find and copy CT volumes
            for ct_file in patient_dir.glob("*CT*"):
                if ct_file.suffix in ['.nii', '.nii.gz', '.dcm']:
                    shutil.copy2(ct_file, target_dir / "CT.nii.gz")

def organize_pathology_data(source_dir: Path, dest_dir: Path):
    """Organize pathology data into standardized structure."""
    for case_dir in source_dir.iterdir():
        if case_dir.is_dir():
            case_id = case_dir.name
            target_dir = dest_dir / case_id
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy all image files
            for img_file in case_dir.rglob("*"):
                if img_file.is_file() and img_file.suffix in ['.png', '.jpg', '.svs', '.tiff']:
                    rel_path = img_file.relative_to(source_dir)
                    target_path = target_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_file, target_path)
```

### Step 3: Verify Data Integrity

```python
def verify_dataset(data_dir: Path):
    """Verify all required files exist."""
    issues = []

    # Check CT data
    for patient_dir in (data_dir / "ct_scans").iterdir():
        ct_file = patient_dir / "CT.nii.gz"
        if not ct_file.exists():
            issues.append(f"Missing CT for {patient_dir.name}")

    # Check pathology data
    for case_dir in (data_dir / "pathology").iterdir():
        if not any(case_dir.iterdir()):
            issues.append(f"Empty case directory: {case_dir.name}")

    return issues
```

---

## Data Preprocessing

### CT Volume Preprocessing

```python
import SimpleITK as sitk
import numpy as np

def preprocess_ct_volume(
    ct_path: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    target_size: tuple = (64, 64, 64)
) -> np.ndarray:
    """
    Preprocess CT volume for model input.

    Steps:
    1. Load NIfTI image
    2. Resample to isotropic spacing
    3. Normalize HU values
    4. Crop/pad to target size
    """
    # Load image
    ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    # Resample to target spacing
    original_spacing = ct.GetSpacing()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetSize(target_size)
    resample_filter.SetInterpolator(sitk.sitkLinear)
    ct_resampled = resample_filter.Execute(ct)

    # Normalize to 0-1 range
    ct_normalized = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
    ct_normalized = np.clip(ct_normalized, -1024, 3071)  # Clip HU range
    ct_normalized = (ct_normalized + 1024) / 4095  # Normalize

    return ct_normalized
```

### Pathology Image Preprocessing

```python
from PIL import Image
import numpy as np

def preprocess_pathology_image(
    image_path: str,
    target_size: tuple = (224, 224),
    normalize_channels: bool = True
) -> np.ndarray:
    """
    Preprocess pathology image for model input.

    Steps:
    1. Load image
    2. Resize to target size
    3. Normalize per channel
    4. Stack multi-channel if needed
    """
    # Load image
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)

    # Convert to numpy
    img_array = np.array(img).astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    return img_array.transpose(2, 0, 1)  # (C, H, W)
```

---

## Dataset Statistics

| Dataset | Modality | Cases | Images | Annotations |
|---------|----------|-------|--------|-------------|
| Head-Neck-PET-CT | CT | 298 | 298 volumes | GTV, survival |
| HNSCC-mIF-mIHC | Pathology | 39 | 1000+ regions | Cell masks, markers |
| TCGA-HNSC | Pathology | ~500 | ~500 WSIs | Clinical |

---

## Citation and Attribution

When using these datasets, please cite:

1. **Head-Neck-PET-CT**:
   ```
   Vallières M, Kay F, Viswanathan A, et al. (2017)
   Data from Head-Neck-PET-CT. The Cancer Imaging Archive.
   ```

2. **HNSCC-mIF-mIHC**:
   ```
   Saltz J, Gupta R, Hou L, et al. (2018)
   Data from Head and Neck Squamous Cell Carcinoma (HNSCC)
   with multiplexed IHC and histology. The Cancer Imaging Archive.
   ```

3. **TCGA**:
   ```
   Cancer Genome Atlas Network. (2015)
   Comprehensive genomic characterization of head and neck squamous cell carcinomas.
   Nature.
   ```

---

## Data Access Links Summary

| Resource | URL | Access Type |
|----------|-----|-------------|
| TCIA Main | https://www.cancerimagingarchive.net | Free (registration) |
| Head-Neck-PET-CT | https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT | Free |
| HNSCC-mIF-mIHC | https://wiki.cancerimagingarchive.net/display/Public/HNSCC-mIF-mIHC | Free |
| GDC | https://portal.gdc.cancer.gov | Free (registration) |
| TCIA Downloader | https://github.com/kirbyju/TCIA_Notebooks | GitHub |

---

## Ethics and Compliance

1. **IRB Approval**: Ensure institutional approval for using clinical data
2. **Data Usage Agreement**: Accept TCIA terms of use
3. **Privacy**: De-identify data per HIPAA guidelines
4. **Attribution**: Include proper citations in publications

---

## Contact

For questions about data access or preprocessing:
- TCIA Support: help@cancerimagingarchive.net

