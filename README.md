
## Model Development Pipeline

The project follows a structured six-stage pipeline, from DICOM data preparation to model evaluation and ensemble learning.

### 1. Metadata Collection and Data Auditing

A centralized metadata table (`CSV`) is created to organize patient-level and scan-level information.

Key metadata fields include:

- `Scan_id`: Unique identifier for each scan
- `Patient_id`: Patient identifier for leakage-free data splitting
- `Scan_index`: Identification of multiple scans belonging to the same patient
- `Label` / `Label_num`: Original and numerical classification labels
- `DICOM_dir` / `NIfTI_dir`: Data storage paths
- DICOM acquisition metadata, including `PixelSpacing`, `SliceThickness`, `SeriesDescription`, and `StudyInstanceUID`

### 2. 3D Volume Reconstruction

- Selection of appropriate 3D MRCP series based on `SeriesDescription` and slice count.
- Reconstruction of volumetric images from DICOM series.
- Conversion to NIfTI format.
- Standardization of image orientation and spatial resolution.
- 3D resampling to a target slice thickness of 2 mm.

### 3. Image Preprocessing and Augmentation

A standardized preprocessing workflow is applied to all image volumes:

- Spatial orientation standardization (RAS).
- Isotropic in-plane resampling and slice-thickness standardization.
- Intensity normalization using percentile clipping and Z-score normalization.
- Foreground-based ROI cropping using segmentation tools such as TotalSegmentator.
- Padding and cropping to a fixed input size of `224 × 224 × 64`.

Training-time data augmentation includes:

- Small-angle rotations
- Scaling and translation
- Gaussian noise injection

Augmentation is applied exclusively to the training set.
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/f6d95f9e-c6f2-40df-a058-ceca72dc8e70" alt="Stone-Stone 3D MRCP Visualization" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/96a4cd6e-1adc-43dd-86d5-c18a05df5382" alt="NoStone-NoStone 3D MRCP Visualization" width="100%">
    </td>
  </tr>
</table>



### 4. Patient-Level Dataset Splitting

The dataset is partitioned using stratified patient-level splitting to prevent data leakage.

- Training: 70%
- Validation: 15%
- Internal Testing: 15%
- Stratification based on classification labels and device groups.
- Multiple scans belonging to the same patient are assigned to the same subset.

The final split is stored in `split_v1.csv`.

### 5. Deep Learning Model Development

Four primary architectures are selected to compare CNN-based and Transformer-based approaches.

| Architecture | Category | Purpose |
|---|---|---|
| 3D ResNet | CNN | Baseline model with pretrained weights |
| 3D ConvNeXt | CNN | Advanced CNN architecture |
| 3D Swin Transformer | Transformer | Hierarchical 3D feature extraction |
| 3D Vision Transformer (ViT) | Transformer | Global attention-based feature learning |

DenseNet121 is also considered as an alternative CNN architecture.

Model-specific training configurations are used, and model checkpoints are saved for reproducibility and subsequent evaluation.

### 6. Model Evaluation and Ensemble Learning

Model performance is evaluated using classification and calibration metrics:

- AUROC
- Accuracy
- Sensitivity
- Specificity
- Positive Predictive Value (PPV)
- Negative Predictive Value (NPV)
- Calibration Curve
- Brier Score

Further analysis includes:

- False Negative (FN) analysis to identify missed CBD stones.
- False Positive (FP) analysis to investigate incorrect detections.
- Comparative evaluation of individual model performance.
- Error analysis and model interpretation.
- Ensemble learning by combining two complementary models to improve predictive performance.

The final objective is to identify the most reliable architecture and investigate whether combining complementary models can improve CBD stone detection.

