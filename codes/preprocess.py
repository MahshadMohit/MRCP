import os
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader

STONE_ROOT = r""
NOSTONE_ROOT = r""
OUTPUT_NII_ROOT = r""
SPLIT_CSV = r""
TARGET_SPACING = (1.0, 1.0, 2.0)
TARGET_SIZE = (224, 224, 64)
ORIENTATION = "RAS"
CLIP_PERCENTILES = (0.5, 99.5)

def read_dicom_series(series_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
    if not series_ids:
        raise ValueError(f"No DICOM series found in {series_dir}")
    file_names = reader.GetGDCMSeriesFileNames(str(series_dir), series_ids[0])
    reader.SetFileNames(file_names)
    return reader.Execute()

def normalize_intensity(image, percentiles=CLIP_PERCENTILES):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    p_low, p_high = np.percentile(arr, percentiles)
    arr = np.clip(arr, p_low, p_high)
    mean, std = arr.mean(), arr.std()
    arr = (arr - mean) / (std + 1e-8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out

def foreground_crop(image, threshold_percentile=1.0, margin=8):
    arr = sitk.GetArrayFromImage(image)
    thresh = np.percentile(arr, threshold_percentile)
    mask = arr > thresh
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return image
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    zmin = max(0, zmin - margin)
    ymin = max(0, ymin - margin)
    xmin = max(0, xmin - margin)
    zmax = min(arr.shape[0], zmax + margin)
    ymax = min(arr.shape[1], ymax + margin)
    xmax = min(arr.shape[2], xmax + margin)
    return image[xmin:xmax, ymin:ymax, zmin:zmax]

def pad_or_crop(image, target_size):
    arr = sitk.GetArrayFromImage(image)
    target_zyx = (target_size[2], target_size[0], target_size[1])
    pads = []
    for cur, tgt in zip(arr.shape, target_zyx):
        diff = max(0, tgt - cur)
        before = diff // 2
        after = diff - before
        pads.append((before, after))
    arr = np.pad(arr, pads, mode="constant", constant_values=arr.min())
    starts = [(s - t) // 2 if s > t else 0 for s, t in zip(arr.shape, target_zyx)]
    arr = arr[
        starts[0]:starts[0] + target_zyx[0],
        starts[1]:starts[1] + target_zyx[1],
        starts[2]:starts[2] + target_zyx[2]
    ]
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(image.GetSpacing())
    out.SetDirection(image.GetDirection())
    out.SetOrigin(image.GetOrigin())
    return out

def process_series(series_dir, output_path):
    image = read_dicom_series(series_dir)
    image = sitk.DICOMOrient(image, ORIENTATION)
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())
    target_spacing = np.array(TARGET_SPACING)
    new_size = (original_size * original_spacing / target_spacing).round().astype(int).tolist()
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    image = resampler.Execute(image)
    image = normalize_intensity(image)
    image = foreground_crop(image)
    image = pad_or_crop(image, TARGET_SIZE)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), useCompression=True)
    return image.GetSize()

class MRCPDataset(Dataset):
    def __init__(self, split_csv, split="train", nii_root=None):
        self.nii_root = Path(nii_root) if nii_root else Path(OUTPUT_NII_ROOT)
        df = pd.read_csv(split_csv)
        df["label"] = df["label"].replace({"nostone": "nonstone"})
        self.df = df[df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load(self, scan_id, label):
        folder = "nostone" if label == "nonstone" else "stone"
        path = self.nii_root / folder / f"{scan_id}.nii.gz"
        if not path.exists():
            path = self.nii_root / label / f"{scan_id}.nii.gz"
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vol = self._load(row["scan_id"], row["label"])
        vol = vol[np.newaxis, ...]
        label = 1.0 if row["label"] == "stone" else 0.0
        return torch.from_numpy(vol), torch.tensor(label, dtype=torch.float32)

def get_loaders(split_csv, batch_size=4, num_workers=4, nii_root=None):
    train_ds = MRCPDataset(split_csv, split="train", nii_root=nii_root)
    val_ds = MRCPDataset(split_csv, split="val", nii_root=nii_root)
    test_ds = MRCPDataset(split_csv, split="test", nii_root=nii_root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader