#!/usr/bin/env python3
"""
مرحله ۳: پیش‌پردازش کامل DICOM → NIfTI
نسخه نهایی بر اساس metadata_stage0.csv و ساختار فعلی پروژه
"""

import os
import yaml
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

def read_dicom_series(series_dir):
    series_dir = str(series_dir)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(series_dir)
    if not series_ids:
        raise ValueError(f"هیچ سری DICOM پیدا نشد: {series_dir}")
    
    file_names = reader.GetGDCMSeriesFileNames(series_dir, series_ids[0])
    reader.SetFileNames(file_names)
    image = reader.Execute()
    return image

def normalize_intensity(image, percentiles=(0.5, 99.5), method="zscore"):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    
    p_low, p_high = np.percentile(arr, percentiles)
    arr = np.clip(arr, p_low, p_high)
    
    if method == "zscore":
        mean, std = arr.mean(), arr.std()
        arr = (arr - mean) / (std + 1e-8)
    else:  # minmax
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    
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
    """target_size = (H, W, D)"""
    arr = sitk.GetArrayFromImage(image)  # (D, H, W)
    target_zyx = (target_size[2], target_size[0], target_size[1])
    
    # Pad
    pads = []
    for cur, tgt in zip(arr.shape, target_zyx):
        diff = max(0, tgt - cur)
        before = diff // 2
        after = diff - before
        pads.append((before, after))
    
    arr = np.pad(arr, pads, mode="constant", constant_values=arr.min())
    
    # Crop از مرکز
    starts = [(s - t)//2 if s > t else 0 for s, t in zip(arr.shape, target_zyx)]
    arr = arr[
        starts[0]:starts[0]+target_zyx[0],
        starts[1]:starts[1]+target_zyx[1],
        starts[2]:starts[2]+target_zyx[2]
    ]
    
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(image.GetSpacing())
    out.SetDirection(image.GetDirection())
    out.SetOrigin(image.GetOrigin())
    return out

def save_qc_image(image, save_path):
    arr = sitk.GetArrayFromImage(image)
    mid_z = arr.shape[0] // 2
    mid_y = arr.shape[1] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(arr[mid_z], cmap="gray")
    axes[0].set_title(f"Axial (z={mid_z})")
    axes[0].axis("off")
    
    axes[1].imshow(arr[:, mid_y, :], cmap="gray")
    axes[1].set_title(f"Coronal (y={mid_y})")
    axes[1].axis("off")
    
    axes[2].imshow(arr.max(axis=0), cmap="gray")
    axes[2].set_title("MIP")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()

def process_one(row, cfg, output_nii_dir, output_qc_dir):
    series_path = Path(row["series_path"])
    scan_id = row["scan_id"]
    label = row["label"]
    
    try:
        # ۱. خواندن
        image = read_dicom_series(series_path)
        
        # ۲. Orientation
        image = sitk.DICOMOrient(image, cfg["orientation"])
        
        # ۳. Resampling
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        target_spacing = np.array(cfg["target_spacing"])
        
        new_size = (original_size * original_spacing / target_spacing).round().astype(int).tolist()
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(cfg["target_spacing"])
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        image = resampler.Execute(image)
        
        # ۴. Intensity Normalization
        image = normalize_intensity(
            image,
            percentiles=cfg["intensity"]["clip_percentiles"],
            method=cfg["intensity"]["method"]
        )
        
        # ۵. ROI Crop
        if cfg["roi"]["method"] == "foreground":
            image = foreground_crop(
                image,
                threshold_percentile=cfg["roi"]["threshold_percentile"],
                margin=cfg["roi"]["margin"]
            )
        
        # ۶. Pad/Crop به اندازه ثابت
        image = pad_or_crop(image, cfg["target_size"])
        
        # ۷. ذخیره NIfTI
        out_nii = output_nii_dir / label / f"{scan_id}.nii.gz"
        out_nii.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(out_nii), useCompression=True)
        
        # ۸. QC
        qc_path = None
        if cfg.get("save_qc", False):
            qc_path = output_qc_dir / label / f"{scan_id}.png"
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            save_qc_image(image, str(qc_path))
        
        return {
            "scan_id": scan_id,
            "success": True,
            "output_nii": str(out_nii),
            "final_size": image.GetSize(),
            "final_spacing": image.GetSpacing(),
            "error": None
        }
        
    except Exception as e:
        return {
            "scan_id": scan_id,
            "success": False,
            "output_nii": None,
            "final_size": None,
            "final_spacing": None,
            "error": str(e)
        }

def main():
    config_path = "E:\MRCP\onfig_preprocess.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    df = pd.read_csv(cfg["metadata_csv"])
    
    # فیلتر تعداد اسلایس
    min_slices = cfg.get("min_slices", 80)
    df = df[df["num_slices"] >= min_slices].copy()
    print(f"تعداد اسکن بعد از فیلتر min_slices={min_slices}: {len(df)}")
    
    output_dir = Path(cfg["output_dir"])
    output_nii_dir = output_dir / "nii"
    output_qc_dir = output_dir / "qc"
    
    results = []
    
    print("="*70)
    print("شروع پیش‌پردازش دسته‌ای")
    print(f"زمان شروع: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # کنترل ذخیره QC
        if cfg.get("save_qc", False) and (idx % cfg.get("qc_every_n", 1) != 0):
            cfg_temp = cfg.copy()
            cfg_temp["save_qc"] = False
        else:
            cfg_temp = cfg
        
        res = process_one(row, cfg_temp, output_nii_dir, output_qc_dir)
        results.append(res)
    
    # ذخیره لاگ
    res_df = pd.DataFrame(results)
    log_path = output_dir / "preprocessing_log.csv"
    res_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    
    success = res_df["success"].sum()
    print("\n" + "="*70)
    print(f"پیش‌پردازش تمام شد")
    print(f"موفق: {success} / {len(res_df)}")
    print(f"ناموفق: {len(res_df) - success}")
    print(f"لاگ ذخیره شد در: {log_path}")
    print("="*70)
    
    if len(res_df) - success > 0:
        print("\nاسکن‌های ناموفق:")
        print(res_df[~res_df["success"]][["scan_id", "error"]])

if __name__ == "__main__":
    main()