#!/usr/bin/env python3
"""
مرحله صفر: ممیزی داده (Data Audit)
نسخه مخصوص ساختار: E:/MRCP/stone/stone1/... / series_folders
"""

import os
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pydicom
from tqdm import tqdm
import yaml
from datetime import datetime

def get_tag(ds, tag, default=None):
    try:
        val = getattr(ds, tag, default)
        if val is None:
            return default
        return str(val).strip()
    except:
        return default

def compute_pixel_hash(dcm_files, max_slices=4):
    """هش از چند اسلایس اول برای پیدا کردن کپی عین‌به‌عین"""
    hashes = []
    for f in sorted(dcm_files)[:max_slices]:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)
            arr = ds.pixel_array.tobytes()
            hashes.append(hashlib.md5(arr).hexdigest())
        except:
            continue
    if not hashes:
        return None
    return hashlib.md5("".join(hashes).encode()).hexdigest()

def extract_metadata(series_dir: Path, label: str, parent_folder: str):
    dcm_files = list(series_dir.glob("*.dcm")) + list(series_dir.glob("*.DCM"))
    if len(dcm_files) == 0:
        return None

    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    except Exception as e:
        print(f"  خطا در خواندن {series_dir.name}: {e}")
        return None

    patient_id = get_tag(ds, "PatientID")
    if not patient_id:
        # اگر PatientID نبود از نام فولدر استفاده کن
        patient_id = series_dir.name

    scan_id = f"{label}_{parent_folder}_{series_dir.name}"

    meta = {
        "scan_id": scan_id,
        "patient_id": patient_id,
        "label": label,
        "parent_folder": parent_folder,          # مثلاً stone1
        "series_folder": series_dir.name,
        "series_path": str(series_dir),
        "StudyInstanceUID": get_tag(ds, "StudyInstanceUID"),
        "SeriesInstanceUID": get_tag(ds, "SeriesInstanceUID"),
        "AcquisitionDate": get_tag(ds, "AcquisitionDate") or get_tag(ds, "StudyDate"),
        "Manufacturer": get_tag(ds, "Manufacturer"),
        "ModelName": get_tag(ds, "ManufacturerModelName"),
        "MagneticFieldStrength": get_tag(ds, "MagneticFieldStrength"),
        "SliceThickness": get_tag(ds, "SliceThickness"),
        "SpacingBetweenSlices": get_tag(ds, "SpacingBetweenSlices"),
        "num_slices": len(dcm_files),
        "SeriesDescription": get_tag(ds, "SeriesDescription"),
        "pixel_hash": compute_pixel_hash(dcm_files),
    }
    return meta

def main():
    config_path = "E:\MRCP\config.yml"
    
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data_root"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_csv = Path(cfg["metadata_csv"])
    min_slices = cfg.get("min_slices", 30)
    classes = cfg.get("classes", ["stone", "nonstone"])

    records = []

    print("="*70)
    print("شروع مرحله صفر - ممیزی داده")
    print(f"زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    for label in classes:
        class_dir = data_root / label
        if not class_dir.exists():
            print(f"\n[!] پوشه {class_dir} وجود ندارد - رد شد")
            continue

        print(f"\nدر حال اسکن کلاس: {label}")
        print(f"مسیر: {class_dir}")

        # سطح اول: stone1, stone2, ...
        parent_folders = [d for d in class_dir.iterdir() if d.is_dir()]
        
        for parent in tqdm(parent_folders, desc=f"  {label}"):
            # سطح دوم: فولدرهای سری (هر کدام ≈۸۷ تا dcm)
            series_dirs = [d for d in parent.iterdir() if d.is_dir()]
            
            for series_dir in series_dirs:
                dcm_count = len(list(series_dir.glob("*.dcm")) + list(series_dir.glob("*.DCM")))
                if dcm_count < min_slices:
                    continue
                
                meta = extract_metadata(series_dir, label, parent.name)
                if meta:
                    records.append(meta)

    if not records:
        print("\nهیچ داده‌ای پیدا نشد! مسیرها را چک کن.")
        return

    df = pd.DataFrame(records)
    df.to_csv(metadata_csv, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*70)
    print("نتیجه ممیزی")
    print("="*70)
    print(f"تعداد کل اسکن‌ها          : {len(df)}")
    print(f"تعداد بیماران یکتا         : {df['patient_id'].nunique()}")
    print(f"فایل ذخیره شد در           : {metadata_csv}")

    # ---------- گزارش‌های مهم ----------
    print("\n--- ۱. بیماران با چند اسکن ---")
    multi = df.groupby("patient_id").size()
    multi = multi[multi > 1].sort_values(ascending=False)
    print(f"تعداد بیمارانی که بیش از یک اسکن دارند: {len(multi)}")
    if len(multi) > 0:
        print(multi.head(10))

    print("\n--- ۲. بررسی کپی عین‌به‌عین (Duplicate پیکسلی) ---")
    hash_counts = df["pixel_hash"].value_counts()
    dups = hash_counts[hash_counts > 1]
    if len(dups) > 0:
        print(f"[!] {len(dups)} گروه کپی پیدا شد:")
        for h in dups.index[:5]:
            print(df[df["pixel_hash"] == h][["scan_id", "patient_id", "parent_folder", "series_folder"]])
    else:
        print("[✓] کپی عین‌به‌عین پیدا نشد.")

    print("\n--- ۳. توزیع متادیتا (خطر Shortcut Learning) ---")
    for col in ["Manufacturer", "MagneticFieldStrength", "ModelName"]:
        if col in df.columns and df[col].notna().any():
            print(f"\n{col}:")
            print(pd.crosstab(df[col].fillna("Unknown"), df["label"]))

    print("\n--- ۴. تعداد اسلایس ---")
    print(df["num_slices"].describe())

    print("\n" + "="*70)
    print("مرحله صفر تمام شد. فایل CSV را باز کن و بررسی کن.")
    print("="*70)

if __name__ == "__main__":
    main()