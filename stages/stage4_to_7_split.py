#!/usr/bin/env python3
"""
قدم‌های ۴ تا ۷:
- تعریف گروه‌های stratification
- تقسیم ۷۰/۱۵/۱۵ بیمارمحور
- برگرداندن تقسیم به سطح اسکن
- ثبت اسکن‌های حذف‌شده
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime

# ====================== مسیرها ======================
PATIENT_TABLE = r"E:\MRCP\processed\patient_level_table.csv"
METADATA_CSV  = r"E:\MRCP\processed\metadata_stage0.csv"
PREPROCESS_LOG = r"E:\MRCP\processed\preprocessing_log.csv"
OUTPUT_DIR    = Path(r"E:\MRCP\processed")

SEED = 42
np.random.seed(SEED)

def map_device_group(device_str):
    """گروه‌بندی نهایی دستگاه‌ها"""
    if pd.isna(device_str):
        return "Unknown"
    d = str(device_str).lower()
    if "philips" in d:
        return "Philips"
    elif "siemens" in d or "magnetom" in d or "avanto" in d:
        return "Siemens"
    else:
        return "Other"

def main():
    print("="*70)
    print("شروع قدم‌های ۴ تا ۷ - ساخت Split")
    print(f"زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {SEED}")
    print("="*70)

    # --------------------------------------------------
    # خواندن جداول
    # --------------------------------------------------
    patient_df = pd.read_csv(PATIENT_TABLE)
    meta_df = pd.read_csv(METADATA_CSV)
    log_df = pd.read_csv(PREPROCESS_LOG)

    # استاندارد کردن نام کلاس
    meta_df["label_clean"] = meta_df["label"].replace({"nostone": "nonstone"})
    patient_df["label_clean"] = patient_df["label"].replace({"both": "both"})

    # --------------------------------------------------
    # قدم ۴: تعریف گروه‌های stratification
    # --------------------------------------------------
    print("\n[قدم ۴] تعریف گروه‌های دستگاه...")
    patient_df["strata_device"] = patient_df["device_group"].apply(map_device_group)

    # حذف بیمار both
    both_patients = patient_df[patient_df["flag_in_both_classes"]]["patient_id"].tolist()
    print(f"حذف بیمار(ان) both: {both_patients}")
    patient_df = patient_df[~patient_df["flag_in_both_classes"]].copy()

    # ساخت کلید stratification = label × device
    patient_df["strata"] = patient_df["label"] + "_" + patient_df["strata_device"]

    print("\nتوزیع strata بعد از حذف both:")
    print(patient_df["strata"].value_counts())

    # --------------------------------------------------
    # قدم ۵: تقسیم ۷۰ / ۱۵ / ۱۵
    # --------------------------------------------------
    print("\n[قدم ۵] تقسیم ۷۰/۱۵/۱۵ ...")

    # اول ۷۰٪ train و ۳۰٪ temp
    train_ids, temp_ids = train_test_split(
        patient_df["patient_id"],
        test_size=0.30,
        stratify=patient_df["strata"],
        random_state=SEED
    )

    # ۳۰٪ باقی‌مانده را نصف می‌کنیم → ۱۵٪ val و ۱۵٪ test
    temp_df = patient_df[patient_df["patient_id"].isin(temp_ids)].copy()
    
    val_ids, test_ids = train_test_split(
        temp_df["patient_id"],
        test_size=0.50,
        stratify=temp_df["strata"],
        random_state=SEED
    )

    # ساخت ستون split
    patient_df["split"] = "train"
    patient_df.loc[patient_df["patient_id"].isin(val_ids), "split"] = "val"
    patient_df.loc[patient_df["patient_id"].isin(test_ids), "split"] = "test"

    print("\nتوزیع بیماران در هر split:")
    print(pd.crosstab(patient_df["split"], patient_df["label"]))
    print("\nتوزیع strata × split:")
    print(pd.crosstab(patient_df["strata"], patient_df["split"]))

    # ذخیره جدول بیمار + split
    patient_split_path = OUTPUT_DIR / "patient_split.csv"
    patient_df.to_csv(patient_split_path, index=False, encoding="utf-8-sig")
    print(f"\nجدول بیمار+split ذخیره شد → {patient_split_path}")

    # --------------------------------------------------
    # قدم ۶: برگرداندن split به سطح اسکن
    # --------------------------------------------------
    print("\n[قدم ۶] نگاشت split به اسکن‌ها...")

    # فقط اسکن‌هایی که پیش‌پردازش موفق داشته‌اند
    success_scans = set(log_df[log_df["success"] == True]["scan_id"])

    # اتصال meta به patient split
    meta_df = meta_df.merge(
        patient_df[["patient_id", "split", "strata_device"]],
        on="patient_id",
        how="left"
    )

    # فقط اسکن‌های موفق و بیمارانی که در split هستند
    scan_split = meta_df[
        (meta_df["scan_id"].isin(success_scans)) &
        (meta_df["split"].notna())
    ].copy()

    scan_split_path = OUTPUT_DIR / "scan_split.csv"
    scan_split.to_csv(scan_split_path, index=False, encoding="utf-8-sig")
    print(f"جدول اسکن+split ذخیره شد → {scan_split_path}")
    print(f"تعداد اسکن نهایی: {len(scan_split)}")

    print("\nتوزیع اسکن‌ها در هر split:")
    print(pd.crosstab(scan_split["split"], scan_split["label_clean"]))

    # --------------------------------------------------
    # قدم ۷: ثبت اسکن‌های حذف‌شده
    # --------------------------------------------------
    print("\n[قدم ۷] ثبت اسکن‌های حذف‌شده...")

    excluded = []

    # ۱. بیمار both
    for pid in both_patients:
        rows = meta_df[meta_df["patient_id"] == pid]
        for _, r in rows.iterrows():
            excluded.append({
                "scan_id": r["scan_id"],
                "patient_id": r["patient_id"],
                "label": r["label_clean"],
                "reason": "بیمار در هر دو کلاس (both)"
            })

    # ۲. اسکن‌هایی که پیش‌پردازش ناموفق بودند
    failed = log_df[log_df["success"] == False]
    for _, r in failed.iterrows():
        excluded.append({
            "scan_id": r["scan_id"],
            "patient_id": None,
            "label": None,
            "reason": f"خطا در پیش‌پردازش: {r.get('error', '')[:80]}"
        })

    # ۳. اسکن‌هایی که در meta هستند ولی به خاطر حذف بیمار یا فیلترهای دیگر نیامده‌اند
    all_meta_ids = set(meta_df["scan_id"])
    kept_ids = set(scan_split["scan_id"])
    already_excluded = set([e["scan_id"] for e in excluded])

    remaining = all_meta_ids - kept_ids - already_excluded
    for sid in remaining:
        row = meta_df[meta_df["scan_id"] == sid].iloc[0]
        excluded.append({
            "scan_id": sid,
            "patient_id": row["patient_id"],
            "label": row["label_clean"],
            "reason": "حذف به دلیل بیمار both یا فیلتر دیگر"
        })

    excluded_df = pd.DataFrame(excluded)
    excluded_path = OUTPUT_DIR / "excluded_scans.csv"
    excluded_df.to_csv(excluded_path, index=False, encoding="utf-8-sig")

    print(f"فهرست اسکن‌های حذف‌شده ذخیره شد → {excluded_path}")
    print(f"تعداد کل حذف‌شده: {len(excluded_df)}")
    print("\nدلیل حذف‌ها:")
    print(excluded_df["reason"].value_counts())

    # --------------------------------------------------
    # خلاصه نهایی
    # --------------------------------------------------
    print("\n" + "="*70)
    print("خلاصه نهایی Split")
    print("="*70)
    print(f"Seed مورد استفاده          : {SEED}")
    print(f"تعداد بیماران نهایی        : {len(patient_df)}")
    print(f"تعداد اسکن نهایی           : {len(scan_split)}")
    print(f"تعداد اسکن حذف‌شده         : {len(excluded_df)}")
    print("\nفایل‌های خروجی:")
    print(f"  - {patient_split_path}")
    print(f"  - {scan_split_path}")
    print(f"  - {excluded_path}")
    print("="*70)

if __name__ == "__main__":
    main()