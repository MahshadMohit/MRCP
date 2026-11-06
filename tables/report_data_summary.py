#!/usr/bin/env python3
"""
گزارش کامل داده‌ها بعد از مرحله صفر و پیش‌پردازش
"""

import pandas as pd
from pathlib import Path

# مسیر فایل متادیتا
CSV_PATH = r"E:\MRCP\processed\metadata_stage0.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    
    # استاندارد کردن نام کلاس (nostone → nonstone برای خوانایی)
    df["label_clean"] = df["label"].replace({"nostone": "nonstone"})
    
    print("="*70)
    print("گزارش کامل داده‌های MRCP")
    print("="*70)
    
    # ----------------------------------------------------------
    # ۱. بیمار به تفکیک دستگاه و کلاس
    # ----------------------------------------------------------
    print("\n۱. تعداد بیمار به تفکیک دستگاه (Manufacturer + Model) و کلاس")
    print("-"*70)
    
    # ترکیب Manufacturer و Model
    df["Device"] = df["Manufacturer"].fillna("Unknown") + " | " + df["ModelName"].fillna("Unknown")
    
    patient_device = df.groupby(["Device", "label_clean"])["patient_id"].nunique().unstack(fill_value=0)
    patient_device["Total"] = patient_device.sum(axis=1)
    print(patient_device.sort_values("Total", ascending=False))
    
    print("\n\nتعداد اسکن به تفکیک دستگاه و کلاس:")
    scan_device = df.groupby(["Device", "label_clean"]).size().unstack(fill_value=0)
    scan_device["Total"] = scan_device.sum(axis=1)
    print(scan_device.sort_values("Total", ascending=False))
    
    # ----------------------------------------------------------
    # ۲. خلاصه آماری تعداد اسلایس به تفکیک کلاس، درون هر دستگاه
    # ----------------------------------------------------------
    print("\n\n۲. خلاصه آماری تعداد اسلایس (num_slices) به تفکیک کلاس و دستگاه")
    print("-"*70)
    
    slice_stats = df.groupby(["Device", "label_clean"])["num_slices"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    ).round(1)
    print(slice_stats)
    
    # ----------------------------------------------------------
    # ۳. محدوده تاریخ اکتساب هر کلاس + تعداد اسکن‌های ۲۰۲۶
    # ----------------------------------------------------------
    print("\n\n۳. محدوده تاریخ اکتساب و تعداد اسکن‌های سال ۲۰۲۶")
    print("-"*70)
    
    # تبدیل تاریخ
    df["AcquisitionDate"] = pd.to_datetime(df["AcquisitionDate"], format="%Y%m%d", errors="coerce")
    
    for label in ["stone", "nonstone"]:
        sub = df[df["label_clean"] == label]
        valid_dates = sub["AcquisitionDate"].dropna()
        
        if len(valid_dates) > 0:
            min_date = valid_dates.min().strftime("%Y-%m-%d")
            max_date = valid_dates.max().strftime("%Y-%m-%d")
            count_2026 = (valid_dates.dt.year == 2026).sum()
            print(f"\nکلاس {label}:")
            print(f"  تعداد اسکن با تاریخ معتبر : {len(valid_dates)}")
            print(f"  محدوده تاریخ             : {min_date}  تا  {max_date}")
            print(f"  تعداد اسکن سال ۲۰۲۶      : {count_2026}")
        else:
            print(f"\nکلاس {label}: تاریخ معتبری پیدا نشد")
    
    # ----------------------------------------------------------
    # ۴. بیمارانی که هم در stone و هم در nonstone هستند
    # ----------------------------------------------------------
    print("\n\n۴. بیمارانی که هم در stone و هم در nonstone حضور دارند")
    print("-"*70)
    
    stone_patients = set(df[df["label_clean"] == "stone"]["patient_id"])
    nonstone_patients = set(df[df["label_clean"] == "nonstone"]["patient_id"])
    
    common_patients = stone_patients.intersection(nonstone_patients)
    
    print(f"تعداد بیمار مشترک: {len(common_patients)}")
    
    if len(common_patients) > 0:
        print("\nلیست بیماران مشترک:")
        common_df = df[df["patient_id"].isin(common_patients)][
            ["patient_id", "label_clean", "AcquisitionDate", "Device", "scan_id"]
        ].sort_values(["patient_id", "label_clean"])
        print(common_df.to_string(index=False))
    else:
        print("هیچ بیماری در هر دو کلاس وجود ندارد (خوب است).")
    
    print("\n" + "="*70)
    print("گزارش تمام شد")
    print("="*70)

if __name__ == "__main__":
    main()