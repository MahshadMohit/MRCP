#!/usr/bin/env python3
"""
ساخت جدول سطح-بیمار (Patient-Level Table)
دقیقاً مطابق مشخصات درخواستی
"""

import pandas as pd
import numpy as np
from pathlib import Path

# مسیر فایل متادیتا
CSV_PATH = r"E:\MRCP\processed\metadata_stage0.csv"
OUTPUT_PATH = r"E:\MRCP\processed\patient_level_table.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    
    # استاندارد کردن نام کلاس
    df["label_clean"] = df["label"].replace({"nostone": "nonstone"})
    
    # ترکیب Manufacturer و Model برای گروه دستگاه
    df["Device"] = (
        df["Manufacturer"].fillna("Unknown").astype(str) + " | " + 
        df["ModelName"].fillna("Unknown").astype(str)
    )
    
    # تبدیل تاریخ
    df["AcquisitionDate"] = pd.to_datetime(
        df["AcquisitionDate"], format="%Y%m%d", errors="coerce"
    )
    
    # مرتب‌سازی بر اساس تاریخ تا "اسکن اول" مشخص شود
    df = df.sort_values(["patient_id", "AcquisitionDate"])
    
    # --------------------------------------------------
    # ساخت جدول سطح-بیمار
    # --------------------------------------------------
    records = []
    
    for patient_id, group in df.groupby("patient_id"):
        labels = set(group["label_clean"].unique())
        devices = group["Device"].unique().tolist()
        dates = group["AcquisitionDate"].dropna()
        
        # برچسب اصلی (اگر در هر دو کلاس باشد → both)
        if len(labels) == 1:
            main_label = list(labels)[0]
            in_both_classes = False
        else:
            main_label = "both"
            in_both_classes = True
        
        # اسکن اول
        first_scan = group.iloc[0]
        first_device = first_scan["Device"]
        first_date = first_scan["AcquisitionDate"]
        
        # پرچم‌ها
        multi_device = len(devices) > 1
        has_2026 = False
        if len(dates) > 0:
            has_2026 = (dates.dt.year == 2026).any()
        
        record = {
            "patient_id": patient_id,
            "label": main_label,                          # stone / nonstone / both
            "num_scans": len(group),
            "device_group": first_device,                   # دستگاه اسکن اول
            "first_scan_year": first_date.year if pd.notna(first_date) else None,
            "first_scan_month": first_date.month if pd.notna(first_date) else None,
            "hospital": None,                             # در متادیتا موجود نیست
            "age": None,                                  # در متادیتا موجود نیست
            "gender": None,                               # در متادیتا موجود نیست
            "flag_multi_device": multi_device,             # دو اسکن روی دو دستگاه متفاوت
            "flag_has_2026": has_2026,                    # دارای اسکن ۲۰۲۶
            "flag_in_both_classes": in_both_classes,      # بیمار در هر دو کلاس
            "all_devices": " | ".join(devices),           # لیست همه دستگاه‌های این بیمار
            "all_labels": " | ".join(sorted(labels)),     # برای بررسی
        }
        records.append(record)
    
    patient_df = pd.DataFrame(records)
    
    # ذخیره
    patient_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    # --------------------------------------------------
    # گزارش خلاصه
    # --------------------------------------------------
    print("="*70)
    print("جدول سطح-بیمار با موفقیت ساخته شد")
    print("="*70)
    print(f"مسیر ذخیره: {OUTPUT_PATH}")
    print(f"تعداد کل بیماران: {len(patient_df)}")
    
    print("\nتوزیع برچسب:")
    print(patient_df["label"].value_counts())
    
    print("\nتعداد بیماران با چند اسکن:")
    print((patient_df["num_scans"] > 1).sum())
    
    print("\nبیماران با پرچم «در هر دو کلاس»:")
    both = patient_df[patient_df["flag_in_both_classes"]]
    print(f"تعداد: {len(both)}")
    if len(both) > 0:
        print(both[["patient_id", "label", "all_labels", "all_devices"]].to_string(index=False))
    
    print("\nبیماران با دو دستگاه متفاوت:")
    multi_dev = patient_df[patient_df["flag_multi_device"]]
    print(f"تعداد: {len(multi_dev)}")
    
    print("\nتوزیع گروه دستگاه (بر اساس اسکن اول):")
    print(patient_df["device_group"].value_counts())
    
    print("\n" + "="*70)
    print("ستون‌های جدول:")
    print(list(patient_df.columns))
    print("="*70)

if __name__ == "__main__":
    main()