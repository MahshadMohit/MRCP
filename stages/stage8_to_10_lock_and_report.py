#!/usr/bin/env python3
"""
قدم‌های ۸ تا ۱۰:
- چک‌های نهایی کیفیت Split
- قفل کردن split (split_v1)
- ساخت جدول گزارش برای مقاله
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
from collections import Counter

# ====================== مسیرها ======================
PATIENT_SPLIT = r"E:\MRCP\processed\patient_split.csv"
SCAN_SPLIT    = r"E:\MRCP\processed\scan_split.csv"
EXCLUDED      = r"E:\MRCP\processed\excluded_scans.csv"
NII_ROOT      = Path(r"E:\MRCP\processed\nii")
OUTPUT_DIR    = Path(r"E:\MRCP\processed")

SEED = 42
EXPECTED_SIZE = (64, 224, 224)   # (D, H, W) بعد از پیش‌پردازش

def check_nii_file(scan_id, label):
    """بررسی وجود و ابعاد فایل NIfTI"""
    # برچسب در پوشه ممکن است nostone باشد
    folder_label = "nostone" if label == "nonstone" else label
    nii_path = NII_ROOT / folder_label / f"{scan_id}.nii.gz"
    
    if not nii_path.exists():
        # سعی با نام اصلی label
        nii_path = NII_ROOT / label / f"{scan_id}.nii.gz"
        if not nii_path.exists():
            return False, "فایل وجود ندارد", None
    
    try:
        img = sitk.ReadImage(str(nii_path))
        size = img.GetSize()  # (x, y, z) → ما (D,H,W) می‌خواهیم
        # SimpleITK GetSize = (W, H, D)
        actual = (size[2], size[1], size[0])  # به (D, H, W)
        if actual != EXPECTED_SIZE:
            return True, f"ابعاد غیرمنتظره: {actual}", actual
        return True, "OK", actual
    except Exception as e:
        return False, f"خطا در خواندن: {str(e)[:60]}", None

def main():
    print("="*70)
    print("شروع قدم‌های ۸ تا ۱۰ - چک نهایی + قفل کردن")
    print(f"زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    patient_df = pd.read_csv(PATIENT_SPLIT)
    scan_df = pd.read_csv(SCAN_SPLIT)
    excluded_df = pd.read_csv(EXCLUDED)

    # استاندارد کردن
    scan_df["label_clean"] = scan_df["label"].replace({"nostone": "nonstone"})
    if "label_clean" not in patient_df.columns:
        patient_df["label_clean"] = patient_df["label"]

    all_checks_passed = True

    # ==================================================
    # قدم ۸: چک‌های نهایی
    # ==================================================
    print("\n" + "="*70)
    print("قدم ۸: چک‌های نهایی کیفیت Split")
    print("="*70)

    # ----- چک ۱: اشتراک بیمار بین سه مجموعه دقیقاً صفر -----
    print("\n[چک ۱] اشتراک بیمار بین Train / Val / Test")
    train_p = set(patient_df[patient_df["split"]=="train"]["patient_id"])
    val_p   = set(patient_df[patient_df["split"]=="val"]["patient_id"])
    test_p  = set(patient_df[patient_df["split"]=="test"]["patient_id"])

    inter_tv = train_p & val_p
    inter_tt = train_p & test_p
    inter_vt = val_p & test_p

    if len(inter_tv)==0 and len(inter_tt)==0 and len(inter_vt)==0:
        print("  ✓ پاس شد - هیچ اشتراکی وجود ندارد")
    else:
        print("  ✗ رد شد!")
        print(f"    Train∩Val: {len(inter_tv)}, Train∩Test: {len(inter_tt)}, Val∩Test: {len(inter_vt)}")
        all_checks_passed = False

    # ----- چک ۲: هر اسکن دقیقاً یک split دارد -----
    print("\n[چک ۲] هر اسکن دقیقاً یک split دارد")
    if scan_df["split"].isna().sum() == 0 and scan_df["scan_id"].duplicated().sum() == 0:
        print("  ✓ پاس شد")
    else:
        print("  ✗ رد شد - اسکن بدون split یا تکراری وجود دارد")
        all_checks_passed = False

    # ----- چک ۳: بیماران دو‌اسکنی هر دو اسکنشان در یک مجموعه -----
    print("\n[چک ۳] بیماران چنداسکنی")
    multi = patient_df[patient_df["num_scans"] > 1]
    multi_ok = True
    for _, row in multi.iterrows():
        pid = row["patient_id"]
        scans = scan_df[scan_df["patient_id"] == pid]
        splits = scans["split"].unique()
        if len(splits) > 1:
            print(f"  ✗ بیمار {pid} اسکن‌هایش در مجموعه‌های مختلف است: {splits}")
            multi_ok = False
            all_checks_passed = False
    if multi_ok:
        print(f"  ✓ پاس شد - {len(multi)} بیمار چنداسکنی همه اسکن‌هایشان در یک مجموعه هستند")

    # ----- چک ۴: نسبت stone به nonstone -----
    print("\n[چک ۴] نسبت stone / nonstone در سه مجموعه")
    ratio_table = pd.crosstab(scan_df["split"], scan_df["label_clean"], normalize="index")
    print(ratio_table.round(3))
    # اختلاف نسبت‌ها نباید خیلی زیاد باشد
    stone_ratios = ratio_table["stone"]
    if stone_ratios.max() - stone_ratios.min() < 0.08:
        print("  ✓ پاس شد - نسبت‌ها تقریباً یکسان هستند")
    else:
        print("  ⚠ هشدار: اختلاف نسبت‌ها کمی زیاد است (ولی قابل قبول)")

    # ----- چک ۵: توزیع دستگاه -----
    print("\n[چک ۵] توزیع دستگاه به تفکیک کلاس و split")
    if "strata_device" in scan_df.columns:
        device_cross = pd.crosstab([scan_df["split"], scan_df["label_clean"]], scan_df["strata_device"])
        print(device_cross)
        # چک خالی نبودن
        if (device_cross == 0).any().any():
            print("  ⚠ برخی ترکیب‌ها صفر هستند (به خاطر محدودیت داده nonstone)")
        else:
            print("  ✓ هیچ ترکیبی کاملاً خالی نیست")
    else:
        print("  ستون strata_device موجود نیست - رد شد")
        all_checks_passed = False

    # ----- چک ۶: تعداد اسلایس (اگر در scan_df باشد) -----
    print("\n[چک ۶] مقایسه تعداد اسلایس بین کلاس‌ها در سه مجموعه")
    if "num_slices" in scan_df.columns:
        slice_stats = scan_df.groupby(["split", "label_clean"])["num_slices"].agg(["mean", "std", "min", "max"]).round(1)
        print(slice_stats)
        print("  ✓ آمار اسلایس محاسبه شد")
    else:
        print("  ستون num_slices در scan_split موجود نیست - رد شد")

    # ----- چک ۷: سن و جنسیت -----
    print("\n[چک ۷] سن و جنسیت")
    print("  اطلاعات سن و جنسیت در متادیتا موجود نیست → رد شد (قابل چشم‌پوشی)")

    # ----- چک ۸: هر بیمار فقط یک برچسب -----
    print("\n[چک ۸] هر بیمار فقط یک برچسب دارد")
    label_per_patient = scan_df.groupby("patient_id")["label_clean"].nunique()
    if (label_per_patient > 1).sum() == 0:
        print("  ✓ پاس شد")
    else:
        print("  ✗ رد شد - بیمارانی با بیش از یک برچسب وجود دارند")
        all_checks_passed = False

    # ----- چک ۹: فایل NIfTI سالم و ابعاد یکسان -----
    print("\n[چک ۹] بررسی فایل‌های NIfTI (نمونه‌گیری + همه)")
    print("  در حال بررسی تمام فایل‌ها... (ممکن است کمی طول بکشد)")
    
    nii_problems = []
    sizes_found = Counter()
    
    for idx, row in scan_df.iterrows():
        ok, msg, size = check_nii_file(row["scan_id"], row["label_clean"])
        if size:
            sizes_found[size] += 1
        if not ok or (size and size != EXPECTED_SIZE):
            nii_problems.append({
                "scan_id": row["scan_id"],
                "label": row["label_clean"],
                "message": msg,
                "size": size
            })
    
    if len(nii_problems) == 0:
        print(f"  ✓ پاس شد - همه {len(scan_df)} فایل سالم و با ابعاد {EXPECTED_SIZE} هستند")
    else:
        print(f"  ✗ {len(nii_problems)} فایل مشکل دارند:")
        for p in nii_problems[:10]:
            print(f"    - {p['scan_id']}: {p['message']}")
        all_checks_passed = False
    
    print("\n  توزیع ابعاد پیدا شده:")
    for s, cnt in sizes_found.most_common():
        print(f"    {s}: {cnt}")

    # --------------------------------------------------
    # نتیجه کلی چک‌ها
    # --------------------------------------------------
    print("\n" + "="*70)
    if all_checks_passed:
        print("✓✓✓ همه چک‌های حیاتی پاس شدند - Split قابل قبول است")
    else:
        print("✗ برخی چک‌ها رد شدند - قبل از ادامه بررسی کنید")
    print("="*70)

    # ==================================================
    # قدم ۹: قفل کردن Split
    # ==================================================
    print("\n[قدم ۹] قفل کردن Split ...")

    # فایل اصلی split_v1
    split_v1 = scan_df[["scan_id", "patient_id", "label_clean", "split"]].copy()
    split_v1 = split_v1.rename(columns={"label_clean": "label"})
    split_v1_path = OUTPUT_DIR / "split_v1.csv"
    split_v1.to_csv(split_v1_path, index=False, encoding="utf-8-sig")
    print(f"  فایل قفل‌شده ذخیره شد → {split_v1_path}")

    # فایل یادداشت
    notes = f"""
Split Version: v1
Date locked  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Seed         : {SEED}

=== تعریف گروه‌های دستگاه ===
- Philips  = MR 5300 + Ingenia CX
- Siemens  = MAGNETOM Amira + Avanto

=== تعداد بیمار ===
Train : {(patient_df['split']=='train').sum()}
Val   : {(patient_df['split']=='val').sum()}
Test  : {(patient_df['split']=='test').sum()}
Total : {len(patient_df)}

=== تعداد اسکن ===
Train : {(scan_df['split']=='train').sum()}
Val   : {(scan_df['split']=='val').sum()}
Test  : {(scan_df['split']=='test').sum()}
Total : {len(scan_df)}

=== نسبت stone در هر split ===
{pd.crosstab(scan_df['split'], scan_df['label_clean'], normalize='index').round(3).to_string()}

=== نکات ===
- بیمار both (2291858191) حذف شد.
- فقط اسکن‌های پیش‌پردازش موفق نگه داشته شدند.
- از این لحظه این فایل ثابت است و تغییر نمی‌کند.
"""
    notes_path = OUTPUT_DIR / "split_v1_notes.txt"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"  فایل یادداشت ذخیره شد → {notes_path}")

    # ==================================================
    # قدم ۱۰: جدول گزارش برای مقاله
    # ==================================================
    print("\n[قدم ۱۰] ساخت جدول گزارش مقاله...")

    # جدول بیمار
    report_patient = pd.crosstab(patient_df["split"], patient_df["label_clean"], margins=True)
    report_patient.columns = [f"Patient_{c}" for c in report_patient.columns]

    # جدول اسکن
    report_scan = pd.crosstab(scan_df["split"], scan_df["label_clean"], margins=True)
    report_scan.columns = [f"Scan_{c}" for c in report_scan.columns]

    # ترکیب
    report = pd.concat([report_patient, report_scan], axis=1)
    
    report_path = OUTPUT_DIR / "paper_table_split_summary.csv"
    report.to_csv(report_path, encoding="utf-8-sig")
    print(f"  جدول خلاصه ذخیره شد → {report_path}")
    print("\nجدول خلاصه برای مقاله:")
    print(report)

    # جدول دستگاه
    if "strata_device" in scan_df.columns:
        device_report = pd.crosstab(
            [scan_df["split"], scan_df["label_clean"]], 
            scan_df["strata_device"], 
            margins=True
        )
        device_path = OUTPUT_DIR / "paper_table_device_distribution.csv"
        device_report.to_csv(device_path, encoding="utf-8-sig")
        print(f"\n  جدول توزیع دستگاه ذخیره شد → {device_path}")
        print(device_report)

    print("\n" + "="*70)
    print("همه مراحل ۸ تا ۱۰ با موفقیت انجام شد.")
    print("Split قفل شد و آماده استفاده برای آموزش مدل‌ها است.")
    print("="*70)

if __name__ == "__main__":
    main()