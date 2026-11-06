#!/usr/bin/env python3
"""
قدم ۱: Dataset و DataLoader
قدم ۲: سه تست سلامت قبل از آموزش جدی
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random

# ====================== تنظیمات ======================
SPLIT_CSV   = r"E:\MRCP\processed\split_v1.csv"
NII_ROOT    = Path(r"E:\MRCP\processed\nii")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE  = 4
NUM_WORKERS = 4
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --------------------------------------------------
# Dataset
# --------------------------------------------------
class MRCPDataset(Dataset):
    def __init__(self, csv_path, split="train", cache_in_ram=False, use_float16=True):
        df = pd.read_csv(csv_path)
        df["label"] = df["label"].replace({"nostone": "nonstone"})
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.cache_in_ram = cache_in_ram
        self.use_float16 = use_float16
        self.cache = {}

        print(f"[{split}] تعداد اسکن: {len(self.df)}")
        
        # چاپ شکل یک نمونه
        sample = self._load_nii(self.df.iloc[0]["scan_id"], self.df.iloc[0]["label"])
        print(f"[{split}] شکل یک نمونه (بدون کانال): {sample.shape}")
        print(f"[{split}] شکل با کانال: {(1, *sample.shape)}")

        if cache_in_ram:
            print(f"[{split}] در حال کش کردن در RAM...")
            for idx in tqdm(range(len(self.df))):
                sid = self.df.iloc[idx]["scan_id"]
                label = self.df.iloc[idx]["label"]
                vol = self._load_nii(sid, label)
                if use_float16:
                    vol = vol.astype(np.float16)
                self.cache[idx] = vol

    def _load_nii(self, scan_id, label):
        folder = "nostone" if label == "nonstone" else "stone"
        path = NII_ROOT / folder / f"{scan_id}.nii.gz"
        if not path.exists():
            path = NII_ROOT / label / f"{scan_id}.nii.gz"
        
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)          # (D, H, W) = (64, 224, 224)
        return arr.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.cache_in_ram and idx in self.cache:
            vol = self.cache[idx].astype(np.float32)
        else:
            vol = self._load_nii(row["scan_id"], row["label"])
        
        # اضافه کردن کانال → (1, 64, 224, 224)
        vol = vol[np.newaxis, ...]
        
        label = 1 if row["label"] == "stone" else 0
        
        return torch.from_numpy(vol), torch.tensor(label, dtype=torch.float32)

# --------------------------------------------------
# تست ۱: بصری
# --------------------------------------------------
def visual_check(loader, n=4):
    print("\n=== تست ۱: بصری ===")
    images, labels = next(iter(loader))
    print(f"Batch shape: {images.shape}")   # باید (B, 1, 64, 224, 224) باشد
    
    fig, axes = plt.subplots(n, 3, figsize=(12, 3*n))
    for i in range(n):
        vol = images[i, 0].numpy()          # (64, 224, 224)
        mid_z = vol.shape[0] // 2
        mid_y = vol.shape[1] // 2
        
        axes[i, 0].imshow(vol[mid_z], cmap="gray")
        axes[i, 0].set_title(f"Axial | label={int(labels[i])}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(vol[:, mid_y, :], cmap="gray")
        axes[i, 1].set_title("Coronal")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(vol.max(axis=0), cmap="gray")
        axes[i, 2].set_title("MIP")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(r"E:\MRCP\processed\visual_check_batch.png", dpi=120)
    print("تصویر ذخیره شد → E:\\MRCP\\processed\\visual_check_batch.png")
    plt.show()

# --------------------------------------------------
# تست ۲: Overfit روی ۱۶ نمونه
# --------------------------------------------------
def overfit_test():
    print("\n=== تست ۲: Overfit روی ۱۶ نمونه ===")
    from monai.networks.nets import resnet
    
    # Dataset کوچک
    full_ds = MRCPDataset(SPLIT_CSV, split="train", cache_in_ram=True)
    indices = list(range(min(16, len(full_ds))))
    subset = torch.utils.data.Subset(full_ds, indices)
    loader = DataLoader(subset, batch_size=4, shuffle=True)
    
    model = resnet.resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    losses = []
    for step in range(100):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        if (step+1) % 20 == 0:
            print(f"  Step {step+1:3d} | loss = {np.mean(losses[-len(loader):]):.4f}")
    
    final_loss = np.mean(losses[-10:])
    print(f"\nLoss نهایی (۱۰ گام آخر): {final_loss:.4f}")
    if final_loss < 0.05:
        print("  ✓ پاس شد - مدل توانست overfit کند")
    else:
        print("  ✗ رد شد - احتمالاً باگ در داده یا مدل وجود دارد")

# --------------------------------------------------
# تست ۳: برچسب تصادفی
# --------------------------------------------------
def random_label_test():
    print("\n=== تست ۳: برچسب تصادفی (نشت داده) ===")
    from monai.networks.nets import resnet
    from sklearn.metrics import roc_auc_score
    
    train_ds = MRCPDataset(SPLIT_CSV, split="train", cache_in_ram=False)
    val_ds   = MRCPDataset(SPLIT_CSV, split="val", cache_in_ram=False)
    
    # برچسب‌های train را تصادفی می‌کنیم
    random_labels = torch.randint(0, 2, (len(train_ds),)).float()
    
    class RandomLabelDataset(Dataset):
        def __init__(self, base_ds, labels):
            self.base = base_ds
            self.labels = labels
        def __len__(self): return len(self.base)
        def __getitem__(self, idx):
            x, _ = self.base[idx]
            return x, self.labels[idx]
    
    train_loader = DataLoader(RandomLabelDataset(train_ds, random_labels), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = resnet.resnet18(spatial_dims=3, n_input_channels=1, num_classes=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        # ارزیابی روی validation واقعی
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                out = torch.sigmoid(model(x)).cpu().numpy().ravel()
                preds.extend(out)
                targets.extend(y.numpy().ravel())
        
        auroc = roc_auc_score(targets, preds)
        print(f"  Epoch {epoch+1:2d} | Val AUROC = {auroc:.3f}")
    
    print(f"\nAUROC نهایی روی validation: {auroc:.3f}")
    if 0.40 <= auroc <= 0.60:
        print("  ✓ پاس شد - نشت داده‌ای دیده نشد")
    else:
        print("  ✗ مشکوک به نشت داده یا مشکل دیگر")

# --------------------------------------------------
# اجرای اصلی
# --------------------------------------------------
if __name__ == "__main__":
    print("Device:", DEVICE)
    
    # ساخت DataLoaderها
    train_ds = MRCPDataset(SPLIT_CSV, split="train", cache_in_ram=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    
    # تست ۱
    visual_check(train_loader, n=4)
    
    # تست ۲
    overfit_test()
    
    # تست ۳
    random_label_test()
    
    print("\nهمه تست‌های سلامت انجام شد.")