"""
Kaggle Training Script for 4x Super-Resolution SwinIR-Medium Model 1:
- 4x Scale Factor (64x64 LR -> 256x256 HR at 15 arc-sec)
- Global Ocean Standardization (Preserves nT units)
- SwinIR-Medium Architecture (6 RSTB Blocks, 60 channels, 4x PixelShuffle)
- Enhanced Physics Losses (lambda_spec=0.20, lambda_cg=0.15, SSIM=0.10)
- Early Stopping (Patience=30)
- Up to 1000 Max Epochs with Cosine Annealing
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append('.')
sys.path.append(os.path.join(os.getcwd(), 'SwinIR'))

from models.network_swinir import SwinIR
from dataset_indian_ocean_4x_sr import IndianOcean4xSRDataset
from physics_loss_4x_sr import AdvancedGeophysicalPhysicsLoss

class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            print(f"  [EARLY STOPPING] Counter: {self.counter}/{self.patience} (Best Val Loss: {self.best_loss:.5f})", flush=True)
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def train():
    print("=" * 70, flush=True)
    print("  KAGGLE 4x SUPER-RESOLUTION SWINIR-MEDIUM MODEL 1 TRAINING", flush=True)
    print("  Features: 256x256 HR | Global Normalization | Early Stopping (Patience=30)", flush=True)
    print("=" * 70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # Detect dataset directory (Kaggle auto-extract or local)
    DATA_DIR = "."
    emag = os.path.join(DATA_DIR, "EMAG2_V3_SeaLevel_DataTiff.tif")
    bathy_dir = DATA_DIR
    grav = os.path.join(DATA_DIR, "grav_SWOT_05.nc")
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    patch_size_hr = 256
    patch_size_lr = 64
    batch_size = 8      # Optimal for Kaggle T4 GPU
    num_patches = 400
    max_epochs = 1000
    patience = 30
    lr = 5e-5

    print(f"\nBuilding 4x SR Dataset ({num_patches} patches, 256px HR)...", flush=True)
    full_dataset = IndianOcean4xSRDataset(
        emag_path=emag,
        bathy_dir=bathy_dir,
        grav_path=grav,
        patch_size_hr=patch_size_hr,
        scale_factor=4,
        num_samples=num_patches,
        augment=True,
        seed=2026
    )

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * 0.20))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset split: {n_train} Train | {n_val} Validation patches", flush=True)

    # num_workers=0 to prevent GDAL concurrency issues
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print("\nInstantiating SwinIR-Medium 4x SR Network...", flush=True)
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=patch_size_lr,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle'
    ).to(device)
    head = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)

    # Load initial warm-start checkpoint if available
    for ckpt_name in ["swinir_model1_4x_sr_best.pth", "swinir_model1_godlevel_v2_best.pth", "swinir_model1_15s_god_level_best.pth"]:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            print(f"Warm-starting from checkpoint: {ckpt_path}", flush=True)
            ckpt = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in ckpt:
                try:
                    model.load_state_dict(ckpt['model_state_dict'], strict=False)
                    head.load_state_dict(ckpt['head_state_dict'], strict=False)
                    print("Pretrained weights loaded successfully!", flush=True)
                except Exception as e:
                    print(f"Warm-start state load warning: {e}", flush=True)
            break

    criterion = AdvancedGeophysicalPhysicsLoss(
        patch_size=patch_size_hr,
        lambda_cg=0.15,
        lambda_spec=0.20,
        lambda_tv=0.002,
        lambda_ssim=0.10
    ).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience)

    best_val_loss = float('inf')
    best_save_path = os.path.join(ckpt_dir, "swinir_model1_4x_sr_kaggle_best.pth")

    print(f"\n{'Epoch':>6} {'Train':>10} {'Val':>10} {'Recon':>9} {'CrossGrad':>10} {'Laplace':>9} {'SSIM':>9} {'Time':>7}", flush=True)
    print("-" * 75, flush=True)

    for ep in range(1, max_epochs + 1):
        t0 = time.time()

        # --- TRAINING PASS ---
        model.train(); head.train()
        tr_loss = tr_recon = tr_cg = tr_spec = tr_ssim = 0.0
        n_tr = 0

        for x, y, b in train_loader:
            x, y, b = x.to(device), y.to(device), b.to(device)
            optimizer.zero_grad()

            pred = head(model(x))
            loss, r_l, cg_l, spec_l, tv_l, ssim_l = criterion(pred, y, b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item()
            tr_recon += r_l.item()
            tr_cg += cg_l.item()
            tr_spec += spec_l.item()
            tr_ssim += ssim_l.item()
            n_tr += 1

        # --- VALIDATION PASS ---
        model.eval(); head.eval()
        val_loss = 0.0; n_val_b = 0
        with torch.no_grad():
            for x, y, b in val_loader:
                x, y, b = x.to(device), y.to(device), b.to(device)
                pred = head(model(x))
                loss, *_ = criterion(pred, y, b)
                val_loss += loss.item()
                n_val_b += 1

        scheduler.step()

        avg_tr = tr_loss / max(1, n_tr)
        avg_val = val_loss / max(1, n_val_b)
        avg_recon = tr_recon / max(1, n_tr)
        avg_cg = tr_cg / max(1, n_tr)
        avg_spec = tr_spec / max(1, n_tr)
        avg_ssim = tr_ssim / max(1, n_tr)
        dt = time.time() - t0

        print(f"[{ep:04d}/{max_epochs:04d}] {avg_tr:10.5f} {avg_val:10.5f} {avg_recon:9.5f} {avg_cg:10.5f} {avg_spec:9.5f} {avg_ssim:9.5f} {dt:6.1f}s", flush=True)

        is_best = early_stopping(avg_val)
        if is_best:
            best_val_loss = avg_val
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'head_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'scale_factor': 4
            }, best_save_path)
            print(f"  >>> SAVED NEW BEST CHECKPOINT (Val Loss: {best_val_loss:.5f})", flush=True)

        if early_stopping.early_stop:
            print(f"\n[EARLY STOPPING TRIGGERED] Training stopped early at Epoch {ep}.", flush=True)
            break

    print(f"\n[SUCCESS] 4x SR Kaggle Training Finished! Best Val Loss: {best_val_loss:.5f}", flush=True)
    print(f"Optimal Model Saved to: {best_save_path}", flush=True)

if __name__ == '__main__':
    train()
