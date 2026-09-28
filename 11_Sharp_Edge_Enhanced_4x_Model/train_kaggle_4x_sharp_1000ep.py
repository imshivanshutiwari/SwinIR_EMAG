"""
Fixed Kaggle Training Script for Ultra-Sharp 4x Model (Option A):
- Continuous Unclipped Global Normalization (mean = 193.03 nT, std = 69.98 nT)
- Anti-Quantization Filtered Sobel Edge Loss (lambda_edge = 0.08)
- Cross-Gradient Seafloor Alignment (lambda_cg = 0.20)
- High-Wavenumber Laplace Loss (lambda_spec = 0.20)
- SwinIR-Medium 4x (in_chans=4, Conv2d(4, 1) head)
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
from dataset_indian_ocean_4x_sharp import IndianOcean4xSharpDataset
from physics_loss_4x_sharp import SharpPhysicsLoss

class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return True
        else:
            self.counter += 1
            print(f"  [EARLY STOPPING] {self.counter}/{self.patience} (Best Val Loss: {self.best_loss:.5f})", flush=True)
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def train():
    print("=" * 75, flush=True)
    print("  KAGGLE FIXED ULTRA-SHARP 4x SUPER-RESOLUTION SWINIR-MEDIUM (OPTION A)", flush=True)
    print("  Features: Unclipped Global Norm | Anti-Quantized Sobel | Balanced Physics", flush=True)
    print("=" * 75, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    DATA_DIR = "."
    emag      = os.path.join(DATA_DIR, "EMAG2_V3_SeaLevel_DataTiff.tif")
    bathy_dir = DATA_DIR
    grav      = os.path.join(DATA_DIR, "grav_SWOT_05.nc")
    ckpt_dir  = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    patch_size_hr = 256
    patch_size_lr = 64
    batch_size    = 8
    num_patches   = 400
    max_epochs    = 1000
    patience      = 30
    lr            = 5e-5

    print(f"\nBuilding 4-Channel Unclipped Dataset ({num_patches} patches)...", flush=True)
    full_dataset = IndianOcean4xSharpDataset(
        emag_path=emag,
        bathy_dir=bathy_dir,
        grav_path=grav,
        patch_size_hr=patch_size_hr,
        scale_factor=4,
        num_samples=num_patches,
        augment=True,
        seed=9999
    )

    n_val   = max(1, int(len(full_dataset) * 0.20))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset split: {n_train} Train | {n_val} Validation patches", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print("\nInstantiating 4-Channel SwinIR-Medium Network...", flush=True)
    model = SwinIR(
        upscale=4,
        in_chans=4,
        img_size=patch_size_lr,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle'
    ).to(device)
    head = torch.nn.Conv2d(4, 1, kernel_size=3, padding=1).to(device)

    criterion = SharpPhysicsLoss(
        lambda_edge=0.08,
        lambda_cg=0.20,
        lambda_spec=0.20
    ).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience)

    best_val_loss  = float('inf')
    best_save_path = os.path.join(ckpt_dir, "swinir_model1_4x_sharp_best.pth")

    print(f"\n{'Epoch':>6} {'Train':>10} {'Val':>10} {'Recon':>9} {'SobelEdge':>10} {'CrossGrad':>10} {'Laplace':>9} {'Time':>7}", flush=True)
    print("-" * 80, flush=True)

    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        model.train(); head.train()
        tr_loss = tr_rec = tr_edge = tr_cg = tr_spec = 0.0
        n_tr = 0

        for x, y, b in train_loader:
            x, y, b = x.to(device), y.to(device), b.to(device)
            optimizer.zero_grad()

            pred = head(model(x))
            loss, r_l, ed_l, cg_l, sp_l = criterion(pred, y, b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item(); tr_rec  += r_l.item()
            tr_edge += ed_l.item(); tr_cg   += cg_l.item()
            tr_spec += sp_l.item(); n_tr += 1

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

        avg_tr   = tr_loss / max(1, n_tr)
        avg_val  = val_loss / max(1, n_val_b)
        avg_rec  = tr_rec  / max(1, n_tr)
        avg_edge = tr_edge / max(1, n_tr)
        avg_cg   = tr_cg   / max(1, n_tr)
        avg_spec = tr_spec / max(1, n_tr)
        dt = time.time() - t0

        print(f"[{ep:04d}/{max_epochs:04d}] {avg_tr:10.5f} {avg_val:10.5f} {avg_rec:9.5f} {avg_edge:10.5f} {avg_cg:10.5f} {avg_spec:9.5f} {dt:6.1f}s", flush=True)

        is_best = early_stopping(avg_val)
        if is_best:
            best_val_loss = avg_val
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'head_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'in_chans': 4,
                'scale_factor': 4
            }, best_save_path)
            print(f"  >>> SAVED NEW BEST SHARP CHECKPOINT (Val Loss: {best_val_loss:.5f})", flush=True)

        if early_stopping.early_stop:
            print(f"\n[EARLY STOPPING TRIGGERED] Training stopped early at Epoch {ep}.", flush=True)
            break

    print(f"\n[SUCCESS] Fixed Ultra-Sharp 4x Training Finished! Best Val Loss: {best_val_loss:.5f}", flush=True)
    print(f"Optimal Model Saved to: {best_save_path}", flush=True)

if __name__ == '__main__':
    train()
