"""
4x Super-Resolution SwinIR Training Script — All 5 Fixes Applied.

Summary of Changes vs Previous Run:
  FIX 1: Global channel-wise standardization (mean/std) instead of per-patch min-max
  FIX 2: 4x SR scale (64x64 LR -> 256x256 HR) — patch covers 115km x 115km
  FIX 3: SwinIR with pixelshuffle upsampler for true 4x upscaling
  FIX 4: Physics loss weights: lambda_spec=0.20, lambda_cg=0.15 (was 0.04, 0.08)
  FIX 5: SwinIR-Medium architecture: depths=[6,6,6,6,6,6], embed_dim=60
          (More capacity for multi-modal fusion without OOM on Kaggle GPU T4)
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader, random_split

CURRENT_DIR = r"C:\Users\shiva\Downloads\Indian_Ocean_Features"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, 'SwinIR'))

from models.network_swinir import SwinIR
from dataset_indian_ocean_4x_sr import IndianOcean4xSRDataset
from physics_loss_4x_sr import AdvancedGeophysicalPhysicsLoss


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
            print(f"  [EarlyStopping] {self.counter}/{self.patience} "
                  f"(best val: {self.best_loss:.5f})", flush=True)
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train():
    print("=" * 70, flush=True)
    print("  4x SR SwinIR-Medium — All 5 Fixes Applied (Local Validation Run)", flush=True)
    print("=" * 70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    emag      = os.path.join(CURRENT_DIR, "04_Marine_Magnetic_and_Rasters", "EMAG2_V3_SeaLevel_DataTiff.tif")
    bathy_dir = os.path.join(CURRENT_DIR, "02_NetCDF_Ocean_DataGrids")
    grav      = os.path.join(CURRENT_DIR, "02_NetCDF_Ocean_DataGrids", "grav_SWOT_05.nc")
    ckpt_dir  = os.path.join(CURRENT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── FIX 2 + 3: 4x SR configuration ───────────────────────────────────────
    patch_size_hr = 256     # HR target at 15 arc-sec
    patch_size_lr = 64      # LR input (4x downsampled)
    batch_size    = 4       # 256x256 patches are large — keep batch small for CPU
    num_patches   = 100     # Local validation run (Kaggle uses 400+)
    epochs        = 5       # Quick local check — Kaggle runs 1000 epochs
    lr            = 5e-5

    print(f"\nBuilding 4x SR dataset (HR={patch_size_hr}px, LR={patch_size_lr}px)...", flush=True)
    full_dataset = IndianOcean4xSRDataset(
        emag_path=emag,
        bathy_dir=bathy_dir,
        grav_path=grav,
        patch_size_hr=patch_size_hr,
        scale_factor=4,
        num_samples=num_patches,
        augment=True,
        seed=4444
    )

    n_val   = max(1, int(len(full_dataset) * 0.20))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    print(f"Split: {n_train} train | {n_val} validation", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── FIX 5: SwinIR-Medium with pixelshuffle 4x upsampler ──────────────────
    print("\nBuilding SwinIR-Medium 4x SR Network...", flush=True)
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=patch_size_lr,      # 64x64 LR input
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],  # FIX 5: 6 blocks (was 4)
        embed_dim=60,                # FIX 5: 60 channels (was 48)
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle'     # FIX 3: 4x pixel-shuffle upsampler
    ).to(device)
    head = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model Parameters: {total_params:.2f}M", flush=True)

    # ── FIX 4: Physics loss with higher weights ───────────────────────────────
    criterion = AdvancedGeophysicalPhysicsLoss(
        patch_size=patch_size_hr,
        lambda_cg=0.15,    # was 0.08
        lambda_spec=0.20,  # was 0.04
        lambda_tv=0.002,
        lambda_ssim=0.10
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stop   = EarlyStopping(patience=30)
    best_val     = float('inf')
    best_path    = os.path.join(ckpt_dir, "swinir_model1_4x_sr_best.pth")

    print(f"\n{'Ep':>4} {'Train':>10} {'Val':>10} {'Recon':>9} {'CG':>8} {'Laplace':>9} {'SSIM':>9} {'Time':>7}", flush=True)
    print("-" * 75, flush=True)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train(); head.train()
        tr_tot = tr_rec = tr_cg = tr_spec = tr_ssim = 0.0
        n_tr = 0

        for x, y, b in train_loader:
            x, y, b = x.to(device), y.to(device), b.to(device)
            optimizer.zero_grad()
            pred = head(model(x))
            loss, r, cg, sp, tv, ss = criterion(pred, y, b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=1.0)
            optimizer.step()
            tr_tot  += loss.item(); tr_rec  += r.item()
            tr_cg   += cg.item();  tr_spec += sp.item()
            tr_ssim += ss.item();  n_tr += 1

        model.eval(); head.eval()
        val_tot = 0.0; n_v = 0
        with torch.no_grad():
            for x, y, b in val_loader:
                x, y, b = x.to(device), y.to(device), b.to(device)
                pred = head(model(x))
                loss, *_ = criterion(pred, y, b)
                val_tot += loss.item(); n_v += 1

        scheduler.step()
        avg_tr  = tr_tot  / max(1, n_tr)
        avg_val = val_tot / max(1, n_v)
        dt = time.time() - t0

        print(f"[{ep:02d}/{epochs:02d}] {avg_tr:10.5f} {avg_val:10.5f} "
              f"{tr_rec/max(1,n_tr):9.5f} {tr_cg/max(1,n_tr):8.5f} "
              f"{tr_spec/max(1,n_tr):9.5f} {tr_ssim/max(1,n_tr):9.5f} {dt:6.1f}s", flush=True)

        is_best = early_stop(avg_val)
        if is_best:
            best_val = avg_val
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'head_state_dict':  head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val,
                'scale_factor': 4,
                'patch_size_lr': patch_size_lr,
                'patch_size_hr': patch_size_hr,
            }, best_path)
            print(f"  >>> SAVED BEST (Val: {best_val:.5f})", flush=True)

        if early_stop.early_stop:
            print(f"\n[EARLY STOP] Epoch {ep} — No improvement for {early_stop.patience} epochs.", flush=True)
            break

    print(f"\n[DONE] 4x SR Training Complete! Best Val Loss: {best_val:.5f}", flush=True)
    print(f"Checkpoint: {best_path}", flush=True)


if __name__ == '__main__':
    train()
