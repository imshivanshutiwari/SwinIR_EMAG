"""
Scaled Multi-Modal SwinIR Training Script for Indian Ocean Magnetics Enhancement.
Option 1: Scales up training to 500 representative deep-ocean patches across the entire
Indian Ocean (Lat: -55S to +20N, Lon: 25E to 140E) for 10 epochs.
Uses AdamW with Cosine Annealing Learning Rate Schedule and Gallardo-Meju Physics Loss.
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, 'SwinIR'))

from models.network_swinir import SwinIR
from dataset_indian_ocean import IndianOceanMultiModalDataset
from physics_loss import GeophysicalPhysicsLoss

def train_scaled():
    print("=== STARTING OPTION 1: SCALED MULTI-MODAL SWINIR TRAINING (INDIAN OCEAN) ===")
    
    # 1. Dataset Paths
    emag_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    bathy_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\ETOPO_2022_v1_60s_N90W180_bed.nc"
    grav_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\grav_SWOT_05.nc"
    ckpt_dir = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Scaled Hyperparameters
    patch_size = 64
    batch_size = 8
    num_samples = 300  # 300 diverse ocean floor patches covering ridges, basins, fracture zones
    epochs = 10
    lr = 3e-4
    
    # 3. Build Dataset
    print(f"Sampling {num_samples} patches across the Indian Ocean basin...")
    dataset = IndianOceanMultiModalDataset(
        emag_path=emag_path,
        bathy_path=bathy_path,
        grav_path=grav_path,
        patch_size_hr=patch_size,
        scale_factor=2,
        num_samples=num_samples,
        seed=42
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 4. SwinIR Architecture
    print("Configuring SwinIR Transformer backbone...")
    model = SwinIR(
        upscale=1,
        in_chans=3,
        img_size=patch_size,
        window_size=8,
        img_range=1.0,
        depths=[4, 4, 4, 4],
        embed_dim=48,
        num_heads=[4, 4, 4, 4],
        mlp_ratio=2,
        upsampler=''
    ).to(device)
    head = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)
    
    # Load weights from sanity checkpoint if available
    sanity_ckpt = os.path.join(ckpt_dir, "swinir_indian_ocean_sanity.pth")
    if os.path.exists(sanity_ckpt):
        ckpt = torch.load(sanity_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        head.load_state_dict(ckpt['head_state_dict'])
        print("Loaded initial weights from sanity checkpoint.")
        
    criterion = GeophysicalPhysicsLoss(lambda_cg=0.08, lambda_tv=0.002).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    print("\n--- BEGIN SCALED TRAINING LOOP ---")
    model.train()
    head.train()
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_cg = 0.0
        steps = 0
        
        for x, y, b in dataloader:
            x = x.to(device)
            y = y.to(device)
            b = b.to(device)
            
            optimizer.zero_grad()
            pred = head(model(x))
            
            loss, r_l, cg_l, tv_l = criterion(pred, y, b)
            loss.backward()
            optimizer.step()
            
            epoch_total += loss.item()
            epoch_recon += r_l.item()
            epoch_cg += cg_l.item()
            steps += 1
            
        scheduler.step()
        dt = time.time() - t0
        avg_tot = epoch_total / steps
        avg_rec = epoch_recon / steps
        avg_cg = epoch_cg / steps
        cur_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | LR: {cur_lr:.6f} | Total Loss: {avg_tot:.5f} | Recon L1: {avg_rec:.5f} | CrossGrad: {avg_cg:.5f}")
        
    out_ckpt = os.path.join(ckpt_dir, "swinir_indian_ocean_scaled.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'head_state_dict': head.state_dict(),
        'final_loss': avg_tot
    }, out_ckpt)
    print(f"\nSUCCESS: Scaled Training Complete! Saved checkpoint: {out_ckpt}")

if __name__ == '__main__':
    train_scaled()
