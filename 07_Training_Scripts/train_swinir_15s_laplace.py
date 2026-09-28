"""
Advanced Multi-Modal SwinIR Training Script.
Integrates:
1. 15-Arc-Second (~450m) ETOPO Seafloor Bathymetry Tiles
2. NOAA EMAG2 v3 Marine Magnetics
3. SWOT Satellite Free-Air Gravity
4. Advanced Geophysical Loss:
   - Charbonnier Reconstruction Loss
   - Gallardo-Meju Structural Cross-Gradient
   - 2D Spectral Laplace Frequency Loss (Potential-Field Harmonic Decay)
   - Total Variation Loss
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
from dataset_indian_ocean_15s import IndianOcean15sDataset
from physics_loss_advanced import AdvancedGeophysicalPhysicsLoss

def train_advanced():
    print("=== STARTING ADVANCED SWINIR TRAINING (15-ARC-SECOND + SPECTRAL LAPLACE LOSS) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    emag_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    tiles_dir = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\15s_Bathymetry_Tiles"
    grav_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\grav_SWOT_05.nc"
    ckpt_dir = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    patch_size = 64
    batch_size = 8
    num_samples = 150  # 150 high-res 450m patches for advanced fine-tuning
    epochs = 5
    lr = 1.5e-4
    
    # 1. Dataset & Loader
    print("Loading 15-arc-second (~450m) multi-modal dataset...")
    dataset = IndianOcean15sDataset(
        emag_path=emag_path,
        tiles_dir=tiles_dir,
        grav_path=grav_path,
        patch_size=patch_size,
        num_samples=num_samples,
        seed=101
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 2. Build Model & Load Epoch 31 Best Checkpoint
    print("Initializing SwinIR Transformer and loading Epoch 31 weights...")
    model = SwinIR(
        upscale=1, in_chans=3, img_size=patch_size, window_size=8, img_range=1.0,
        depths=[4, 4, 4, 4], embed_dim=48, num_heads=[4, 4, 4, 4], mlp_ratio=2, upsampler=''
    ).to(device)
    head = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)
    
    best_ckpt = os.path.join(ckpt_dir, "swinir_best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        head.load_state_dict(ckpt['head_state_dict'])
        print(f"Successfully warm-started from Kaggle Best Checkpoint (Epoch {ckpt.get('epoch', 31)})!")
        
    # 3. Advanced Physics Loss (with Spectral Laplace)
    criterion = AdvancedGeophysicalPhysicsLoss(
        patch_size=patch_size,
        lambda_cg=0.08,
        lambda_spec=0.05,
        lambda_tv=0.002
    ).to(device)
    
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    print("\n--- BEGIN ADVANCED FINE-TUNING (SPECTRAL LAPLACE + 450M TILES) ---")
    model.train()
    head.train()
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tot_loss, rec_l, cg_l, spec_l = 0.0, 0.0, 0.0, 0.0
        steps = 0
        
        for x, y, b in dataloader:
            x = x.to(device)
            y = y.to(device)
            b = b.to(device)
            
            optimizer.zero_grad()
            pred = head(model(x))
            
            loss, r, cg, sp, tv = criterion(pred, y, b)
            loss.backward()
            optimizer.step()
            
            tot_loss += loss.item()
            rec_l += r.item()
            cg_l += cg.item()
            spec_l += sp.item()
            steps += 1
            
        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | Total: {tot_loss/steps:.5f} | Recon: {rec_l/steps:.5f} | CrossGrad: {cg_l/steps:.5f} | SpectralLaplace: {spec_l/steps:.5f}")
        
    save_path = os.path.join(ckpt_dir, "swinir_model1_advanced_15s_laplace.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'head_state_dict': head.state_dict(),
        'final_loss': tot_loss / steps
    }, save_path)
    print(f"\nSUCCESS: Advanced Fine-Tuning Complete! Saved model to: {save_path}")

if __name__ == '__main__':
    train_advanced()
