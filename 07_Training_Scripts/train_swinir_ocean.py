"""
Training Script for Multi-Modal Geophysical Super-Resolution using Official SwinIR backbone.
Combines:
1. SwinIR Vision Transformer (from official repo: SwinIR.models.network_swinir)
2. IndianOceanMultiModalDataset (dataset_indian_ocean.py)
3. GeophysicalPhysicsLoss (physics_loss.py) with Gallardo-Meju structural cross-gradient constraint
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader

# Add current directory and official SwinIR repo to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, 'SwinIR'))

from models.network_swinir import SwinIR
from dataset_indian_ocean import IndianOceanMultiModalDataset
from physics_loss import GeophysicalPhysicsLoss

def train():
    print("=== STARTING SWINIR GEOPHYSICAL TRAINING PIPELINE ===")
    
    # 1. Paths
    emag_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    bathy_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\ETOPO_2022_v1_60s_N90W180_bed.nc"
    grav_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\grav_SWOT_05.nc"
    ckpt_dir = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using computation device: {device}")
    
    # 2. Hyperparameters
    patch_size = 64
    batch_size = 4
    num_samples = 40  # Verified fast sanity dataset size
    epochs = 3
    lr = 2e-4
    
    # 3. Dataset & DataLoader
    print("Building multi-modal Indian Ocean dataset...")
    dataset = IndianOceanMultiModalDataset(
        emag_path=emag_path,
        bathy_path=bathy_path,
        grav_path=grav_path,
        patch_size_hr=patch_size,
        scale_factor=2,
        num_samples=num_samples,
        seed=123
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 4. Instantiate Official SwinIR Network
    print("Initializing Official SwinIR Transformer model...")
    model = SwinIR(
        upscale=1,  # Since input is already multi-modal 3-channel spatially aligned tensor
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
    
    # Final 1x1 conv layer to project embed_dim (or output channels) to 1-channel high-res magnetics
    head = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)
    
    # 5. Physics-Informed Criterion & Optimizer
    criterion = GeophysicalPhysicsLoss(lambda_cg=0.08, lambda_tv=0.002).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    
    print("\n=== STARTING TRAINING LOOP ===")
    model.train()
    head.train()
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_cg_loss = 0.0
        steps = 0
        
        for batch_idx, (x, y, b) in enumerate(dataloader):
            x = x.to(device) # [B, 3, 64, 64]
            y = y.to(device) # [B, 1, 64, 64] Target HR Magnetics
            b = b.to(device) # [B, 1, 64, 64] Guide HR Bathymetry
            
            optimizer.zero_grad()
            
            # SwinIR forward pass
            features = model(x)
            pred_m = head(features) # [B, 1, 64, 64]
            
            # Compute Composite Physics Loss (Charbonnier + Gallardo-Meju Cross Gradient + Total Variation)
            total_loss, recon_l, cg_l, tv_l = criterion(pred_m, y, b)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_l.item()
            epoch_cg_loss += cg_l.item()
            steps += 1
            
        dt = time.time() - t0
        avg_total = epoch_total_loss / steps
        avg_recon = epoch_recon_loss / steps
        avg_cg = epoch_cg_loss / steps
        print(f"Epoch [{epoch}/{epochs}] ({dt:.1f}s) | Total Loss: {avg_total:.5f} | Recon L1: {avg_recon:.5f} | Cross-Grad: {avg_cg:.5f}")
        
    # Save checkpoint
    save_path = os.path.join(ckpt_dir, "swinir_indian_ocean_sanity.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_total
    }, save_path)
    print(f"\nSUCCESS: Training complete! Checkpoint saved to: {save_path}")

if __name__ == '__main__':
    train()
