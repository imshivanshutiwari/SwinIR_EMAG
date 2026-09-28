"""
Advanced Diagnostic Verification Script.
Evaluates the fine-tuned SwinIR model with 15-arc-second (~450m) ETOPO bathymetry
and Spectral Laplace loss, comparing high-frequency wavenumber spectrum recovery
and spatial detail against raw NOAA EMAG2.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, 'SwinIR'))

from models.network_swinir import SwinIR
from dataset_indian_ocean_15s import IndianOcean15sDataset

def verify_advanced_model():
    print("=== VERIFYING ADVANCED 15-ARC-SECOND + SPECTRAL LAPLACE MODEL ===")
    
    device = torch.device('cpu')
    output_dir = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\01_Maps_and_Visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    emag = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    tiles_dir = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\15s_Bathymetry_Tiles"
    grav = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\grav_SWOT_05.nc"
    
    patch_size = 64
    dataset = IndianOcean15sDataset(emag, tiles_dir, grav, patch_size=patch_size, num_samples=20, seed=555)
    x, y, b = dataset[0]  # Sample 1 high-res patch
    
    ckpt_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints\swinir_model1_advanced_15s_laplace.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints\swinir_best_model.pth"
        
    model = SwinIR(upscale=1, in_chans=3, img_size=patch_size, window_size=8, img_range=1.0, depths=[4, 4, 4, 4], embed_dim=48, num_heads=[4, 4, 4, 4], mlp_ratio=2, upsampler='').to(device)
    head = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    head.load_state_dict(ckpt['head_state_dict'])
    model.eval()
    head.eval()
    
    with torch.no_grad():
        x_in = x.unsqueeze(0).to(device)
        pred = head(model(x_in)).squeeze().numpy()
        
    lr_mag = x[0].numpy()
    guide_bathy_15s = x[1].numpy()
    guide_grav = x[2].numpy()
    target_hr = y.squeeze().numpy()
    
    # 2D Fourier Spectral Amplitude
    fft_lr = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(lr_mag))))
    fft_pred = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(pred))))
    fft_gt = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(target_hr))))
    
    # Plot 2x3 Diagnostic Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    
    im0 = axes[0, 0].imshow(lr_mag, cmap='turbo')
    axes[0, 0].set_title("A. Low-Res Input Magnetics", fontsize=10, fontweight='bold')
    fig.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(guide_bathy_15s, cmap='turbo')
    axes[0, 1].set_title("B. High-Res 15-Arc-Sec Bathymetry (~450m)", fontsize=10, fontweight='bold')
    fig.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(pred, cmap='turbo')
    axes[0, 2].set_title("C. Enhanced Magnetics (SwinIR + Laplace)", fontsize=10, fontweight='bold')
    fig.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[1, 0].imshow(fft_lr, cmap='inferno')
    axes[1, 0].set_title("D. 2D Wavenumber Spectrum: Low-Res Input", fontsize=10, fontweight='bold')
    fig.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(fft_pred, cmap='inferno')
    axes[1, 1].set_title("E. 2D Spectrum: Spectral Laplace Output", fontsize=10, fontweight='bold')
    fig.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[1, 2].imshow(fft_gt, cmap='inferno')
    axes[1, 2].set_title("F. 2D Spectrum: Ground-Truth HR", fontsize=10, fontweight='bold')
    fig.colorbar(im5, ax=axes[1, 2])
    
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.suptitle("Model 1 Advanced Verification: 15-Arc-Second Bathymetry Guidance & Potential-Field Laplace Spectral Recovery", fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = os.path.join(output_dir, "Model1_Advanced_15s_Spectral_Laplace_Verification.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print("SUCCESS: Advanced verification map saved to:", out_path)

if __name__ == '__main__':
    verify_advanced_model()
