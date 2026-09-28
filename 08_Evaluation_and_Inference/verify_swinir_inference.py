import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, 'SwinIR'))

from models.network_swinir import SwinIR
from dataset_indian_ocean import IndianOceanMultiModalDataset

def test_and_plot():
    device = torch.device('cpu')
    ckpt_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints\swinir_indian_ocean_sanity.pth"
    output_dir = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\01_Maps_and_Visualizations"
    
    # Load model
    model = SwinIR(upscale=1, in_chans=3, img_size=64, window_size=8, img_range=1.0, depths=[4, 4, 4, 4], embed_dim=48, num_heads=[4, 4, 4, 4], mlp_ratio=2, upsampler='')
    head = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    head.load_state_dict(ckpt['head_state_dict'])
    model.eval()
    head.eval()
    
    emag = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    bathy = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\ETOPO_2022_v1_60s_N90W180_bed.nc"
    grav = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\grav_SWOT_05.nc"
    
    ds = IndianOceanMultiModalDataset(emag, bathy, grav, patch_size_hr=64, scale_factor=2, num_samples=10, seed=777)
    x, y, b = ds[0] # Sample 1 patch
    
    with torch.no_grad():
        x_in = x.unsqueeze(0)
        pred = head(model(x_in)).squeeze().numpy()
        
    lr_mag = x[0].numpy()
    guide_bathy = x[1].numpy()
    guide_grav = x[2].numpy()
    target_hr = y.squeeze().numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=300)
    
    im0 = axes[0].imshow(lr_mag, cmap='turbo')
    axes[0].set_title("1. Low-Res Input Magnetics", fontsize=9, fontweight='bold')
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(guide_bathy, cmap='turbo')
    axes[1].set_title("2. Guide Bathymetry (ETOPO)", fontsize=9, fontweight='bold')
    fig.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(guide_grav, cmap='turbo')
    axes[2].set_title("3. Guide Gravity (SWOT 05)", fontsize=9, fontweight='bold')
    fig.colorbar(im2, ax=axes[2])
    
    im3 = axes[3].imshow(pred, cmap='turbo')
    axes[3].set_title("4. SwinIR Enhanced Magnetics", fontsize=9, fontweight='bold')
    fig.colorbar(im3, ax=axes[3])
    
    im4 = axes[4].imshow(target_hr, cmap='turbo')
    axes[4].set_title("5. Ground Truth HR Magnetics", fontsize=9, fontweight='bold')
    fig.colorbar(im4, ax=axes[4])
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.suptitle("Official SwinIR Multi-Modal Geophysical Super-Resolution (Indian Ocean Seafloor Patch)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_fig = os.path.join(output_dir, "SwinIR_Indian_Ocean_Inference_Verification.png")
    plt.savefig(save_fig, bbox_inches='tight')
    plt.close()
    print("SUCCESS: Inference verification map saved to:", save_fig)

if __name__ == '__main__':
    test_and_plot()
