"""
Full Regional Seamless Inference & Comparison Script (Option 2).
Uses 50% overlapping Hanning-window blending to eliminate patch boundary seams,
producing a continuous, high-resolution scientific map.
"""

import os
import sys
import torch
import rasterio
import rasterio.windows
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, 'SwinIR'))

from models.network_swinir import SwinIR

def generate_seamless_comparison_map():
    print("=== OPTION 2: GENERATING SEAMLESS BLENDED SUPER-RESOLUTION MAP ===")
    
    device = torch.device('cpu')
    output_dir = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\01_Maps_and_Visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Target Evaluation Region: Central Indian Ridge Spreading Center
    lat_min, lat_max = -25.0, -15.0
    lon_min, lon_max = 65.0, 75.0
    
    emag_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    bathy_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\ETOPO_2022_v1_60s_N90W180_bed.nc"
    grav_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\grav_SWOT_05.nc"
    
    grid_size = 256
    patch_size = 64
    stride = 32  # 50% overlap for seamless blending
    
    with rasterio.open(emag_path) as src:
        win = rasterio.windows.from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        m_lr = src.read(1, window=win, out_shape=(grid_size, grid_size), resampling=rasterio.enums.Resampling.cubic).astype(np.float32)
        
    ds_b = xr.open_dataset(bathy_path)
    b_sub = ds_b.z.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values
    b_t = torch.tensor(b_sub, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    b_hr = torch.nn.functional.interpolate(b_t, size=(grid_size, grid_size), mode='bilinear', align_corners=False).squeeze().numpy()
    
    ds_g = xr.open_dataset(grav_path)
    g_sub = ds_g.z.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values
    g_t = torch.tensor(g_sub, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    g_hr = torch.nn.functional.interpolate(g_t, size=(grid_size, grid_size), mode='bilinear', align_corners=False).squeeze().numpy()
    
    def norm(a):
        a = np.nan_to_num(a)
        mi, ma = np.min(a), np.max(a)
        return (a - mi) / (ma - mi + 1e-7), mi, ma
        
    m_lr_n, mi_m, ma_m = norm(m_lr)
    b_hr_n, _, _ = norm(b_hr)
    g_hr_n, _, _ = norm(g_hr)
    
    # Checkpoint selection (scaled or sanity)
    ckpt_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints\swinir_indian_ocean_scaled.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints\swinir_indian_ocean_sanity.pth"
        
    model = SwinIR(upscale=1, in_chans=3, img_size=patch_size, window_size=8, img_range=1.0, depths=[4, 4, 4, 4], embed_dim=48, num_heads=[4, 4, 4, 4], mlp_ratio=2, upsampler='').to(device)
    head = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    head.load_state_dict(ckpt['head_state_dict'])
    model.eval()
    head.eval()
    
    # 2D Hanning window for smooth spatial reconstruction
    hann_1d = np.hanning(patch_size)
    hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)
    
    pred_acc = np.zeros((grid_size, grid_size), dtype=np.float32)
    weight_acc = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    for r in range(0, grid_size - patch_size + 1, stride):
        for c in range(0, grid_size - patch_size + 1, stride):
            p_m = m_lr_n[r:r+patch_size, c:c+patch_size]
            p_b = b_hr_n[r:r+patch_size, c:c+patch_size]
            p_g = g_hr_n[r:r+patch_size, c:c+patch_size]
            
            p_in = np.stack([p_m, p_b, p_g], axis=0).astype(np.float32)
            p_tensor = torch.tensor(p_in).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = head(model(p_tensor)).squeeze().numpy()
                
            pred_acc[r:r+patch_size, c:c+patch_size] += out * hann_2d
            weight_acc[r:r+patch_size, c:c+patch_size] += hann_2d
            
    # Normalize by accumulated weights
    weight_acc[weight_acc < 1e-5] = 1.0
    pred_seamless = pred_acc / weight_acc
    
    # Denormalize
    pred_physical = pred_seamless * (ma_m - mi_m) + mi_m
    m_lr_physical = m_lr_n * (ma_m - mi_m) + mi_m
    
    # Render Master 4-Panel Publication Comparison Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    # Panel A: Original LR EMAG2
    im0 = axes[0, 0].imshow(m_lr_physical, cmap='turbo', extent=extent, origin='lower')
    axes[0, 0].set_title("A. Original Low-Resolution EMAG2 Marine Magnetics (Sea-Level)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Longitude (°E)")
    axes[0, 0].set_ylabel("Latitude (°S)")
    fig.colorbar(im0, ax=axes[0, 0], label="Intensity")
    
    # Panel B: High-Res Bathymetry Guide
    im1 = axes[0, 1].imshow(b_hr, cmap='turbo', extent=extent, origin='lower')
    axes[0, 1].set_title("B. Conditioning High-Res Bathymetry Seafloor (ETOPO 2022)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Longitude (°E)")
    axes[0, 1].set_ylabel("Latitude (°S)")
    fig.colorbar(im1, ax=axes[0, 1], label="Depth (m)")
    
    # Panel C: SWOT Free-Air Gravity Guide
    im2 = axes[1, 0].imshow(g_hr, cmap='turbo', extent=extent, origin='lower')
    axes[1, 0].set_title("C. Conditioning SWOT Satellite Free-Air Gravity Anomaly", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Longitude (°E)")
    axes[1, 0].set_ylabel("Latitude (°S)")
    fig.colorbar(im2, ax=axes[1, 0], label="mGal")
    
    # Panel D: SwinIR Seamless Enhanced Magnetics
    im3 = axes[1, 1].imshow(pred_physical, cmap='turbo', extent=extent, origin='lower')
    axes[1, 1].set_title("D. Physics-Informed SwinIR Enhanced High-Res Magnetics (Seamless)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Longitude (°E)")
    axes[1, 1].set_ylabel("Latitude (°S)")
    fig.colorbar(im3, ax=axes[1, 1], label="Enhanced Intensity")
    
    plt.suptitle("Central Indian Ridge Spreading Center (15°S-25°S, 65°E-75°E)\nOfficial SwinIR Multi-Modal Geophysical Super-Resolution (Seamless Tiled Inference)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_file = os.path.join(output_dir, "SwinIR_Central_Indian_Ridge_SuperResolution_Comparison.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print("SUCCESS: Seamless comparison map saved to:", out_file)

if __name__ == '__main__':
    generate_seamless_comparison_map()
