"""
Render Full Continuous Indian Ocean Map (20°E to 140°E, -60°S to 30°N).
Full-basin continuous sliding-window inference with Model 1 (4x SR SwinIR-Medium).
Rendered using the 'turbo' rainbow colormap as explicitly requested.
"""

import sys
sys.path.append(r"C:\Users\shiva\Downloads\Indian_Ocean_Features")
sys.path.append(r"C:\Users\shiva\Downloads\Indian_Ocean_Features\SwinIR")
sys.path.append(r"C:\Users\shiva\.gemini\antigravity-cli\brain\158f5b70-2534-4c90-9450-291ea8601c65\scratch")

import os
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from models.network_swinir import SwinIR

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    emag_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    ckpt_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints\swinir_model1_4x_sr_kaggle_best.pth"
    out_map   = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\01_Maps_and_Visualizations\Full_Indian_Ocean_4x_SR_Turbo_Map.png"
    
    # 1. Load Model 1 SwinIR-Medium
    print("Loading SwinIR-Medium 4x Model...", flush=True)
    model = SwinIR(
        upscale=4, in_chans=3, img_size=64, window_size=8,
        img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=60,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle'
    ).to(device)
    head = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        head.load_state_dict(ckpt['head_state_dict'])
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval(); head.eval()
    
    # 2. Read full raster bounds & data
    print("Opening EMAG2 full raster dataset...", flush=True)
    with rasterio.open(emag_path) as src:
        bounds = src.bounds
        lon_min, lon_max = max(20.0, bounds.left), min(140.0, bounds.right)
        lat_min, lat_max = max(-60.0, bounds.bottom), min(30.0, bounds.top)
        
        # Read Indian Ocean extent at high resolution
        win = rasterio.windows.from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        # Resample full basin grid to 1000x800 for high-detail global rendering
        full_hr = src.read(1, window=win, out_shape=(800, 1200), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        
    print(f"Full Basin Extent: Lon [{lon_min:.1f}E to {lon_max:.1f}E], Lat [{lat_min:.1f}S to {lat_max:.1f}N]", flush=True)
    
    # Mask nodata values
    full_hr[full_hr == 255.0] = np.nan
    full_hr[full_hr == -9999.0] = np.nan
    
    # Generate Low-Res input grid by 4x downsampling
    full_lr = torch.nn.functional.interpolate(
        torch.tensor(full_hr).unsqueeze(0).unsqueeze(0),
        scale_factor=0.25, mode='bilinear', align_corners=False
    ).squeeze().numpy()
    
    # 3. Perform tiled 4x SR inference across full grid
    print("Running tiled 4x Super-Resolution inference across entire Indian Ocean Basin...", flush=True)
    lr_h, lr_w = full_lr.shape
    sr_full = np.zeros((lr_h * 4, lr_w * 4), dtype=np.float32)
    
    tile_lr = 64
    stride = 32
    
    # Standardize
    mag_mean, mag_std = 117.45, 16.36
    norm_lr = np.nan_to_num((full_lr - mag_mean) / mag_std)
    
    with torch.no_grad():
        for i in range(0, lr_h - tile_lr + 1, stride):
            for j in range(0, lr_w - tile_lr + 1, stride):
                patch_lr = norm_lr[i:i+tile_lr, j:j+tile_lr]
                dummy_b = np.zeros((tile_lr, tile_lr), dtype=np.float32)
                dummy_g = np.zeros((tile_lr, tile_lr), dtype=np.float32)
                
                x_in = np.stack([patch_lr, dummy_b, dummy_g], axis=0).astype(np.float32)
                x_t = torch.tensor(x_in).unsqueeze(0).to(device)
                
                pred_patch = head(model(x_t)).squeeze().cpu().numpy()
                
                # Un-standardize back to real nT units
                pred_nT = (pred_patch * mag_std) + mag_mean
                
                sr_full[i*4:(i+tile_lr)*4, j*4:(j+tile_lr)*4] = pred_nT

    # Replace nan regions
    sr_full[np.isnan(full_hr)] = np.nan
    
    # Robust percentiles for vibrant turbo colormap (-150 nT to +150 nT)
    vmin, vmax = np.nanpercentile(full_hr, 2), np.nanpercentile(full_hr, 98)
    
    # 4. Render Side-by-Side Full-Basin Comparison in TURBO Colormap
    print("Rendering 2-Panel Full Indian Ocean Basin Map in TURBO Colormap...", flush=True)
    fig, axes = plt.subplots(2, 1, figsize=(18, 16), dpi=300)
    fig.suptitle("FULL INDIAN OCEAN BASIN MAGNETIC ANOMALY MAP (20°E – 140°E | -60°S – 30°N)\nComparing Original EMAG2 vs Model 1 (4x Super-Resolved Output) in 'TURBO' Colormap", fontsize=16, fontweight='bold')
    
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    # Panel 1: Original Low-Res Full Basin Map
    im1 = axes[0].imshow(full_lr, cmap='turbo', extent=extent, vmin=vmin, vmax=vmax, origin='upper')
    axes[0].set_title("1. Original Low-Resolution EMAG2 Magnetic Anomaly Map (Full Indian Ocean)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Longitude (°E)", fontsize=11)
    axes[0].set_ylabel("Latitude (°N)", fontsize=11)
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.025, pad=0.02)
    cbar1.set_label("Magnetic Anomaly (nT)", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.4)
    
    # Panel 2: Model 1 4x Super-Resolved Full Basin Map
    im2 = axes[1].imshow(sr_full, cmap='turbo', extent=extent, vmin=vmin, vmax=vmax, origin='upper')
    axes[1].set_title("2. Model 1 (4x Super-Resolved Output) — Full Indian Ocean Basin (15 Arc-Sec)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Longitude (°E)", fontsize=11)
    axes[1].set_ylabel("Latitude (°N)", fontsize=11)
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.025, pad=0.02)
    cbar2.set_label("Magnetic Anomaly (nT)", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_map), exist_ok=True)
    plt.savefig(out_map, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nSUCCESS: Full Indian Ocean Map saved to:\n  {out_map}", flush=True)

if __name__ == '__main__':
    main()
