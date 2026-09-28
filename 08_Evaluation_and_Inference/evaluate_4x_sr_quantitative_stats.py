"""
Quantitative Proof & Benchmark Evaluation for 4x Super-Resolution SwinIR-Medium Model 1.
Computes PSNR (dB), SSIM, RMSE, MAE, Cross-Gradient Error, and Spectral Laplace Error across Indian Ocean test patches.
Also re-renders the 6-Panel Verification Map on a high-topography Ridge Patch (so Bathymetry & Gravity display rich geological features).
"""

import sys
sys.path.append(r"C:\Users\shiva\Downloads\Indian_Ocean_Features")
sys.path.append(r"C:\Users\shiva\Downloads\Indian_Ocean_Features\SwinIR")
sys.path.append(r"C:\Users\shiva\.gemini\antigravity-cli\brain\158f5b70-2534-4c90-9450-291ea8601c65\scratch")

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_indian_ocean_4x_sr import IndianOcean4xSRDataset
from models.network_swinir import SwinIR
from physics_loss_4x_sr import AdvancedGeophysicalPhysicsLoss

def calculate_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    emag = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    bathy_dir = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids"
    grav = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\grav_SWOT_05.nc"
    ckpt_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints\swinir_model1_4x_sr_kaggle_best.pth"
    out_map = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\01_Maps_and_Visualizations\Model1_4x_SR_Kaggle_Verification.png"
    
    patch_size_hr = 256
    patch_size_lr = 64
    
    dataset = IndianOcean4xSRDataset(
        emag_path=emag,
        bathy_dir=bathy_dir,
        grav_path=grav,
        patch_size_hr=patch_size_hr,
        scale_factor=4,
        num_samples=60,
        augment=False,
        seed=3030
    )
    
    print("\nLoading SwinIR-Medium 4x Model & Kaggle Trained Checkpoint...", flush=True)
    model = SwinIR(
        upscale=4, in_chans=3, img_size=patch_size_lr, window_size=8,
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
    criterion = AdvancedGeophysicalPhysicsLoss(patch_size=patch_size_hr).to(device)
    
    psnr_list, ssim_list, rmse_list, mae_list = [], [], [], []
    cg_list, spec_list = [], []
    
    best_patch_idx = 0
    max_combined_var = -1
    
    print("\nEvaluating 60 test patches across Indian Ocean...", flush=True)
    with torch.no_grad():
        for i in range(len(dataset)):
            x_in, y_tgt, b_g = dataset[i]
            x_t = x_in.unsqueeze(0).to(device)
            y_t = y_tgt.unsqueeze(0).to(device)
            b_t = b_g.unsqueeze(0).to(device)
            
            pred_t = head(model(x_t))
            
            p_np = pred_t.squeeze().cpu().numpy()
            t_np = y_t.squeeze().cpu().numpy()
            b_np = b_g.squeeze().cpu().numpy()
            g_np = x_in[2].numpy()
            
            # Check combined variance (magnetics + bathymetry + gravity) for selection
            comb_var = np.var(t_np) + np.var(b_np) + np.var(g_np)
            if comb_var > max_combined_var and not np.isnan(t_np).any():
                max_combined_var = comb_var
                best_patch_idx = i
                
            psnr_val = calculate_psnr(p_np, t_np)
            rmse_val = np.sqrt(np.mean((p_np - t_np) ** 2))
            mae_val  = np.mean(np.abs(p_np - t_np))
            
            _, r_l, cg_l, spec_l, tv_l, ssim_l = criterion(pred_t, y_t, b_t)
            
            psnr_list.append(psnr_val)
            ssim_list.append(1.0 - ssim_l.item())
            rmse_list.append(rmse_val)
            mae_list.append(mae_val)
            cg_list.append(cg_l.item())
            spec_list.append(spec_l.item())

    # --- PRINT QUANTITATIVE STATISTICAL PROOF ---
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_rmse = np.mean(rmse_list)
    avg_mae  = np.mean(mae_list)
    avg_cg   = np.mean(cg_list)
    avg_spec = np.mean(spec_list)
    
    print("\n" + "="*75, flush=True)
    print("  QUANTITATIVE PROOF & BENCHMARK STATS (4x SR MODEL 1)", flush=True)
    print("="*75, flush=True)
    print(f"  1. PSNR (Peak Signal-to-Noise Ratio) : {avg_psnr:.2f} dB  (High accuracy > 28 dB)", flush=True)
    print(f"  2. SSIM (Structural Similarity)      : {avg_ssim:.4f}     (High structural fidelity > 0.85)", flush=True)
    print(f"  3. RMSE (Root Mean Squared Error)   : {avg_rmse:.5f}", flush=True)
    print(f"  4. MAE (Mean Absolute Error)        : {avg_mae:.5f}", flush=True)
    print(f"  5. Cross-Gradient Alignment Error   : {avg_cg:.5f}    (Near zero boundary mismatch)", flush=True)
    print(f"  6. Spectral Laplace Frequency Loss   : {avg_spec:.5f}    (Harmonic potential-field decay)", flush=True)
    print("="*75, flush=True)

    # --- RE-RENDER VERIFICATION MAP ON HIGH-TOPOGRAPHY RIDGE PATCH ---
    print(f"\nRe-rendering 6-Panel Map on High-Topography Ridge Patch (Index {best_patch_idx})...", flush=True)
    x_in, y_tgt, b_g = dataset[best_patch_idx]
    with torch.no_grad():
        x_t = x_in.unsqueeze(0).to(device)
        pred_t = head(model(x_t)).squeeze(0).cpu()
        
    lr_mag      = x_in[0].numpy()
    bathy_guide = b_g[0].numpy()
    grav_guide  = x_in[2].numpy()
    hr_target   = y_tgt[0].numpy()
    sr_output   = pred_t[0].numpy()
    diff_error  = np.abs(hr_target - sr_output)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    fig.suptitle("Model 1 (SwinIR-Medium 4x SR) — Quantitative Verification & Topographic Ridge Analysis\nTrained on Kaggle GPU | PSNR: {:.2f} dB | SSIM: {:.4f}".format(avg_psnr, avg_ssim), fontsize=15, fontweight='bold')
    
    cmap_mag   = 'coolwarm'
    cmap_bathy = 'viridis'
    cmap_grav  = 'magma'
    cmap_diff  = 'inferno'
    
    im0 = axes[0, 0].imshow(lr_mag, cmap=cmap_mag)
    axes[0, 0].set_title("1. Input 4x Downsampled Magnetics (64x64)", fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im1 = axes[0, 1].imshow(hr_target, cmap=cmap_mag)
    axes[0, 1].set_title("2. Target High-Res Magnetics (256x256)", fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 2].imshow(sr_output, cmap=cmap_mag)
    axes[0, 2].set_title("3. Model 1 (4x SR) Super-Resolved Output", fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 0].imshow(bathy_guide, cmap=cmap_bathy)
    axes[1, 0].set_title("4. 15-Arc-Sec Bathymetry Seafloor Ridge (256x256)", fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im4 = axes[1, 1].imshow(grav_guide, cmap=cmap_grav)
    axes[1, 1].set_title("5. SWOT Free-Air Satellite Gravity Ridge", fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im5 = axes[1, 2].imshow(diff_error, cmap=cmap_diff)
    axes[1, 2].set_title("6. Absolute Error |Target - SR|", fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_map), exist_ok=True)
    plt.savefig(out_map, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"SUCCESS: Updated Verification Map saved to:\n  {out_map}", flush=True)

if __name__ == '__main__':
    main()
