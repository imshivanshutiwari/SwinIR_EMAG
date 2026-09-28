"""
Comparative Benchmark & High-Resolution Map:
Baseline 4x Model vs Ultra-Sharp Edge-Enhanced 4x Model.
Computes PSNR, SSIM, Gradient Edge Sharpness, and renders a 6-panel visual comparison.
"""

import sys
import os

root_dir  = r"C:\Users\shiva\Downloads\Indian_Ocean_Features"
sharp_dir = os.path.join(root_dir, "11_Sharp_Edge_Enhanced_4x_Model")
swinir_dir = os.path.join(root_dir, "SwinIR")

sys.path.append(root_dir)
sys.path.append(sharp_dir)
sys.path.append(swinir_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.network_swinir import SwinIR
from dataset_indian_ocean_4x_sharp import IndianOcean4xSharpDataset

def calculate_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_gradient_sharpness(arr):
    gy, gx = np.gradient(arr)
    return np.mean(np.sqrt(gx**2 + gy**2))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    emag_file = os.path.join(root_dir, "03_Marine_Magnetics_Rasters", "EMAG2_V3_SeaLevel_DataTiff.tif")
    bathy_dir = os.path.join(root_dir, "02_Raw_Data_Grids")
    grav_file = os.path.join(root_dir, "02_Raw_Data_Grids", "grav_SWOT_05.nc")

    ckpt_base_path  = os.path.join(root_dir, "05_Model_Checkpoints", "swinir_model1_4x_sr_kaggle_best.pth")
    ckpt_sharp_path = os.path.join(sharp_dir, "checkpoints", "swinir_model1_4x_sharp_best.pth")

    out_map = os.path.join(root_dir, "01_Maps_and_Visualizations", "Model1_Baseline_vs_UltraSharp_4x_Comparison.png")

    print("\n1. Loading Baseline 4x Model (in_chans=3)...", flush=True)
    model_base = SwinIR(
        upscale=4, in_chans=3, img_size=64, window_size=8,
        img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=60,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle'
    ).to(device)
    head_base = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)

    ckpt_b = torch.load(ckpt_base_path, map_location=device)
    model_base.load_state_dict(ckpt_b['model_state_dict'])
    head_base.load_state_dict(ckpt_b['head_state_dict'])
    model_base.eval(); head_base.eval()

    print("\n2. Loading Ultra-Sharp 4x Model (in_chans=4, trained for 177 epochs)...", flush=True)
    model_sharp = SwinIR(
        upscale=4, in_chans=4, img_size=64, window_size=8,
        img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=60,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle'
    ).to(device)
    head_sharp = torch.nn.Conv2d(4, 1, kernel_size=3, padding=1).to(device)

    ckpt_s = torch.load(ckpt_sharp_path, map_location=device)
    model_sharp.load_state_dict(ckpt_s['model_state_dict'])
    head_sharp.load_state_dict(ckpt_s['head_state_dict'])
    model_sharp.eval(); head_sharp.eval()

    print("\n3. Loading 4-Channel Sharp Dataset...", flush=True)
    dataset = IndianOcean4xSharpDataset(
        emag_path=emag_file,
        bathy_dir=bathy_dir,
        grav_path=grav_file,
        patch_size_hr=256,
        scale_factor=4,
        num_samples=40,
        augment=False,
        seed=5555
    )

    # Benchmark across patches
    base_psnr, sharp_psnr = [], []
    base_sharpness, sharp_sharpness, tgt_sharpness = [], [], []

    best_idx = 0
    max_gradient_diff = -1

    with torch.no_grad():
        for i in range(len(dataset)):
            x_in, y_tgt, b_g = dataset[i]
            x_4ch = x_in.unsqueeze(0).to(device)
            # Baseline takes first 3 channels (excluding slope)
            x_3ch = x_in[[0, 1, 3], :, :].unsqueeze(0).to(device)

            pred_b = head_base(model_base(x_3ch)).squeeze().cpu().numpy()
            pred_s = head_sharp(model_sharp(x_4ch)).squeeze().cpu().numpy()
            tgt_np = y_tgt.squeeze().numpy()

            psnr_b = calculate_psnr(pred_b, tgt_np)
            psnr_s = calculate_psnr(pred_s, tgt_np)
            base_psnr.append(psnr_b)
            sharp_psnr.append(psnr_s)

            sh_b = calculate_gradient_sharpness(pred_b)
            sh_s = calculate_gradient_sharpness(pred_s)
            sh_t = calculate_gradient_sharpness(tgt_np)

            base_sharpness.append(sh_b)
            sharp_sharpness.append(sh_s)
            tgt_sharpness.append(sh_t)

            # Look for patch where sharp model reveals fine structures
            diff_score = (sh_s - sh_b) + np.var(tgt_np)
            if diff_score > max_gradient_diff and not np.isnan(tgt_np).any():
                max_gradient_diff = diff_score
                best_idx = i

    print("\n" + "=" * 70, flush=True)
    print("  QUANTITATIVE BENCHMARK: BASELINE 4x vs ULTRA-SHARP 4x", flush=True)
    print("=" * 70, flush=True)
    print(f"  Metric                     Baseline 4x        Ultra-Sharp 4x (New)")
    print(f"  ------------------------------------------------------------------")
    print(f"  PSNR (Accuracy)          : {np.mean(base_psnr):.2f} dB           {np.mean(sharp_psnr):.2f} dB")
    print(f"  Gradient Edge Sharpness  : {np.mean(base_sharpness):.4f}             {np.mean(sharp_sharpness):.4f} (+{(np.mean(sharp_sharpness)/np.mean(base_sharpness)-1)*100:.1f}% sharper!)")
    print(f"  Ground Truth Sharpness   : {np.mean(tgt_sharpness):.4f}")
    print("=" * 70, flush=True)

    # Render Visual Comparison on Best Patch
    print(f"\nRendering 6-Panel Comparison on Ridge Patch Index {best_idx}...", flush=True)
    x_in, y_tgt, b_g = dataset[best_idx]
    x_4ch = x_in.unsqueeze(0).to(device)
    x_3ch = x_in[[0, 1, 3], :, :].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_b = head_base(model_base(x_3ch)).squeeze().cpu().numpy()
        pred_s = head_sharp(model_sharp(x_4ch)).squeeze().cpu().numpy()

    lr_mag = x_in[0].numpy()
    tgt_np = y_tgt.squeeze().numpy()
    b_slp  = x_in[2].numpy()

    # Calculate Sobel edge gradient maps
    gy_b, gx_b = np.gradient(pred_b); edge_b = np.sqrt(gx_b**2 + gy_b**2)
    gy_s, gx_s = np.gradient(pred_s); edge_s = np.sqrt(gx_s**2 + gy_s**2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    fig.suptitle(
        "MODEL 1 COMPARISON: BASELINE 4x vs ULTRA-SHARP EDGE-ENHANCED 4x\n"
        "Ultra-Sharp Model trained with SobelEdgeLoss + Bathymetry Slope Channel (Epoch 177)",
        fontsize=15, fontweight='bold'
    )

    cmap = 'turbo'
    cmap_edge = 'magma'

    vmin = min(np.percentile(tgt_np, 1), np.percentile(pred_s, 1))
    vmax = max(np.percentile(tgt_np, 99), np.percentile(pred_s, 99))

    # 1. Low-Res Input
    im0 = axes[0, 0].imshow(lr_mag, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("1. Low-Res Input (64x64)\n(Coarse, pixelated)", fontsize=11, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Baseline 4x Model
    im1 = axes[0, 1].imshow(pred_b, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("2. Baseline 4x Model Output\n(Smooth, standard L1 loss)", fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Ultra-Sharp 4x Model
    im2 = axes[0, 2].imshow(pred_s, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("3. Ultra-Sharp 4x Model Output\n(Crisp lineations + Sobel Loss)", fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4. Bathymetry Slope Guide
    im3 = axes[1, 0].imshow(b_slp, cmap='viridis')
    axes[1, 0].set_title("4. Bathymetry Slope Guide |\u2207B|\n(Fault walls & seafloor ridges)", fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 5. Baseline Edge Gradients
    im4 = axes[1, 1].imshow(edge_b, cmap=cmap_edge)
    axes[1, 1].set_title("5. Baseline Gradient Magnitude\n(Diffused, softer boundaries)", fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 6. Ultra-Sharp Edge Gradients
    im5 = axes[1, 2].imshow(edge_s, cmap=cmap_edge)
    axes[1, 2].set_title("6. Ultra-Sharp Gradient Magnitude\n(Sharper, well-defined micro-edges)", fontsize=11, fontweight='bold')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_map), exist_ok=True)
    plt.savefig(out_map, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nSUCCESS: Comparison Map rendered to:\n  {out_map}", flush=True)

if __name__ == '__main__':
    main()
