"""
Scientific Benchmark Evaluation Script.
Evaluates model accuracy on 50 unseen validation patches across the Indian Ocean:
Metrics computed:
1. PSNR (Peak Signal-to-Noise Ratio in dB)
2. SSIM (Structural Similarity Index, -1 to 1)
3. RMSE (Root Mean Squared Error)
4. Gallardo-Meju Cross-Gradient Error
Compares:
- Baseline Standard Bicubic Interpolation
- Physics-Informed SwinIR Model
"""

import os
import sys
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, 'SwinIR'))

from models.network_swinir import SwinIR
from dataset_indian_ocean import IndianOceanMultiModalDataset
from physics_loss import CrossGradientLoss

def run_benchmark():
    print("=== STARTING QUANTITATIVE BENCHMARK EVALUATION (50 UNSEEN VALIDATION PATCHES) ===")
    
    device = torch.device('cpu')
    patch_size = 64
    
    # 1. Load 50 completely unseen validation patches using a different random seed
    emag = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    bathy = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\ETOPO_2022_v1_60s_N90W180_bed.nc"
    grav = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\grav_SWOT_05.nc"
    
    val_dataset = IndianOceanMultiModalDataset(emag, bathy, grav, patch_size_hr=patch_size, scale_factor=2, num_samples=50, seed=9999)
    
    # 2. Load SwinIR scaled checkpoint
    ckpt_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\checkpoints\swinir_indian_ocean_scaled.pth"
    model = SwinIR(upscale=1, in_chans=3, img_size=patch_size, window_size=8, img_range=1.0, depths=[4, 4, 4, 4], embed_dim=48, num_heads=[4, 4, 4, 4], mlp_ratio=2, upsampler='').to(device)
    head = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    head.load_state_dict(ckpt['head_state_dict'])
    model.eval()
    head.eval()
    
    cg_loss_fn = CrossGradientLoss()
    
    # Accumulators
    psnr_bicubic_list, psnr_swinir_list = [], []
    ssim_bicubic_list, ssim_swinir_list = [], []
    rmse_bicubic_list, rmse_swinir_list = [], []
    cg_bicubic_list, cg_swinir_list = [], []
    
    print("Evaluating 50 unseen patches...")
    for idx in range(len(val_dataset)):
        x, y, b = val_dataset[idx]
        
        # Bicubic baseline (Channel 0 of input is bicubic upsampled LR)
        bicubic_pred = x[0].numpy()
        target_hr = y.squeeze().numpy()
        guide_bathy = b.squeeze().numpy()
        
        # SwinIR Prediction
        with torch.no_grad():
            x_in = x.unsqueeze(0).to(device)
            swinir_pred = head(model(x_in)).squeeze().numpy()
            
        # Metrics: Bicubic
        psnr_b = compute_psnr(target_hr, bicubic_pred, data_range=1.0)
        ssim_b = compute_ssim(target_hr, bicubic_pred, data_range=1.0)
        rmse_b = np.sqrt(np.mean((target_hr - bicubic_pred)**2))
        
        # Metrics: SwinIR
        psnr_s = compute_psnr(target_hr, swinir_pred, data_range=1.0)
        ssim_s = compute_ssim(target_hr, swinir_pred, data_range=1.0)
        rmse_s = np.sqrt(np.mean((target_hr - swinir_pred)**2))
        
        # Cross-gradient error
        t_b_pred = torch.tensor(bicubic_pred).unsqueeze(0).unsqueeze(0)
        t_s_pred = torch.tensor(swinir_pred).unsqueeze(0).unsqueeze(0)
        t_target = y.unsqueeze(0)
        t_guide = b.unsqueeze(0)
        
        cg_b = cg_loss_fn(t_b_pred, t_guide).item()
        cg_s = cg_loss_fn(t_s_pred, t_guide).item()
        
        psnr_bicubic_list.append(psnr_b)
        psnr_swinir_list.append(psnr_s)
        ssim_bicubic_list.append(ssim_b)
        ssim_swinir_list.append(ssim_s)
        rmse_bicubic_list.append(rmse_b)
        rmse_swinir_list.append(rmse_s)
        cg_bicubic_list.append(cg_b)
        cg_swinir_list.append(cg_s)
        
    # Summarize Benchmark Results
    print("\n" + "="*70)
    print("       SCIENTIFIC ACCURACY BENCHMARK ON UNSEEN VALIDATION DATA       ")
    print("="*70)
    print(f"{'Metric':<30} | {'Bicubic Baseline':<16} | {'Physics SwinIR':<16} | {'Improvement':<12}")
    print("-" * 70)
    
    mean_psnr_b, mean_psnr_s = np.mean(psnr_bicubic_list), np.mean(psnr_swinir_list)
    mean_ssim_b, mean_ssim_s = np.mean(ssim_bicubic_list), np.mean(ssim_swinir_list)
    mean_rmse_b, mean_rmse_s = np.mean(rmse_bicubic_list), np.mean(rmse_swinir_list)
    mean_cg_b, mean_cg_s = np.mean(cg_bicubic_list), np.mean(cg_swinir_list)
    
    print(f"{'1. PSNR (Peak SNR in dB) [Higher is better]':<45} | {mean_psnr_b:>10.2f} dB | {mean_psnr_s:>10.2f} dB | {mean_psnr_s - mean_psnr_b:>+8.2f} dB")
    print(f"{'2. SSIM (Structural Similarity) [Higher is better]':<45} | {mean_ssim_b:>13.4f} | {mean_ssim_s:>13.4f} | {(mean_ssim_s - mean_ssim_b):>+10.4f}")
    print(f"{'3. RMSE (Normalized Error) [Lower is better]':<45} | {mean_rmse_b:>13.4f} | {mean_rmse_s:>13.4f} | {((mean_rmse_s - mean_rmse_b)/mean_rmse_b*100):>+8.1f} %")
    print(f"{'4. Cross-Gradient Error [Lower is better]':<45} | {mean_cg_b:>13.6f} | {mean_cg_s:>13.6f} | {((mean_cg_s - mean_cg_b)/mean_cg_b*100):>+8.1f} %")
    print("="*70)
    
    # Save report to text file
    report_path = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\01_Maps_and_Visualizations\SwinIR_Validation_Benchmark_Report.txt"
    with open(report_path, "w") as f:
        f.write("=== SCIENTIFIC ACCURACY BENCHMARK ON UNSEEN VALIDATION DATA ===\n")
        f.write(f"Number of unseen test patches: 50\n\n")
        f.write(f"1. PSNR (dB): Bicubic = {mean_psnr_b:.2f} dB, SwinIR = {mean_psnr_s:.2f} dB (Gain: {mean_psnr_s - mean_psnr_b:+.2f} dB)\n")
        f.write(f"2. SSIM:     Bicubic = {mean_ssim_b:.4f}, SwinIR = {mean_ssim_s:.4f} (Gain: {mean_ssim_s - mean_ssim_b:+.4f})\n")
        f.write(f"3. RMSE:     Bicubic = {mean_rmse_b:.4f}, SwinIR = {mean_rmse_s:.4f} (Reduction: {(mean_rmse_s - mean_rmse_b)/mean_rmse_b*100:+.1f}%)\n")
        f.write(f"4. CG Error: Bicubic = {mean_cg_b:.6f}, SwinIR = {mean_cg_s:.6f} (Reduction: {(mean_cg_s - mean_cg_b)/mean_cg_b*100:+.1f}%)\n")
    print(f"Report saved to: {report_path}")

if __name__ == '__main__':
    run_benchmark()
