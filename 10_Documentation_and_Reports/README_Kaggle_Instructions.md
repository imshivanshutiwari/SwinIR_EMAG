# 🚀 4x Super-Resolution Master Kaggle Package (SwinIR-Medium)

This package contains **100% of everything** to train the 4x Super-Resolution Model 1 on Kaggle:
- **4x Scale Factor**: 64x64 LR Input -> 256x256 HR Target at 15-arc-seconds (~115 km span)
- **Global Standardization**: Preserves absolute nanotesla (nT) magnetic values across patches
- **SwinIR-Medium**: 6 RSTB Blocks, 60 Channels, 4x PixelShuffle Upsampler
- **Enhanced Physics Losses**: lambda_spec=0.20, lambda_cg=0.15, SSIM=0.10 (FP32 numerical stability)
- **Early Stopping**: Patience = 30 epochs
- **All Data Included**: 9 bathymetry tiles, EMAG2 magnetics, SWOT gravity NetCDF
- **Pretrained Checkpoint**: `swinir_model1_4x_sr_best.pth` warm-start

---

## 📋 Instructions for Kaggle:

1. **Upload `Kaggle_4xSR_GodLevel_Master_Package.zip`** to your Kaggle Notebook.

2. **Unzip & Detector Cell**:
   ```python
   import os, glob, shutil, zipfile
   zip_matches = glob.glob("**/Kaggle_4xSR_GodLevel_Master_Package.zip", recursive=True)
   if zip_matches:
       with zipfile.ZipFile(zip_matches[0], "r") as z:
           z.extractall(".")
   else:
       script_matches = glob.glob("/kaggle/input/**/train_kaggle_4x_sr_1000ep.py", recursive=True)
       if script_matches:
           s_dir = os.path.dirname(script_matches[0])
           for item in os.listdir(s_dir):
               s_path = os.path.join(s_dir, item)
               d_path = os.path.join(".", item)
               if os.path.isdir(s_path):
                   if not os.path.exists(d_path): shutil.copytree(s_path, d_path)
               else: shutil.copy2(s_path, d_path)
   ```

3. **Enable GPU T4 / P100** in Kaggle settings.

4. **Launch Training**:
   ```bash
   !PYTHONPATH=.:SwinIR python train_kaggle_4x_sr_1000ep.py
   ```
