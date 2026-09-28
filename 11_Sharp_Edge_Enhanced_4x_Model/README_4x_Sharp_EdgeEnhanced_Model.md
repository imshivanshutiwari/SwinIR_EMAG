# ⚡ Model 1 Extension: Ultra-Sharp Edge-Enhanced 4x Super-Resolution Model

This directory contains the **isolated, self-contained implementation** of the **Ultra-Sharp Edge-Enhanced 4x Super-Resolution Model**. 

It is designed to solve the spatial smoothing limitation of standard L1 loss by introducing **Sobel Edge High-Pass Loss** and **Bathymetry Slope Channels** to recover crisp magnetic anomaly lineations without AI hallucinations.

---

## 🔬 Key Architectural & Physics Upgrades

1. **4 Input Channels (with Bathymetry Slope $|\nabla B|$):**
   - Channel 0: Low-Res Magnetics ($64 \times 64$)
   - Channel 1: Low-Res Bathymetry Depth ($64 \times 64$)
   - Channel 2: **Low-Res Bathymetry Slope Magnitude $|\nabla B|$ ($64 \times 64$)**
   - Channel 3: Low-Res SWOT Satellite Gravity ($64 \times 64$)

2. **SobelEdgeLoss ($L_{\text{edge}}$):**
   High-pass spatial gradient loss $\big\| \nabla M_{\text{pred}} - \nabla M_{\text{target}} \big\|_1$ enforcing sharp high-frequency magnetic dipole lineation recovery.

3. **High-Wavenumber Boosted Laplace Loss ($(1.0 + 4.0 \cdot K^2)$):**
   Directs $80\%$ of spectral gradient updates into spatial wavenumbers $k > 32\text{ cycles/patch}$.

4. **SwinIR-Medium Architecture:**
   Configured with `in_chans=4`, `embed_dim=60`, and `depths=[6, 6, 6, 6, 6, 6]` ($1.67\text{M parameters}$).

---

## 📁 Directory Files

- **`dataset_indian_ocean_4x_sharp.py`**: 4-channel dataset loader with Sobel slope calculation.
- **`physics_loss_4x_sharp.py`**: Composite loss with SobelEdgeLoss and High-Wavenumber Laplace boosting.
- **`train_kaggle_4x_sharp_1000ep.py`**: Kaggle 1000-epoch training script with early stopping (patience=30).
- **`create_sharp_4x_kaggle_package.py`**: Packaging script for Kaggle ZIP.

---

## 🚀 How to Train on Kaggle:

1. Run `create_sharp_4x_kaggle_package.py` to generate `Kaggle_4xSharp_EdgeEnhanced_Master_Package.zip` in `09_Kaggle_Upload_Packages/`.
2. Upload the ZIP package to Kaggle GPU.
3. Run training command:
   ```bash
   !PYTHONPATH=.:SwinIR python train_kaggle_4x_sharp_1000ep.py
   ```
