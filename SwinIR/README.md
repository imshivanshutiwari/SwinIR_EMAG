# SwinIR_EMAG: Geophysical Earth Magnetic Anomaly Grid Super-Resolution

**SwinIR_EMAG** is an advanced adaptation of the Swin Transformer (SwinIR) engineered for high-resolution geophysical potential field restoration, specifically super-resolving global satellite magnetic grids (EMAG2 $4\,\text{km}$) to continental airborne survey resolutions ($1\,\text{km}$).

Developed by **Shivanshu Tiwari**.

---

## 🌟 Key Innovations

1. **v3 DC-Offset Bicubic Skip Connection:**
   - Satellite EMAG2 observations exhibit a non-zero regional background level ($\sim +180\,\text{nT}$) relative to mean-zero high-resolution aeromagnetic surveys.
   - Standard residual connections cause amplitude crushing. SwinIR_EMAG implements a zero-centered global bicubic skip ($\text{Bicubic}(x) - \text{mean}(DC)$) that preserves residual learning without structural collapse.

2. **4-Component Geophysical Physics Loss Suite (`AdvancedGeophysicalPhysicsLoss`):**
   - **$\mathcal{L}_{L1}$ Pixel Loss (1.0):** Guarantees accurate regional scalar magnetic intensity.
   - **$\mathcal{L}_{CG}$ Continental Gradient Loss (0.15):** Aligns structural geological strike directions and magnetic dipoles.
   - **$\mathcal{L}_{spec}$ Spectral Fourier Loss (0.20):** Enforces high-wavenumber energy preservation in Fourier space.
   - **$\mathcal{L}_{SSIM}$ Structural Similarity (0.10):** Retains curvilinear magnetic lineaments and tectonic boundary sharpness.

3. **2D Hann-Window Sliding Inference for Continental Mosaics:**
   - Evaluates large grids (e.g. Australia $3688 \times 3556$) using $64\times64$ patches with $75\%$ overlap ($stride=32$).
   - Blends overlapping predictions with a 2D Hann taper kernel ($W_{2D} = \text{outer}(w, w) + 10^{-4}$) to completely eliminate boundary seam artifacts.

---

## 📊 Interactive Architecture & Pipeline Diagrams

The repository includes standalone, explorable Archify interactive HTML diagrams with dark/light themes, relationship tracing, and zoom/pan:

* 🌐 **Model Architecture:** [`swinir_architecture.html`](swinir_architecture.html) — Explores shallow feature extraction (`conv_first`), Swin Transformer Layers with W-MSA and SW-MSA, dual residual skips, and PixelShuffle upsampling.
* 🌐 **Inference Workflow:** [`swinir_inference_workflow.html`](swinir_inference_workflow.html) — Traces the complete sliding-window partition, 2D Hann taper blending, and continental evaluation pipeline.
* 🌐 **Training Pipeline Data Flow:** [`swinir_training_dataflow.html`](swinir_training_dataflow.html) — Illustrates multi-modal data ingestion (EMAG2 + SWOT gravity), patch sampling, and joint supervision through the 4-component physics loss suite.

---

## 📁 Repository Structure

```
SwinIR_EMAG/
├── datasets/
│   └── dataset_emag.py             # Multi-modal paired patch sampler (64 LR -> 256 HR)
├── losses/
│   └── physics_loss.py             # 4-component Geophysical Physics Loss suite
├── models/
│   ├── network_swinir.py           # Standard SwinIR backbone
│   └── swinir_emag_v3.py           # SwinIR v3 with DC-offset bicubic skip
├── train_emag.py                   # Full training script with CosineAnnealingLR & AdamW
├── inference_emag_tiled.py         # 2D Hann-window sliding inference engine
├── main_test_swinir.py             # Standard evaluation benchmark script
├── swinir_architecture.html        # Interactive Archify Model Architecture diagram
├── swinir_inference_workflow.html  # Interactive Archify Inference Workflow diagram
├── swinir_training_dataflow.html   # Interactive Archify Training Dataflow diagram
└── README.md                       # Project documentation
```

---

## 🚀 Quickstart

### 1. Training SwinIR_EMAG
```bash
python train_emag.py
```
Trains the SwinIR-Medium network ($6\times$ RSTB blocks, $embed\_dim=60$) on multi-modal magnetic and altimetry rasters using AdamW, gradient clipping (`max_norm=1.0`), and CosineAnnealingLR decay.

### 2. Continental Tiled Inference
```bash
python inference_emag_tiled.py
```
Applies 2D Hann-window overlap blending across full continental rasters to generate seamless, high-resolution $1\,\text{km}$ grids.

---

## 📜 References
- Liang et al., *SwinIR: Image Restoration Using Swin Transformer*, ICCV 2021.
- EMAG2 V3: Earth Magnetic Anomaly Grid (2-arc-minute resolution), NOAA / NCEI.
