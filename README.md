# 🌊 Indian Ocean Potential-Field Super-Resolution & SwinIR-EMAG Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research and production-grade framework for **4× potential-field super-resolution** over the Indian Ocean lithosphere, integrating **SwinIR transformer backbones**, **geophysical physics-informed loss suites**, **2D Hann-window sliding-window inference**, and multi-altitude aeromagnetic/marine benchmark datasets.

---

## 📂 Project Architecture & Directory Layout

```
Indian_Ocean_Features/
├── 05_Benchmark_Codebases/             # Physics benchmark baselines
│   ├── 01_DownwardContinuation_Geodata/ # Classical FFT upward/downward continuation
│   ├── 01_Harmonica_EquivalentLayer/    # Harmonica equivalent layer inversion
│   ├── 02_CrossGradient_SimPEG/        # SimPEG joint cross-gradient inversion
│   └── 03_FourierNeuralOperator_FNO/   # FNO potential field baseline
├── 06_Core_Pipeline_Code/              # Reusable dataset loaders & physics loss modules
│   ├── dataset_indian_ocean_4x_sr.py   # Multi-modal EMAG2v3 + ETOPO2022 dataset loader
│   ├── physics_loss_4x_sr.py           # 4-component physics loss suite
│   ├── physics_loss_advanced.py        # Spectral Fourier + edge coherence loss
│   └── dataset_indian_ocean_15s_v2.py  # High-resolution 15-arc-sec bathymetry loader
├── 07_Training_Scripts/                # PyTorch local & Kaggle GPU training pipelines
│   ├── train_4x_sr_swinir.py           # Local workstation training loop
│   ├── train_kaggle_4x_sr_1000ep.py    # 1000-epoch Kaggle distributed training
│   ├── Kaggle_Training_SwinIR.ipynb    # Interactive Jupyter notebook for Kaggle GPUs
│   └── train_swinir_ocean.py           # Deep ocean patch fine-tuning
├── 08_Evaluation_and_Inference/        # Verification, proof, & full-basin inference
│   ├── benchmark_metrics.py            # PSNR, SSIM, Spectral Slope, RMSE metrics
│   ├── inference_whole_indian_ocean.py # Full-basin 4x inference with 2D Hann tiling
│   ├── verify_physics_proof.py         # Taylor series & Laplace boundary verification
│   └── render_full_indian_ocean_turbo.py # Publication-quality geophysical rendering
├── 10_Documentation_and_Reports/       # Scientific reports, audit documentation, & guides
│   ├── Model1_4x_SuperResolution_Report.md
│   ├── Brutal_Audit_Baseline_vs_OptionA_Report.md
│   └── High_Resolution_Marine_Magnetic_GroundTruth_Guide.md
├── 11_Sharp_Edge_Enhanced_4x_Model/    # Sharp-edge enhanced 4x model suite
│   ├── dataset_indian_ocean_4x_sharp.py
│   ├── physics_loss_4x_sharp.py
│   └── compare_baseline_vs_sharp_4x.py
├── diagrams/                           # Interactive Archify SVG architecture & pipeline diagrams
│   ├── swinir_architecture.html        # Interactive model architecture visualization
│   ├── swinir_inference_workflow.html  # End-to-end inference flow & Hann-window tiling
│   └── swinir_training_dataflow.html   # Training dataflow & 4-component loss suite
└── SwinIR/                             # Core SwinIR transformer backbone implementation
    ├── models/
    │   ├── swinir_emag_v3.py           # SwinIR v3 with DC-offset bicubic skip connection
    │   └── network_swinir.py           # Standard SwinIR backbone
    ├── losses/
    │   └── physics_loss.py             # L1 + Continental Gradient + Spectral + SSIM
    ├── datasets/
    │   └── dataset_emag.py             # Multi-modal PyTorch dataset class
    ├── train_emag.py                   # Self-contained standalone training script
    └── inference_emag_tiled.py         # Ripple-compensated 2D Hann-tiling inference
```

---

## 🔬 Core Innovations

### 1. Global DC-Offset Bicubic Skip (`swinir_emag_v3.py`)
Standard super-resolution networks struggle with potential-field datasets because magnetic and gravity fields have absolute regional DC baselines. Our v3 model architecture injects an explicit **residual bicubic path**:
$$I_{SR} = f_{SwinIR}(I_{LR}) + \text{Bicubic}(I_{LR})$$
forcing the deep Swin Transformer layers to focus solely on high-frequency lithospheric anomalies and crustal faults.

### 2. 4-Component Geophysical Physics Loss (`physics_loss.py`)
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_1 + \lambda_{grad} \mathcal{L}_{grad} + \lambda_{spec} \mathcal{L}_{spec} + \lambda_{SSIM} \mathcal{L}_{SSIM}$$
- **$L_1$ Loss**: Preserves absolute potential-field amplitudes in nanoTeslas ($nT$).
- **Continental Gradient Loss**: Enforces horizontal derivative consistency ($\nabla_x, \nabla_y$) along mid-ocean ridges and fracture zones.
- **Spectral Fourier Loss**: Penalizes spectral slope roll-off in the radially averaged power spectrum (RAPS).
- **Structural Similarity (SSIM)**: Retains high-frequency tectonic fabric and magnetic lineations.

### 3. Boundary-Artifact-Free 2D Hann Tiling (`inference_emag_tiled.py`)
To process basin-scale rasters (thousands of kilometers) on single GPUs, we employ a 2D Hann-window sliding tile engine with 75% overlap:
$$w_{2D}(x, y) = w_{Hann}(x) \otimes w_{Hann}(y)$$
With exact weight-matrix normalization, this completely eliminates tile boundary seams and high-frequency edge ringing.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/imshivanshutiwari/SwinIR_EMAG.git
cd SwinIR_EMAG
pip install torch torchvision numpy scipy tifffile matplotlib
```

### 2. Model Inference with 2D Hann Tiling
```bash
python SwinIR/inference_emag_tiled.py \
    --input "path/to/lr_magnetic_grid.tif" \
    --weights "path/to/swinir_model1_4x_sr.pth" \
    --output "path/to/sr_output.tif" \
    --tile_size 128 --stride 32
```

### 3. Training
```bash
python SwinIR/train_emag.py \
    --data_dir "data/patches" \
    --batch_size 16 \
    --epochs 100 \
    --lr 2e-4
```

### 4. Interactive Architecture Diagrams
The `diagrams/` folder contains standalone interactive SVG diagrams viewable in any browser:
- `diagrams/swinir_architecture.html`
- `diagrams/swinir_inference_workflow.html`
- `diagrams/swinir_training_dataflow.html`

---

## 📜 Citation & License
This project is licensed under the MIT License. Developed for Indian Ocean geophysical potential-field mapping and deep-sea mineral exploration.
