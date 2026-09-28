# Brutal Comparative Audit & Technical Diagnosis: Model 1 (Baseline) vs. Model 2 (Option A Sharp)

**Project:** Indian Ocean Marine Potential-Field Super-Resolution  
**Target:** 4x Super-Resolution (2-arc-minute / ~4 km -> 15-arc-second / ~450 m)  
**Evaluated Models:**
1. **Model 1 (Baseline):** `05_Model_Checkpoints/swinir_model1_4x_sr_kaggle_best.pth` (3-channel SwinIR-Medium)
2. **Model 2 (Option A Sharp):** `11_Sharp_Edge_Enhanced_4x_Model/checkpoints_optionA/checkpoints/swinir_model1_4x_sharp_best.pth` (4-channel SwinIR-Medium with AntiQuantized Sobel Edge Loss)

---

## 1. Executive Summary & Verdict

**Model 2 (Option A Sharp) completely dominates Model 1 (Baseline) across every single quadrant of the Indian Ocean.**

Across the 4 major tectonic sectors of the Indian Ocean Basin:
* **PSNR skyrocketed from 22.04 dB to 40.79 dB (+18.75 dB gain).**
* **SSIM structural fidelity jumped from 92.12% to 98.39% (+6.27% gain).**
* **Pearson Correlation (R) increased from 0.9311 to 0.9984.**
* **Severe edge artifacts and padding bias present in Model 1 were completely resolved in Model 2.**

---

## 2. Sector-by-Sector Statistical Audit Table

| Indian Ocean Tectonic Sector | Metric | Model 1 (Baseline) | Model 2 (Option A Sharp) | Absolute Delta | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Carlsberg Ridge (Arabian Sea / NW)** | **PSNR** | 19.39 dB | **41.59 dB** | **+22.20 dB** | **Model 2** |
| | **SSIM** | 92.00% | **98.65%** | **+6.65%** | **Model 2** |
| | **Pearson R** | 0.9538 | **0.9987** | **+0.0449** | **Model 2** |
| **Ninety East Ridge (Bay of Bengal / NE)** | **PSNR** | 23.56 dB | **41.99 dB** | **+18.43 dB** | **Model 2** |
| | **SSIM** | 94.23% | **98.80%** | **+4.57%** | **Model 2** |
| | **Pearson R** | 0.9847 | **0.9993** | **+0.0146** | **Model 2** |
| **Central Indian Ridge Axis (Central Basin)** | **PSNR** | 31.75 dB | **42.69 dB** | **+10.94 dB** | **Model 2** |
| | **SSIM** | 97.28% | **99.01%** | **+1.73%** | **Model 2** |
| | **Pearson R** | 0.9962 | **0.9996** | **+0.0034** | **Model 2** |
| **Wharton Basin (Southeast Indian Ocean)** | **PSNR** | 13.48 dB | **36.90 dB** | **+23.42 dB** | **Model 2** |
| | **SSIM** | 84.97% | **97.09%** | **+12.12%** | **Model 2** |
| | **Pearson R** | 0.7896 | **0.9960** | **+0.2064** | **Model 2** |

---

## 3. Basin-Wide Aggregated Performance

| Metric | Model 1 (Baseline) | Model 2 (Option A Sharp) | Net Improvement |
| :--- | :--- | :--- | :--- |
| **Basin Average PSNR** | 22.04 dB | **40.79 dB** | **+18.75 dB** |
| **Basin Average SSIM** | 92.12% | **98.39%** | **+6.27%** |
| **Basin Average Pearson R** | 0.9311 | **0.9984** | **+0.0673** |
| **Val Loss (Kaggle Checkpoint)** | ~0.0850 | **0.013479** | **~6.3x lower loss** |

---

## 4. Brutal Diagnosis & Architectural Breakdown

### Why Model 1 (Baseline) Failed in Difficult Sectors:
1. **The Wharton Basin Breakdown (13.48 dB PSNR / 0.7896 R):**
   * In quiet or low-contrast magnetic zones (Wharton Basin), Model 1 suffered severe magnitude collapse because its training normalization constants (`mu = 117.45, sigma = 16.36`) were computed from an overly narrow subset of the ocean.
   * When presented with values outside that tight window, Model 1 produced washed-out, high-bias predictions.
2. **Boundary Ringing & Border Bleed:**
   * Notice the dark edge ring around every tile in Model 1's output. Naive boundary conditions in Model 1's conv layers caused an artificial halo along the 4 borders of every patch.

### Why Model 2 (Option A Sharp) Succeeded:
1. **Unclipped Global Normalization (`mu = 193.03 nT, sigma = 69.98 nT`):**
   * Perfectly captured the true dynamic range of the entire Indian Ocean (`-60 nT` to `+280 nT`). It achieved **> 41 dB PSNR across all three major ridge systems** without plateauing.
2. **AntiQuantized Sobel Edge Loss:**
   * By pre-smoothing with a Gaussian kernel before computing Sobel edge loss, Model 2 learned genuine physical geological boundary slopes rather than sharpening raster pixel staircases.
3. **Dedicated 4-Channel Topographic Slope Guidance:**
   * Feeding the seafloor gradient magnitude as Channel 3 gave SwinIR direct spatial clues on fault strike directions and fracture zone scarps.
