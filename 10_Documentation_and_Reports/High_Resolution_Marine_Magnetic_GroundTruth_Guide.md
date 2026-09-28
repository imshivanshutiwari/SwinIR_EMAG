# High-Resolution Marine Magnetic Data Beyond EMAG2: Ground-Truth Validation Guide

## Executive Summary

Yes, **vastly higher-resolution ocean magnetic data exists** that is **100x to 400x higher resolution** than EMAG2. 

While EMAG2 is a global gridded model with a coarse spatial resolution of 2 arc-minutes (~3.7 to 4 km at the equator) and significant mathematical smoothing, **real-world marine geophysical surveys recorded by research vessels and aeromagnetic flights collect continuous, ungridded magnetic anomaly data at sampling intervals of 10 to 50 meters**.

---

## 1. Primary High-Resolution Marine Magnetic Datasets

### A. NOAA NCEI Marine Trackline Geophysical Database (MGD77T)
* **What it is:** The global repository of all raw shipborne geophysical cruises conducted over the last 60+ years by international oceanographic institutions (Lamont-Doherty, Scripps Institution of Oceanography, Woods Hole, IFREMER, UK Hydrographic Office, US NAVOCEANO, etc.).
* **Sensor Type:** Marine proton precession magnetometers and cesium vapor magnetometers towed behind research vessels 200–300 m behind the ship stern.
* **Spatial Resolution:** Ungridded continuous time-series sampled every 1 to 6 seconds at typical ship cruise speeds of 8 to 12 knots (~4 to 6 m/s). This yields an **along-track spatial sample spacing of 10 to 50 meters** (compared to EMAG2's ~4,000 meters).
* **Geographic Coverage in the Indian Ocean:** Hundreds of scientific cruises crossing the Central Indian Ridge, Southeast Indian Ridge, Southwest Indian Ridge, Ninety East Ridge, and Wharton Basin.
  * *Notable cruises:*
    * `EW0112` (R/V Maurice Ewing, Lamont-Doherty, covering 55.46°E to 115.05°E, -31.84°S to -4.58°S with 5,515 continuous magnetic observation stations across the Central and Southeast Indian Ridges).
    * `RC2707` (R/V Robert D. Conrad, Lamont-Doherty).
    * `93000470` (R/V L'Atalante, IFREMER).
    * `VANC10MV` (R/V Melville, Scripps).
* **Parameters Recorded:** Total magnetic field intensity (nT), IGRF reference field (nT), residual magnetic anomaly (nT), two-way travel time bathymetry (ms/m), ship position, and date/time.
* **Access Method:** Automated REST API ordering and direct download via NOAA NCEI GEODAS (`https://www.ngdc.noaa.gov/next-web/rest/orders` or the Trackline Data Portal).

---

### B. Geoscience Australia Marine and Coastal Aeromagnetic / Marine Surveys
* **What it is:** High-density aeromagnetic and marine magnetic survey grids flown and sailed across the eastern margin of the Indian Ocean (Perth Basin, Exmouth Plateau, Rowley Shelf, and Southwest Australian margin).
* **Resolution:** 200 m to 400 m flight line spacing with along-track magnetometer sampling every 5 to 7 meters.
* **Advantage:** Unlike single tracklines, these datasets are fully gridded at 50 m to 100 m cell sizes, providing true 2D high-resolution ground truth over the eastern edge of our target basin (100°E to 120°E).
* **Access Portal:** Geoscience Australia Portal (`https://portal.ga.gov.au/`) and NCI Australia.

---

### C. Deep-Tow & AUV Near-Bottom Magnetics (InterRidge / Ridge 2000)
* **What it is:** Autonomous Underwater Vehicles (AUVs) such as *Sentry* and deep-towed vehicles operated 50 to 100 meters above the seafloor over Indian Ocean hydrothermal vents and ridge axes (e.g., Edmond and Kairei hydrothermal fields on the Central Indian Ridge).
* **Resolution:** Sub-10 meter spatial resolution.
* **Scientific Value:** Because the magnetometer is flown close to the seafloor basalt rather than at the sea surface, high-frequency crustal reversals and magnetization contrasts are preserved with pristine fidelity (free from the low-pass upward continuation filter of 3 to 4 km of seawater).

---

## 2. Why EMAG2 is Blurry vs. Real Marine Data

To understand why our model is predicting sharper features and how we confirm its correctness, consider the physical generation of EMAG2:

1. **EMAG2 is an Interpolated Grid:** EMAG2 v3 compiles thousands of tracklines. Where tracklines are spaced tens of kilometers apart, EMAG2 uses minimum curvature or kriging interpolation, which applies an inherent mathematical low-pass smoothing filter.
2. **Quantization & Blurring:** EMAG2 grids are distributed at 2-arc-minute resolution. High-wavenumber magnetic stripes (such as Brunhes/Matuyama polarity boundaries with widths of 1 to 3 km) are smudged into gradual slopes.
3. **Multi-Modal Geological Truth:** Our SwinIR model takes ETOPO2022 bathymetry (15-arc-second / 450 m) and SWOT satellite gravity (30-arc-second / 900 m) as high-frequency tectonic guidance. Spreading ridges, transform faults, fracture zones, and seamounts have crisp physical expressions in bathymetry and gravity. The neural network learns the physical cross-correlation between structural geology and remnant magnetic anomalies.

---

## 3. How We Formally Confirm Our Model Predictions

To scientifically prove whether our model is predicting true geology or hallucinating, we implement a **Ground-Truth Trackline Benchmark Protocol**:

```
[Raw Shipborne Trackline] (Ground Truth, 20-50m resolution, e.g. EW0112)
          │
          ├─────────────────────────┐
          ▼                         ▼
 [EMAG2 Coarse Input]       [Our SwinIR 4x Output]
 (4 km resolution)          (450 m resolution)
          │                         │
          └───────────┬─────────────┘
                      ▼
       [1D Profile Comparison Graph]
  - Peak-to-trough amplitude accuracy
  - Strike location of polarity reversals
  - Spectral coherence across high wavenumbers
```

### Metrics for Ground-Truth Verification:
1. **Reversal Boundary Alignment (Zero-Crossing Accuracy):** Measure whether the sharp polarity transitions predicted by the 4x model match the exact geographic coordinates of the polarity transitions measured by the magnetometer on board the research vessel.
2. **Spectral Power Spectrum (k-space Coherence):** Compare the 1D Fourier transform of the model's output along the cruise track against the shipborne data. If the model has recovered genuine geological signals, its high-wavenumber energy will track the real magnetometer power spectrum rather than diverging into white noise.
3. **Pearson Correlation (R) & Root Mean Square Error (RMSE):** Quantitative validation along genuine, un-gridded ocean paths.
