# Mathematical \& Statistical Gravity Correlation Data

This document contains the rigorous spatial and statistical correlation metrics between seafloor geomorphological features and marine Free-Air gravity anomalies. All values are derived by mathematically extracting gravity pixels interior to the feature footprint and comparing them against a localized 20km background control ring.

## 1\. Global Correlation Summary (Class Level)

This table summarizes the statistical relationship between the morphometric footprint area (km²) of the features and their observed Gravity Contrast (Interior Gravity minus Local Background Gravity).

|Feature Class|N|Mean Contrast (mGal)|Cohen's d (Effect Size)|Welch t-test (adj-p)|Pearson r|Spearman rho|Reg R²|
|-|-|-|-|-|-|-|-|
|Mid Ocean Ridge|195|+11.94|+0.56|4.40e-04|+0.04|+0.18|0.002|
|Escarpment|38|+7.60|+0.30|1.49e-01|+0.29|+0.14|0.086|
|Ocean Trench|34|-27.66|-0.87|6.23e-02|-0.09|-0.19|0.008|
|Underwater Plateau|16|+7.34|+0.18|1.83e-01|+0.14|+0.51|0.021|
|Fracture Fault Zone|108|-7.91|-0.34|1.43e-01|+0.06|+0.06|0.004|
|Seamount|328|+15.33|+0.86|1.64e-07|+0.05|+0.27|0.002|
|Ocean Basin|102|-10.19|-0.58|1.49e-01|+0.10|+0.14|0.009|
|Submarine Volcano|17|+33.37|+1.34|4.56e-02|+0.01|+0.39|0.000|

## 2\. Detailed Mathematical Breakdown by Feature Class

### Mid Ocean Ridge

#### A. Zonal Statistics (Averages across all features in class)

* **Mean Interior Gravity:** +17.53 mGal (σ = 19.55)
* **Mean Background Gravity:** +5.59 mGal (σ = 23.78)
* **Absolute Gravity Contrast (Δ):** +11.94 mGal

#### B. Morphometric Correlation Metrics (Area vs Gravity Contrast)

* **Pearson Correlation Coefficient (r):** +0.042 (adj-p = 6.95e-01)
* **Spearman Rank Correlation (rho):** +0.181
* **Linear Regression R²:** 0.0018
* **Regression Coefficient (Slope):** +3.7557e-05 mGal / km²

#### C. Statistical Difference (Interior vs Local Background Control)

* **Welch's t-statistic:** +3.909
* **Welch's p-value (FDR Adjusted):** 4.40e-04
* **Mean Cohen's d Effect Size:** +0.556

\---

### Escarpment

#### A. Zonal Statistics (Averages across all features in class)

* **Mean Interior Gravity:** +15.21 mGal (σ = 16.68)
* **Mean Background Gravity:** +7.61 mGal (σ = 21.58)
* **Absolute Gravity Contrast (Δ):** +7.60 mGal

#### B. Morphometric Correlation Metrics (Area vs Gravity Contrast)

* **Pearson Correlation Coefficient (r):** +0.294 (adj-p = 5.87e-01)
* **Spearman Rank Correlation (rho):** +0.142
* **Linear Regression R²:** 0.0863
* **Regression Coefficient (Slope):** +5.2174e-04 mGal / km²

#### C. Statistical Difference (Interior vs Local Background Control)

* **Welch's t-statistic:** +1.535
* **Welch's p-value (FDR Adjusted):** 1.49e-01
* **Mean Cohen's d Effect Size:** +0.296

\---

### Ocean Trench

#### A. Zonal Statistics (Averages across all features in class)

* **Mean Interior Gravity:** -53.78 mGal (σ = 25.97)
* **Mean Background Gravity:** -26.12 mGal (σ = 34.70)
* **Absolute Gravity Contrast (Δ):** -27.66 mGal

#### B. Morphometric Correlation Metrics (Area vs Gravity Contrast)

* **Pearson Correlation Coefficient (r):** -0.091 (adj-p = 6.95e-01)
* **Spearman Rank Correlation (rho):** -0.185
* **Linear Regression R²:** 0.0083
* **Regression Coefficient (Slope):** -2.1958e-04 mGal / km²

#### C. Statistical Difference (Interior vs Local Background Control)

* **Welch's t-statistic:** -2.204
* **Welch's p-value (FDR Adjusted):** 6.23e-02
* **Mean Cohen's d Effect Size:** -0.869

\---

### Underwater Plateau

#### A. Zonal Statistics (Averages across all features in class)

* **Mean Interior Gravity:** +14.41 mGal (σ = 19.07)
* **Mean Background Gravity:** +7.07 mGal (σ = 21.24)
* **Absolute Gravity Contrast (Δ):** +7.34 mGal

#### B. Morphometric Correlation Metrics (Area vs Gravity Contrast)

* **Pearson Correlation Coefficient (r):** +0.144 (adj-p = 6.95e-01)
* **Spearman Rank Correlation (rho):** +0.509
* **Linear Regression R²:** 0.0208
* **Regression Coefficient (Slope):** +1.1628e-05 mGal / km²

#### C. Statistical Difference (Interior vs Local Background Control)

* **Welch's t-statistic:** +1.364
* **Welch's p-value (FDR Adjusted):** 1.83e-01
* **Mean Cohen's d Effect Size:** +0.175

\---

### Fracture Fault Zone

#### A. Zonal Statistics (Averages across all features in class)

* **Mean Interior Gravity:** -18.40 mGal (σ = 19.38)
* **Mean Background Gravity:** -10.49 mGal (σ = 25.50)
* **Absolute Gravity Contrast (Δ):** -7.91 mGal

#### B. Morphometric Correlation Metrics (Area vs Gravity Contrast)

* **Pearson Correlation Coefficient (r):** +0.065 (adj-p = 6.95e-01)
* **Spearman Rank Correlation (rho):** +0.063
* **Linear Regression R²:** 0.0042
* **Regression Coefficient (Slope):** +5.6967e-05 mGal / km²

#### C. Statistical Difference (Interior vs Local Background Control)

* **Welch's t-statistic:** -1.706
* **Welch's p-value (FDR Adjusted):** 1.43e-01
* **Mean Cohen's d Effect Size:** -0.341

\---

### Seamount

#### A. Zonal Statistics (Averages across all features in class)

* **Mean Interior Gravity:** +25.84 mGal (σ = 14.35)
* **Mean Background Gravity:** +10.52 mGal (σ = 21.78)
* **Absolute Gravity Contrast (Δ):** +15.33 mGal

#### B. Morphometric Correlation Metrics (Area vs Gravity Contrast)

* **Pearson Correlation Coefficient (r):** +0.047 (adj-p = 6.95e-01)
* **Spearman Rank Correlation (rho):** +0.272
* **Linear Regression R²:** 0.0022
* **Regression Coefficient (Slope):** +4.9362e-04 mGal / km²

#### C. Statistical Difference (Interior vs Local Background Control)

* **Welch's t-statistic:** +5.680
* **Welch's p-value (FDR Adjusted):** 1.64e-07
* **Mean Cohen's d Effect Size:** +0.859

\---

### Ocean Basin

#### A. Zonal Statistics (Averages across all features in class)

* **Mean Interior Gravity:** -19.47 mGal (σ = 10.54)
* **Mean Background Gravity:** -9.28 mGal (σ = 17.43)
* **Absolute Gravity Contrast (Δ):** -10.19 mGal

#### B. Morphometric Correlation Metrics (Area vs Gravity Contrast)

* **Pearson Correlation Coefficient (r):** +0.096 (adj-p = 6.95e-01)
* **Spearman Rank Correlation (rho):** +0.137
* **Linear Regression R²:** 0.0091
* **Regression Coefficient (Slope):** +3.5076e-06 mGal / km²

#### C. Statistical Difference (Interior vs Local Background Control)

* **Welch's t-statistic:** -1.518
* **Welch's p-value (FDR Adjusted):** 1.49e-01
* **Mean Cohen's d Effect Size:** -0.575

\---

### Submarine Volcano

#### A. Zonal Statistics (Averages across all features in class)

* **Mean Interior Gravity:** +46.21 mGal (σ = 21.36)
* **Mean Background Gravity:** +12.84 mGal (σ = 23.82)
* **Absolute Gravity Contrast (Δ):** +33.37 mGal

#### B. Morphometric Correlation Metrics (Area vs Gravity Contrast)

* **Pearson Correlation Coefficient (r):** +0.008 (adj-p = 9.75e-01)
* **Spearman Rank Correlation (rho):** +0.391
* **Linear Regression R²:** 0.0001
* **Regression Coefficient (Slope):** +2.5973e-05 mGal / km²

#### C. Statistical Difference (Interior vs Local Background Control)

* **Welch's t-statistic:** +2.520
* **Welch's p-value (FDR Adjusted):** 4.56e-02
* **Mean Cohen's d Effect Size:** +1.344

\---

