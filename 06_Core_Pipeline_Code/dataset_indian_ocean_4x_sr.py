"""
UPGRADED 4x Super-Resolution Dataset Loader for Indian Ocean.

Fixes Applied vs Previous Version:
  FIX 1: Global channel-wise normalization (computed from data, not per-patch min-max)
          This preserves real nT physics values across patches.
  FIX 2: 4x Super-Resolution scale (64x64 LR -> 256x256 HR at 15 arc-sec)
          Uses true 4x downsampling to create challenging LR inputs
  FIX 3: Larger 256x256 HR patch size (covers ~115km x 115km at 15 arc-sec)
          Allows SpectralLaplace loss to see full low-freq wavenumber decay
  FIX 4: All guide channels (bathy, gravity) are LR at 64x64 for proper fusion

Architecture:
  Input:  [64x64]  3-ch: [LR_mag, LR_bathy, LR_gravity]
  Target: [256x256] 1-ch: HR magnetic anomaly
  Guide:  [256x256] 1-ch: HR bathymetry for CrossGradient loss
"""

import os
import glob
import random
import rasterio
import rasterio.windows
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset

EMAG_NODATA = 255.0
EMAG_NODATA_FRAC_MAX = 0.10


class IndianOcean4xSRDataset(Dataset):
    def __init__(self,
                 emag_path,
                 bathy_dir,
                 grav_path,
                 patch_size_hr=256,   # HR patch: 256x256 at 15 arc-sec (~115km)
                 scale_factor=4,       # 4x Super-Resolution
                 num_samples=300,
                 augment=True,
                 seed=42):
        super().__init__()
        self.patch_size_hr  = patch_size_hr
        self.patch_size_lr  = patch_size_hr // scale_factor   # 64x64
        self.scale_factor   = scale_factor
        self.augment        = augment
        np.random.seed(seed)
        random.seed(seed)

        print("Opening EMAG2 & SWOT Gravity datasets...", flush=True)
        self.src_emag  = rasterio.open(emag_path)
        self.ds_grav   = xr.open_dataset(grav_path)

        tile_files = sorted(glob.glob(os.path.join(bathy_dir, "ETOPO2022_15s_*.tif")))
        print(f"Found {len(tile_files)} 15-arc-second bathymetry GeoTIFF tiles.", flush=True)
        self.src_bathy_tiles = [rasterio.open(f) for f in tile_files]

        self.lon_min, self.lon_max = 25.0, 140.0
        self.lat_min, self.lat_max = -55.0, 25.0

        # ── FIX 1: Compute global channel stats from data sample ──────────────
        print("Computing global channel statistics for normalization...", flush=True)
        self.mag_mean, self.mag_std, self.bathy_mean, self.bathy_std, \
            self.grav_mean, self.grav_std = self._compute_global_stats(n_sample=50)
        print(f"  EMAG2   — mean: {self.mag_mean:.2f} nT   std: {self.mag_std:.2f} nT", flush=True)
        print(f"  Bathy   — mean: {self.bathy_mean:.1f} m   std: {self.bathy_std:.1f} m", flush=True)
        print(f"  Gravity — mean: {self.grav_mean:.3f}   std: {self.grav_std:.3f}", flush=True)

        # ── Pre-generate valid patch coordinates ──────────────────────────────
        print(f"Generating {num_samples} valid 256px 15s patch locations...", flush=True)
        self.coords = []
        attempts = 0
        while len(self.coords) < num_samples and attempts < num_samples * 40:
            attempts += 1
            lon_c = np.random.uniform(self.lon_min, self.lon_max)
            lat_c = np.random.uniform(self.lat_min, self.lat_max)

            # 256 pixels at 15 arc-sec = 256 * (15/3600) degrees
            d_deg     = patch_size_hr * (15.0 / 3600.0)
            b_lon_min = lon_c - d_deg / 2
            b_lon_max = lon_c + d_deg / 2
            b_lat_min = lat_c - d_deg / 2
            b_lat_max = lat_c + d_deg / 2

            # Must be deep ocean
            sub_b = self._read_bathy_patch(b_lon_min, b_lat_min, b_lon_max, b_lat_max, 64)
            if sub_b is None or np.nanmean(sub_b) >= -200:
                continue

            # FIX 1: Reject patches with >10% EMAG nodata
            win = rasterio.windows.from_bounds(
                b_lon_min, b_lat_min, b_lon_max, b_lat_max, self.src_emag.transform)
            raw_m = self.src_emag.read(
                1, window=win,
                out_shape=(patch_size_hr, patch_size_hr),
                resampling=rasterio.enums.Resampling.bilinear
            ).astype(np.float32)
            if np.mean(raw_m == EMAG_NODATA) > EMAG_NODATA_FRAC_MAX:
                continue

            self.coords.append((b_lon_min, b_lat_min, b_lon_max, b_lat_max))

        print(f"Generated {len(self.coords)} valid 4x SR patches "
              f"(rejected {attempts - len(self.coords)} nodata/land patches).", flush=True)

    def _compute_global_stats(self, n_sample=50):
        """Sample random deep-ocean patches to compute global mean/std per channel."""
        mag_vals, bathy_vals, grav_vals = [], [], []
        attempts = 0
        while len(mag_vals) < n_sample and attempts < n_sample * 20:
            attempts += 1
            lon_c = np.random.uniform(self.lon_min, self.lon_max)
            lat_c = np.random.uniform(self.lat_min, self.lat_max)
            d_deg     = 64 * (15.0 / 3600.0)
            bln = lon_c - d_deg / 2; blx = lon_c + d_deg / 2
            blan = lat_c - d_deg / 2; blax = lat_c + d_deg / 2

            sub_b = self._read_bathy_patch(bln, blan, blx, blax, 32)
            if sub_b is None or np.nanmean(sub_b) >= -200:
                continue

            try:
                win = rasterio.windows.from_bounds(bln, blan, blx, blax, self.src_emag.transform)
                m = self.src_emag.read(1, window=win, out_shape=(32, 32),
                                       resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
                m[m == EMAG_NODATA] = np.nan
                if np.nanmean(np.isnan(m)) > 0.3:
                    continue
                mag_vals.extend(m[~np.isnan(m)].tolist())
                bathy_vals.extend(sub_b[~np.isnan(sub_b)].tolist())

                sub_g = self.ds_grav.z.sel(lat=slice(blan, blax), lon=slice(bln, blx)).values
                if sub_g.size > 0 and not np.all(np.isnan(sub_g)):
                    grav_vals.extend(sub_g[~np.isnan(sub_g)].flatten().tolist())
            except Exception:
                continue

        mag_mean  = float(np.nanmean(mag_vals))  if mag_vals   else 0.0
        mag_std   = float(np.nanstd(mag_vals))   if mag_vals   else 1.0
        b_mean    = float(np.nanmean(bathy_vals)) if bathy_vals else -3000.0
        b_std     = float(np.nanstd(bathy_vals))  if bathy_vals else 1500.0
        g_mean    = float(np.nanmean(grav_vals))  if grav_vals  else 0.0
        g_std     = float(np.nanstd(grav_vals))   if grav_vals  else 40.0

        # Protect against zero std
        mag_std  = max(mag_std, 1.0)
        b_std    = max(b_std,   1.0)
        g_std    = max(g_std,   1.0)
        return mag_mean, mag_std, b_mean, b_std, g_mean, g_std

    def _read_bathy_patch(self, lon_min, lat_min, lon_max, lat_max, size):
        for src in self.src_bathy_tiles:
            b = src.bounds
            if b.left <= lon_min and b.right >= lon_max and \
               b.bottom <= lat_min and b.top >= lat_max:
                win  = rasterio.windows.from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
                data = src.read(1, window=win, out_shape=(size, size),
                                resampling=rasterio.enums.Resampling.bilinear)
                return data.astype(np.float32)
        return None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        try:
            return self._get_item(idx)
        except Exception:
            return self._get_item(0)

    def _get_item(self, idx):
        b_lon_min, b_lat_min, b_lon_max, b_lat_max = self.coords[idx]
        LR = self.patch_size_lr   # 64
        HR = self.patch_size_hr   # 256

        # ── 1. EMAG2 Magnetic — HR target + LR input ─────────────────────────
        win_emag = rasterio.windows.from_bounds(
            b_lon_min, b_lat_min, b_lon_max, b_lat_max, self.src_emag.transform)
        m_hr = self.src_emag.read(1, window=win_emag, out_shape=(HR, HR),
                                  resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        m_lr = self.src_emag.read(1, window=win_emag, out_shape=(LR, LR),
                                  resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        # FIX 1: mask nodata then global normalize
        m_hr[m_hr == EMAG_NODATA] = np.nan
        m_lr[m_lr == EMAG_NODATA] = np.nan

        # ── 2. Bathymetry — HR guide + LR input channel ───────────────────────
        b_hr = self._read_bathy_patch(b_lon_min, b_lat_min, b_lon_max, b_lat_max, HR)
        b_lr = self._read_bathy_patch(b_lon_min, b_lat_min, b_lon_max, b_lat_max, LR)
        if b_hr is None: b_hr = np.zeros((HR, HR), dtype=np.float32)
        if b_lr is None: b_lr = np.zeros((LR, LR), dtype=np.float32)

        # ── 3. SWOT Gravity — LR input channel ───────────────────────────────
        try:
            sub_g = self.ds_grav.z.sel(
                lat=slice(b_lat_min, b_lat_max),
                lon=slice(b_lon_min, b_lon_max)
            ).values
            if sub_g.size == 0 or np.all(np.isnan(sub_g)):
                g_lr = np.zeros((LR, LR), dtype=np.float32)
            else:
                g_t  = torch.tensor(sub_g, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                g_lr = torch.nn.functional.interpolate(
                    g_t, size=(LR, LR), mode='bilinear', align_corners=False
                ).squeeze().numpy()
        except Exception:
            g_lr = np.zeros((LR, LR), dtype=np.float32)

        # ── FIX 1: Global standardization (not per-patch min-max) ─────────────
        def standardize(arr, mean, std):
            arr = np.nan_to_num(arr, nan=mean)
            return np.clip((arr - mean) / std, -3.0, 3.0)   # clip at 3-sigma

        m_hr_n = standardize(m_hr, self.mag_mean,   self.mag_std)
        m_lr_n = standardize(m_lr, self.mag_mean,   self.mag_std)
        b_hr_n = standardize(b_hr, self.bathy_mean, self.bathy_std)
        b_lr_n = standardize(b_lr, self.bathy_mean, self.bathy_std)
        g_lr_n = standardize(g_lr, self.grav_mean,  self.grav_std)

        # ── FIX 2: 4x SR — LR input stays at 64x64, target at 256x256 ────────
        x_input  = np.stack([m_lr_n, b_lr_n, g_lr_n], axis=0).astype(np.float32)
        y_target = np.expand_dims(m_hr_n, axis=0).astype(np.float32)
        b_guide  = np.expand_dims(b_hr_n, axis=0).astype(np.float32)

        # ── Augmentation (flip/rotate — applied to HR and LR together) ────────
        if self.augment:
            if random.random() > 0.5:
                x_input  = x_input[:, :, ::-1].copy()
                y_target = y_target[:, :, ::-1].copy()
                b_guide  = b_guide[:,  :, ::-1].copy()
            if random.random() > 0.5:
                x_input  = x_input[:, ::-1, :].copy()
                y_target = y_target[:, ::-1, :].copy()
                b_guide  = b_guide[:,  ::-1, :].copy()
            k = random.randint(0, 3)
            if k > 0:
                x_input  = np.rot90(x_input,  k, axes=(1, 2)).copy()
                y_target = np.rot90(y_target, k, axes=(1, 2)).copy()
                b_guide  = np.rot90(b_guide,  k, axes=(1, 2)).copy()

        return (torch.tensor(x_input),
                torch.tensor(y_target),
                torch.tensor(b_guide))
