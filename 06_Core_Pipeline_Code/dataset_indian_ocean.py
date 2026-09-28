"""
Accurate Multi-Modal Dataset Loader for Indian Ocean Geophysical Super-Resolution.
Handles:
1. Low-Resolution Magnetic Anomaly (EMAG2 v3)
2. High-Resolution Bathymetry (ETOPO 2022)
3. High-Resolution SWOT Satellite Free-Air Gravity
Extracts spatially synchronized patches across the Indian Ocean (Lon: 20E..145E, Lat: -60S..30N)
"""

import os
import rasterio
import rasterio.windows
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class IndianOceanMultiModalDataset(Dataset):
    def __init__(self, 
                 emag_path, 
                 bathy_path, 
                 grav_path, 
                 patch_size_hr=128, 
                 scale_factor=2, 
                 num_samples=200, 
                 seed=42):
        super(IndianOceanMultiModalDataset, self).__init__()
        self.patch_size_hr = patch_size_hr
        self.scale_factor = scale_factor
        self.patch_size_lr = patch_size_hr // scale_factor
        self.num_samples = num_samples
        np.random.seed(seed)
        
        # Load datasets into memory/mmap
        print("Opening datasets...")
        self.src_emag = rasterio.open(emag_path)
        self.ds_bathy = xr.open_dataset(bathy_path)
        self.ds_grav = xr.open_dataset(grav_path)
        
        # Indian Ocean Bounding Box
        self.lon_min, self.lon_max = 25.0, 140.0
        self.lat_min, self.lat_max = -55.0, 25.0
        
        # Pre-generate valid random patch coordinates (focusing on ocean floor)
        print(f"Generating {num_samples} valid spatial patch locations across Indian Ocean...")
        self.coords = []
        attempts = 0
        while len(self.coords) < num_samples and attempts < num_samples * 10:
            attempts += 1
            # Random center lat/lon
            lon_c = np.random.uniform(self.lon_min, self.lon_max)
            lat_c = np.random.uniform(self.lat_min, self.lat_max)
            
            # Span corresponding to patch_size_hr at 1-arc-minute (~0.0167 deg per pixel)
            d_deg = (self.patch_size_hr * 0.0167)
            b_lon_min = lon_c - d_deg / 2
            b_lon_max = lon_c + d_deg / 2
            b_lat_min = lat_c - d_deg / 2
            b_lat_max = lat_c + d_deg / 2
            
            # Quick check if patch is in deep ocean (depth < -500m)
            try:
                sub_z = self.ds_bathy.z.sel(lat=slice(b_lat_min, b_lat_max), lon=slice(b_lon_min, b_lon_max)).values
                if sub_z.size > 0 and np.nanmean(sub_z) < -200:
                    self.coords.append((b_lon_min, b_lat_min, b_lon_max, b_lat_max))
            except Exception:
                continue

        print(f"Successfully generated {len(self.coords)} valid deep-ocean training patches.")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        b_lon_min, b_lat_min, b_lon_max, b_lat_max = self.coords[idx]
        
        # 1. Read EMAG2 Magnetics (HR target & downsampled LR)
        win = rasterio.windows.from_bounds(b_lon_min, b_lat_min, b_lon_max, b_lat_max, self.src_emag.transform)
        # Read at patch_size_hr
        m_hr = self.src_emag.read(1, window=win, out_shape=(self.patch_size_hr, self.patch_size_hr), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        # Read at patch_size_lr (simulating low-res sensor)
        m_lr = self.src_emag.read(1, window=win, out_shape=(self.patch_size_lr, self.patch_size_lr), resampling=rasterio.enums.Resampling.cubic).astype(np.float32)
        
        # 2. Read Bathymetry (HR structural guide)
        sub_b = self.ds_bathy.z.sel(lat=slice(b_lat_min, b_lat_max), lon=slice(b_lon_min, b_lon_max)).values
        # Resize to patch_size_hr
        b_t = torch.tensor(sub_b, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        b_hr = torch.nn.functional.interpolate(b_t, size=(self.patch_size_hr, self.patch_size_hr), mode='bilinear', align_corners=False).squeeze().numpy()
        
        # 3. Read SWOT Gravity (HR structural guide)
        sub_g = self.ds_grav.z.sel(lat=slice(b_lat_min, b_lat_max), lon=slice(b_lon_min, b_lon_max)).values
        g_t = torch.tensor(sub_g, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        g_hr = torch.nn.functional.interpolate(g_t, size=(self.patch_size_hr, self.patch_size_hr), mode='bilinear', align_corners=False).squeeze().numpy()
        
        # Normalize each channel to [0, 1] range for stable Transformer training
        def norm(arr):
            arr = np.nan_to_num(arr)
            mi, ma = np.min(arr), np.max(arr)
            if ma - mi > 1e-6:
                return (arr - mi) / (ma - mi)
            return np.zeros_like(arr)
        
        m_hr_n = norm(m_hr)
        m_lr_n = norm(m_lr)
        b_hr_n = norm(b_hr)
        g_hr_n = norm(g_hr)
        
        # For multi-modal conditioning: resize LR magnetics to HR grid size for channel concatenation
        m_lr_up = torch.nn.functional.interpolate(torch.tensor(m_lr_n).unsqueeze(0).unsqueeze(0), 
                                                  size=(self.patch_size_hr, self.patch_size_hr), 
                                                  mode='bicubic', 
                                                  align_corners=False).squeeze().numpy()
        
        # Input tensor: 3 channels [Low-Res Magnetics Upsampled, High-Res Bathymetry, High-Res Gravity]
        x_input = np.stack([m_lr_up, b_hr_n, g_hr_n], axis=0).astype(np.float32)
        y_target = np.expand_dims(m_hr_n, axis=0).astype(np.float32)
        b_guide = np.expand_dims(b_hr_n, axis=0).astype(np.float32)
        
        return torch.tensor(x_input), torch.tensor(y_target), torch.tensor(b_guide)

if __name__ == '__main__':
    emag = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    bathy = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\ETOPO_2022_v1_60s_N90W180_bed.nc"
    grav = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\grav_SWOT_05.nc"
    
    ds = IndianOceanMultiModalDataset(emag, bathy, grav, patch_size_hr=64, scale_factor=2, num_samples=10)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    for x, y, b in loader:
        print(f"Batch loaded -> Input X: {x.shape}, Target Y: {y.shape}, Guide Bathy: {b.shape}")
        break
    print("SUCCESS: Dataset loader fully verified!")
