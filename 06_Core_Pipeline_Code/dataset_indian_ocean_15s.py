"""
High-Resolution (15-Arc-Second / 450m) Multi-Modal Dataset Loader for Indian Ocean.
Integrates:
1. 15-Arc-Second (~450m resolution) ETOPO Bathymetry Tiles from:
   02_NetCDF_Ocean_DataGrids/15s_Bathymetry_Tiles/
2. NOAA EMAG2 v3 Marine Magnetics (High-Res target & bicubic downsampled input)
3. SWOT Free-Air Satellite Gravity Anomaly
Extracts synchronized 450m-resolution spatial patches across the active tectonic ridges
and basins of the Indian Ocean.
"""

import os
import glob
import rasterio
import rasterio.windows
import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class IndianOcean15sDataset(Dataset):
    def __init__(self, 
                 emag_path, 
                 tiles_dir, 
                 grav_path, 
                 patch_size=64, 
                 num_samples=500, 
                 seed=42):
        super(IndianOcean15sDataset, self).__init__()
        self.patch_size = patch_size
        self.num_samples = num_samples
        np.random.seed(seed)
        
        # 1. Open NOAA EMAG2 raster & SWOT gravity
        self.src_emag = rasterio.open(emag_path)
        self.ds_grav = xr.open_dataset(grav_path)
        
        # 2. Index all 15-arc-second bathymetry tiles
        tile_files = sorted(glob.glob(os.path.join(tiles_dir, "*.nc")))
        self.tiles = []
        for tf in tile_files:
            ds = xr.open_dataset(tf)
            lat_arr = ds.lat.values
            lon_arr = ds.lon.values
            self.tiles.append({
                'path': tf,
                'ds': ds,
                'lat_min': float(lat_arr.min()),
                'lat_max': float(lat_arr.max()),
                'lon_min': float(lon_arr.min()),
                'lon_max': float(lon_arr.max())
            })
            
        print(f"Indexed {len(self.tiles)} high-res 15-arc-second (~450m) bathymetry tiles.")
        
        # 3. Generate random spatial patch locations strictly within 15s tile coverage
        self.coords = []
        # Span corresponding to 64 pixels at 15-arc-seconds (64 * 0.25 arc-min = 16 arc-min = 0.2667 deg)
        d_deg = (patch_size * 15.0) / 3600.0  # 0.2667 degrees
        
        print(f"Sampling {num_samples} 15-arc-second (~450m) deep-ocean patches...")
        attempts = 0
        while len(self.coords) < num_samples and attempts < num_samples * 10:
            attempts += 1
            # Pick a random tile
            tile = np.random.choice(self.tiles)
            lon_c = np.random.uniform(tile['lon_min'] + d_deg, tile['lon_max'] - d_deg)
            lat_c = np.random.uniform(tile['lat_min'] + d_deg, tile['lat_max'] - d_deg)
            
            b_lon_min = lon_c - d_deg / 2
            b_lon_max = lon_c + d_deg / 2
            b_lat_min = lat_c - d_deg / 2
            b_lat_max = lat_c + d_deg / 2
            
            try:
                sub_z = tile['ds'].z.sel(lat=slice(b_lat_min, b_lat_max), lon=slice(b_lon_min, b_lon_max)).values
                if sub_z.size > 0 and np.nanmean(sub_z) < -200:
                    self.coords.append((tile['path'], b_lon_min, b_lat_min, b_lon_max, b_lat_max))
            except Exception:
                continue

        print(f"Successfully generated {len(self.coords)} 15-arc-second deep-ocean training patches.")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        tile_path, b1, b2, b3, b4 = self.coords[idx]
        
        # 1. Read High-Res 15s Bathymetry from the corresponding tile
        ds_tile = xr.open_dataset(tile_path)
        sub_b = ds_tile.z.sel(lat=slice(b2, b4), lon=slice(b1, b3)).values
        b_t = torch.tensor(sub_b, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        b_hr = F.interpolate(b_t, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=False).squeeze().numpy()
        
        # 2. Read NOAA EMAG2 Magnetics
        win = rasterio.windows.from_bounds(b1, b2, b3, b4, self.src_emag.transform)
        m_hr = self.src_emag.read(1, window=win, out_shape=(self.patch_size, self.patch_size), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        m_lr = self.src_emag.read(1, window=win, out_shape=(self.patch_size // 2, self.patch_size // 2), resampling=rasterio.enums.Resampling.cubic).astype(np.float32)
        
        # 3. Read SWOT Gravity
        sub_g = self.ds_grav.z.sel(lat=slice(b2, b4), lon=slice(b1, b3)).values
        g_t = torch.tensor(sub_g, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        g_hr = F.interpolate(g_t, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=False).squeeze().numpy()
        
        def norm(a):
            a = np.nan_to_num(a)
            mi, ma = np.min(a), np.max(a)
            return (a - mi) / (ma - mi + 1e-7)
            
        m_up = F.interpolate(torch.tensor(norm(m_lr)).unsqueeze(0).unsqueeze(0), size=(self.patch_size, self.patch_size), mode='bicubic', align_corners=False).squeeze().numpy()
        
        x_in = np.stack([m_up, norm(b_hr), norm(g_hr)], axis=0).astype(np.float32)
        y_tgt = np.expand_dims(norm(m_hr), axis=0).astype(np.float32)
        b_guide = np.expand_dims(norm(b_hr), axis=0).astype(np.float32)
        
        return torch.tensor(x_in), torch.tensor(y_tgt), torch.tensor(b_guide)

if __name__ == '__main__':
    emag = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\04_Marine_Magnetic_and_Rasters\EMAG2_V3_SeaLevel_DataTiff.tif"
    tiles_dir = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\15s_Bathymetry_Tiles"
    grav = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\02_NetCDF_Ocean_DataGrids\grav_SWOT_05.nc"
    
    ds = IndianOcean15sDataset(emag, tiles_dir, grav, patch_size=64, num_samples=10)
    x, y, b = ds[0]
    print(f"Verified 15s Sample: Input={x.shape}, Target={y.shape}, Guide={b.shape}")
    print("SUCCESS: 15-arc-second dataset loader is operational!")
