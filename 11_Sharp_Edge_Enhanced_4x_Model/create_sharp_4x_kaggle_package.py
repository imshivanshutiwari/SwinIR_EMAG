import os
import zipfile
import glob
import time

def create_sharp_4x_zip():
    zip_filename = r"C:\Users\shiva\Downloads\Indian_Ocean_Features\09_Kaggle_Upload_Packages\Kaggle_4xSharp_EdgeEnhanced_Master_Package.zip"
    print("==================================================================")
    print("  CREATING ULTRA-SHARP 4x EDGE-ENHANCED KAGGLE MASTER PACKAGE")
    print("==================================================================")
    print(f"Target Zip File: {zip_filename}\n")

    root_dir  = r"C:\Users\shiva\Downloads\Indian_Ocean_Features"
    sharp_dir = os.path.join(root_dir, "11_Sharp_Edge_Enhanced_4x_Model")
    raw_dir   = os.path.join(root_dir, "02_Raw_Data_Grids")
    mag_dir   = os.path.join(root_dir, "03_Marine_Magnetics_Rasters")

    code_files = [
        (os.path.join(sharp_dir, "train_kaggle_4x_sharp_1000ep.py"), "train_kaggle_4x_sharp_1000ep.py"),
        (os.path.join(sharp_dir, "train_kaggle_4x_sharp_1000ep.py"), "train_kaggle_1000ep.py"),
        (os.path.join(sharp_dir, "dataset_indian_ocean_4x_sharp.py"), "dataset_indian_ocean_4x_sharp.py"),
        (os.path.join(sharp_dir, "physics_loss_4x_sharp.py"), "physics_loss_4x_sharp.py"),
    ]

    # Data files
    data_files = []
    tile_files = sorted(glob.glob(os.path.join(raw_dir, "ETOPO2022_15s_*.tif")))
    for tf in tile_files:
        data_files.append((tf, os.path.basename(tf)))

    emag_file = os.path.join(mag_dir, "EMAG2_V3_SeaLevel_DataTiff.tif")
    if os.path.exists(emag_file):
        data_files.append((emag_file, "EMAG2_V3_SeaLevel_DataTiff.tif"))

    grav_file = os.path.join(raw_dir, "grav_SWOT_05.nc")
    if os.path.exists(grav_file):
        data_files.append((grav_file, "grav_SWOT_05.nc"))

    readme_content = """# ⚡ Ultra-Sharp 4x Edge-Enhanced Kaggle Master Package

Features of this Sharp Model:
- **4 Input Channels**: [LR_Mag, LR_Bathy, LR_Bathy_Slope |\nabla B|, LR_Gravity]
- **SobelEdgeLoss (L_edge)**: High-pass spatial gradient matching for crisp magnetic lineations
- **High-Wavenumber Boosted Laplace Loss**: (1.0 + 4.0 * K^2) directs 80% weight into high-frequency details
- **SwinIR-Medium 4x**: in_chans=4, embed_dim=60, 6 RSTB Blocks

---

## 📋 Launch Cell on Kaggle:

```bash
!PYTHONPATH=.:SwinIR python train_kaggle_4x_sharp_1000ep.py
```
"""

    start_time = time.time()
    total_bytes = 0

    os.makedirs(os.path.dirname(zip_filename), exist_ok=True)
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        zipf.writestr("README_Kaggle_Instructions.md", readme_content)

        print("Packing Sharp Python Scripts...")
        for src, arc in code_files:
            if os.path.exists(src):
                size = os.path.getsize(src)
                total_bytes += size
                zipf.write(src, arc)
                print(f"  + Added: {arc}")

        swinir_dir = os.path.join(root_dir, "SwinIR")
        if os.path.exists(swinir_dir):
            print("\nPacking SwinIR Backbone Package...")
            for r, d, files in os.walk(swinir_dir):
                if '__pycache__' in r: continue
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(r, file)
                        arc_path  = os.path.relpath(file_path, root_dir)
                        size = os.path.getsize(file_path)
                        total_bytes += size
                        zipf.write(file_path, arc_path)

        print("\nPacking Raw Data Grids & Tiles...")
        for src, arc in data_files:
            if os.path.exists(src):
                size = os.path.getsize(src)
                total_bytes += size
                print(f"  + Compressing Data File: {arc} ({size/1e6:.2f} MB)...", flush=True)
                zipf.write(src, arc)

    elapsed = time.time() - start_time
    zip_size = os.path.getsize(zip_filename)

    print("\n==================================================================")
    print("  ULTRA-SHARP 4x KAGGLE MASTER PACKAGE CREATED SUCCESSFULLY!")
    print("==================================================================")
    print(f"  Output File : {zip_filename}")
    print(f"  Compressed  : {zip_size/1e9:.2f} GB")
    print(f"  Time Taken  : {elapsed:.1f} seconds")
    print("==================================================================")

if __name__ == '__main__':
    create_sharp_4x_zip()
