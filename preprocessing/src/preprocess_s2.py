import os
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


PATCH_SIZE = 256

BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
ALL_BANDS = BANDS_10M + BANDS_20M


def find_band(safe_path, band_name):
    files = glob.glob(
        os.path.join(
            safe_path,
            "GRANULE",
            "*",
            "IMG_DATA",
            "**",
            f"*_{band_name}_*.jp2"
        ),
        recursive=True
    )

    if not files:
        raise FileNotFoundError(f"{band_name} not found in {safe_path}")

    return files[0]


def process_s2_safe(safe_path, output_root):
    product_name = os.path.basename(safe_path).replace(".SAFE", "")
    out_dir = os.path.join(output_root, product_name)
    os.makedirs(out_dir, exist_ok=True)

    band_paths = {band: find_band(safe_path, band) for band in ALL_BANDS}

    ref_band = band_paths["B02"]

    with rasterio.open(ref_band) as ref:
        ref_profile = ref.profile
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_height = ref.height
        ref_width = ref.width

    def read_resampled(path):
        with rasterio.open(path) as src:
            data = np.empty((ref_height, ref_width), dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_width=ref_width,
                dst_height=ref_height,
                resampling=Resampling.bilinear
            )

            return data

    stacked = []

    for band in ALL_BANDS:
        arr = read_resampled(band_paths[band])
        stacked.append(arr)

    s2_stack = np.stack(stacked, axis=0)

    stack_profile = ref_profile.copy()
    stack_profile.update({
        "driver": "GTiff",
        "count": s2_stack.shape[0],
        "dtype": "float32"
    })

    patch_count = 0

    for y in range(0, ref_height - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, ref_width - PATCH_SIZE + 1, PATCH_SIZE):
            patch = s2_stack[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            if patch.shape[1] != PATCH_SIZE or patch.shape[2] != PATCH_SIZE:
                continue

            patch_transform = window_transform(
                Window(x, y, PATCH_SIZE, PATCH_SIZE),
                ref_transform
            )

            patch_profile = stack_profile.copy()
            patch_profile.update({
                "height": PATCH_SIZE,
                "width": PATCH_SIZE,
                "transform": patch_transform
            })

            patch_path = os.path.join(
                out_dir,
                f"{product_name}_patch_{patch_count:04d}.tif"
            )

            with rasterio.open(patch_path, "w", **patch_profile) as dst:
                dst.write(patch.astype(np.float32))

            patch_count += 1

    print(f"Done: {product_name} | Stack: {s2_stack.shape} | Patches: {patch_count}")

    return patch_count


def process_s2_root(s2_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    safe_folders = glob.glob(os.path.join(s2_root, "**", "*.SAFE"), recursive=True)

    print("Found Sentinel-2 SAFE folders:", len(safe_folders))

    total_patches = 0

    for safe_path in safe_folders:
        patch_count = process_s2_safe(safe_path, output_root)
        total_patches += patch_count

    print("All Sentinel-2 files processed.")
    print("Total Sentinel-2 patches saved:", total_patches)

    return total_patches


if __name__ == "__main__":
    print("preprocess_s2 module loaded successfully")
