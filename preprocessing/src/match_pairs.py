import os
import glob
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling


def find_s1_bands(safe_path):
    measurement_dir = os.path.join(safe_path, "measurement")

    vv_files = glob.glob(os.path.join(measurement_dir, "*vv*.tiff"))
    vh_files = glob.glob(os.path.join(measurement_dir, "*vh*.tiff"))

    if not vv_files:
        raise FileNotFoundError("VV band not found")

    if not vh_files:
        raise FileNotFoundError("VH band not found")

    return vv_files[0], vh_files[0]


def clip_s1_to_s2_patch(s2_patch_path, vv_file, vh_file, output_dir, include_ratio=True):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(s2_patch_path) as s2:
        target_crs = s2.crs
        target_transform = s2.transform
        target_width = s2.width
        target_height = s2.height
        target_profile = s2.profile

    def reproject_band(s1_band_path):
        with rasterio.open(s1_band_path) as src:
            dst = np.empty((target_height, target_width), dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_width=target_width,
                dst_height=target_height,
                resampling=Resampling.bilinear,
            )

            return dst

    vv_clip = reproject_band(vv_file)
    vh_clip = reproject_band(vh_file)

    if include_ratio:
        ratio = vv_clip / (vh_clip + 1e-8)
        s1_stack = np.stack([vv_clip, vh_clip, ratio], axis=0)
    else:
        s1_stack = np.stack([vv_clip, vh_clip], axis=0)

    s2_patch_name = os.path.splitext(os.path.basename(s2_patch_path))[0]
    output_path = os.path.join(output_dir, f"{s2_patch_name}_S1_match.tif")

    profile = target_profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "height": target_height,
            "width": target_width,
            "count": s1_stack.shape[0],
            "dtype": "float32",
            "crs": target_crs,
            "transform": target_transform,
        }
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(s1_stack.astype(np.float32))

    return output_path


def create_matched_s1_patches(s2_patch_root, s1_safe_path, output_dir, include_ratio=True):
    s2_patches = sorted(glob.glob(os.path.join(s2_patch_root, "**", "*.tif"), recursive=True))

    vv_file, vh_file = find_s1_bands(s1_safe_path)

    saved_files = []

    for i, s2_patch in enumerate(s2_patches):
        output_path = clip_s1_to_s2_patch(
            s2_patch,
            vv_file,
            vh_file,
            output_dir,
            include_ratio=include_ratio,
        )
        saved_files.append(output_path)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(s2_patches)}")

    print(f"Total S1 matched patches saved: {len(saved_files)}")

    return saved_files


def create_patch_pairs_csv(s2_patch_root, s1_match_root, csv_path):
    s2_patches = sorted(glob.glob(os.path.join(s2_patch_root, "**", "*.tif"), recursive=True))
    s1_patches = sorted(glob.glob(os.path.join(s1_match_root, "*.tif")))

    s1_lookup = {
        os.path.basename(path).replace("_S1_match.tif", ""): path
        for path in s1_patches
    }

    rows = []

    for s2_path in s2_patches:
        s2_name = os.path.splitext(os.path.basename(s2_path))[0]
        s1_path = s1_lookup.get(s2_name)

        if s1_path:
            rows.append({"s2_path": s2_path, "s1_path": s1_path})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"Pairs saved: {len(df)}")

    return df


if __name__ == "__main__":
    print("match_pairs module loaded successfully")
