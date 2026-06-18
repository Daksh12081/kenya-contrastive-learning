import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


PATCH_SIZE = 256


def find_s1_bands(safe_path):
    measurement_dir = os.path.join(safe_path, "measurement")

    vv_files = glob.glob(os.path.join(measurement_dir, "*vv*.tiff"))
    vh_files = glob.glob(os.path.join(measurement_dir, "*vh*.tiff"))

    if not vv_files:
        raise FileNotFoundError("VV band not found")

    if not vh_files:
        raise FileNotFoundError("VH band not found")

    return vv_files[0], vh_files[0]


def process_s1_safe(safe_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    vv_file, vh_file = find_s1_bands(safe_path)

    with rasterio.open(vv_file) as vv_src:
        vv = vv_src.read(1).astype(np.float32)
        profile = vv_src.profile
        transform = vv_src.transform
        crs = vv_src.crs
        height = vv_src.height
        width = vv_src.width
        resolution = vv_src.res

    with rasterio.open(vh_file) as vh_src:
        vh = vh_src.read(1).astype(np.float32)

    print("VV shape:", vv.shape)
    print("VH shape:", vh.shape)
    print("Resolution:", resolution)
    print("CRS:", crs)

    s1_stack = np.stack([vv, vh], axis=0)

    print("Sentinel-1 stack shape:", s1_stack.shape)

    stack_profile = profile.copy()

    stack_profile.update({
        "driver": "GTiff",
        "count": 2,
        "dtype": "float32"
    })

    patch_count = 0

    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):

            patch = s1_stack[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            patch_transform = window_transform(
                Window(x, y, PATCH_SIZE, PATCH_SIZE),
                transform
            )

            patch_profile = stack_profile.copy()

            patch_profile.update({
                "height": PATCH_SIZE,
                "width": PATCH_SIZE,
                "transform": patch_transform
            })

            patch_path = os.path.join(
                output_dir,
                f"s1_patch_{patch_count:04d}.tif"
            )

            with rasterio.open(patch_path, "w", **patch_profile) as dst:
                dst.write(patch.astype(np.float32))

            patch_count += 1

    print("Total Sentinel-1 patches saved:", patch_count)

    return patch_count


if __name__ == "__main__":
    print("preprocess_s1 module loaded successfully")