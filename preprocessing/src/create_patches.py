import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


PATCH_SIZE = 256


def save_patches(stack, profile, output_dir, product_name, patch_size=PATCH_SIZE):
    os.makedirs(output_dir, exist_ok=True)

    _, height, width = stack.shape

    stack_profile = profile.copy()
    stack_profile.update({
        "driver": "GTiff",
        "count": stack.shape[0],
        "dtype": "float32"
    })

    patch_count = 0

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = stack[:, y:y+patch_size, x:x+patch_size]

            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                continue

            patch_transform = window_transform(
                Window(x, y, patch_size, patch_size),
                profile["transform"]
            )

            patch_profile = stack_profile.copy()
            patch_profile.update({
                "height": patch_size,
                "width": patch_size,
                "transform": patch_transform
            })

            patch_path = os.path.join(
                output_dir,
                f"{product_name}_patch_{patch_count:04d}.tif"
            )

            with rasterio.open(patch_path, "w", **patch_profile) as dst:
                dst.write(patch.astype(np.float32))

            patch_count += 1

    print(f"Done: {product_name} | Stack: {stack.shape} | Patches: {patch_count}")

    return patch_count


if __name__ == "__main__":
    print("create_patches module loaded successfully")
