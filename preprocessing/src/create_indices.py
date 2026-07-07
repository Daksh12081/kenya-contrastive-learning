import numpy as np


def add_indices(patch):
    """
    Add vegetation indices to a 10-channel Sentinel-2 patch.

    Input:
        patch: numpy array of shape (10, H, W)

    Output:
        numpy array of shape (14, H, W)
        Original 10 bands + NDVI + EVI + MSAVI + LAI
    """

    BLUE = patch[0]   # B02
    RED = patch[2]    # B04
    NIR = patch[3]    # B08

    eps = 1e-8

    NDVI = (NIR - RED) / (NIR + RED + eps)

    EVI = 2.5 * (
        (NIR - RED)
        /
        (NIR + 6 * RED - 7.5 * BLUE + 1 + eps)
    )

    MSAVI = (
        (2 * NIR + 1)
        - np.sqrt(
            np.maximum(
                (2 * NIR + 1) ** 2 - 8 * (NIR - RED),
                0
            )
        )
    ) / 2

    LAI = 3.618 * EVI - 0.118

    patch14 = np.concatenate(
        [
            patch,
            NDVI[np.newaxis, :, :],
            EVI[np.newaxis, :, :],
            MSAVI[np.newaxis, :, :],
            LAI[np.newaxis, :, :]
        ],
        axis=0
    )

    return patch14


if __name__ == "__main__":
    dummy_patch = np.random.rand(10, 256, 256).astype(np.float32)
    output = add_indices(dummy_patch)
    print(f"Input shape: {dummy_patch.shape}")
    print(f"Output shape: {output.shape}")
