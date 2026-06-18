

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess_s2 import process_s2_root
from src.preprocess_s1 import process_s1_safe
from src.match_pairs import create_matched_s1_patches, create_patch_pairs_csv


S2_ROOT = "data/sentinel2"
S1_SAFE_PATH = "data/sentinel1/sample.SAFE"
S2_PATCH_OUTPUT = "outputs/s2_patches"
S1_PATCH_OUTPUT = "outputs/s1_patches"
S1_MATCH_OUTPUT = "outputs/s1_matched"
PATCH_PAIRS_CSV = "data/patch_pairs.csv"


def main():
    print("Starting preprocessing pipeline...")

    print("\n[1/4] Processing Sentinel-2 SAFE files")
    process_s2_root(S2_ROOT, S2_PATCH_OUTPUT)

    print("\n[2/4] Processing Sentinel-1 SAFE file")
    process_s1_safe(S1_SAFE_PATH, S1_PATCH_OUTPUT)

    print("\n[3/4] Creating matched S1 patches")
    create_matched_s1_patches(
        S2_PATCH_OUTPUT,
        S1_SAFE_PATH,
        S1_MATCH_OUTPUT,
        include_ratio=True,
    )

    print("\n[4/4] Generating patch_pairs.csv")
    create_patch_pairs_csv(
        S2_PATCH_OUTPUT,
        S1_MATCH_OUTPUT,
        PATCH_PAIRS_CSV,
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()