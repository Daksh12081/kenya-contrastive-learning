import os
import sys
import pandas as pd
import torch
import rasterio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from preprocessing.src.create_indices import add_indices


def read_tif(path):
    with rasterio.open(path) as src:
        array = src.read()

    return array


def create_training_tensors(pairs_csv, output_dir):
    s2_output_dir = os.path.join(output_dir, "s2")
    s1_output_dir = os.path.join(output_dir, "s1")

    os.makedirs(s2_output_dir, exist_ok=True)
    os.makedirs(s1_output_dir, exist_ok=True)

    df = pd.read_csv(pairs_csv)

    rows = []

    for idx, row in df.iterrows():
        s2_patch = read_tif(row["s2_patch_path"])
        s1_patch = read_tif(row["s1_patch_path"])

        if s2_patch.shape[0] == 10:
            s2_patch = add_indices(s2_patch)
        elif s2_patch.shape[0] != 14:
            raise ValueError(f"Unexpected S2 channel count: {s2_patch.shape[0]}")

        s2_tensor = torch.tensor(s2_patch, dtype=torch.float32)
        s1_tensor = torch.tensor(s1_patch, dtype=torch.float32)

        s2_tensor_path = os.path.join(s2_output_dir, f"s2_{idx:06d}.pt")
        s1_tensor_path = os.path.join(s1_output_dir, f"s1_{idx:06d}.pt")

        torch.save(s2_tensor, s2_tensor_path)
        torch.save(s1_tensor, s1_tensor_path)

        rows.append({
            "s2_tensor": s2_tensor_path,
            "s1_tensor": s1_tensor_path,
        })

        if (idx + 1) % 50 == 0:
            print(f"Created {idx + 1}/{len(df)} tensor pairs")

    tensor_pairs = pd.DataFrame(rows)
    manifest_path = os.path.join(output_dir, "tensor_pairs.csv")
    tensor_pairs.to_csv(manifest_path, index=False)

    print("Tensor generation complete")
    print("S2 tensor shape:", tuple(s2_tensor.shape))
    print("S1 tensor shape:", tuple(s1_tensor.shape))
    print("Manifest saved:", manifest_path)

    return manifest_path


if __name__ == "__main__":
    create_training_tensors(
        pairs_csv="data/patch_pairs.csv",
        output_dir="outputs/tensors",
    )