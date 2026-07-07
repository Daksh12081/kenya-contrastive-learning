

import pandas as pd
import torch
from torch.utils.data import Dataset


class TensorPairDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        s2_tensor = torch.load(row["s2_tensor"], map_location="cpu")
        s1_tensor = torch.load(row["s1_tensor"], map_location="cpu")

        return {
            "s2": s2_tensor.float(),
            "s1": s1_tensor.float(),
        }


if __name__ == "__main__":
    dataset = TensorPairDataset("outputs/tensors/tensor_pairs.csv")

    sample = dataset[0]

    print("S2 shape:", sample["s2"].shape)
    print("S1 shape:", sample["s1"].shape)
    print("Total pairs:", len(dataset))