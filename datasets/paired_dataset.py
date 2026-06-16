import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np


class PairedDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        s2_path = self.df.iloc[idx]["s2_patch_path"]
        s1_path = self.df.iloc[idx]["s1_patch_path"]

        with rasterio.open(s2_path) as src:
            s2 = src.read().astype(np.float32)

        with rasterio.open(s1_path) as src:
            s1 = src.read().astype(np.float32)

        s2 = torch.from_numpy(s2)
        s1 = torch.from_numpy(s1)

        return {
            "s2": s2,
            "s1": s1
        }