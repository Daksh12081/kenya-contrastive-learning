import os
import pickle
import random

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


S2_DROP_CIRRUS_INDEX = 9

FOREST_INDICES = [0, 1, 2, 3, 4]

SHRUB_INDICES = [5, 6]

CROP_INDICES = [11, 13]

GRASS_INDICES = [7, 8, 9]


NONVEGETATION_INDICES = [10, 12, 14, 15, 16]

CLASS_NAMES = {
    0: 'Forest',
    1: 'Shrub',
    2: 'Crop',
    3: 'Grass',
    4: 'NonVegetation',
}


class SEN12MSCROMADataset(Dataset):
    def __init__(
        self,
        root_dir=os.environ.get("SEN12MS_ROOT", "datasets/sen12ms-subset"),
        metadata_dir="datasets/SEN12MS/metadata",
        max_samples=None,
        seed=42,
    ):
        self.root_dir = root_dir
        self.metadata_dir = metadata_dir
        self.max_samples = max_samples
        self.seed = seed

        label_path = os.path.join(metadata_dir, "IGBP_probability_labels_spring.pkl")
        train_list_path = os.path.join(metadata_dir, "train_list_spring.pkl")

        with open(label_path, "rb") as f:
            self.labels = pickle.load(f)

        with open(train_list_path, "rb") as f:
            self.train_list = set(pickle.load(f))

        self.samples = self._find_samples()

        if self.max_samples is not None:
            random.seed(self.seed)
            random.shuffle(self.samples)
            self.samples = self.samples[: self.max_samples]

        print(f"Loaded SEN12MS CROMA samples: {len(self.samples)}")

    def _find_samples(self):
        samples = []

        if not os.path.exists(self.root_dir):
            print(f"WARNING: root_dir does not exist: {self.root_dir}")
            return samples

        # Layout 1: official extracted SEN12MS archive
        # root_dir/ROIs1158_spring/s1_1/ROIs1158_spring_s1_1_p100.tif
        if os.path.basename(self.root_dir) == "ROIs1158_spring":
            official_root = self.root_dir
        else:
            official_root = os.path.join(self.root_dir, "ROIs1158_spring")
        if os.path.exists(official_root):
            print(f"Scanning official SEN12MS directory: {official_root}")
            for folder in os.listdir(official_root):
                if not folder.startswith("s1_"):
                    continue

                roi = folder.replace("s1_", "")
                s1_dir = os.path.join(official_root, f"s1_{roi}")
                s2_dir = os.path.join(official_root, f"s2_{roi}")

                if not os.path.isdir(s1_dir) or not os.path.isdir(s2_dir):
                    continue

                for filename in os.listdir(s1_dir):
                    if not filename.endswith(".tif"):
                        continue

                    s1_path = os.path.join(s1_dir, filename)
                    s2_filename = filename.replace("_s1_", "_s2_")
                    s2_path = os.path.join(s2_dir, s2_filename)

                    if not os.path.exists(s2_path):
                        continue

                    label_key = s2_filename
                    if label_key not in self.train_list:
                        continue
                    if label_key not in self.labels:
                        continue

                    samples.append(
                        {
                            "s1_path": s1_path,
                            "s2_path": s2_path,
                            "label_key": label_key,
                        }
                    )

            print(f"Found {len(samples)} official SEN12MS samples")
            return samples

        # Layout 2: local subset layout used by earlier experiments
        # root_dir/..._s1/s1_1/ROIs1158_spring_s1_1_p100.tif
        for folder in os.listdir(self.root_dir):
            if not folder.endswith("_s1"):
                continue

            s1_root = os.path.join(self.root_dir, folder)
            s2_root = os.path.join(self.root_dir, folder.replace("_s1", "_s2"))

            if not os.path.exists(s2_root):
                continue

            for s1_subfolder in os.listdir(s1_root):
                s1_dir = os.path.join(s1_root, s1_subfolder)
                s2_dir = os.path.join(s2_root, s1_subfolder.replace("s1_", "s2_"))

                if not os.path.isdir(s1_dir) or not os.path.isdir(s2_dir):
                    continue

                for filename in os.listdir(s1_dir):
                    if not filename.endswith(".tif"):
                        continue

                    s1_path = os.path.join(s1_dir, filename)
                    s2_filename = filename.replace("_s1_", "_s2_")
                    s2_path = os.path.join(s2_dir, s2_filename)

                    if not os.path.exists(s2_path):
                        continue

                    label_key = s2_filename
                    if label_key not in self.train_list:
                        continue
                    if label_key not in self.labels:
                        continue

                    samples.append(
                        {
                            "s1_path": s1_path,
                            "s2_path": s2_path,
                            "label_key": label_key,
                        }
                    )

        return samples

    def _read_s1(self, path):
        with rasterio.open(path) as src:
            s1 = src.read().astype(np.float32)

        s1 = np.nan_to_num(s1)
        s1 = np.clip(s1, -25, 0)
        s1 = s1 / 25.0 + 1.0
        return torch.from_numpy(s1).float()

    def _read_s2(self, path):
        with rasterio.open(path) as src:
            s2 = src.read().astype(np.float32)

        s2 = np.delete(s2, S2_DROP_CIRRUS_INDEX, axis=0)
        s2 = np.clip(s2, 0, 10000)
        s2 = s2 / 10000.0
        return torch.from_numpy(s2).float()

    def _resize(self, x, size=120):
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    def _multiclass_label(self, label_key):
        label_vector = self.labels[label_key]
        dominant_class = int(np.argmax(label_vector))

        if dominant_class in FOREST_INDICES:
            return torch.tensor(0, dtype=torch.long)
        elif dominant_class in SHRUB_INDICES:
            return torch.tensor(1, dtype=torch.long)
        elif dominant_class in CROP_INDICES:
            return torch.tensor(2, dtype=torch.long)
        elif dominant_class in GRASS_INDICES:
            return torch.tensor(3, dtype=torch.long)
        else:
            return torch.tensor(4, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        s1 = self._read_s1(sample["s1_path"])
        s2 = self._read_s2(sample["s2_path"])

        s1 = self._resize(s1, 120)
        s2 = self._resize(s2, 120)

        label = self._multiclass_label(sample["label_key"])

        return {
            "s1": s1,
            "s2": s2,
            "label": label,
            "id": sample["label_key"],
        }


if __name__ == "__main__":
    print("SEN12MS_ROOT:", os.environ.get("SEN12MS_ROOT"))
    dataset = SEN12MSCROMADataset(max_samples=10)

    if len(dataset) == 0:
        print("No samples found.")
        print("Check that root_dir points to the extracted SEN12MS folder on this machine.")
    else:
        sample = dataset[0]

        print("S1 shape:", sample["s1"].shape)
        print("S2 shape:", sample["s2"].shape)
        print("Label:", sample["label"])
        print("ID:", sample["id"])
        print("""
Classes:
0 = Forest
1 = Shrubs
2 = Crops
3 = Grass
4 = NonVegetation
""")