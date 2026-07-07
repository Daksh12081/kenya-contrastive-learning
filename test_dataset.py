from datasets.paired_dataset import PairedDataset

dataset = PairedDataset(
    "data/patch_pairs.csv"
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("S2 shape:", sample["s2"].shape)
print("S1 shape:", sample["s1"].shape)