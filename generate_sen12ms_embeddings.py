import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.sen12ms_croma_dataset import SEN12MSCROMADataset
from use_croma import PretrainedCROMA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "outputs/sen12ms_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "CROMA_base.pt"


def main():
    dataset = SEN12MSCROMADataset(max_samples=5000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = PretrainedCROMA(
        pretrained_path=MODEL_PATH,
        size="base",
        modality="both",
        image_resolution=120,
    ).to(DEVICE)

    model.eval()

    embeddings = []
    labels = []
    ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            s1 = batch["s1"].to(DEVICE)
            s2 = batch["s2"].to(DEVICE)

            outputs = model(SAR_images=s1, optical_images=s2)
            joint = outputs["joint_GAP"]

            embeddings.append(joint.cpu())
            labels.append(batch["label"])
            ids.extend(batch["id"])

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    torch.save(embeddings, os.path.join(OUTPUT_DIR, "croma_embeddings.pt"))
    torch.save(labels, os.path.join(OUTPUT_DIR, "labels.pt"))

    with open(os.path.join(OUTPUT_DIR, "ids.txt"), "w") as f:
        for sample_id in ids:
            f.write(sample_id + "\n")

    print("Saved embeddings:", embeddings.shape)
    print("Saved labels:", labels.shape)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
