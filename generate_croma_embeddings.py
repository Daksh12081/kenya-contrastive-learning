import os
import pandas as pd
import torch
import torch.nn.functional as F
from use_croma import PretrainedCROMA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KENYA_ROOT = "/Users/dakshanjaria/Kenya_Contrastive_model"
CSV_PATH = os.path.join(KENYA_ROOT, "outputs/tensors/tensor_pairs.csv")
MODEL_PATH = "CROMA_base.pt"

OUTPUT_DIR = "outputs/croma_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def resize_tensor(x, size=120):
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def adapt_for_croma(s1, s2):
    s1 = s1[:2]
    s2 = s2[:12]

    s1 = resize_tensor(s1, 120)
    s2 = resize_tensor(s2, 120)

    return s1, s2


def main():
    df = pd.read_csv(CSV_PATH)

    print(f"Total pairs: {len(df)}")
    print(f"Device: {DEVICE}")
    print("Loading CROMA...")

    model = PretrainedCROMA(
        pretrained_path=MODEL_PATH,
        size="base",
        modality="both",
        image_resolution=120,
    ).to(DEVICE)

    model.eval()

    embeddings = []

    with torch.no_grad():
        for idx, row in df.iterrows():
            s1_path = os.path.join(KENYA_ROOT, row["s1_tensor"])
            s2_path = os.path.join(KENYA_ROOT, row["s2_tensor"])

            s1 = torch.load(s1_path, map_location="cpu").float()
            s2 = torch.load(s2_path, map_location="cpu").float()

            s1, s2 = adapt_for_croma(s1, s2)

            s1 = s1.unsqueeze(0).to(DEVICE)
            s2 = s2.unsqueeze(0).to(DEVICE)

            outputs = model(
                SAR_images=s1,
                optical_images=s2,
            )

            joint_embedding = outputs["joint_GAP"].squeeze(0).cpu()
            embeddings.append(joint_embedding)

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(df)}")

    embeddings = torch.stack(embeddings)

    output_path = os.path.join(OUTPUT_DIR, "croma_joint_embeddings.pt")
    torch.save(embeddings, output_path)

    print("Done.")
    print("Saved:", output_path)
    print("Embeddings shape:", embeddings.shape)


if __name__ == "__main__":
    main()
