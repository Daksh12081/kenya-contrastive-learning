import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.paired_dataset import PairedDataset
from models.dual_encoder import DualEncoderContrastiveModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = DualEncoderContrastiveModel(
    embedding_dim=768,
    projection_dim=256
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "convnext_dual_encoder_contrastive.pth",
        map_location=DEVICE
    )
)

model.eval()

dataset = PairedDataset("data/patch_pairs.csv")

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

s2_embeddings = []
s1_embeddings = []

with torch.no_grad():
    for batch in dataloader:
        s2 = batch["s2"].float().to(DEVICE)
        s1 = batch["s1"].float().to(DEVICE)

        z_s2, z_s1 = model(s2, s1)

        s2_embeddings.append(z_s2.cpu())
        s1_embeddings.append(z_s1.cpu())

s2_embeddings = torch.cat(s2_embeddings, dim=0)
s1_embeddings = torch.cat(s1_embeddings, dim=0)

s2_embeddings = F.normalize(s2_embeddings, dim=1)
s1_embeddings = F.normalize(s1_embeddings, dim=1)

similarity_matrix = torch.matmul(s2_embeddings, s1_embeddings.T)

positive_scores = similarity_matrix.diag()
mask = ~torch.eye(len(dataset), dtype=torch.bool)
negative_scores = similarity_matrix[mask]

print("\n===== Similarity Statistics =====")
print("Mean Positive Similarity:", positive_scores.mean().item())
print("Mean Negative Similarity:", negative_scores.mean().item())

ranked_indices = torch.argsort(
    similarity_matrix,
    dim=1,
    descending=True
)

correct_indices = torch.arange(len(dataset))

top1_correct = ranked_indices[:, 0] == correct_indices

top5_correct = (
    ranked_indices[:, :5]
    == correct_indices.unsqueeze(1)
).any(dim=1)

top10_correct = (
    ranked_indices[:, :10]
    == correct_indices.unsqueeze(1)
).any(dim=1)

print("\n===== Retrieval Accuracy =====")
print("Top-1 :", top1_correct.float().mean().item())
print("Top-5 :", top5_correct.float().mean().item())
print("Top-10:", top10_correct.float().mean().item())
