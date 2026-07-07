import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from xjepa.dataset import TensorPairDataset
from xjepa.models.xjepa import XJEPA

CSV_PATH = "outputs/tensors/tensor_pairs.csv"
MODEL_PATH = "xjepa_model.pth"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_retrieval_metrics(similarity_matrix, ks=[1, 5, 10]):
    results = {}
    n = similarity_matrix.shape[0]

    for k in ks:
        correct = 0

        for i in range(n):
            topk = torch.topk(similarity_matrix[i], k=k).indices
            if i in topk:
                correct += 1

        results[f"Top-{k}"] = 100.0 * correct / n

    return results


print("Loading dataset...")
dataset = TensorPairDataset(CSV_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Loading model...")
model = XJEPA().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_s2_embeddings = []
all_s1_embeddings = []

print("Generating embeddings...")
with torch.no_grad():
    for batch in dataloader:
        s2 = batch["s2"].to(DEVICE)
        s1 = batch["s1"].to(DEVICE)

        z_s2 = model.predictor(
            model.s2_encoder(s2)
        )

        z_s1 = model.s1_target_encoder(s1)

        z_s2 = F.normalize(z_s2, dim=1)
        z_s1 = F.normalize(z_s1, dim=1)

        all_s2_embeddings.append(z_s2.cpu())
        all_s1_embeddings.append(z_s1.cpu())

s2_embeddings = torch.cat(all_s2_embeddings, dim=0)
s1_embeddings = torch.cat(all_s1_embeddings, dim=0)

print(f"S2 embeddings: {s2_embeddings.shape}")
print(f"S1 embeddings: {s1_embeddings.shape}")

print("Computing similarity matrix...")
similarity_matrix = torch.matmul(s2_embeddings, s1_embeddings.T)

metrics = compute_retrieval_metrics(similarity_matrix)

print("\n===== Retrieval Results =====")
for metric, value in metrics.items():
    print(f"{metric} Accuracy: {value:.2f}%")