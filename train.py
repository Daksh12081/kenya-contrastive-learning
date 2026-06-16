import torch
from torch.utils.data import DataLoader

from datasets.paired_dataset import PairedDataset
from models.dual_encoder import DualEncoderContrastiveModel
from losses.contrastive_loss import InfoNCELoss

CSV_PATH = "data/patch_pairs.csv"
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = PairedDataset(CSV_PATH)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print(f"Dataset Size: {len(dataset)}")
print(f"Device: {DEVICE}")

model = DualEncoderContrastiveModel(
    embedding_dim=768,
    projection_dim=256
).to(DEVICE)

criterion = InfoNCELoss(
    temperature=0.07
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch in dataloader:

        s2 = batch["s2"].float().to(DEVICE)
        s1 = batch["s1"].float().to(DEVICE)

        optimizer.zero_grad()

        z_s2, z_s1 = model(s2, s1)

        loss = criterion(z_s2, z_s1)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "convnext_dual_encoder_contrastive.pth"
)

print("Training Complete.")
print("Model saved as convnext_dual_encoder_contrastive.pth")