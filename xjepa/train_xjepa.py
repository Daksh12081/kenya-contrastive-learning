import torch
from torch.utils.data import DataLoader

from xjepa.dataset import TensorPairDataset
from xjepa.models.xjepa import XJEPA


CSV_PATH = "outputs/tensors/tensor_pairs.csv"
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    dataset = TensorPairDataset(CSV_PATH)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = XJEPA().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Device: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            s2 = batch["s2"].to(DEVICE)
            s1 = batch["s1"].to(DEVICE)

            optimizer.zero_grad()

            loss, _, _ = model(s2, s1)

            loss.backward()
            optimizer.step()

            model.update_target_encoder()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Batch [{batch_idx+1}/{len(dataloader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        epoch_loss = running_loss / len(dataloader)

        print(
            f"Epoch {epoch+1} Complete | "
            f"Average Loss: {epoch_loss:.6f}"
        )

        torch.save(
            model.state_dict(),
            f"xjepa_epoch_{epoch+1}.pth"
        )

        print(
            f"Checkpoint saved: xjepa_epoch_{epoch+1}.pth"
        )

    torch.save(model.state_dict(), "xjepa_model.pth")
    print("Model saved: xjepa_model.pth")


if __name__ == "__main__":
    train()