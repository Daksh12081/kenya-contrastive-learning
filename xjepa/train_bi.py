import torch
from torch.utils.data import DataLoader

from xjepa.dataset import TensorPairDataset
from xjepa.models.xjepa_bi import BidirectionalXJEPA


CSV_PATH = "outputs/tensors/tensor_pairs.csv"
BATCH_SIZE = 2
EPOCHS = 1
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

    model = BidirectionalXJEPA().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Device: {DEVICE}")
    print("Training Bidirectional X-JEPA")

    for epoch in range(EPOCHS):
        model.train()

        running_total_loss = 0.0
        running_s2_to_s1 = 0.0
        running_s1_to_s2 = 0.0

        for batch_idx, batch in enumerate(dataloader):

            s2 = batch["s2"].to(DEVICE)
            s1 = batch["s1"].to(DEVICE)

            optimizer.zero_grad()

            loss, metrics = model(s2, s1)

            loss.backward()
            optimizer.step()

            model.update_target_encoders()

            running_total_loss += metrics["total_loss"]
            running_s2_to_s1 += metrics["loss_s2_to_s1"]
            running_s1_to_s2 += metrics["loss_s1_to_s2"]

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Batch [{batch_idx+1}/{len(dataloader)}] "
                    f"Total Loss: {metrics['total_loss']:.6f} | "
                    f"S2→S1: {metrics['loss_s2_to_s1']:.6f} | "
                    f"S1→S2: {metrics['loss_s1_to_s2']:.6f}"
                )

        avg_total_loss = running_total_loss / len(dataloader)
        avg_s2_to_s1 = running_s2_to_s1 / len(dataloader)
        avg_s1_to_s2 = running_s1_to_s2 / len(dataloader)

        print(
            f"Epoch {epoch+1} Complete | "
            f"Avg Total Loss: {avg_total_loss:.6f} | "
            f"Avg S2→S1: {avg_s2_to_s1:.6f} | "
            f"Avg S1→S2: {avg_s1_to_s2:.6f}"
        )

        torch.save(
            model.state_dict(),
            f"xjepa_bi_epoch_{epoch+1}.pth"
        )

        print(
            f"Checkpoint saved: xjepa_bi_epoch_{epoch+1}.pth"
        )

    torch.save(
        model.state_dict(),
        "xjepa_bi_model.pth"
    )

    print("Model saved: xjepa_bi_model.pth")


if __name__ == "__main__":
    train()