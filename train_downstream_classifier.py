import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDINGS_PATH = "outputs/sen12ms_embeddings/croma_embeddings.pt"
LABELS_PATH = "outputs/sen12ms_embeddings/labels.pt"
OUTPUT_DIR = "outputs/classifier_checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DownstreamClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(embeddings)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, matrix


def main():
    set_seed(SEED)

    embeddings = torch.load(EMBEDDINGS_PATH, map_location="cpu").float()
    labels = torch.load(LABELS_PATH, map_location="cpu").long()

    print("Embeddings:", embeddings.shape)
    print("Labels:", labels.shape)
    print("Class counts:", torch.bincount(labels))
    print("Classes: 0=Forest, 1=Cropland, 2=Waterbody, 3=NonForest")
    print("Device:", DEVICE)

    indices = np.arange(len(labels))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=labels.numpy(),
    )

    train_dataset = TensorDataset(embeddings[train_idx], labels[train_idx])
    test_dataset = TensorDataset(embeddings[test_idx], labels[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DownstreamClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_embeddings)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        accuracy, precision, recall, f1, _ = evaluate(model, test_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {accuracy:.4f} | "
            f"F1: {f1:.4f}"
        )

    accuracy, precision, recall, f1, matrix = evaluate(model, test_loader)

    print("\n===== Final Evaluation =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(matrix)

    output_path = os.path.join(OUTPUT_DIR, "sen12ms_4class_classifier.pth")
    torch.save(model.state_dict(), output_path)
    print("Saved classifier:", output_path)


if __name__ == "__main__":
    main()