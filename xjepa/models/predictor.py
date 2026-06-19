import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=1024):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.predictor(x)


if __name__ == "__main__":
    import torch

    predictor = Predictor()

    x = torch.randn(2, 768)

    y = predictor(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)