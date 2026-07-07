import timm
import torch.nn as nn


class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_channels, model_name="convnext_tiny", embedding_dim=768):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=in_channels,
            num_classes=0,
        )

        self.projector = nn.Linear(
            self.backbone.num_features,
            embedding_dim,
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projector(features)
        return embeddings


if __name__ == "__main__":
    import torch

    encoder = ConvNeXtEncoder(in_channels=14)

    x = torch.randn(2, 14, 256, 256)

    z = encoder(x)

    print("Embedding shape:", z.shape)