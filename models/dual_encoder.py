import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny


class ConvNeXtEncoder(nn.Module):

    def __init__(self, in_channels: int, embedding_dim: int = 768):
        super().__init__()

        self.backbone = convnext_tiny(weights=None)

        original_first_conv = self.backbone.features[0][0]

        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias is not None,
        )

        self.backbone.classifier = nn.Identity()

        self.fc = nn.Linear(768, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x


class ProjectionHead(nn.Module):

    def __init__(self, embedding_dim: int = 768, projection_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, dim=1)
        return x


class DualEncoderContrastiveModel(nn.Module):

    def __init__(self, embedding_dim: int = 768, projection_dim: int = 256):
        super().__init__()

        self.s2_encoder = ConvNeXtEncoder(
            in_channels=14,
            embedding_dim=embedding_dim,
        )

        self.s1_encoder = ConvNeXtEncoder(
            in_channels=3,
            embedding_dim=embedding_dim,
        )

        self.s2_projection = ProjectionHead(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
        )

        self.s1_projection = ProjectionHead(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
        )

    def forward(self, s2: torch.Tensor, s1: torch.Tensor):
        s2_embedding = self.s2_encoder(s2)
        s1_embedding = self.s1_encoder(s1)

        z_s2 = self.s2_projection(s2_embedding)
        z_s1 = self.s1_projection(s1_embedding)

        return z_s2, z_s1