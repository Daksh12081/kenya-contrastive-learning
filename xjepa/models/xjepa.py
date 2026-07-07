import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from xjepa.models.encoder import ConvNeXtEncoder
from xjepa.models.predictor import Predictor


class XJEPA(nn.Module):
    def __init__(
        self,
        s2_channels=14,
        s1_channels=3,
        model_name="convnext_tiny",
        embedding_dim=768,
        ema_momentum=0.996,
    ):
        super().__init__()

        self.s2_encoder = ConvNeXtEncoder(
            in_channels=s2_channels,
            model_name=model_name,
            embedding_dim=embedding_dim,
        )

        self.s1_encoder = ConvNeXtEncoder(
            in_channels=s1_channels,
            model_name=model_name,
            embedding_dim=embedding_dim,
        )

        self.s1_target_encoder = copy.deepcopy(self.s1_encoder)

        for param in self.s1_target_encoder.parameters():
            param.requires_grad = False

        self.predictor = Predictor(
            embedding_dim=embedding_dim,
            hidden_dim=1024,
        )

        self.ema_momentum = ema_momentum

    def forward(self, s2, s1):
        z_s2 = self.s2_encoder(s2)
        pred_s1 = self.predictor(z_s2)

        with torch.no_grad():
            target_s1 = self.s1_target_encoder(s1)

        loss = F.mse_loss(pred_s1, target_s1)

        return loss, pred_s1, target_s1

    @torch.no_grad()
    def update_target_encoder(self):
        for online_param, target_param in zip(
            self.s1_encoder.parameters(),
            self.s1_target_encoder.parameters(),
        ):
            target_param.data = (
                self.ema_momentum * target_param.data
                + (1.0 - self.ema_momentum) * online_param.data
            )


if __name__ == "__main__":
    model = XJEPA()

    s2 = torch.randn(2, 14, 256, 256)
    s1 = torch.randn(2, 3, 256, 256)

    loss, pred_s1, target_s1 = model(s2, s1)

    print("Loss:", loss.item())
    print("Predicted S1 embedding shape:", pred_s1.shape)
    print("Target S1 embedding shape:", target_s1.shape)