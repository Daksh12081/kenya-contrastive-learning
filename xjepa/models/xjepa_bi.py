import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from xjepa.models.encoder import ConvNeXtEncoder
from xjepa.models.predictor import Predictor


class BidirectionalXJEPA(nn.Module):
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

        self.s2_target_encoder = copy.deepcopy(self.s2_encoder)
        self.s1_target_encoder = copy.deepcopy(self.s1_encoder)

        for param in self.s2_target_encoder.parameters():
            param.requires_grad = False

        for param in self.s1_target_encoder.parameters():
            param.requires_grad = False

        self.predictor_s2_to_s1 = Predictor(
            embedding_dim=embedding_dim,
            hidden_dim=1024,
        )

        self.predictor_s1_to_s2 = Predictor(
            embedding_dim=embedding_dim,
            hidden_dim=1024,
        )

        self.ema_momentum = ema_momentum

    def forward(self, s2, s1):
        z_s2 = self.s2_encoder(s2)
        pred_s1 = self.predictor_s2_to_s1(z_s2)

        z_s1 = self.s1_encoder(s1)
        pred_s2 = self.predictor_s1_to_s2(z_s1)

        with torch.no_grad():
            target_s1 = self.s1_target_encoder(s1)
            target_s2 = self.s2_target_encoder(s2)

        loss_s2_to_s1 = F.mse_loss(pred_s1, target_s1)
        loss_s1_to_s2 = F.mse_loss(pred_s2, target_s2)

        total_loss = loss_s2_to_s1 + loss_s1_to_s2

        metrics = {
            "loss_s2_to_s1": loss_s2_to_s1.item(),
            "loss_s1_to_s2": loss_s1_to_s2.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics

    @torch.no_grad()
    def update_target_encoders(self):
        for online_param, target_param in zip(
            self.s2_encoder.parameters(),
            self.s2_target_encoder.parameters(),
        ):
            target_param.data = (
                self.ema_momentum * target_param.data
                + (1.0 - self.ema_momentum) * online_param.data
            )

        for online_param, target_param in zip(
            self.s1_encoder.parameters(),
            self.s1_target_encoder.parameters(),
        ):
            target_param.data = (
                self.ema_momentum * target_param.data
                + (1.0 - self.ema_momentum) * online_param.data
            )


if __name__ == "__main__":
    model = BidirectionalXJEPA()

    s2 = torch.randn(2, 14, 256, 256)
    s1 = torch.randn(2, 3, 256, 256)

    loss, metrics = model(s2, s1)

    print("Loss:", loss.item())
    print("Metrics:", metrics)