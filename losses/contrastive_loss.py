import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_s2, z_s1):
        batch_size = z_s2.shape[0]

        z_s2 = F.normalize(z_s2, dim=1)
        z_s1 = F.normalize(z_s1, dim=1)

        logits = torch.matmul(
            z_s2,
            z_s1.T
        ) / self.temperature

        labels = torch.arange(
            batch_size,
            device=z_s2.device
        )

        loss_s2 = F.cross_entropy(
            logits,
            labels
        )

        loss_s1 = F.cross_entropy(
            logits.T,
            labels
        )

        loss = (loss_s2 + loss_s1) / 2

        return loss