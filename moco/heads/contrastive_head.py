import torch
from torch._C import dtype
import torch.nn as nn


class ContrastiveHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-paramter that
        controls the concentration of the distribution.
        default: 0.1.
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
        """
        N = pos.size(0)
        logits = torch.cat([pos, neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros((N,), dtype=torch.long).cuda()
        losses = dict()
        losses["loss_contra"] = self.criterion(logits, labels)
        return losses
