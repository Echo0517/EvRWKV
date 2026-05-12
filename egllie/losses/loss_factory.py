from absl.logging import info
from torch.nn.modules.loss import _Loss
import torch.nn as nn

from .image_loss import (
    L1CharbonnierLoss,
    PerceptualLoss,
    SSIMLoss,
    MSSSIMLoss,
)


def get_single_loss(config):
    if config.NAME == "normal-light-reconstructed-loss": 
        return L1CharbonnierLoss()
    elif config.NAME == "normal-light-perceptual-loss":
        return PerceptualLoss()
    elif config.NAME == "ssim_loss":
        return SSIMLoss(window_size=9, size_average=True, value_range=1.0)
    elif config.NAME == "ms_ssim_loss":
        return MSSSIMLoss(value_range=1.0, size_average=True)
    else:
        raise ValueError(f"Unknown loss: {config.NAME}")


class MixedLoss(nn.Module):
    def __init__(self, configs):
        super(MixedLoss, self).__init__()
        self.loss_names = []
        self.weights = []
        self.criteria = []
        for item in configs:
            self.loss_names.append(item.NAME)
            self.weights.append(item.WEIGHT)
            self.criteria.append(get_single_loss(item))
        print(f"Initialized Mixed Loss with: {configs}")

    def forward(self, batch):
        total_loss = 0
        loss_details = []
        for name, weight, criterion in zip(self.loss_names, self.weights, self.criteria):
            loss_val = criterion(batch)
            loss_details.append((name, loss_val))
            total_loss += weight * loss_val
        return total_loss, loss_details
