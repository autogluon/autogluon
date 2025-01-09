import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureLoss(nn.Module):
    """
    Structure Loss based on https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py
    The loss represent the weighted IoU loss and binary cross entropy (BCE) loss for the global restriction and local (pixel-level) restriction.

    References:
        [1] https://arxiv.org/abs/2006.11392
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if input.dim() == 3:
            input = input.unsqueeze(1)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(input, target, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        input = torch.sigmoid(input)
        inter = ((input * target) * weit).sum(dim=(2, 3))
        union = ((input + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()
