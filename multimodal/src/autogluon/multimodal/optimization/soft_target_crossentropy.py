import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    The soft target CrossEntropy from timm.
    https://github.com/rwightman/pytorch-image-models/blob/e4360e6125bb0bb4279785810c8eb33b40af3ebd/timm/loss/cross_entropy.py
    It works under the mixup.
    It can calculate the crossentropy of input and label with one-hot.
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(input, dim=-1), dim=-1)
        return loss.mean()
