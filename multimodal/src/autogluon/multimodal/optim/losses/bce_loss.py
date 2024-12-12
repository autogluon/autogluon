import torch
import torch.nn as nn


class BBCEWithLogitLoss(nn.Module):
    """
    Balanced BCEWithLogitLoss based on https://github.com/NiFangBaAGe/Explicit-Visual-Prompt/blob/latest_branch/models/segformer.py
    """

    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if input.dim() == 3:
            input = input.unsqueeze(1)
        eps = 1e-10
        count_pos = torch.sum(target) + eps
        count_neg = torch.sum(1.0 - target)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(input, target)

        return loss
