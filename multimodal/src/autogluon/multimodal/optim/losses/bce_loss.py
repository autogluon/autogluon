import torch
import torch.nn as nn

DEFAULT_IGNORE_LABEL_VALUE = -1

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


class BCEWithLogitsLossIgnoreLabel(nn.BCEWithLogitsLoss):
    def __init__(self, ignore_label=DEFAULT_IGNORE_LABEL_VALUE, weight=None, reduction='mean', pos_weight=None):
        # Initialize with reduction='none' since we'll handle reduction manually
        super().__init__(weight=weight, reduction='none', pos_weight=pos_weight)
        self.ignore_label = ignore_label
        self.requested_reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute element-wise loss without reduction
        unreduced_loss = super().forward(input, target)

        if self.ignore_label_value is not None:
            # Create mask for valid entries (where target != ignore_label_value)
            valid_mask = (target != self.ignore_label)

            # Apply mask to unreduced loss
            masked_loss = unreduced_loss * valid_mask.float()

            # Apply reduction
            if self.requested_reduction == 'mean':
                # Compute mean over non-ignored elements
                num_valid = valid_mask.sum()
                return masked_loss.sum() / (num_valid if num_valid > 0 else 1)
            elif self.requested_reduction == 'sum':
                return masked_loss.sum()
            else:  # 'none'
                return masked_loss
        else:
            # If no ignore_label_value is specified, apply reduction directly
            if self.requested_reduction == 'mean':
                return unreduced_loss.mean()
            elif self.requested_reduction == 'sum':
                return unreduced_loss.sum()
            else:  # 'none'
                return unreduced_loss