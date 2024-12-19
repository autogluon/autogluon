from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss based on https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: Optional[float] = 2.0,
        reduction: Optional[str] = "mean",
        eps: Optional[float] = 1e-6,
    ):
        """

        Parameters
        ----------
        alpha
            weighting factor for each class. Should be of shape (num_classes)
        gamma
            the focal parameter for calculating weights on easy/hard samples
        reduction
            the reduction to apply to the final loss output. Default: "mean". Options:
                "mean", "sum"
        eps
            epsilon for numerical stability
        """
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        if alpha is not None:
            if isinstance(alpha, str):  # handles Ray Tune HPO sampled hyperparameter
                try:
                    numbers = alpha.strip("()").split(",")
                    alpha = [float(num) for num in numbers]
                except:
                    raise ValueError(f"{type(alpha)} {alpha} is not in a supported format.")
            alpha = torch.tensor(alpha)
        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if not torch.is_tensor(input):
            raise TypeError("input type is not a torch.Tensor. Got {}".format(type(input)))
        if input.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            num_class = input.shape[1]
            input = input.permute(0, *range(2, input.ndim), 1).reshape(-1, num_class)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            target = target.view(-1)

        pt = F.softmax(input, dim=-1)

        # -alpha_t * log(pt) term
        log_p = torch.log_softmax(input, dim=-1)
        ce = self.nll_loss(log_p, target)

        # (1 - pt)^gamma term
        all_rows = torch.arange(input.shape[0])
        pt = pt[all_rows, target]
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()

        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
