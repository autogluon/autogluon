import torch
import torch.nn as nn
import torch.nn.functional as F

from ...constants import BINARY, REGRESSION


class LemdaLoss(nn.Module):
    def __init__(self, mse_weight, kld_weight, consist_weight, consist_threshold, problem_type):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.mse_weight = mse_weight
        self.kld_weight = kld_weight
        self.consist_weight = consist_weight
        self.consist_threshold = consist_threshold
        self.problem_type = problem_type

    def consist_loss(self, p_logits, q_logits):
        p = F.softmax(p_logits, dim=1)
        logp = F.log_softmax(p_logits, dim=1)
        logq = F.log_softmax(q_logits, dim=1)
        loss = torch.sum(p * (logp - logq), dim=-1)
        q = F.softmax(q_logits, dim=1)
        q_largest = torch.max(q, dim=1)[0]
        loss_mask = torch.gt(q_largest, self.consist_threshold).float()
        loss = loss * loss_mask
        return torch.mean(loss)

    def forward(self, pre_aug, post_aug, vae_mean, vae_var, ori_logits, aug_logits):
        mse_loss = self.mse_loss(pre_aug, post_aug) * self.mse_weight
        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.mean(1 + vae_var - vae_mean.pow(2) - vae_var.exp()) * self.kld_weight
        if self.problem_type in [REGRESSION, BINARY]:
            consist_loss = self.mse_loss(ori_logits, aug_logits) * self.consist_weight
        else:
            consist_loss = self.consist_loss(ori_logits, aug_logits) * self.consist_weight

        return mse_loss + kld_loss + consist_loss
