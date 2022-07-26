# Reference: https://github.com/lenscloth/RKD/blob/master/metric/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import pdist
from typing import Optional


class RKDDistance(nn.Module):
    def forward(self, student: Optional[torch.Tensor], teacher: Optional[torch.Tensor]):
        """
        Compute RKD Distance Loss
        Refer to https://github.com/lenscloth/RKD/blob/master/metric/loss.py
        Paper Link https://arxiv.org/abs/1904.05068

        Parameters
        ----------
        student
            Output feature of student model
        teacher
            Output feature of teacher model
        Returns
        -------
        The RKD Distance Loss between teacher and student
        """
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction="elementwise_mean")

        return loss


class RKDAngle(nn.Module):
    def forward(self, student: Optional[torch.Tensor], teacher: Optional[torch.Tensor]):
        """
        Compute RKD Angle Loss
        Refer to https://github.com/lenscloth/RKD/blob/master/metric/loss.py
        Paper Link https://arxiv.org/abs/1904.05068

        Parameters
        ----------
        student
            Output feature of student model
        teacher
            Output feature of teacher model
        Returns
        -------
        The RKD Angle Loss between teacher and student
        """

        with torch.no_grad():
            td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction="elementwise_mean")

        return loss
