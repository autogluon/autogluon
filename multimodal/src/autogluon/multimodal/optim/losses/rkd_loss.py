from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RKDLoss(nn.Module):
    """
    Compute RKD Distance Loss.
    Paper Refer to: Relational Knowledge Disitllation, CVPR2019. https://arxiv.org/abs/1904.05068
    Code Refer to: https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/RKD.py
    and https://github.com/lenscloth/RKD/blob/master/metric/loss.py
    """

    def __init__(self, distance_loss_weight: Optional[float] = 25.0, angle_loss_weight: Optional[float] = 50.0):
        """
        Parameters
        ----------
        distance_loss_weight
            Weight of RKD distance loss
        angle_loss_weight
            Weight of RKD angle loss
        Returns
        -------
        """
        super(RKDLoss, self).__init__()
        self.distance_loss_weight = distance_loss_weight
        self.angle_loss_weight = angle_loss_weight

    def forward(self, feature_student: Optional[torch.Tensor], feature_teacher: Optional[torch.Tensor]):
        """
        Parameters
        ----------
        feature_student
            Output feature of student model, shape: (N, D)
        feature_teacher
            Output feature of teacher model, shape: (N, D)
        Returns
        -------
        The RKD Loss between teacher and student
        """
        # RKD loss
        if self.distance_loss_weight > 0:
            with torch.no_grad():
                t_dist = self.pdist(feature_teacher, squared=False)
                mean_td = t_dist[t_dist > 0].mean()
                t_dist = t_dist / mean_td

            s_dist = self.pdist(feature_student, squared=False)
            mean_d = s_dist[s_dist > 0].mean()
            s_dist = s_dist / mean_d

            loss_distance = F.smooth_l1_loss(s_dist, t_dist)

        # RKD Angle loss
        if self.angle_loss_weight > 0:
            with torch.no_grad():
                td = feature_teacher.unsqueeze(0) - feature_teacher.unsqueeze(1)
                norm_td = F.normalize(td, p=2, dim=2)
                t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

            sd = feature_student.unsqueeze(0) - feature_student.unsqueeze(1)
            norm_sd = F.normalize(sd, p=2, dim=2)
            s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

            loss_angle = F.smooth_l1_loss(s_angle, t_angle)

        loss = ((self.distance_loss_weight * loss_distance) if self.distance_loss_weight > 0 else 0) + (
            (self.angle_loss_weight * loss_angle) if self.angle_loss_weight > 0 else 0
        )

        return loss

    @staticmethod
    def pdist(embeddings: Optional[torch.Tensor], squared: Optional[bool] = False, eps: Optional[float] = 1e-12):
        """
        Compute pairwise Euclidean distances between embeddings in n-dimensional space.

        Parameters
        ----------
        embeddings
            The embeddings to compute pairwise distance between. Shape: (N,D)
        squared
            If the result is square of Euclidean distance.
        eps
            Min value of each entry.

        Returns
        -------
        Pairwise Euclidean distances. Shape: (N,N)
        """
        e_square = embeddings.pow(2).sum(dim=1)
        prod = embeddings @ embeddings.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(embeddings)), range(len(embeddings))] = 0

        return res
