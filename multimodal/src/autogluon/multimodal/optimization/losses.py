from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


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


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class MultiNegativesSoftmaxLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            # rank=0,
            # world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        # self.rank = rank
        # self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, rank=0, world_size=1):
        device = image_features.device
        if world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, rank, world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if world_size > 1 and self.local_loss:
                labels = labels + num_logits * rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss
