import logging
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

from ..constants import AUTOMM

logger = logging.getLogger(AUTOMM)


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
    image_features, text_features, local_loss=False, gather_with_grad=False, rank=0, world_size=1, use_horovod=False
):
    """
    Gather features across GPUs.

    Parameters
    ----------
    image_features
        image features of the current process.
    text_features
        text features of the current process.
    local_loss
        If False, make sure the features on the current GPU have gradients.
    gather_with_grad
        Whether to gather all features with gradients enabled.
    rank
        Rank of the current process (it should be a number between 0 and world_size-1).
    world_size
        Number of processes participating in the job.
    use_horovod
        Whether to use horovod.

    Returns
    -------
    Gathered image and text features from all processes.
    """
    assert has_distributed, "torch.distributed did not import correctly, please use a PyTorch version with support."
    if use_horovod:
        assert hvd is not None, "Please install horovod"
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
    """
    This loss expects as input a batch consisting of pairs (a_1, p_1), (a_2, p_2)…, (a_n, p_n) where
        we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
        For each a_i, it uses all other p_j as negative samples, i.e., for a_i,
        we have 1 positive example (p_i) and n-1 negative examples (p_j).
        It then minimizes the negative log-likehood for softmax normalized scores.
        It can also support gather negatives across processes.
    """

    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        use_horovod=False,
    ):
        """
        Parameters
        ----------
        local_loss
            Whether to compute the loss only for the current process's samples.
        gather_with_grad
            Whether to gather all features with gradients enabled.
        cache_labels
            Whether to cache labels for loss in next iterations.
        use_horovod
            Whether to use horovod.
        """
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, features_a, features_b, logit_scale, rank=0, world_size=1):
        device = features_a.device
        if world_size > 1:
            all_features_a, all_features_b = gather_features(
                features_a, features_b, self.local_loss, self.gather_with_grad, rank, world_size, self.use_horovod
            )

            if self.local_loss:
                logits_per_a = logit_scale * features_a @ all_features_b.T
                logits_per_b = logit_scale * features_b @ all_features_a.T
            else:
                logits_per_a = logit_scale * all_features_a @ all_features_b.T
                logits_per_b = logits_per_a.T
        else:
            logits_per_a = logit_scale * features_a @ features_b.T
            logits_per_b = logit_scale * features_b @ features_a.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_a.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if world_size > 1 and self.local_loss:
                labels = labels + num_logits * rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (F.cross_entropy(logits_per_a, labels) + F.cross_entropy(logits_per_b, labels)) / 2
        return total_loss


class DistillLoss(nn.Module):
    """
    Distillation loss to encourage two predictions to be close.
    In self-distillation, we encourage the predictions from the original sample and the corrupted sample to be close to each other.
    """
    def __init__(self, temperature=1):
        """
        Parameters
        ----------
        temperature (float, optional):
            scaling factor of the similarity metric.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Parameters
        ----------
        z_i (torch.tensor)
            anchor batch of samples
        z_j (torch.tensor)
            positive batch of samples
        Returns:
            float: loss
        """
        z_i, z_j = z_i.flatten(1), z_j.flatten(1)

        if z_i.size(1) == 1:
            return F.mse_loss(z_i, z_j)
        else:
            z_i, z_j = z_i / self.temperature, z_j / self.temperature
            z_i = F.softmax(z_i, dim=-1)
            return F.cross_entropy(z_j, z_i)


class NTXent(nn.Module):
    """
    The contrastive loss as used in the SCARF paper (https://arxiv.org/abs/2106.15147)
    NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
    Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation
    """
    def __init__(self, temperature=0.1):
        """
        Parameters
        ----------
        temperature (float, optional):
            scaling factor of the similarity metric.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch
        Parameters
        ----------
        z_i (torch.tensor)
            anchor batch of samples
        z_j (torch.tensor)
            positive batch of samples
        Returns:
            float: loss
        """
        batch_size = z_i.size(0)
        z_i, z_j = z_i.flatten(1), z_j.flatten(1)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        numerator = torch.exp(positives / self.temperature)

        denominator = torch.exp(similarity / self.temperature)  # * mask
        all_losses = -torch.log(numerator / torch.mean(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss


class ReconstructionLoss:
    """
    The reconstruction loss used to measure how well a sample can be reconstructed from a corrupted embedding.
    For tables,
        reconstruction loss is cross_entropy() for categorical columns reconstruction;
        reconstruction loss is mse_loss() for numerical columns reconstruction;
    """
    def __init__(self, model):
        """
        Parameters
        ----------
        model:
            AutoMM model with categorical_key, numerical_key, or both (FT_transformer).
        """
        self.model = model

    def __call__(self, batch, batch_):
        """
        Parameters
        ----------
        batch
            original batch
        batch_
            reconstructed batch
        Returns:
            float: loss
        """
        batch_ = batch_[self.model.prefix]["logits"]
        loss = 0
        for permodel in self.model.model:
            if hasattr(permodel, "categorical_key"):
                for y_, y in zip(batch_["cat_out"], batch[permodel.categorical_key]):
                    loss += F.cross_entropy(y_, y.long())
            if hasattr(permodel, "numerical_key"):
                y = batch[permodel.numerical_key]
                y_ = batch_["num_out"]
                loss += F.mse_loss(y_, y)
        return loss