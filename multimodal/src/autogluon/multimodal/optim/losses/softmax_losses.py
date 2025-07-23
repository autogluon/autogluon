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
            all_features_a, all_features_b = self.gather_features(
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

    @staticmethod
    def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
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
        assert (
            has_distributed
        ), "torch.distributed did not import correctly, please use a PyTorch version with support."
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
