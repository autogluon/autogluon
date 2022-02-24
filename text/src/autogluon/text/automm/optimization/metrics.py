import torch
from torch.nn import functional as F
from torchmetrics import Metric


class CrossEntropy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        self.loss += F.cross_entropy(
            input=preds,
            target=target,
            reduction="sum",
        )
        self.total += len(target)

    def compute(self):
        return self.loss.float() / self.total
