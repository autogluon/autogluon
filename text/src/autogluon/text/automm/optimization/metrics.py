import torch
from torch.nn import functional as F
from torchmetrics import Metric


class CrossEntropy(Metric):
    """
    A torchmetrics.Metric to compute cross entropy, which corresponds to sklearn's log_loss
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    """

    def __init__(self, dist_sync_on_step=False):
        """
        Each state variable should be called using self.add_state(...).

        Parameters
        ----------
        dist_sync_on_step
            Synchronize metric state across processes at each forward() before returning the value at the step.
            Refer to https://torchmetrics.readthedocs.io/en/stable/pages/implement.html#torchmetrics.Metric
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the state given inputs of one batch.
        It needs to accumulate metric scores over multiple batches.

        Parameters
        ----------
        preds
            Logits output of a model.
        target
            Ground-truth labels.
        """
        self.loss += F.cross_entropy(
            input=preds,
            target=target,
            reduction="sum",
        )
        self.total += len(target)

    def compute(self):
        """
        Computes the average cross entropy loss over all samples.

        Returns
        -------
        Average loss.
        """
        return self.loss.float() / self.total
