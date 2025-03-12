import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class Coverage(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("pos_probs", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.tp_threshold = 0.97
        self.tn_threshold = 0.99
        higher_is_better = True

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.dim() == 1
        assert target.dim() == 1
        self.pos_probs.append(preds)
        self.targets.append(target)

    def compute(self):
        # parse inputs
        pos_probs = dim_zero_cat(self.pos_probs)
        targets = dim_zero_cat(self.targets)
        y_pos = targets[torch.where(pos_probs >= self.tp_threshold)]
        y_neg = targets[torch.where(pos_probs <= 1 - self.tn_threshold)]
        tp = sum(y_pos == 1)
        tn = sum(y_neg == 0)
        if len(y_pos) == 0 or len(targets) == 0:
            tp_precision, tp_coverage = 0, 0
        else:
            tp_precision, tp_coverage = tp / len(y_pos), len(y_pos) / len(targets)
            if tp_precision < self.tp_threshold:
                tp_coverage = 0
        if len(y_neg) == 0 or len(targets) == 0:
            tn_precision, tn_coverage = 0, 0
        else:
            tn_precision, tn_coverage = tn / len(y_neg), len(y_neg) / len(targets)
            if tn_precision < self.tn_threshold:
                tn_coverage = 0

        return torch.tensor(tp_coverage) + torch.tensor(tn_coverage)
