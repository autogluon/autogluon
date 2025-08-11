import einops
import torch

from ..._internal.config.config_pretrain import ConfigPretrain
from ..._internal.config.config_run import ConfigRun
from ..._internal.config.enums import LossName, Task


class CrossEntropyLossExtraBatch(torch.nn.Module):
    def __init__(self, label_smoothing: float):
        super().__init__()

        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, input, target):
        """
        Input has shape (batch_size, num_samples, num_classes)
        Target has shape (batch_size, num_samples)

        Compared to the original CrossEntropyLoss, accepts (batch_size, num_samples) as batch
        """

        input = einops.rearrange(input, "b s c -> (b s) c")
        target = einops.rearrange(target, "b s -> (b s)")

        return self.loss(input, target)


def get_loss(cfg: ConfigRun):
    if cfg.task == Task.REGRESSION and cfg.hyperparams["regression_loss"] == LossName.MSE:
        return torch.nn.MSELoss()
    elif cfg.task == Task.REGRESSION and cfg.hyperparams["regression_loss"] == LossName.MAE:
        return torch.nn.L1Loss()
    elif cfg.task == Task.REGRESSION and cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY:
        return CrossEntropyLossExtraBatch(cfg.hyperparams["label_smoothing"])
    elif cfg.task == Task.CLASSIFICATION:
        return CrossEntropyLossExtraBatch(cfg.hyperparams["label_smoothing"])
    else:
        raise ValueError(f"Unsupported task {cfg.task} and (regression) loss {cfg.hyperparams['regression_loss']}")


def get_loss_pretrain(cfg: ConfigPretrain):
    if cfg.data.task == Task.REGRESSION and cfg.optim.regression_loss == LossName.MSE:
        return torch.nn.MSELoss()
    elif cfg.data.task == Task.REGRESSION and cfg.optim.regression_loss == LossName.MAE:
        return torch.nn.L1Loss()
    elif cfg.data.task == Task.REGRESSION and cfg.optim.regression_loss == LossName.CROSS_ENTROPY:
        return CrossEntropyLossExtraBatch(cfg.optim.label_smoothing)
    elif cfg.data.task == Task.CLASSIFICATION:
        return CrossEntropyLossExtraBatch(cfg.optim.label_smoothing)
    else:
        raise ValueError(f"Unsupported task {cfg.data.task} and (regression) loss {cfg.optim.regression_loss}")
