import logging
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .utils import (
    get_optimizer,
    get_lr_scheduler,
    apply_two_stages_lr,
    apply_layerwise_lr_decay,
    apply_single_lr,
)
from ..constants import LOGITS, WEIGHT
from typing import Union, Optional, List, Dict
import torchmetrics
from torch.nn.modules.loss import _Loss

logger = logging.getLogger(__name__)


class LitModule(pl.LightningModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            model: nn.Module,
            optim_type: Optional[str] = None,
            lr_choice: Optional[str] = None,
            lr_schedule: Optional[str] = None,
            lr: Optional[float] = None,
            lr_decay: Optional[float] = None,
            end_lr: Optional[Union[float, int]] = None,
            lr_mult: Optional[Union[float, int]] = None,
            weight_decay: Optional[float] = None,
            warmup_steps: Optional[int] = None,
            loss_func: Optional[_Loss] = None,
            val_metric: Optional[torchmetrics.Metric] = None,
            test_metric: Optional[torchmetrics.Metric] = None,
    ):
        """
        Parameters
        ----------
        model
            A Pytorch model
        optim_type
            Optimizer type. We now support:
            - adamw
            - adam
            - sgd
        lr_choice
            How to set each layer's learning rate. If not specified, the default is a single
            learnng rate for all layers. Otherwise, we now support two choices:
            - two_stages
                The layers in the pretrained models have a small learning rate (lr * lr_mult),
                while the newly added head layers use the provided learning rate.
            - layerwise_decay
                The layers have decreasing learning rate from the output end to the input end.
                The intuition is that later layers are more task-related, hence larger learning rates.
        lr_schedule
            Learning rate schedule. We now support:
            - cosine_decay
                Linear warmup followed by cosine decay
            - polynomial_decay
                Linear warmup followed by polynomial decay
        lr
            Learning rate.
        lr_decay
            The learning rate decay factor (0, 1). It is used only when lr_choice is "layerwise_decay".
        end_lr
            The final learning rate after decay.
        lr_mult
            The learning rate multiplier (0, 1). It is used only when lr_choice is "two_stages".
        weight_decay
            The weight decay to regularize layer weights' l2 norm.
        warmup_steps
            How many steps to warmup learning rate. If a float (0, 1), it would represent the
            percentage of steps over all the training steps. The actual number is calculated as
            "int(warmup_steps * max_steps)". If an integer, it would be the exact step number.
        loss_func
            A Pytorch loss module, e.g., nn.CrossEntropyLoss().
        val_metric
            A torchmetrics module used in the validation stage, e.g., torchmetrics.Accuracy().
        test_metric
            A torchmetrics module used in the test stage, e.g., torchmetrics.Accuracy().
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "val_metric", "test_metric", "loss_func"])
        self.model = model
        self.val_metric = val_metric
        if val_metric is not None:
            self.val_metric_name = f"val_{val_metric.__class__.__name__}"
        self.loss_func = loss_func

    def _compute_loss(
            self,
            output: Union[Dict, List[Dict]],
            label: torch.Tensor,
    ):
        if isinstance(output, dict):
            output = [output]

        loss = 0
        for per_output in output:
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            loss += self.loss_func(per_output[LOGITS].squeeze(dim=1), label) * weight
        return loss

    def _compute_metric(
            self,
            output: Union[Dict, List[Dict]],
            label: torch.Tensor,
    ):
        if isinstance(output, dict):
            logits = output[LOGITS]
        else:
            # use only the last logits, which is the fusion logits
            logits = output[-1][LOGITS]

        if isinstance(self.val_metric, torchmetrics.AUROC):
            prob = F.softmax(logits.float(), dim=1)
            return self.val_metric(preds=prob[:, 1], target=label)  # only for binary classification
        else:
            return self.val_metric(logits.squeeze(dim=1), label)

    def _shared_step(
            self,
            batch: dict,
    ):
        output = self.model(batch)
        label = batch[self.model.label_key]
        loss = self._compute_loss(output=output, label=label)
        return output, loss

    def training_step(self, batch, batch_idx):
        """
        Per training step. This function is registered by pl.LightningModule.
        Refer to https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data, including both input data and
            ground-truth labels. The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix. The
            ground-truth labels are used here to compute the training loss.
        batch_idx
            Index of mini-batch.

        Returns
        -------
        Average loss of the mini-batch data.
        """
        output, loss = self._shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Per validation step. This function is registered by pl.LightningModule.
        Refer to https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validation

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data, including both input data and
            ground-truth labels. The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix. The
            ground-truth labels are used here to compute the validation loss and metric.
            The validation metric is used for top k model selection and early stopping.
        batch_idx
            Index of mini-batch.
        """
        output, loss = self._shared_step(batch)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        self.log(
            self.val_metric_name,
            self._compute_metric(output=output, label=batch[self.model.label_key]),
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Per prediction step. This function is registered by pl.LightningModule.
        Refer to https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#prediction-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data.
            The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix.
            Ground-truth labels are not needed for prediction.
        batch_idx
            Index of mini-batch.
        dataloader_idx
            Index of dataloader.
        Returns
        -------
        A dictionary with the mini-batch's logits and features.
        """
        output = self.model(batch)
        if isinstance(output, dict):
            ret = output
        else:
            ret = output[-1]
        return ret

    def configure_optimizers(self):
        """
        Configure optimizer. This function is registered by pl.LightningModule.
        Refer to https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        Returns
        -------
        [optimizer]
            Optimizer.
        [sched]
            Learning rate scheduler.
        """
        if self.hparams.lr_choice == "two_stages":
            logger.debug("applying 2-stage learning rate...")
            grouped_parameters = apply_two_stages_lr(
                model=self.model,
                lr=self.hparams.lr,
                lr_mult=self.hparams.lr_mult,
                weight_decay=self.hparams.weight_decay,
                return_params=True,
            )
        elif self.hparams.lr_choice == "layerwise_decay":
            logger.debug("applying layerwise learning rate decay...")
            grouped_parameters = apply_layerwise_lr_decay(
                model=self.model,
                lr=self.hparams.lr,
                lr_decay=self.hparams.lr_decay,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            logger.debug("applying single learning rate...")
            grouped_parameters = apply_single_lr(
                model=self.model,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

        optimizer = get_optimizer(
            optim_type=self.hparams.optim_type,
            optimizer_grouped_parameters=grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        logger.debug(f"trainer.max_steps: {self.trainer.max_steps}")
        if self.trainer.max_steps is None or -1:
            max_steps = (
                    len(self.trainer.datamodule.train_dataloader())
                    * self.trainer.max_epochs
                    // self.trainer.accumulate_grad_batches
            )
            logger.debug(f"len(trainer.datamodule.train_dataloader()): "
                  f"{len(self.trainer.datamodule.train_dataloader())}")
            logger.debug(f"trainer.max_epochs: {self.trainer.max_epochs}")
            logger.debug(f"trainer.accumulate_grad_batches: {self.trainer.accumulate_grad_batches}")
        else:
            max_steps = self.trainer.max_steps

        logger.debug(f"max steps: {max_steps}")

        warmup_steps = self.hparams.warmup_steps
        if isinstance(warmup_steps, float):
            warmup_steps = int(max_steps * warmup_steps)

        logger.debug(f"warmup steps: {warmup_steps}")
        logger.debug(f"lr_schedule: {self.hparams.lr_schedule}")
        scheduler = get_lr_scheduler(optimizer=optimizer,
                                     num_max_steps=max_steps,
                                     num_warmup_steps=warmup_steps,
                                     lr_schedule=self.hparams.lr_schedule,
                                     end_lr=self.hparams.end_lr)

        sched = {"scheduler": scheduler, "interval": "step"}
        logger.debug("done configuring optimizer and scheduler")
        return [optimizer], [sched]
