import json
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
from ..constants import LABEL, OUTPUT, LOSS, LOGITS, WEIGHT
from typing import Union, Optional, List, Dict
import torchmetrics
from torch.nn.modules.loss import _Loss


class LitModule(pl.LightningModule):
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
        output, loss = self._shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output, loss = self._shared_step(batch)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        self.log(
            self.val_metric_name,
            self._compute_metric(output=output, label=batch[self.model.label_key]),
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self.model(batch)
        if isinstance(output, dict):
            ret = output
        else:
            ret = output[-1]
        return ret

    def configure_optimizers(self):
        if self.hparams.lr_choice == "two_stages":
            print("applying 2-stage learning rate...")
            grouped_parameters = apply_two_stages_lr(
                model=self.model,
                lr=self.hparams.lr,
                lr_mult=self.hparams.lr_mult,
                weight_decay=self.hparams.weight_decay,
                return_params=True,
            )
        elif self.hparams.lr_choice == "layerwise_decay":
            print("applying layerwise learning rate decay...")
            grouped_parameters = apply_layerwise_lr_decay(
                model=self.model,
                lr=self.hparams.lr,
                lr_decay=self.hparams.lr_decay,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            print("applying single learning rate...")
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

        print(f"trainer.max_steps: {self.trainer.max_steps}")
        if self.trainer.max_steps is None or -1:
            max_steps = (
                    len(self.trainer.datamodule.train_dataloader())
                    * self.trainer.max_epochs
                    // self.trainer.accumulate_grad_batches
            )
            print(f"len(trainer.datamodule.train_dataloader()): "
                  f"{len(self.trainer.datamodule.train_dataloader())}")
            print(f"trainer.max_epochs: {self.trainer.max_epochs}")
            print(f"trainer.accumulate_grad_batches: {self.trainer.accumulate_grad_batches}")
        else:
            max_steps = self.trainer.max_steps

        print(f"max steps: {max_steps}")

        warmup_steps = self.hparams.warmup_steps
        if isinstance(warmup_steps, float):
            warmup_steps = int(max_steps * warmup_steps)

        print(f"warmup steps: {warmup_steps}")
        print(f"lr_schedule: {self.hparams.lr_schedule}")
        scheduler = get_lr_scheduler(optimizer=optimizer,
                                     num_max_steps=max_steps,
                                     num_warmup_steps=warmup_steps,
                                     lr_schedule=self.hparams.lr_schedule,
                                     end_lr=self.hparams.end_lr)

        sched = {"scheduler": scheduler, "interval": "step"}
        print("done configuring optimizer and scheduler")
        return [optimizer], [sched]

