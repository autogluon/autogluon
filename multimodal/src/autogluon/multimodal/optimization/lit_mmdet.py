import logging
from typing import Callable, Optional, Union

import lightning.pytorch as pl
import torchmetrics
from lightning.pytorch.utilities import grad_norm
from torch.nn.modules.loss import _Loss
from torchmetrics.aggregation import BaseAggregator

from ..constants import BBOX, IMAGE, LABEL
from .utils import (
    apply_layerwise_lr_decay,
    apply_single_lr,
    apply_two_stages_lr,
    get_lr_scheduler,
    get_optimizer,
    remove_parameters_without_grad,
)

try:
    import mmdet
    from mmcv import ConfigDict
except ImportError as e:
    mmdet = None
    ConfigDict = None

logger = logging.getLogger(__name__)


class MMDetLitModule(pl.LightningModule):
    def __init__(
        self,
        model,
        optim_type: Optional[str] = None,
        lr_choice: Optional[str] = None,
        lr_schedule: Optional[str] = None,
        lr: Optional[float] = None,
        lr_decay: Optional[float] = None,
        end_lr: Optional[Union[float, int]] = None,
        lr_mult: Optional[Union[float, int]] = None,
        weight_decay: Optional[float] = None,
        warmup_steps: Optional[int] = None,
        validation_metric: Optional[torchmetrics.Metric] = None,
        validation_metric_name: Optional[str] = None,
        custom_metric_func: Callable = None,
        test_metric: Optional[torchmetrics.Metric] = None,
        track_grad_norm: Optional[Union[int, str]] = -1,
    ):
        super().__init__()  # TODO: inherit LitModule
        self.save_hyperparameters(
            ignore=[
                "model",
                "validation_metric",
                "test_metric",
            ]
        )
        self.model = model
        self.validation_metric = validation_metric
        self.validation_metric_name = f"val_{validation_metric_name}"
        self.use_loss = isinstance(validation_metric, BaseAggregator)
        self.id2label = self.model.id2label
        self.input_data_key = self.model.prefix + "_" + IMAGE
        self.input_label_key = self.model.prefix + "_" + LABEL
        self.track_grad_norm = track_grad_norm

    def _base_step(self, batch, mode):
        ret = self.model(batch=batch[self.input_data_key], mode=mode)

        return ret

    def _predict_step(self, batch):
        return self._base_step(batch=batch, mode="predict")

    def _loss_step(self, batch):
        return self._base_step(batch=batch, mode="loss")

    def _get_map_input(self, pred_results):
        preds = []
        target = []

        batch_size = len(pred_results)

        for i in range(batch_size):
            if hasattr(pred_results[i][BBOX], "masks"):
                # has one additional dimension with 2 outputs: img_result=img_result[0], mask_result=img_result[1]
                raise NotImplementedError(
                    "Do not support training for models with masks like mask r-cnn, "
                    "because most custom datasets do not have a ground truth mask."
                    " However, you can still inference with this model."
                )
            preds.append(
                dict(
                    boxes=pred_results[i][BBOX].bboxes,  # .float().to(self.device)?
                    scores=pred_results[i][BBOX].scores,  # .float().to(self.device)?
                    labels=pred_results[i][BBOX].labels,  # .long().to(self.device)?
                )
            )
            target.append(
                dict(
                    boxes=pred_results[i][LABEL].bboxes,
                    labels=pred_results[i][LABEL].labels,
                )
            )

        return preds, target

    def evaluate(self, sample, stage=None):
        """
        sample: dict
            Single data sample.
        """
        pred_results = self._predict_step(sample)

        preds, target = self._get_map_input(pred_results)

        # use MeanAveragePrecision, example code: https://github.com/Lightning-AI/metrics/blob/master/examples/detection_map.py
        self.validation_metric.update(preds, target)

        return pred_results

    def sum_and_log_step_results(self, losses, logging=True):
        # losses is a dict of several type of losses, e.g. ['loss_cls', 'loss_conf', 'loss_xy', 'loss_wh']
        # each type of losses may have multiple channels due to multiple resolution settings
        total_loss = 0.0
        for loss_key, loss_values in losses.items():
            curr_loss = 0.0
            if isinstance(loss_values, list) or isinstance(loss_values, tuple):  # is a collection of shape 0 tensors
                for loss_chanel_idx, loss_val in enumerate(loss_values):
                    if logging:
                        self.log(f"step/{loss_key}_{loss_chanel_idx}", loss_val)
                    curr_loss += loss_val
            else:  # is a tensor
                curr_loss += loss_values.sum()

            if logging:
                self.log(f"step/{loss_key}", curr_loss)
            total_loss += curr_loss

        return total_loss

    def training_step(self, batch, batch_idx):
        losses = self._loss_step(batch=batch)
        # sum and log step losses
        total_loss = self.sum_and_log_step_results(losses)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.use_loss:
            losses = self._loss_step(batch=batch)
            total_loss = self.sum_and_log_step_results(losses, logging=False)
            self.validation_metric.update(total_loss)
        else:
            self.evaluate(batch, "val")

    def on_validation_epoch_end(self):
        val_result = self.validation_metric.compute()
        if self.use_loss:
            self.log_dict({"val_direct_loss": val_result}, sync_dist=True)
        else:
            # TODO: add mAP/mAR_per_class
            val_result.pop("classes", None)  # introduced in torchmetrics v1.0.0
            mAPs = {"val_" + k: v for k, v in val_result.items()}
            mAPs["val_mAP"] = mAPs["val_map"]
            self.log_dict(mAPs, sync_dist=True)
        self.validation_metric.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError("test with lit_mmdet is not implemented yet.")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self._predict_step(batch)

        return pred

    def configure_optimizers(self):
        """
        Configure optimizer. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        Returns
        -------
        [optimizer]
            Optimizer.
        [sched]
            Learning rate scheduler.
        """
        kwargs = dict(
            model=self.model,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_choice == "two_stages":
            logger.debug("applying 2-stage learning rate...")
            grouped_parameters = apply_two_stages_lr(
                lr_mult=self.hparams.lr_mult,
                return_params=True,
                **kwargs,
            )
        elif self.hparams.lr_choice == "layerwise_decay":
            logger.debug("applying layerwise learning rate decay...")
            grouped_parameters = apply_layerwise_lr_decay(
                lr_decay=self.hparams.lr_decay,
                **kwargs,
            )
        else:
            logger.debug("applying single learning rate...")
            grouped_parameters = apply_single_lr(
                **kwargs,
            )

        grouped_parameters = remove_parameters_without_grad(grouped_parameters=grouped_parameters)

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
            logger.debug(
                f"len(trainer.datamodule.train_dataloader()): " f"{len(self.trainer.datamodule.train_dataloader())}"
            )
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
        scheduler = get_lr_scheduler(
            optimizer=optimizer,
            num_max_steps=max_steps,
            num_warmup_steps=warmup_steps,
            lr_schedule=self.hparams.lr_schedule,
            end_lr=self.hparams.end_lr,
        )

        sched = {"scheduler": scheduler, "interval": "step"}
        logger.debug("done configuring optimizer and scheduler")
        return [optimizer], [sched]

    def on_before_optimizer_step(self, optimizer):
        # If using mixed precision, the gradients are already unscaled here
        if self.track_grad_norm != -1:
            self.log_dict(grad_norm(self, norm_type=self.track_grad_norm))
