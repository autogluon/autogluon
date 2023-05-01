import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn
from torch.nn.modules.loss import _Loss
from torchmetrics.aggregation import BaseAggregator

try:
    import mmdet
    from mmcv import ConfigDict
except ImportError as e:
    import warnings

    pass

from ..constants import AUTOMM, IMAGE, LABEL
from .utils import apply_layerwise_lr_decay, apply_single_lr, apply_two_stages_lr, get_lr_scheduler, get_optimizer

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
        loss_func: Optional[_Loss] = None,
        validation_metric: Optional[torchmetrics.Metric] = None,
        validation_metric_name: Optional[str] = None,
        custom_metric_func: Callable = None,
        test_metric: Optional[torchmetrics.Metric] = None,
        efficient_finetune: Optional[str] = None,
    ):
        super().__init__()  # TODO: inherit LitModule
        self.save_hyperparameters(
            ignore=[
                "model",
                "validation_metric",
                "test_metric",
                "loss_func",
                "matches",
            ]
        )
        self.model = model
        self.validation_metric = validation_metric
        self.validation_metric_name = f"val_{validation_metric_name}"
        self.use_loss = isinstance(validation_metric, BaseAggregator)
        self.id2label = self.model.id2label
        self.input_data_key = self.model.prefix + "_" + IMAGE
        self.input_label_key = self.model.prefix + "_" + LABEL

    def _base_step(self, batch, mode):
        self.model.set_data_preprocessor_device()
        data = self.model.data_preprocessor(batch[self.input_data_key])
        ret = self.model.model(
            inputs=data["inputs"],
            data_samples=data["data_samples"],
            mode=mode,
        )

        return ret

    def _predict_step(self, batch):
        return self._base_step(batch=batch, mode="predict")

    def _loss_step(self, batch):
        return self._base_step(batch=batch, mode="loss")

    def _get_map_input(self, pred_results, sample):
        # print(pred_results)
        # print(pred_results[0].gt_instances)
        # print(pred_results[0].ignored_instances)
        # print(pred_results[0].keys())
        # exit()

        preds = []
        target = []

        batch_size = len(pred_results)

        for i in range(batch_size):
            if hasattr(pred_results[i].pred_instances, "masks"):
                # has one additional dimension with 2 outputs: img_result=img_result[0], mask_result=img_result[1]
                raise NotImplementedError(
                    "Do not support training for models with masks like mask r-cnn, "
                    "because most custom datasets do not have a ground truth mask."
                    " However, you can still inference with this model."
                )
            preds.append(
                dict(
                    boxes=pred_results[i].pred_instances.bboxes,  # .float().to(self.device)?
                    scores=pred_results[i].pred_instances.scores,  # .float().to(self.device)?
                    labels=pred_results[i].pred_instances.labels,  # .long().to(self.device)?
                )
            )
            target.append(
                dict(
                    boxes=pred_results[i].gt_instances.bboxes,
                    labels=pred_results[i].gt_instances.labels,
                )
            )

        return preds, target

    def evaluate(self, sample, stage=None):
        """
        sample: dict
            Single data sample.
        """
        pred_results = self._predict_step(sample)

        preds, target = self._get_map_input(pred_results, sample)

        # use MeanAveragePrecision, example code: https://github.com/Lightning-AI/metrics/blob/master/examples/detection_map.py
        self.validation_metric.update(preds, target)

        return pred_results

    def compute_loss(self, img, img_metas, gt_bboxes, gt_labels, *args, **kwargs):
        """
        Equivalent to `val_step` and `train_step` of `self.model`.
        https://github.com/open-mmlab/mmdetection/blob/56e42e72cdf516bebb676e586f408b98f854d84c/mmdet/models/detectors/base.py#L221
        https://github.com/open-mmlab/mmdetection/blob/56e42e72cdf516bebb676e586f408b98f854d84c/mmdet/models/detectors/base.py#L256
        Parameters
        ----------
        img
            Image Tensor
            torch.Tensor, Size: [batch_size, C, W, H]
        img_metas
            List of image metadata dict
            [
                {
                    'filename': 'data/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
                    'ori_filename': 'JPEGImages/000001.jpg',
                    'ori_shape': (500, 353, 3),
                    ...
                },
                ...(batch size times)
            ]
        gt_bboxes
            List of ground-truth bounding boxes position tensors
            [
                torch.Tensor, Size: [# objects in image, 4],
                ...(batch size times)
            ]
        gt_labels
            List of ground-truth bounding boxes label tensors
            [
                torch.Tensor, Size: [# objects in image],
                ...(batch size times)
            ]
        """
        losses = self.model.forward_train(
            img=img,
            img_metas=img_metas,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            *args,
            **kwargs,
        )
        return self.parse_losses(losses)

    def parse_losses(self, losses):
        # `_parse_losses`: https://github.com/open-mmlab/mmdetection/blob/
        # 56e42e72cdf516bebb676e586f408b98f854d84c/mmdet/models/detectors/base.py#L176
        loss, log_vars = self.model._parse_losses(losses)
        return loss, log_vars

    def log_step_results(self, losses):
        map_loss_names = {
            "loss": "total_loss",
            "acc": "classification-accuracy",
            "loss_rpn_cls": "loss/rpn_cls",
            "loss_rpn_bbox": "loss/rpn_bbox",
            "loss_bbox": "loss/bbox_reg",
            "loss_cls": "loss/classification",
        }
        for key in losses:
            # map metric names same name with epoch-wise metrics defined in:
            # `configs/vision/object-detection/mmdet/mmdet-base.yaml`
            loss_name = map_loss_names.get(key, key)
            self.log(f"step/{loss_name}", losses[key])

    def training_step(self, batch, batch_idx):
        print(len(batch))
        losses = self._loss_step(batch=batch)  # TODO: sum and log losses
        print(f"losses:\n{losses}", flush=True)
        exit()
        return losses
        # log step losses
        self.log_step_results(log_vars)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.use_loss:
            losses = self._loss_step(batch=batch)
            self.validation_metric.update(val_loss)
        else:
            self.evaluate(batch, "val")

    def validation_epoch_end(self, validation_step_outputs):
        val_result = self.validation_metric.compute()
        if self.use_loss:
            self.log_dict({"val_direct_loss": val_result}, sync_dist=True)
        else:
            # TODO: add mAP/mAR_per_class
            mAPs = {"val_" + k: v for k, v in val_result.items()}
            mAPs["val_mAP"] = mAPs["val_map"]
            self.log_dict(mAPs, sync_dist=True)
        self.validation_metric.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError("test with lit_mmdet is not implemented yet.")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self._predict_step(batch)
        if "mmdet_image_label" in batch:
            return {"bbox": pred, "label": batch[self.input_label_key]}
        else:
            return {"bbox": pred}

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
        # TODO: add freeze layer and different lr for head and backbone.
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
