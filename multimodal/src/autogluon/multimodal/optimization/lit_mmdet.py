"""
Useful documentation for mmdetection:
    https://mmdetection.readthedocs.io/en/latest/api.html?
Originally, the `DataLoaders` in mmdetection handle multi-gpu by custom collate functions and importantly the
mmcv.parallel.MMDataParallel class which wraps the dataloader which interpret the specific format of input. Since
pytorch-lightning handles such multi-gpu training, we use a classic `torch.utils.data.DataLoader` object to batch data
but parse the output of collate function during `training_step` and `validation_step`.
Specifically, these include:
    - the logic that receives `_sample` and converts it into `sample` in `MMDetectionTrainer::evaluate`
    - the logic that receives `_batch` and converts it into `batch` in `MMDetectionTrainer::_training_step`
"""

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
except ImportError:
    pass

from ..constants import AUTOMM, PROBABILITY
from .utils import (
    apply_layerwise_lr_decay,
    apply_single_lr,
    apply_two_stages_lr,
    compute_probability,
    gather_column_features,
    generate_metric_learning_labels,
    get_lr_scheduler,
    get_metric_learning_loss_funcs,
    get_metric_learning_miner_funcs,
    get_optimizer,
)

from ..utils import (
    send_datacontainers_to_device,
    unpack_datacontainers,
)

logger = logging.getLogger(AUTOMM)


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
        """
        In essence, `MMDetectionTrainer` is a pytorch-lightning version of the runners in `mmcv`. The implementation
        was especially influenced by the implementation of `EpochBasedRunner`.
        https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/epoch_based_runner.html
        """
        super().__init__()
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
        if isinstance(validation_metric, BaseAggregator) and custom_metric_func is None:
            raise ValueError(
                f"validation_metric {validation_metric} is an aggregation metric,"
                f"which must be used with a customized metric function."
            )
        self.custom_metric_func = custom_metric_func
        self.id2label = dict(zip(range(100), range(100))) # TODO: replace with real id2label

    def forward(self, x):
        """
        x: dict
            batch of data. For example,
            {
                "img":
                "img_metas":
                "gt_bboxes":
                "gt_labels":
            }
        """
        # out = self.model.forward(x)
        pass

    def _predict_step(self, batch, batch_idx=0):
        assert not self.model.training
        assert not self.model.model.training
        from mmcv.parallel import scatter
        from mmcv.ops import RoIPool

        data = batch["mmdet_image_image"]
        data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
        data["img"] = [img.data[0] for img in data["img"]]
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(m, RoIPool), "CPU inference with RoIPool is not supported currently."
        pred_results = self.model.model(return_loss=False, rescale=True, **data)
        return pred_results

    def _training_step(self, batch, batch_idx=0):
        assert self.model.training
        assert self.model.model.training
        # train dataloader.
        # send_datacontainers_to_device(data=batch, device=self.device)
        batch = unpack_datacontainers(batch)

        img_metas = batch["mmdet_image_image"]["img_metas"][0]
        img = batch["mmdet_image_image"]["img"][0].float().to(self.device)
        batch_size = img.shape[0]
        gt_bboxes = []
        gt_labels = []
        for i in range(batch_size):
            gt_bboxes.append(torch.tensor(batch["mmdet_image_image"]["gt_bboxes"][0][i]).float().to(self.device))
            gt_labels.append(torch.tensor(batch["mmdet_image_image"]["gt_labels"][0][i]).long().to(self.device))

        loss, log_vars = self.compute_loss(
            img=img,
            img_metas=img_metas,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
        )

        # log step losses
        self.log_step_results(log_vars)

        return loss, log_vars

    def evaluate(self, sample, stage=None):
        """
        sample: dict
            Single data sample.
        """
        pred_results = self._predict_step(sample)

        preds = []
        for img_idx, img_result in enumerate(pred_results):
            img_result = img_result
            boxes = []
            scores = []
            labels = []
            for category_idx, category_result in enumerate(img_result):
                for item_idx, item_result in enumerate(category_result):
                    boxes.append(item_result[:4])
                    scores.append(float(item_result[4]))
                    labels.append(category_idx)
            preds.append(
                dict(
                    boxes=torch.tensor(np.array(boxes).astype(float)).float().to(self.device),
                    scores=torch.tensor(scores).float().to(self.device),
                    labels=torch.tensor(labels).long().to(self.device),
                )
            )

        '''
        batch = unpack_datacontainers(sample)

        img_metas = batch["mmdet_image_image"]["img_metas"][0]
        imgs = [batch["mmdet_image_image"]["img"][0][0].float().to(self.device)]


        batch_result = self.model.forward_test(
            imgs=imgs,
            img_metas=img_metas,
        )  # batch_size, 80, (n, 5)

        preds = []
        target = []

        for img_idx, img_result in enumerate(batch_result):
            boxes = []
            scores = []
            labels = []
            for category_idx, category_result in enumerate(img_result):
                for item_idx, item_result in enumerate(category_result):
                    boxes.append(item_result[:4])
                    scores.append(float(item_result[4]))
                    labels.append(category_idx)
            preds.append(
                dict(
                    boxes=torch.tensor(np.array(boxes).astype(float)).float().to(self.device),
                    scores=torch.tensor(scores).float().to(self.device),
                    labels=torch.tensor(labels).long().to(self.device),
                )
            )
        '''

        target = []

        batch_size = len(preds)
        batch = unpack_datacontainers(sample)
        gt = batch["mmdet_image_label"] # batch_size, (n, 5)
        for i in range(batch_size):
            img_gt = np.array(gt[i])
            boxes = img_gt[:,:4]
            labels = img_gt[:,4]
            target.append(
                dict(
                    boxes=torch.tensor(boxes).float().to(self.device),
                    labels=torch.tensor(labels).long().to(self.device),
                )
            )

        # use MeanAveragePrecision, example code: https://github.com/Lightning-AI/metrics/blob/master/examples/detection_map.py
        self.validation_metric.update(preds, target)

        return pred_results

    def compute_loss(self, img, img_metas, gt_bboxes, gt_labels, *args, **kwargs):
        """
        Equivalent to `val_step` and `train_step` of `self.model`.
        https://github.com/open-mmlab/mmdetection/blob/56e42e72cdf516bebb676e586f408b98f854d84c/mmdet/models/detectors/base.py#L221
        https://github.com/open-mmlab/mmdetection/blob/56e42e72cdf516bebb676e586f408b98f854d84c/mmdet/models/detectors/base.py#L256
        x: dict
            batch of data. For example,
            ```
            {
                "img": torch.Tensor, Size: [batch_size, C, W, H],
                "img_metas": [
                    {
                        'filename': 'data/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
                        'ori_filename': 'JPEGImages/000001.jpg',
                        'ori_shape': (500, 353, 3),
                        ...
                    },
                    ...(batch size times)
                ],
                "gt_bboxes": [
                    torch.Tensor, Size: [# objects in image, 4],
                    ...(batch size times)
                ],
                "gt_labels": [
                    torch.Tensor, Size: [# objects in image],
                    ...(batch size times)
                ],
            }
            ```
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
        loss, log_vars = self._training_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.evaluate(batch, "val")

    def validation_epoch_end(self, validation_step_outputs):
        mAPs = {"val_" + k: v for k, v in self.validation_metric.compute().items()}
        self.print(mAPs)
        #mAPs_per_class = mAPs.pop("val_map_per_class")
        #mARs_per_class = mAPs.pop("val_mar_100_per_class")
        self.log_dict(mAPs, sync_dist=True)
        '''
        self.log_dict(
            {
                f"val_map_{label}": value
                for label, value in zip(self.id2label.values(), mAPs_per_class)
            },
            sync_dist=True,
        )
        self.log_dict(
            {
                f"val_mar_100_{label}": value
                for label, value in zip(self.id2label.values(), mARs_per_class)
            },
            sync_dist=True,
        )
        '''
        self.validation_metric.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError("test with lit_mmdet is not implemented yet.")
        res = self.evaluate(batch, "test")
        return res

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self._predict_step(batch, batch_idx)
        return {"bbox": pred, "label": batch["mmdet_image_label"]}

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
