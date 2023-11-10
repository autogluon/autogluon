import logging
from typing import Callable, Dict, List, Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities import grad_norm
from torch import nn
from torch.nn.modules.loss import _Loss
from torchmetrics.aggregation import BaseAggregator
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import CLASS_LOGITS, LM_TARGET, LOGITS, MASK_SEMANTIC_INFER, T_FEW, TEMPLATE_LOGITS, WEIGHT
from ..data.mixup import MixupModule, multimodel_mixup
from ..models.utils import run_model
from .lit_module import LitModule
from .semantic_seg_metrics import COD, Balanced_Error_Rate
from .utils import apply_layerwise_lr_decay, apply_single_lr, apply_two_stages_lr, get_lr_scheduler, get_optimizer

logger = logging.getLogger(__name__)


class SemanticSegmentationLitModule(LitModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def _compute_loss(self, output: Dict, label: torch.Tensor, **kwargs):
        loss = 0
        for _, per_output in output.items():
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            if (
                TEMPLATE_LOGITS in per_output and self.model.prefix == T_FEW
            ):  # Do only add template loss if T-Few. #TODO Add compatibility to Fusion models.
                loss += self._compute_template_loss(per_output, label) * weight
            elif isinstance(self.loss_func, Mask2FormerLoss):
                dict_loss = self.loss_func(
                    masks_queries_logits=per_output[LOGITS],  # bs, num_q, h, w
                    class_queries_logits=per_output[CLASS_LOGITS],  # bs, num_q, num_labels
                    mask_labels=kwargs["mask_labels"].to(per_output[LOGITS]),
                    class_labels=kwargs["class_labels"],
                )
                for v in dict_loss.values():
                    loss += v
            else:
                loss += (
                    self.loss_func(
                        input=per_output[LOGITS].squeeze(dim=1),
                        target=label,
                    )
                    * weight
                )
        return loss

    def _compute_metric_score(
        self,
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        logits: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ):
        if isinstance(
            metric, (torchmetrics.classification.BinaryAUROC, torchmetrics.classification.BinaryAveragePrecision)
        ):
            prob = F.softmax(logits.float(), dim=1)
            metric.update(preds=prob[:, 1], target=label)  # for binary classification only
        elif isinstance(metric, BaseAggregator):
            metric.update(custom_metric_func(logits, label))
        elif (
            isinstance(metric, torchmetrics.classification.BinaryJaccardIndex)
            or isinstance(metric, Balanced_Error_Rate)
            or isinstance(metric, COD)
        ):
            metric.update(logits.float(), label)
        elif isinstance(metric, torchmetrics.classification.MulticlassJaccardIndex):
            bs, num_classes = kwargs["processed_results"].shape[0:2]
            processed_results = kwargs["processed_results"].float().reshape(bs, num_classes, -1)
            label = label.reshape(bs, -1)
            metric.update(processed_results, label)
        else:
            metric.update(logits.squeeze(dim=1).float(), label)

    def _shared_step(
        self,
        batch: Dict,
    ):
        label = batch[self.model.label_key]
        # prepare_targets
        if self.mixup_fn is not None:
            self.mixup_fn.mixup_enabled = self.training & (self.current_epoch < self.hparams.mixup_off_epoch)
            batch, label = multimodel_mixup(batch=batch, model=self.model, mixup_fn=self.mixup_fn)
        output = run_model(self.model, batch)
        if isinstance(self.loss_func, Mask2FormerLoss):
            loss = self._compute_loss(
                output=output,
                label=label,
                mask_labels=batch[self.model.mask_label_key],
                class_labels=batch[self.model.class_label_key],
            )
        else:
            loss = self._compute_loss(
                output=output,
                label=label,
            )

        return output, loss

    def validation_step(self, batch, batch_idx, **kwargs):
        """
        Per validation step. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-loop

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
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        if isinstance(self.loss_func, Mask2FormerLoss):
            self._compute_metric_score(
                metric=self.validation_metric,
                custom_metric_func=self.custom_metric_func,
                logits=output[self.model.prefix][LOGITS],
                label=batch[self.model.label_key],
                processed_results=output[self.model.prefix][MASK_SEMANTIC_INFER],
            )
        else:
            self._compute_metric_score(
                metric=self.validation_metric,
                custom_metric_func=self.custom_metric_func,
                logits=output[self.model.prefix][LOGITS],
                label=batch[self.model.label_key],
            )

        self.log(
            self.validation_metric_name,
            self.validation_metric,
            on_step=False,
            on_epoch=True,
        )
