import logging
from typing import Callable, List, Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch import nn
from torch.nn.modules.loss import _Loss
from torchmetrics.aggregation import BaseAggregator

from ..constants import FEATURES, LOGITS, WEIGHT
from ..models.utils import run_model
from .utils import apply_layerwise_lr_decay, apply_single_lr, apply_two_stages_lr, get_lr_scheduler, get_optimizer

logger = logging.getLogger(__name__)


class DistillerLitModule(pl.LightningModule):
    """
    Knowledge distillation loops for training and evaluation. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        matches: List[DictConfig],
        critics: nn.ModuleList,
        baseline_funcs: nn.ModuleList,
        hard_label_weight: float,
        soft_label_weight: float,
        softmax_regression_weight: float,
        temperature: float,
        output_feature_loss_weight: float,
        optim_type: Optional[str] = None,
        lr_choice: Optional[str] = None,
        lr_schedule: Optional[str] = None,
        lr: Optional[float] = None,
        lr_decay: Optional[float] = None,
        end_lr: Optional[Union[float, int]] = None,
        lr_mult: Optional[Union[float, int]] = None,
        weight_decay: Optional[float] = None,
        warmup_steps: Optional[int] = None,
        hard_label_loss_func: Optional[_Loss] = None,
        soft_label_loss_func: Optional[_Loss] = None,
        softmax_regression_loss_func: Optional[_Loss] = None,
        output_feature_adaptor: Optional[nn.Module] = None,
        output_feature_loss_func: Optional[_Loss] = None,
        rkd_loss_func: Optional[nn.Module] = None,
        validation_metric: Optional[torchmetrics.Metric] = None,
        validation_metric_name: Optional[str] = None,
        custom_metric_func: Callable = None,
        test_metric: Optional[torchmetrics.Metric] = None,
        track_grad_norm: Optional[Union[int, str]] = -1,
    ):
        """
        Parameters
        ----------
        student_model
            The student model in knowledge distillation.
        teacher_model
            The teacher model in knowledge distillation.
        matches
            Teacher/student layer matches to compute the intermediate loss.
        critics
            The critics used in computing mutual information loss.
        baseline_funcs
            The baseline functions used in computing mutual information loss.
        hard_label_weight
            Weight for hard label loss.
        soft_label_weight
            Weight for soft label loss.
        softmax_regression_weight_label_weight
            Weight for softmax regression loss. Ref: https://www.adrianbulat.com/downloads/ICLR2021/knowledge_distillation_via_softmax_regression_representation_learning.pdf
        temperature
            A scalar to scale teacher and student logits in soft label loss.
        output_feature_loss_weight
            Weight for output_feature layer's loss.
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
        hard_label_loss_func
            A Pytorch loss module, e.g., nn.CrossEntropyLoss(), for hard labels.
        soft_label_loss_func
            A Pytorch loss module, e.g., nn.CrossEntropyLoss(), for soft labels.
        softmax_regression_loss_func
            A Pytorch loss module, e.g., nn.CrossEntropyLoss(), for softmax regression.
            Refer to: https://www.adrianbulat.com/downloads/ICLR2021/knowledge_distillation_via_softmax_regression_representation_learning.pdf
        output_feature_adaptor
            A Pytorch Module, e.g. nn.Linear, for adapting student output feature to the shape of teacher's.
        output_feature_loss_func
            A Pytorch loss module, e.g., nn.MSELoss(), for output_feature distance between teacher and student.
        rkd_loss_func
            A Pytorch loss module, i.e., RKDLoss in .rkd_loss, for rkd loss of output_feature between teacher and student.
        validation_metric
            A torchmetrics module used in the validation stage, e.g., torchmetrics.Accuracy().
        validation_metric_name
            Name of validation metric in case that validation_metric is a aggregation metric,
            e.g., torchmetrics.MeanMetric, whose name can't reflect the real metric name.
        custom_metric_func
            A customized metric function in case that torchmetrics doesn't have the metric.
            It is generally used together with torchmetrics' aggregators, e.g., torchmetrics.MeanMetric.
            Refer to https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/aggregation.py
        test_metric
            A torchmetrics module used in the test stage, e.g., torchmetrics.Accuracy().
        track_grad_norm
            Track the p-norm of gradients during training. May be set to ‘inf’ infinity-norm.
            If using Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.
        """
        super().__init__()
        self.optim_type = optim_type
        self.save_hyperparameters(
            ignore=[
                "student_model",
                "teacher_model",
                "validation_metric",
                "hard_label_loss_func",
                "soft_label_loss_func",
                "custom_metric_func",
                "test_metric",
                "matches",
                "critics",
                "baseline_funcs",
                "output_feature_adaptor",
                "output_feature_loss_func",
                "rkd_loss_func",
            ]
        )
        if matches:
            assert len(matches) == len(critics)
            assert len(matches) == len(baseline_funcs)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.matches = matches
        self.critics = critics
        self.baseline_funcs = baseline_funcs
        self.validation_metric = validation_metric
        self.validation_metric_name = f"val_{validation_metric_name}"
        self.temperature = temperature
        self.hard_label_weight = hard_label_weight
        self.soft_label_weight = soft_label_weight
        self.softmax_regression_weight = softmax_regression_weight
        self.output_feature_loss_weight = output_feature_loss_weight
        self.hard_label_loss_func = hard_label_loss_func
        self.soft_label_loss_func = soft_label_loss_func
        self.softmax_regression_loss_func = softmax_regression_loss_func
        self.output_feature_loss_func = output_feature_loss_func
        if isinstance(validation_metric, BaseAggregator) and custom_metric_func is None:
            raise ValueError(
                f"validation_metric {validation_metric} is an aggregation metric,"
                "which must be used with a customized metric function."
            )
        self.custom_metric_func = custom_metric_func

        self.output_feature_adaptor = output_feature_adaptor
        self.rkd_loss_func = rkd_loss_func
        self.track_grad_norm = track_grad_norm

    def _compute_hard_label_loss(
        self,
        output: dict,
        label: torch.Tensor,
    ):
        loss = 0
        for per_output in output.values():
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            loss += (
                self.hard_label_loss_func(
                    input=per_output[LOGITS].squeeze(dim=1),
                    target=label,
                )
                * weight
            )

        return loss

    def _compute_soft_label_loss(
        self,
        student_output: dict,
        teacher_output: dict,
    ):
        student_logits = student_output[self.student_model.prefix][LOGITS].squeeze(dim=1)
        soft_labels = teacher_output[self.teacher_model.prefix][LOGITS].squeeze(dim=1)
        student_logits = student_logits / self.temperature
        soft_labels = soft_labels / self.temperature

        if isinstance(self.soft_label_loss_func, nn.CrossEntropyLoss):
            soft_labels = F.softmax(soft_labels, dim=-1)

        loss = self.soft_label_loss_func(
            input=student_logits,
            target=soft_labels,
        )
        return loss

    def _compute_output_feature_loss(
        self,
        student_output: dict,
        teacher_output: dict,
    ):
        student_result = student_output[self.student_model.prefix][FEATURES].squeeze(dim=1)
        teacher_result = teacher_output[self.teacher_model.prefix][FEATURES].squeeze(dim=1)

        student_result = self.output_feature_adaptor(student_result)

        loss = self.output_feature_loss_func(
            input=student_result,
            target=teacher_result,
        )
        return loss

    def _compute_rkd_loss(
        self,
        student_output: dict,
        teacher_output: dict,
    ):
        student_result = student_output[self.student_model.prefix][FEATURES].squeeze(dim=1)
        teacher_result = teacher_output[self.teacher_model.prefix][FEATURES].squeeze(dim=1)

        student_result = self.output_feature_adaptor(student_result)

        loss = self.rkd_loss_func(
            feature_student=student_result,
            feature_teacher=teacher_result,
        )

        return loss

    def _compute_softmax_regression_loss(
        self,
        student_output: dict,
        teacher_output: dict,
    ):
        student_feature = student_output[self.student_model.prefix][FEATURES].squeeze(dim=1)
        student_feature = self.output_feature_adaptor(student_feature)

        student_logits = self.teacher_model.head(student_feature)
        soft_labels = teacher_output[self.teacher_model.prefix][LOGITS].squeeze(dim=1)

        student_logits = student_logits
        soft_labels = soft_labels

        if isinstance(self.softmax_regression_loss_func, nn.CrossEntropyLoss):
            soft_labels = F.softmax(soft_labels, dim=-1)

        loss = self.softmax_regression_loss_func(
            input=student_logits,
            target=soft_labels,
        )
        return loss

    def _compute_loss(
        self,
        student_output: dict,
        teacher_output: dict,
        label: torch.Tensor,
    ):
        loss = 0
        hard_label_loss = self._compute_hard_label_loss(
            output=student_output,
            label=label,
        )
        loss += hard_label_loss * self.hard_label_weight

        if self.soft_label_weight > 0:
            soft_label_loss = self._compute_soft_label_loss(
                student_output=student_output,
                teacher_output=teacher_output,
            )
            loss += soft_label_loss * self.soft_label_weight

        if self.softmax_regression_weight > 0:
            softmax_regression_loss = self._compute_softmax_regression_loss(
                student_output=student_output,
                teacher_output=teacher_output,
            )
            loss += softmax_regression_loss * self.softmax_regression_weight

        if self.output_feature_loss_weight > 0:
            output_feature_loss = self._compute_output_feature_loss(
                student_output=student_output,
                teacher_output=teacher_output,
            )
            loss += output_feature_loss * self.output_feature_loss_weight

        if self.rkd_loss_func.distance_loss_weight > 0 or self.rkd_loss_func.angle_loss_weight > 0:
            rkd_loss = self._compute_rkd_loss(
                student_output=student_output,
                teacher_output=teacher_output,
            )
            loss += rkd_loss

        return loss

    def _compute_metric_score(
        self,
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        logits: torch.Tensor,
        label: torch.Tensor,
    ):
        if isinstance(
            metric, (torchmetrics.classification.BinaryAUROC, torchmetrics.classification.BinaryAveragePrecision)
        ):
            prob = F.softmax(logits.float(), dim=1)
            metric.update(preds=prob[:, 1], target=label)  # only for binary classification
        elif isinstance(metric, BaseAggregator):
            metric.update(custom_metric_func(logits, label))
        else:
            metric.update(logits.squeeze(dim=1), label)

    def _shared_step(
        self,
        batch: dict,
    ):
        student_output = run_model(self.student_model, batch)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_output = run_model(self.teacher_model, batch)
        label = batch[self.student_model.label_key]
        loss = self._compute_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            label=label,
        )
        return student_output, loss

    def training_step(self, batch, batch_idx):
        """
        Per training step. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop

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
        _, loss = self._shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
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
        student_output, loss = self._shared_step(batch)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        self._compute_metric_score(
            metric=self.validation_metric,
            custom_metric_func=self.custom_metric_func,
            logits=student_output[self.student_model.prefix][LOGITS],
            label=batch[self.student_model.label_key],
        ),
        self.log(
            self.validation_metric_name,
            self.validation_metric,
            on_step=False,
            on_epoch=True,
        )

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
            model=self.student_model,
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

        if self.critics:  # to handle None
            for per_model_critics in self.critics:
                for per_critic in per_model_critics:
                    critics_parameters = apply_single_lr(
                        model=per_critic,
                        lr=self.hparams.lr,
                        weight_decay=self.hparams.weight_decay,
                    )
                    grouped_parameters.extend(critics_parameters)

        if self.baseline_funcs:  # to handle None
            for per_model_baseline_funcs in self.baseline_funcs:
                for per_baseline_func in per_model_baseline_funcs:
                    baseline_func_params = apply_single_lr(
                        model=per_baseline_func,
                        lr=self.hparams.lr,
                        weight_decay=self.hparams.weight_decay,
                    )
                    grouped_parameters.extend(baseline_func_params)

        adaptor_params = apply_single_lr(
            model=self.output_feature_adaptor,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        grouped_parameters.extend(adaptor_params)

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
                f"len(trainer.datamodule.train_dataloader()): {len(self.trainer.datamodule.train_dataloader())}"
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
