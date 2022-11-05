import logging
from typing import Callable, Dict, List, Optional, Union

import torchmetrics
from torch import nn
from torch.nn.modules.loss import _Loss

from ..constants import AUTOMM, LM_TARGET, LOGITS, T_FEW, TEMPLATE_LOGITS, WEIGHT
from ..data.mixup import MixupModule, multimodel_mixup
from ..data.tabular_pretrain import ContrastiveTransformations
from .lit_module import LitModule
from .losses import DistillLoss, NTXent, ReconstructionLoss

logger = logging.getLogger(AUTOMM)


class PretrainLitModule(LitModule):
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
        validation_metric: Optional[torchmetrics.Metric] = None,
        validation_metric_name: Optional[str] = None,
        custom_metric_func: Callable = None,
        test_metric: Optional[torchmetrics.Metric] = None,
        efficient_finetune: Optional[str] = None,
        trainable_param_names: Optional[List] = None,
        mixup_fn: Optional[MixupModule] = None,
        mixup_off_epoch: Optional[int] = 0,
        model_postprocess_fn: Callable = None,
        pretrain_epochs: Optional[int] = 5,
        problem_type: Optional[str] = None,
        augmentation_mode: Optional[str] = None,
        corruption_rate: Optional[float] = None,
        start_loss_coefficient: Optional[float] = 0.1,
        end_loss_coefficient: Optional[float] = 0.1,
        decay_loss_coefficient: Optional[float] = 0.6,
        pretrain_objective: Optional[str] = None,
        row_attention_weight_decay: Optional[float] = None,
        temperature: Optional[float] = 1,
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
        efficient_finetune
            Whether to use efficient finetuning strategies. This will be helpful for fast finetuning of large backbones.
            We support options such as:

            - bit_fit (only finetune the bias terms)
            - norm_fit (only finetune the weights in norm layers / bias layer)
            - lora, lora_bias, lora_norm (only finetunes decomposition matrices inserted into model, in combination with either bit_fit or norm_fit)
            - ia3, ia3_bias, ia3_norm (adds vector that scales activations by learned vectors, in combination with either bit_fit or norm_fit)
            - None (do not use efficient finetuning strategies)

        """
        super().__init__(
            model=model,
            optim_type=optim_type,
            lr_choice=lr_choice,
            lr_schedule=lr_schedule,
            lr=lr,
            lr_decay=lr_decay,
            end_lr=end_lr,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            loss_func=loss_func,
            validation_metric=validation_metric,
            validation_metric_name=validation_metric_name,
            custom_metric_func=custom_metric_func,
            test_metric=test_metric,
            efficient_finetune=efficient_finetune,
            trainable_param_names=trainable_param_names,
            mixup_fn=mixup_fn,
            mixup_off_epoch=mixup_off_epoch,
            model_postprocess_fn=model_postprocess_fn,
            row_attention_weight_decay=row_attention_weight_decay,
        )
        self.contrastive_loss = (
            DistillLoss(temperature=temperature)
            if pretrain_objective in ["self_distill"]
            else NTXent(temperature=temperature)
        )
        self.reconstruction_loss = ReconstructionLoss(model)
        self.contrastive_fn = ContrastiveTransformations(
            model,
            mode=augmentation_mode,
            problem_type=problem_type,
            corruption_rate=corruption_rate,
        )
        self.start_loss_coefficient = start_loss_coefficient
        self.end_loss_coefficient = end_loss_coefficient
        self.decay_loss_coefficient = decay_loss_coefficient
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_objective = pretrain_objective

    def _compute_pretrain_loss(
        self,
        batch: Dict,
        output: Dict,
        positive: Dict,
        reconstruction: Dict,
    ):
        loss = 0
        if output and positive:
            for per_key, _ in output.items():
                per_output = output[per_key]
                per_positive = positive[per_key]
                weight = per_output[WEIGHT] if WEIGHT in per_output else 1
                loss += (
                    self.contrastive_loss(
                        z_i=per_output[LOGITS],
                        z_j=per_positive[LOGITS],
                    )
                    * weight
                )

        if reconstruction:
            loss += self.reconstruction_loss(batch, reconstruction)

        return loss

    def _shared_step(
        self,
        batch: Dict,
    ):
        label = batch[self.model.label_key]
        if self.mixup_fn is not None:
            self.mixup_fn.mixup_enabled = self.training & (self.current_epoch < self.hparams.mixup_off_epoch)
            batch, label = multimodel_mixup(batch=batch, model=self.model, mixup_fn=self.mixup_fn)
        corrupted_batch = self.contrastive_fn(batch)
        output = self.model(batch)
        original_view, corrupted_view, reconstruction = None, None, None
        if self.pretrain_objective in ["both", "reconstruction"]:
            reconstruction = self.model(corrupted_batch, head="reconstruction")
        if self.pretrain_objective in ["both", "contrastive"]:
            original_view = self.model(batch, head="contrastive")
            corrupted_view = self.model(corrupted_batch, head="contrastive")

        if self.pretrain_objective in ["self_distill"]:
            original_view = output  # self.model(batch, require_grad=True)
            corrupted_view = self.model(corrupted_batch)

        pretrain_data = (batch, original_view, corrupted_view, reconstruction)
        loss = self._compute_loss(output=output, label=label)
        return output, loss, pretrain_data

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
        output, loss, pretrain_data = self._shared_step(batch)
        self.log("train_loss", loss)

        pretrain_loss = self._compute_pretrain_loss(*pretrain_data)

        if self.current_epoch < self.pretrain_epochs:
            return pretrain_loss
        else:
            lam = self.start_loss_coefficient * (
                self.decay_loss_coefficient ** (self.current_epoch - self.pretrain_epochs + 1)
            )
            lam = max(lam, self.end_loss_coefficient)
            return pretrain_loss * lam + loss  # * (1-lam)

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
        output, loss, _ = self._shared_step(batch)
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        self._compute_metric_score(
            metric=self.validation_metric,
            custom_metric_func=self.custom_metric_func,
            logits=output[self.model.prefix][LOGITS],
            label=batch[self.model.label_key],
        ),
        self.log(
            self.validation_metric_name,
            self.validation_metric,
            on_step=False,
            on_epoch=True,
        )
