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

from ..constants import (
    AUG_LOGITS,
    LM_TARGET,
    LOGITS,
    MULTIMODAL_FEATURES,
    MULTIMODAL_FEATURES_POST_AUG,
    MULTIMODAL_FEATURES_PRE_AUG,
    ORI_LOGITS,
    T_FEW,
    TEMPLATE_LOGITS,
    VAE_MEAN,
    VAE_VAR,
    WEIGHT,
)
from ..data.mixup import MixupModule, multimodel_mixup
from ..models.utils import run_model
from .lr import apply_layerwise_lr_decay, apply_single_lr, apply_two_stages_lr, get_lr_scheduler
from .metrics import Coverage
from .utils import get_optimizer

logger = logging.getLogger(__name__)


class LitModule(pl.LightningModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
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
        peft: Optional[str] = None,
        trainable_param_names: Optional[List] = None,
        mixup_fn: Optional[MixupModule] = None,
        mixup_off_epoch: Optional[int] = 0,
        model_postprocess_fn: Callable = None,
        skip_final_val: Optional[bool] = False,
        track_grad_norm: Optional[Union[int, str]] = -1,
        cross_modal_align: Optional[str] = None,
        cross_modal_align_weight: Optional[float] = 0,
        automatic_optimization: Optional[bool] = True,
        accumulate_grad_batches: Optional[int] = None,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
        use_aug_optim: Optional[bool] = False,
        aug_loss_func: Optional[_Loss] = None,
        aug_lr: Optional[float] = None,
        aug_weight_decay: Optional[float] = None,
        aug_optim_type: Optional[str] = None,
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
        peft
            Whether to use efficient finetuning strategies. This will be helpful for fast finetuning of large backbones.
            We support options such as:

            - bit_fit (only finetune the bias terms)
            - norm_fit (only finetune the weights in norm layers / bias layer)
            - lora, lora_bias, lora_norm (only finetunes decomposition matrices inserted into model, in combination with either bit_fit or norm_fit)
            - ia3, ia3_bias, ia3_norm (adds vector that scales activations by learned vectors, in combination with either bit_fit or norm_fit)
            - None (do not use efficient finetuning strategies)
        track_grad_norm
            Track the p-norm of gradients during training. May be set to 'inf'  infinity-norm.
            If using Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.

        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "validation_metric",
                "test_metric",
                "loss_func",
                "model_postprocess_fn",
                "mixup_fn",
                "trainable_param_names",
                "custom_metric_func",
                "aug_loss_func",
            ]
        )
        self.model = model
        self.validation_metric = validation_metric
        self.validation_metric_name = f"val_{validation_metric_name}"
        self.loss_func = loss_func
        self.mixup_fn = mixup_fn
        if isinstance(validation_metric, BaseAggregator) and custom_metric_func is None:
            raise ValueError(
                f"validation_metric {validation_metric} is an aggregation metric,"
                "which must be used with a customized metric function."
            )
        self.custom_metric_func = custom_metric_func
        self.model_postprocess_fn = model_postprocess_fn
        self.trainable_param_names = trainable_param_names if trainable_param_names else []
        self.skip_final_val = skip_final_val
        self.automatic_optimization = automatic_optimization
        self.aug_loss_func = aug_loss_func
        if self.hparams.cross_modal_align:
            assert self.hparams.cross_modal_align_weight > 0
            logger.debug(f"Cross modal alignment mode: {self.hparams.cross_modal_align}")
            logger.debug(f"Cross modal alignment loss weight: {self.hparams.cross_modal_align_weight}")

    def _compute_template_loss(
        self,
        per_output: Dict,
        label: torch.Tensor,
    ):
        logits = per_output[TEMPLATE_LOGITS]
        choices_scores = per_output[LOGITS]
        lm_target = per_output[LM_TARGET]

        bs = lm_target.size(0)
        num_choices = lm_target.size(1)

        lm_loss = F.cross_entropy(
            logits[range(bs), label].flatten(0, 1),
            lm_target[range(bs), label].flatten(0, 1),
        )
        if self.model.mc_loss > 0:
            mc_loss = F.cross_entropy(choices_scores, label)
        else:
            mc_loss = 0.0

        if self.model.unlikely_loss > 0:
            cand_loglikely = -F.cross_entropy(logits.flatten(0, 2), lm_target.flatten(0, 2), reduction="none").view(
                bs, num_choices, -1
            )
            cand_loglikely += (lm_target < 0) * -100
            cand_loglikely[range(bs), label] = -100
            unlikely_loss = -torch.log(1 - torch.exp(cand_loglikely) + 1e-2).sum() / (cand_loglikely != -100).sum()
        else:
            unlikely_loss = 0.0

        return lm_loss + mc_loss * self.model.mc_loss + unlikely_loss * self.model.unlikely_loss

    def _compute_cross_modal_align_loss(self, multimodal_features):
        if self.hparams.cross_modal_align == "positive_only":
            kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss = 0
            num = 0
            for i in range(len(multimodal_features)):
                # input should be a distribution in the log space
                a = F.log_softmax(multimodal_features[i], dim=1)
                # kl divergence is not symmetric, so need to compute both (i, j) and (j, i)
                for j in range(len(multimodal_features)):
                    if i == j:
                        continue
                    # input should be a distribution in the log space
                    b = F.log_softmax(multimodal_features[j], dim=1)
                    loss += kl_loss(a, b)
                    num += 1
            return self.hparams.cross_modal_align_weight * loss / num
        else:
            raise ValueError(f"Unsupported cross modal alignment loss: {self.hparams.cross_modal_align}.")

    def _compute_loss(
        self,
        output: Dict,
        label: torch.Tensor,
    ):
        loss = 0
        for per_prefix, per_output in output.items():
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            if (
                TEMPLATE_LOGITS in per_output and self.model.prefix == T_FEW
            ):  # Do only add template loss if T-Few. #TODO Add compatibility to Fusion models.
                loss += self._compute_template_loss(per_output, label) * weight
            else:
                if self.training and self.hparams.use_aug_optim and per_prefix.startswith("fusion"):
                    label = label.tile((2,))
                loss += (
                    self.loss_func(
                        input=per_output[LOGITS].squeeze(dim=1),
                        target=label,
                    )
                    * weight
                )

        if self.hparams.cross_modal_align:
            loss += self._compute_cross_modal_align_loss(
                multimodal_features=output[self.model.prefix][MULTIMODAL_FEATURES]
            )

        if self.training and self.hparams.use_aug_optim:
            loss += self.aug_loss_func(
                pre_aug=output[self.model.prefix][MULTIMODAL_FEATURES_PRE_AUG],
                post_aug=output[self.model.prefix][MULTIMODAL_FEATURES_POST_AUG],
                vae_mean=output[self.model.prefix][VAE_MEAN],
                vae_var=output[self.model.prefix][VAE_VAR],
                ori_logits=output[self.model.prefix][ORI_LOGITS],
                aug_logits=output[self.model.prefix][AUG_LOGITS],
            )

        return loss

    def _compute_metric_score(
        self,
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        logits: torch.Tensor,
        label: torch.Tensor,
    ):
        if isinstance(
            metric,
            (
                torchmetrics.classification.BinaryAUROC,
                torchmetrics.classification.BinaryAveragePrecision,
                torchmetrics.classification.BinaryF1Score,
                Coverage,
            ),
        ):
            prob = F.softmax(logits.float(), dim=1)
            metric.update(preds=prob[:, 1], target=label)  # only for binary classification
        elif isinstance(metric, BaseAggregator):
            metric.update(custom_metric_func(logits, label))
        else:
            metric.update(logits.squeeze(dim=1).float(), label)

    def _shared_step(
        self,
        batch: Dict,
    ):
        label = batch[self.model.label_key]
        if self.mixup_fn is not None:
            self.mixup_fn.mixup_enabled = self.training & (self.current_epoch < self.hparams.mixup_off_epoch)
            batch, label = multimodel_mixup(batch=batch, model=self.model, mixup_fn=self.mixup_fn)
        output = run_model(self.model, batch)
        loss = self._compute_loss(output=output, label=label)
        return output, loss

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
        output, loss = self._shared_step(batch)

        if not self.automatic_optimization:
            if self.hparams.use_aug_optim:
                optimizer, aug_optimizer = self.optimizers()
            else:
                optimizer = self.optimizers()
                aug_optimizer = None

            lr_scheduler = self.lr_schedulers()
            loss = loss / self.hparams.accumulate_grad_batches
            self.manual_backward(loss)

            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0 or self.trainer.is_last_batch:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                if aug_optimizer is not None:
                    aug_optimizer.step()
                    aug_optimizer.zero_grad()

        self.log("train_loss", loss)
        return loss

    def on_validation_start(self) -> None:
        if self.skip_final_val and self.trainer.should_stop:
            self.log(
                self.validation_metric_name,
                self.validation_metric,
                on_step=False,
                on_epoch=True,
            )
            return None
        else:
            return super().on_validation_start()

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
        output, loss = self._shared_step(batch)
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Per prediction step. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#prediction-loop

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
        output = run_model(self.model, batch)
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)

        return output[self.model.prefix]

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
            exclude_keys=[
                "augmenter"
            ],  # exclude augmenter parameters from the optimizer as they would use an independent optimizer
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
                peft=self.hparams.peft,
                trainable_param_names=self.trainable_param_names,
                **kwargs,
            )
        else:
            logger.debug("applying single learning rate...")
            grouped_parameters = apply_single_lr(
                peft=self.hparams.peft,
                trainable_param_names=self.trainable_param_names,
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
            if isinstance(self.trainer.strategy, DeepSpeedStrategy):
                max_steps = 1
            else:
                accumulate_grad_batches = (
                    self.trainer.accumulate_grad_batches
                    if self.automatic_optimization
                    else self.hparams.accumulate_grad_batches
                )
                max_steps = (
                    len(self.trainer.datamodule.train_dataloader())
                    * self.trainer.max_epochs
                    // accumulate_grad_batches
                    // self.trainer.num_devices  # Account for distributed training
                )
                logger.debug(
                    f"len(trainer.datamodule.train_dataloader()): {len(self.trainer.datamodule.train_dataloader())}"
                )
                logger.debug(f"trainer.max_epochs: {self.trainer.max_epochs}")
                logger.debug(f"accumulate_grad_batches: {accumulate_grad_batches}")
                logger.debug(f"num_devices: {self.trainer.num_devices}")
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
        ret_optimizers = [optimizer]
        ret_schedulers = [sched]
        if self.hparams.use_aug_optim:
            logger.debug("initializing augment optimizer")
            # augmenter's optimizer
            aug_grouped_parameters = apply_single_lr(
                model=self.model.augmenter,
                lr=self.hparams.aug_lr,
                weight_decay=self.hparams.aug_weight_decay,
            )
            aug_optimizer = get_optimizer(
                optim_type=self.hparams.aug_optim_type,
                optimizer_grouped_parameters=aug_grouped_parameters,
                lr=self.hparams.aug_lr,
                weight_decay=self.hparams.aug_weight_decay,
            )
            ret_optimizers.append(aug_optimizer)

        logger.debug("done configuring optimizer and scheduler")
        return ret_optimizers, ret_schedulers

    def on_before_optimizer_step(self, optimizer):
        # If using mixed precision, the gradients are already unscaled here
        # TODO: apply gradient clip only to the target optimizer
        if not self.automatic_optimization and self.hparams.gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.hparams.gradient_clip_val,
                gradient_clip_algorithm=self.hparams.gradient_clip_algorithm,
            )
        if self.hparams.track_grad_norm != -1:
            self.log_dict(grad_norm(self, norm_type=self.hparams.track_grad_norm))
