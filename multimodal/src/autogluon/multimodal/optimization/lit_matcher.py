import logging
from typing import Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn
from torch.nn.modules.loss import _Loss
from torchmetrics.aggregation import BaseAggregator

from ..constants import AUTOMM, FEATURES, LOGIT_SCALE, PROBABILITY, QUERY, RESPONSE
from ..models.utils import run_model
from ..utils.matcher import compute_matching_probability
from .losses import MultiNegativesSoftmaxLoss
from .utils import (
    CustomHitRate,
    apply_layerwise_lr_decay,
    apply_single_lr,
    apply_two_stages_lr,
    generate_metric_learning_labels,
    get_lr_scheduler,
    get_optimizer,
)

logger = logging.getLogger(AUTOMM)


class MatcherLitModule(pl.LightningModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        query_model: nn.Module,
        response_model: nn.Module,
        signature: Optional[str] = None,
        match_label: Optional[int] = None,
        matches: Optional[List[DictConfig]] = None,
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
        miner_func: Optional[_Loss] = None,
        validation_metric: Optional[torchmetrics.Metric] = None,
        validation_metric_name: Optional[str] = None,
        custom_metric_func: Callable = None,
        test_metric: Optional[torchmetrics.Metric] = None,
    ):
        """
        Parameters
        ----------
        query_model
            The query model.
        response_model
            The response model.
        signature
            query or response.
        match_label
            The label of match class.
        matches
            A list of DictConfigs, each of which defines one pair of feature column match and the configs
            to compute the matching loss.
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
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "query_model",
                "response_model",
                "validation_metric",
                "test_metric",
                "loss_func",
                "miner_func",
                "matches",
            ]
        )
        self.query_model = query_model
        self.response_model = response_model
        if signature:
            assert signature in [QUERY, RESPONSE]
        self.signature = signature
        self.validation_metric = validation_metric
        self.validation_metric_name = f"val_{validation_metric_name}"

        if isinstance(validation_metric, BaseAggregator) and custom_metric_func is None:
            raise ValueError(
                f"validation_metric {validation_metric} is an aggregation metric,"
                f"which must be used with a customized metric function."
            )
        self.custom_metric_func = custom_metric_func

        self.matches = matches
        self.match_label = match_label
        self.reverse_prob = match_label == 0

        logger.debug(f"match label: {match_label}")
        logger.debug(f"reverse probability: {self.reverse_prob}")

        self.loss_func = loss_func
        self.miner_func = miner_func

    def _compute_loss(
        self,
        query_embeddings: torch.Tensor,
        response_embeddings: torch.Tensor,
        label: torch.Tensor,
        logit_scale: Optional[torch.tensor] = None,
    ):
        assert query_embeddings.shape == response_embeddings.shape

        if isinstance(self.loss_func, MultiNegativesSoftmaxLoss):
            loss = self.loss_func(
                features_a=query_embeddings,
                features_b=response_embeddings,
                logit_scale=logit_scale,
                rank=self.global_rank,
                world_size=self.trainer.world_size,
            )
        else:
            embeddings = torch.cat([query_embeddings, response_embeddings], dim=0)  # (b*2, d)

            metric_learning_labels = generate_metric_learning_labels(
                num_samples=len(query_embeddings),
                match_label=self.match_label,
                labels=label,
            )
            indices_tuple = self.miner_func(
                embeddings=embeddings,
                labels=metric_learning_labels,
            )
            loss = self.loss_func(
                embeddings=embeddings,
                labels=metric_learning_labels,
                indices_tuple=indices_tuple,
            )

        return loss

    @staticmethod
    def _compute_metric_score(
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        label: torch.Tensor,
        query_embeddings: torch.Tensor,
        response_embeddings: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        reverse_prob: Optional[bool] = False,
    ):

        if isinstance(metric, BaseAggregator):
            metric.update(custom_metric_func(query_embeddings, response_embeddings, label))
        elif isinstance(metric, CustomHitRate):
            metric.update(
                batch_query_embeds=query_embeddings.cpu(),
                batch_response_embeds=response_embeddings.cpu(),
                logit_scale=logit_scale.cpu() if logit_scale else None,
            )
        else:
            metric.update(
                compute_matching_probability(
                    embeddings1=query_embeddings,
                    embeddings2=response_embeddings,
                    reverse_prob=reverse_prob,
                ),
                label,
            )

    def _get_label(self, batch: Dict):
        label = None
        if self.response_model.label_key in batch:
            label = batch[self.response_model.label_key]
        return label

    def _shared_step(
        self,
        batch: Dict,
    ):
        query_outputs = run_model(self.query_model, batch)[self.query_model.prefix]
        query_embeddings = query_outputs[FEATURES]

        response_outputs = run_model(self.response_model, batch)[self.response_model.prefix]
        response_embeddings = response_outputs[FEATURES]

        logit_scale = (response_outputs[LOGIT_SCALE] if LOGIT_SCALE in response_outputs else None,)

        if isinstance(logit_scale, tuple):
            logit_scale = logit_scale[0]

        loss = self._compute_loss(
            query_embeddings=query_embeddings,
            response_embeddings=response_embeddings,
            label=self._get_label(batch),
            logit_scale=logit_scale,
        )
        return query_embeddings, response_embeddings, logit_scale, loss

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
        _, _, _, loss = self._shared_step(batch)
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
        query_embeddings, response_embeddings, logit_scale, loss = self._shared_step(batch)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)

        self._compute_metric_score(
            metric=self.validation_metric,
            custom_metric_func=self.custom_metric_func,
            query_embeddings=query_embeddings,
            response_embeddings=response_embeddings,
            label=self._get_label(batch),
            logit_scale=logit_scale,
            reverse_prob=self.reverse_prob,
        )

        self.log(
            self.validation_metric_name,
            self.validation_metric,
            on_step=False,
            on_epoch=True,
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
        if self.signature == QUERY:
            embeddings = run_model(self.query_model, batch)[self.query_model.prefix][FEATURES]
            return {FEATURES: embeddings}
        elif self.signature == RESPONSE:
            embeddings = run_model(self.response_model, batch)[self.response_model.prefix][FEATURES]
            return {FEATURES: embeddings}
        else:
            query_embeddings = run_model(self.query_model, batch)[self.query_model.prefix][FEATURES]
            response_embeddings = run_model(self.response_model, batch)[self.response_model.prefix][FEATURES]

        match_prob = compute_matching_probability(
            embeddings1=query_embeddings,
            embeddings2=response_embeddings,
        )
        if self.match_label == 0:
            probability = torch.stack([match_prob, 1 - match_prob]).t()
        else:
            probability = torch.stack([1 - match_prob, match_prob]).t()

        return {PROBABILITY: probability}

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
        # TODO: need to consider query_model and response_model in the optimizer
        # TODO: how to avoid pass one parameter multiple times in the optimizer?
        # TODO: in the late-fusion siamese setting, one shared parameter may have different layer ids in the query and reponse models.
        kwargs = dict(
            model=self.query_model,
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
