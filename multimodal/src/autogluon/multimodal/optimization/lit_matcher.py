import logging
from typing import Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn
from torch.nn.modules.loss import _Loss
from torchmetrics.aggregation import BaseAggregator

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

logger = logging.getLogger(AUTOMM)


class MatcherLitModule(pl.LightningModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: nn.Module,
        matches: List[DictConfig],
        match_label: Optional[int] = None,
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
        Parameters
        ----------
        model
            A Pytorch model
        matches
            A list of DictConfigs, each of which defines one pair of feature column match and the configs
            to compute the matching loss.
        match_label
            The label of match class.
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
        self.loss_func = loss_func
        if isinstance(validation_metric, BaseAggregator) and custom_metric_func is None:
            raise ValueError(
                f"validation_metric {validation_metric} is an aggregation metric,"
                f"which must be used with a customized metric function."
            )
        self.custom_metric_func = custom_metric_func
        assert len(matches) > 0
        self.matches = matches
        self.match_label = match_label
        self.reverse_prob = match_label == 0
        logger.debug(f"match num: {len(matches)}")
        logger.debug(f"match label: {match_label}")
        logger.debug(f"reverse probability: {self.reverse_prob}")
        for per_match in matches:
            logger.debug(f"per_match.pair[0]: {per_match.pair[0]}")
            logger.debug(f"per_match.pair[1]: {per_match.pair[1]}")
            # assert no duplicate column names
            if isinstance(per_match.pair[0], list):
                assert len(per_match.pair[0]) == len(set(per_match.pair[0]))
            if isinstance(per_match.pair[1], list):
                assert len(per_match.pair[1]) == len(set(per_match.pair[1]))

        self.metric_learning_loss_funcs = get_metric_learning_loss_funcs(matches)
        self.metric_learning_miner_funcs = get_metric_learning_miner_funcs(matches)

        # TODO: support validation metric on multiple matches
        # TODO: each match should use an independent torchmetric function
        assert (
            sum([per_match.use_label for per_match in matches]) == 1
        ), f"We only support one match to have labels currently."

    def _compute_loss(
        self,
        output: Dict,
        label: torch.Tensor,
    ):
        loss = 0
        for per_match, per_loss_func, per_miner_func in zip(
            self.matches,
            self.metric_learning_loss_funcs,
            self.metric_learning_miner_funcs,
        ):

            assert len(per_match.pair) == 2
            embeddings1 = gather_column_features(
                output=output,
                column_names=per_match.pair[0],
            )
            embeddings2 = gather_column_features(
                output=output,
                column_names=per_match.pair[1],
            )
            assert embeddings1.shape == embeddings2.shape
            embeddings = torch.cat([embeddings1, embeddings2], dim=0)  # (b*2, d)

            metric_learning_labels = generate_metric_learning_labels(
                num_samples=len(embeddings1),
                match_label=self.match_label if per_match.use_label else None,
                labels=label if per_match.use_label else None,
            )
            indices_tuple = per_miner_func(
                embeddings=embeddings,
                labels=metric_learning_labels,
            )
            per_loss = per_loss_func(
                embeddings=embeddings,
                labels=metric_learning_labels,
                indices_tuple=indices_tuple,
            )
            loss += per_loss * per_match.loss.weight

        return loss

    @staticmethod
    def _compute_metric_score(
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        label: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        embeddings1: Optional[torch.Tensor] = None,
        embeddings2: Optional[torch.Tensor] = None,
        reverse_prob: Optional[bool] = False,
    ):
        if logits is not None:
            if isinstance(metric, torchmetrics.AUROC):
                prob = compute_probability(logits=logits)
                metric.update(preds=prob, target=label)  # only for binary classification
            elif isinstance(metric, BaseAggregator):
                metric.update(custom_metric_func(logits, label))
            else:
                metric.update(logits.squeeze(dim=1), label)
        else:
            if isinstance(metric, BaseAggregator):
                metric.update(custom_metric_func(embeddings1, embeddings2, label))
            else:
                metric.update(
                    compute_probability(
                        embeddings1=embeddings1,
                        embeddings2=embeddings2,
                        reverse_prob=reverse_prob,
                    ),
                    label,
                )

    def _shared_step(
        self,
        batch: Dict,
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
        for per_match in self.matches:
            if per_match.use_label:
                embeddings1 = gather_column_features(
                    output=output,
                    column_names=per_match.pair[0],
                )
                embeddings2 = gather_column_features(
                    output=output,
                    column_names=per_match.pair[1],
                )
                self._compute_metric_score(
                    metric=self.validation_metric,
                    custom_metric_func=self.custom_metric_func,
                    embeddings1=embeddings1,
                    embeddings2=embeddings2,
                    label=batch[self.model.label_key],
                    reverse_prob=self.reverse_prob,
                )
                # TODO: support validation metric on multiple matches
                # TODO: each match should use an independent torchmetric function
                break

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
        output = self.model(batch)
        for per_match in self.matches:
            if per_match.use_label:
                embeddings1 = gather_column_features(
                    output=output,
                    column_names=per_match.pair[0],
                )
                embeddings2 = gather_column_features(
                    output=output,
                    column_names=per_match.pair[1],
                )
                match_prob = compute_probability(
                    embeddings1=embeddings1,
                    embeddings2=embeddings2,
                )
                if self.match_label == 0:
                    probability = torch.stack([match_prob, 1 - match_prob]).t()
                else:
                    probability = torch.stack([1 - match_prob, match_prob]).t()
                break

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
                efficient_finetune=self.hparams.efficient_finetune,
                trainable_param_names=self.hparams.trainable_param_names,
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
