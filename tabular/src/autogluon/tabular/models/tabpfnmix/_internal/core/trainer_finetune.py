from __future__ import annotations

import logging
import time

import einops
import numpy as np
import torch
from sklearn.base import BaseEstimator

from numpy.random import Generator

from autogluon.core.metrics import Scorer

from .callbacks import Checkpoint, EarlyStopping, TrackOutput
from .collator import CollatorWithPadding
from .enums import Task
from .get_loss import get_loss
from .get_optimizer import get_optimizer
from .get_scheduler import get_scheduler
from .y_transformer import create_y_transformer
from ..config.config_run import ConfigRun
from ..data.dataset_finetune import DatasetFinetune, DatasetFinetuneGenerator
from ..data.preprocessor import Preprocessor
from ..results.prediction_metrics import PredictionMetrics

logger = logging.getLogger(__name__)


class TrainerFinetune(BaseEstimator):
    def __init__(
        self,
        cfg: ConfigRun,
        model: torch.nn.Module,
        n_classes: int,
        stopping_metric: Scorer,
        use_best_epoch: bool = True,
        compute_train_metrics: bool = False,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.model.to(self.cfg.device)
        self.n_classes = n_classes

        self.loss = get_loss(self.cfg.task)
        self.optimizer = get_optimizer(self.cfg.hyperparams, self.model)
        self.scheduler = get_scheduler(self.cfg.hyperparams, self.optimizer)
        self.use_best_epoch = use_best_epoch
        self.compute_train_metrics = compute_train_metrics

        self.early_stopping = EarlyStopping(patience=self.cfg.hyperparams["early_stopping_patience"])
        self.preprocessor = Preprocessor(
            use_quantile_transformer=self.cfg.hyperparams["use_quantile_transformer"],
            use_feature_count_scaling=self.cfg.hyperparams["use_feature_count_scaling"],
            max_features=self.cfg.hyperparams["n_features"],
            task=self.cfg.task,
        )

        self.stopping_metric = stopping_metric
        self.best_epoch = None

    def set_random_seed(self) -> Generator:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        rng = np.random.default_rng(seed=self.cfg.seed)
        return rng

    def reset_optimizer(self):
        self.optimizer = get_optimizer(self.cfg.hyperparams, self.model)
        self.scheduler = get_scheduler(self.cfg.hyperparams, self.optimizer)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray = None,
        y_val: np.ndarray = None,
        time_limit: float = None,
    ):
        time_start = time.time()
        if self.optimizer is None:
            self.reset_optimizer()
        # FIXME: Figure out best way to seed model
        rng = self.set_random_seed()
        use_val = x_val is not None

        checkpoint = Checkpoint(save_best=self.use_best_epoch, in_memory=True)

        self.preprocessor.fit(x_train, y_train)
        x_train = self.preprocessor.transform(x_train)

        if use_val:
            x_val = self.preprocessor.transform(x_val)
        self.y_transformer = create_y_transformer(y_train, self.cfg.task)

        dataset_train_generator = DatasetFinetuneGenerator(
            self.cfg,
            x=x_train,
            y=self.y_transformer.transform(y_train),
            task=self.cfg.task,
            max_samples_support=self.cfg.hyperparams["max_samples_support"],
            max_samples_query=self.cfg.hyperparams["max_samples_query"],
            split=0.8,
            random_state=rng,
        )

        if use_val:
            dataset_valid = DatasetFinetune(
                self.cfg,
                x_support=x_train,
                y_support=self.y_transformer.transform(y_train),
                x_query=x_val,
                y_query=y_val,
                max_samples_support=self.cfg.hyperparams["max_samples_support"],
                max_samples_query=self.cfg.hyperparams["max_samples_query"],
            )
            loader_valid = self.make_loader(dataset_valid, training=False)
        else:
            loader_valid = None

        if use_val and self.use_best_epoch:
            checkpoint.reset()

        max_epochs = self.cfg.hyperparams["max_epochs"]

        epoch = 0
        if max_epochs != 0 and use_val:
            metrics_valid = self.test_epoch(loader_valid, y_val)

            log_msg = f"Epoch 000"
            if self.compute_train_metrics:
                log_msg += f" | Train error: -.---- | Train score: -.---- |"
            if metrics_valid is not None:
                log_msg += f" | Val error: {metrics_valid.loss:.4f} | Val score: {metrics_valid.score:.4f}"

            logger.log(15, log_msg)
            if self.use_best_epoch:
                checkpoint(self.model, metrics_valid.loss, epoch=0)

            if time_limit is not None:
                time_cur = time.time()
                time_elapsed = time_cur - time_start
                time_left = time_limit - time_elapsed
                if time_left < (time_elapsed * 3 + 3):
                    # Fine-tuning an epoch will take longer than this, so triple the time required
                    logger.log(15, "Early stopping due to running out of time...")
                    max_epochs = 0

        for epoch in range(1, max_epochs + 1):
            dataset_train = next(dataset_train_generator)
            loader_train = self.make_loader(dataset_train, training=True)

            metrics_train = self.train_epoch(loader_train, return_metrics=self.compute_train_metrics)
            if use_val:
                metrics_valid = self.test_epoch(loader_valid, y_val)
            else:
                metrics_valid = None

            log_msg = f"Epoch {epoch:03d}"
            if metrics_train is not None:
                log_msg += f" | Train error: {metrics_train.loss:.4f} | Train score: {metrics_train.score:.4f}"
            if metrics_valid is not None:
                log_msg += f" | Val error: {metrics_valid.loss:.4f} | Val score: {metrics_valid.score:.4f}"

            logger.log(15, log_msg)
            if metrics_valid is not None:
                if self.use_best_epoch:
                    checkpoint(self.model, metrics_valid.loss, epoch=epoch)

                self.early_stopping(metrics_valid.loss)
                if self.early_stopping.we_should_stop():
                    logger.info("Early stopping")
                    break
                self.scheduler.step(
                    metrics_valid.loss
                )  # TODO: Make scheduler work properly during refit with no val data, to mimic scheduler in OG fit

            if time_limit is not None:
                time_cur = time.time()
                time_elapsed = time_cur - time_start

                time_per_epoch = time_elapsed / epoch
                time_left = time_limit - time_elapsed
                if time_left < (time_per_epoch + 3):
                    logger.log(15, "Early stopping due to running out of time...")
                    break

        if use_val and self.use_best_epoch and checkpoint.best_model is not None:
            # TODO: Can do a trick: Skip saving and loading best epoch if best epoch is the final epoch, will save around ~0.5 seconds
            self.best_epoch = checkpoint.best_epoch
            self.model.load_state_dict(checkpoint.load())
        else:
            self.best_epoch = epoch

    def minimize_for_inference(self):
        # delete unnecessary objects for inference
        self.optimizer = None
        self.scheduler = None

    def train_epoch(
        self, dataloader: torch.utils.data.DataLoader, return_metrics: bool = False
    ) -> PredictionMetrics | None:
        """

        Parameters
        ----------
        dataloader
        return_metrics: bool = False
            If True, will calculate and return metrics on the train data.
            Note that this can slow down training speed by >10%.

        Returns
        -------

        """
        assert self.optimizer is not None
        self.model.train()

        if return_metrics:
            output_tracker = TrackOutput()
        else:
            output_tracker = None

        for batch in dataloader:
            self.optimizer.zero_grad()

            x_support = batch["x_support"].to(self.cfg.device)
            y_support = batch["y_support"].to(self.cfg.device)
            x_query = batch["x_query"].to(self.cfg.device)
            y_query = batch["y_query"].to(self.cfg.device)

            if self.cfg.task == Task.REGRESSION:
                x_support, y_support, x_query, y_query = (
                    x_support.float(),
                    y_support.float(),
                    x_query.float(),
                    y_query.float(),
                )

            y_hat = self.model(x_support, y_support, x_query)

            if self.cfg.task == Task.REGRESSION:
                y_hat = y_hat[0, :, 0]
            else:
                y_hat = y_hat[0, :, : self.n_classes]

            y_query = y_query[0, :]

            loss = self.loss(y_hat, y_query)
            loss.backward()
            self.optimizer.step()

            if return_metrics:
                output_tracker.update(einops.asnumpy(y_query), einops.asnumpy(y_hat))

        if return_metrics:
            y_true, y_pred = output_tracker.get()
            y_pred = self.y_transformer.inverse_transform(y_pred)
            prediction_metrics = PredictionMetrics.from_prediction(
                y_pred, y_true, self.cfg.task, metric=self.stopping_metric
            )
            return prediction_metrics
        else:
            return None

    def test_epoch(self, dataloader: torch.utils.data.DataLoader, y_test: np.ndarray) -> PredictionMetrics:
        # FIXME: test_epoch might be better if it uses the for loop logic with n_ensembles
        y_hat = self.predict_epoch(dataloader)
        y_hat_finish = self.y_transformer.inverse_transform(y_hat)

        prediction_metrics = PredictionMetrics.from_prediction(
            y_hat_finish, y_test, self.cfg.task, metric=self.stopping_metric
        )
        return prediction_metrics

    def _get_memory_size(self) -> int:
        import gc
        import sys
        import pickle

        gc.collect()  # Try to avoid OOM error
        return sys.getsizeof(pickle.dumps(self, protocol=4))

    def predict(self, x_support: np.ndarray, y_support: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        """
        Give a prediction for the query set.
        """

        x_support = self.preprocessor.transform(x_support)
        x_query = self.preprocessor.transform(x_query)

        dataset = DatasetFinetune(
            self.cfg,
            x_support=x_support,
            y_support=self.y_transformer.transform(y_support),
            x_query=x_query,
            y_query=None,
            max_samples_support=self.cfg.hyperparams["max_samples_support"],
            max_samples_query=self.cfg.hyperparams["max_samples_query"],
        )

        loader = self.make_loader(dataset, training=False)

        y_hat_ensembles = []

        for _ in range(self.cfg.hyperparams["n_ensembles"]):
            y_hat = self.predict_epoch(loader)
            y_hat_ensembles.append(y_hat)

        y_hat_ensembled = sum(y_hat_ensembles) / self.cfg.hyperparams["n_ensembles"]
        y_hat_finish = self.y_transformer.inverse_transform(y_hat_ensembled)

        return y_hat_finish

    def predict_epoch(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Returns the predictions for the data in the dataloader.
        The predictions are in the original state as they come from the model, i.e. not transformed.
        """

        self.model.eval()

        y_hat_list = []

        with torch.no_grad():
            for batch in dataloader:
                x_support = batch["x_support"].to(self.cfg.device)
                y_support = batch["y_support"].to(self.cfg.device)
                x_query = batch["x_query"].to(self.cfg.device)

                if self.cfg.task == Task.REGRESSION:
                    y_support = y_support.float()

                y_hat = self.model(x_support, y_support, x_query)

                if self.cfg.task == Task.REGRESSION:
                    y_hat = y_hat[0, :, 0]
                else:
                    y_hat = y_hat[0, :, : self.n_classes]

                y_hat_list.append(einops.asnumpy(y_hat))

        y_hat = np.concatenate(y_hat_list, axis=0)
        return y_hat

    def make_loader(self, dataset, training):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=training,
            pin_memory=True,
            num_workers=0,
            drop_last=False,
            collate_fn=CollatorWithPadding(pad_to_n_support_samples=None),
        )
