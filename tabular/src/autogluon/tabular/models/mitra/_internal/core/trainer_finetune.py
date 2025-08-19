import time

import numpy as np
import torch
from loguru import logger
from sklearn.base import BaseEstimator

from ..._internal.config.config_run import ConfigRun
from ..._internal.config.enums import LossName, MetricName, ModelName, Task
from ..._internal.core.callbacks import Checkpoint, EarlyStopping
from ..._internal.core.get_loss import get_loss
from ..._internal.core.get_optimizer import GradScaler, get_optimizer
from ..._internal.core.get_scheduler import get_scheduler
from ..._internal.core.prediction_metrics import PredictionMetrics, PredictionMetricsTracker
from ..._internal.data.collator import CollatorWithPadding
from ..._internal.data.dataset_finetune import DatasetFinetune, DatasetFinetuneGenerator
from ..._internal.data.preprocessor import Preprocessor


class TrainerFinetune(BaseEstimator):
    def __init__(
        self,
        cfg: ConfigRun,
        model: torch.nn.Module,
        n_classes: int,
        device: str,
        rng: np.random.RandomState = None,
        verbose: bool = True,
    ):
        self.cfg = cfg
        if rng is None:
            rng = np.random.RandomState(self.cfg.seed)
        self.rng = rng
        self.verbose = verbose
        self.device = device
        self.model = model.to(self.device, non_blocking=True)
        self.n_classes = n_classes

        self.loss = get_loss(self.cfg)
        self.optimizer = get_optimizer(self.cfg.hyperparams, self.model)
        self.scheduler_warmup, self.scheduler_reduce_on_plateau = get_scheduler(self.cfg.hyperparams, self.optimizer)
        self.scaler = GradScaler(
            enabled=self.cfg.hyperparams["grad_scaler_enabled"],
            scale_init=self.cfg.hyperparams["grad_scaler_scale_init"],
            scale_min=self.cfg.hyperparams["grad_scaler_scale_min"],
            growth_interval=self.cfg.hyperparams["grad_scaler_growth_interval"],
            device=self.device,
        )

        self.early_stopping = EarlyStopping(patience=self.cfg.hyperparams["early_stopping_patience"])
        self.checkpoint = Checkpoint()
        self.preprocessor = Preprocessor(
            dim_embedding=self.cfg.hyperparams["dim_embedding"],
            n_classes=self.n_classes,
            dim_output=self.cfg.hyperparams["dim_output"],
            use_quantile_transformer=self.cfg.hyperparams["use_quantile_transformer"],
            use_feature_count_scaling=self.cfg.hyperparams["use_feature_count_scaling"],
            use_random_transforms=self.cfg.hyperparams["use_random_transforms"],
            shuffle_classes=self.cfg.hyperparams["shuffle_classes"],
            shuffle_features=self.cfg.hyperparams["shuffle_features"],
            random_mirror_x=self.cfg.hyperparams["random_mirror_x"],
            random_mirror_regression=self.cfg.hyperparams["random_mirror_regression"],
            task=self.cfg.task,
        )

        self.checkpoint.reset(self.model)

        if self.cfg.task == Task.REGRESSION and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY:
            self.bins = torch.linspace(-0.5, 1.5, self.cfg.hyperparams["dim_output"] + 1, device=cfg.device)
            self.bin_width = self.bins[1] - self.bins[0]

        self.metric = self.cfg.hyperparams["metric"]

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
        self.preprocessor.fit(x_train, y_train)

        x_train_transformed = self.preprocessor.transform_X(x_train)
        y_train_transformed = self.preprocessor.transform_y(y_train)

        dataset_train_generator = DatasetFinetuneGenerator(
            self.cfg,
            x=x_train_transformed,
            y=y_train_transformed,
            task=self.cfg.task,
            max_samples_support=self.cfg.hyperparams["max_samples_support"],
            max_samples_query=self.cfg.hyperparams["max_samples_query"],
            rng=self.rng,
        )

        self.checkpoint.reset(self.model)

        metrics_valid = self.evaluate(x_train, y_train, x_val, y_val)
        if self.verbose:
            self.log_start_metrics(metrics_valid)
        self.checkpoint(self.model, metrics_valid.loss)

        start_time = time.time()

        for epoch in range(1, self.cfg.hyperparams["max_epochs"] + 1):
            dataset_train = next(dataset_train_generator)
            loader_train = self.make_loader(dataset_train, training=True)
            self.model.train()

            prediction_metrics_tracker = PredictionMetricsTracker(task=self.cfg.task, preprocessor=self.preprocessor)

            for batch in loader_train:
                with torch.autocast(device_type=self.device, dtype=getattr(torch, self.cfg.hyperparams["precision"])):
                    x_support = batch["x_support"].to(self.device, non_blocking=True)
                    y_support = batch["y_support"].to(self.device, non_blocking=True)
                    x_query = batch["x_query"].to(self.device, non_blocking=True)
                    y_query = batch["y_query"].to(self.device, non_blocking=True)
                    padding_features = batch["padding_features"].to(self.device, non_blocking=True)
                    padding_obs_support = batch["padding_obs_support"].to(self.device, non_blocking=True)
                    padding_obs_query = batch["padding_obs_query"].to(self.device, non_blocking=True)

                    # Convert numerical y_support to bin ids
                    if (
                        self.cfg.task == Task.REGRESSION
                        and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY
                    ):
                        y_support = torch.bucketize(y_support, self.bins) - 1
                        y_support = torch.clamp(y_support, 0, self.cfg.hyperparams["dim_output"] - 1).to(torch.int64)
                        y_query_bin_ids = torch.bucketize(y_query, self.bins) - 1
                        y_query_bin_ids = torch.clamp(y_query_bin_ids, 0, self.cfg.hyperparams["dim_output"] - 1).to(
                            torch.int64
                        )

                    if self.cfg.model_name == ModelName.TABPFN:
                        y_hat = self.model(x_support, y_support, x_query, task=self.cfg.task).squeeze(-1)
                    elif self.cfg.model_name in [ModelName.TAB2D, ModelName.TAB2D_COL_ROW, ModelName.TAB2D_SDPA]:
                        y_hat = self.model(
                            x_support, y_support, x_query, padding_features, padding_obs_support, padding_obs_query
                        )

                    # Convert numerical y_query to bin ids
                    if (
                        self.cfg.task == Task.REGRESSION
                        and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY
                    ):
                        loss = self.loss(y_hat, y_query_bin_ids)
                    elif self.cfg.task == Task.CLASSIFICATION:
                        # for b in range(y_support.shape[0]):
                        #     unique_classes = len(torch.unique(torch.cat((y_support[b], y_query[b]))))
                        #     y_hat[b, :, unique_classes:] = 0
                        loss = self.loss(y_hat, y_query)
                    else:
                        loss = self.loss(y_hat, y_query)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Convert bin id predictions to numerical values
                if (
                    self.cfg.task == Task.REGRESSION
                    and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY
                ):
                    y_hat = torch.argmax(y_hat, dim=-1)
                    y_hat = self.bins[y_hat] + self.bin_width / 2

                y_hat = y_hat.float()
                if self.cfg.task == Task.REGRESSION:
                    prediction_metrics_tracker.update(y_hat, y_query, train=True)
                else:
                    prediction_metrics_tracker.update(y_hat, y_query, train=False)

            metrics_train = prediction_metrics_tracker.get_metrics()
            metrics_valid = self.evaluate(x_train, y_train, x_val, y_val)

            if self.verbose:
                self.log_metrics(epoch, metrics_train, metrics_valid)

            self.checkpoint(self.model, metrics_valid.loss)

            self.early_stopping(metrics_valid.metrics[self.metric])
            if self.early_stopping.we_should_stop():
                if self.verbose:
                    logger.info("Early stopping")
                break

            if (
                self.cfg.hyperparams["budget"] is not None
                and self.cfg.hyperparams["budget"] > 0
                and time.time() - start_time > self.cfg.hyperparams["budget"]
            ):
                logger.info("Time limit reached")
                break

            if epoch < self.cfg.hyperparams["warmup_steps"]:
                self.scheduler_warmup.step()
            else:
                self.scheduler_reduce_on_plateau.step(metrics_valid.loss)

        self.checkpoint.set_to_best(self.model)

    def evaluate(
        self, x_support: np.ndarray, y_support: np.ndarray, x_query: np.ndarray, y_query: np.ndarray
    ) -> PredictionMetrics:
        self.model.eval()

        x_support_transformed = self.preprocessor.transform_X(x_support)
        x_query_transformed = self.preprocessor.transform_X(x_query)
        y_support_transformed = self.preprocessor.transform_y(y_support)
        # y_query_transformed = self.preprocessor.transform_y(y_query)

        dataset = DatasetFinetune(
            self.cfg,
            x_support=x_support_transformed,
            y_support=y_support_transformed,
            x_query=x_query_transformed,
            y_query=y_query,
            max_samples_support=self.cfg.hyperparams["max_samples_support"],
            max_samples_query=self.cfg.hyperparams["max_samples_query"],
            rng=self.rng,
        )

        loader = self.make_loader(dataset, training=False)
        prediction_metrics_tracker = PredictionMetricsTracker(task=self.cfg.task, preprocessor=self.preprocessor)

        with torch.no_grad():
            for batch in loader:
                with torch.autocast(device_type=self.device, dtype=getattr(torch, self.cfg.hyperparams["precision"])):
                    x_s = batch["x_support"].to(self.device, non_blocking=True)
                    y_s = batch["y_support"].to(self.device, non_blocking=True)
                    x_q = batch["x_query"].to(self.device, non_blocking=True)
                    y_q = batch["y_query"].to(self.device, non_blocking=True)
                    padding_features = batch["padding_features"].to(self.device, non_blocking=True)
                    padding_obs_support = batch["padding_obs_support"].to(self.device, non_blocking=True)
                    padding_obs_query = batch["padding_obs_query"].to(self.device, non_blocking=True)

                    # Convert numerical y_support to bin ids
                    if (
                        self.cfg.task == Task.REGRESSION
                        and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY
                    ):
                        y_s = torch.bucketize(y_s, self.bins) - 1
                        y_s = torch.clamp(y_s, 0, self.cfg.hyperparams["dim_output"] - 1).to(torch.int64)

                    if self.cfg.model_name == ModelName.TABPFN:
                        y_hat = self.model(x_s, y_s, x_q, task=self.cfg.task).squeeze(-1)
                    elif self.cfg.model_name in [ModelName.TAB2D, ModelName.TAB2D_COL_ROW, ModelName.TAB2D_SDPA]:
                        y_hat = self.model(x_s, y_s, x_q, padding_features, padding_obs_support, padding_obs_query)

                # Convert bin id predictions to numerical values
                if (
                    self.cfg.task == Task.REGRESSION
                    and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY
                ):
                    y_hat = torch.argmax(y_hat, dim=-1)
                    y_hat = self.bins[y_hat] + self.bin_width / 2

                y_hat = y_hat.float()
                prediction_metrics_tracker.update(y_hat, y_q, train=False)

        metrics_eval = prediction_metrics_tracker.get_metrics()
        return metrics_eval

    def predict(self, x_support: np.ndarray, y_support: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        x_support_transformed = self.preprocessor.transform_X(x_support)
        x_query_transformed = self.preprocessor.transform_X(x_query)
        y_support_transformed = self.preprocessor.transform_y(y_support)

        dataset = DatasetFinetune(
            self.cfg,
            x_support=x_support_transformed,
            y_support=y_support_transformed,
            x_query=x_query_transformed,
            y_query=None,
            max_samples_support=self.cfg.hyperparams["max_samples_support"],
            max_samples_query=self.cfg.hyperparams["max_samples_query"],
            rng=self.rng,
        )

        loader = self.make_loader(dataset, training=False)
        self.model.eval()

        y_pred_list = []

        with torch.no_grad():
            for batch in loader:
                with torch.autocast(device_type=self.device, dtype=getattr(torch, self.cfg.hyperparams["precision"])):
                    x_s = batch["x_support"].to(self.device, non_blocking=True)
                    y_s = batch["y_support"].to(self.device, non_blocking=True)
                    x_q = batch["x_query"].to(self.device, non_blocking=True)
                    padding_features = batch["padding_features"].to(self.device, non_blocking=True)
                    padding_obs_support = batch["padding_obs_support"].to(self.device, non_blocking=True)
                    padding_obs_query = batch["padding_obs_query"].to(self.device, non_blocking=True)

                    # Convert numerical y_support to bin ids
                    if (
                        self.cfg.task == Task.REGRESSION
                        and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY
                    ):
                        y_s = torch.bucketize(y_s, self.bins) - 1
                        y_s = torch.clamp(y_s, 0, self.cfg.hyperparams["dim_output"] - 1).to(torch.int64)

                    if self.cfg.model_name == ModelName.TABPFN:
                        y_hat = self.model(x_s, y_s, x_q, task=self.cfg.task).squeeze(-1)
                    elif self.cfg.model_name in [ModelName.TAB2D, ModelName.TAB2D_COL_ROW, ModelName.TAB2D_SDPA]:
                        y_hat = self.model(x_s, y_s, x_q, padding_features, padding_obs_support, padding_obs_query)

                y_hat = y_hat[0].float().cpu().numpy()

                # Convert bin id predictions to numerical values
                if (
                    self.cfg.task == Task.REGRESSION
                    and self.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY
                ):
                    y_hat = np.argmax(y_hat, axis=-1)
                    y_hat = (self.bins[y_hat] + self.bin_width / 2).cpu().numpy()

                y_hat = self.preprocessor.inverse_transform_y(y_hat)
                y_pred_list.append(y_hat)

        y_pred = np.concatenate(y_pred_list, axis=0)

        return y_pred

    def load_params(self, path):
        self.model.load_state_dict(torch.load(path))

    def make_loader(self, dataset: torch.utils.data.Dataset, training: bool) -> torch.utils.data.DataLoader:
        if self.cfg.model_name == ModelName.TABPFN:
            pad_to_max_features = True
        elif self.cfg.model_name in [ModelName.TAB2D, ModelName.TAB2D_COL_ROW, ModelName.TAB2D_SDPA]:
            pad_to_max_features = False
        else:
            raise NotImplementedError(f"Model {self.cfg.model_name} not implemented")

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=training,
            pin_memory=True,
            num_workers=0,
            drop_last=False,
            collate_fn=CollatorWithPadding(
                max_features=self.cfg.hyperparams["dim_embedding"], pad_to_max_features=pad_to_max_features
            ),
        )

    def log_start_metrics(self, metrics_valid: PredictionMetrics):
        if self.cfg.task == Task.REGRESSION:
            logger.info(
                (
                    f"Epoch 000 "
                    f"| Train MSE: -.---- "
                    f"| Train MAE: -.---- "
                    f"| Train r2: -.---- "
                    f"| Val MSE: {metrics_valid.metrics[MetricName.MSE]:.4f} "
                    f"| Val MAE: {metrics_valid.metrics[MetricName.MAE]:.4f} "
                    f"| Val r2: {metrics_valid.metrics[MetricName.R2]:.4f}"
                )
            )

        elif self.cfg.task == Task.CLASSIFICATION:
            logger.info(
                (
                    f"Epoch 000 "
                    f"| Train CE: -.---- "
                    f"| Train acc: -.---- "
                    f"| Val CE: {metrics_valid.metrics[MetricName.LOG_LOSS]:.4f} "
                    f"| Val acc: {metrics_valid.metrics[MetricName.ACCURACY]:.4f}"
                )
            )

    def log_metrics(self, epoch: int, metrics_train: PredictionMetrics, metrics_valid: PredictionMetrics):
        if self.cfg.task == Task.REGRESSION:
            logger.info(
                (
                    f"Epoch {epoch:03d} "
                    f"| Train MSE: {metrics_train.metrics[MetricName.MSE]:.4f} "
                    f"| Train MAE: {metrics_train.metrics[MetricName.MAE]:.4f} "
                    f"| Train r2: {metrics_train.metrics[MetricName.R2]:.4f} "
                    f"| Val MSE: {metrics_valid.metrics[MetricName.MSE]:.4f} "
                    f"| Val MAE: {metrics_valid.metrics[MetricName.MAE]:.4f} "
                    f"| Val r2: {metrics_valid.metrics[MetricName.R2]:.4f}"
                )
            )
        elif self.cfg.task == Task.CLASSIFICATION:
            logger.info(
                (
                    f"Epoch {epoch:03d} "
                    f"| Train CE: {metrics_train.metrics[MetricName.LOG_LOSS]:.4f} "
                    f"| Train acc: {metrics_train.metrics[MetricName.ACCURACY]:.4f} "
                    f"| Val CE: {metrics_valid.metrics[MetricName.LOG_LOSS]:.4f} "
                    f"| Val acc: {metrics_valid.metrics[MetricName.ACCURACY]:.4f}"
                )
            )
