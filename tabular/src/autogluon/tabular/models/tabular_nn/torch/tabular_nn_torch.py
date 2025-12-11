from __future__ import annotations

import io
import json
import logging
import os
import random
import time
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Union

import numpy as np
import pandas as pd

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT, S_TEXT_AS_CATEGORY, S_TEXT_NGRAM
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_torch
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS
from autogluon.core.hpo.constants import RAY_BACKEND
from autogluon.core.metrics import Scorer
from autogluon.core.models._utils import get_early_stopping_rounds
from autogluon.core.models.abstract.abstract_nn_model import AbstractNeuralNetworkModel
from autogluon.core.utils.early_stopping import AdaptiveES, NoES, SimpleES
from autogluon.core.utils.exceptions import TimeLimitExceeded

from ..compilers.native import TabularNeuralNetTorchNativeCompiler
from ..compilers.onnx import TabularNeuralNetTorchOnnxCompiler
from ..hyperparameters.parameters import get_default_param
from ..hyperparameters.searchspaces import get_default_searchspace
from ..utils.data_preprocessor import create_preprocessor, get_feature_arraycol_map, get_feature_type_map
from ..utils.nn_architecture_utils import infer_y_range

if TYPE_CHECKING:
    from .tabular_torch_dataset import TabularTorchDataset

logger = logging.getLogger(__name__)


# TODO: QuantileTransformer in pipelines accounts for majority of online inference time
class TabularNeuralNetTorchModel(AbstractNeuralNetworkModel):
    """
    PyTorch neural network models for classification/regression with tabular data.

    Extra hyperparameter options:
        ag.early_stop : int | str, default = "default"
            Specifies the early stopping rounds. Defaults to an adaptive strategy. Recommended to keep default.
    """
    ag_key = "NN_TORCH"
    ag_name = "NeuralNetTorch"
    ag_priority = 25
    seed_name = "seed_value"

    # Constants used throughout this class:
    unique_category_str = np.nan  # string used to represent missing values and unknown categories for categorical features.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_arraycol_map = None
        self.feature_type_map = None
        self.features_to_drop = []  # may change between different bagging folds. TODO: consider just removing these from self._features_internal
        self.processor = None  # data processor
        self.num_dataloading_workers = None
        self._architecture_desc = None
        self.optimizer = None
        self.device = None
        self.max_batch_size = None
        self._num_cpus_infer = None

    def _set_default_params(self):
        """Specifies hyperparameter values to use by default"""
        default_params = get_default_param(problem_type=self.problem_type, framework="pytorch")
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_default_searchspace(self):
        return get_default_searchspace(problem_type=self.problem_type, framework="pytorch")

    def _get_num_net_outputs(self):
        if self.problem_type in [MULTICLASS, SOFTCLASS]:
            return self.num_classes
        elif self.problem_type == BINARY:
            return 2
        elif self.problem_type == REGRESSION:
            return 1
        elif self.problem_type == QUANTILE:
            return len(self.quantile_levels)
        else:
            raise ValueError(f"Unknown problem_type: {self.problem_type}")

    def _get_device(self, num_gpus):
        import torch

        if num_gpus is not None and num_gpus >= 1:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.log(15, "Training on GPU (CUDA)")
                if num_gpus > 1:
                    logger.warning(f"{self.__class__.__name__} not yet able to use more than 1 GPU. 'num_gpus' is set to >1, but we will be using only 1 GPU.")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.log(15, "Training on GPU (MPS - Apple Silicon)")
                if num_gpus > 1:
                    logger.warning(f"{self.__class__.__name__} on Apple Silicon can only use 1 GPU (MPS). 'num_gpus' is set to >1, but we will be using only 1 GPU.")
            else:
                device = torch.device("cpu")
                logger.log(15, "Training on CPU")
        else:
            device = torch.device("cpu")
            logger.log(15, "Training on CPU")
        return device

    def _set_net_defaults(self, train_dataset, params):
        params = params.copy()
        y_range_extend = params.pop("y_range_extend", None)
        """ Sets dataset-adaptive default values to use for our neural network """
        if self.problem_type in [REGRESSION, QUANTILE]:
            if params["y_range"] is None:
                params["y_range"] = infer_y_range(y_vals=train_dataset.data_list[train_dataset.label_index], y_range_extend=y_range_extend)
        return params

    def _get_default_loss_function(self):
        import torch

        if self.problem_type == REGRESSION:
            return torch.nn.L1Loss()  # or torch.nn.MSELoss()
        elif self.problem_type in [BINARY, MULTICLASS]:
            return torch.nn.CrossEntropyLoss()
        elif self.problem_type == SOFTCLASS:
            return torch.nn.KLDivLoss()  # compares log-probability prediction vs probability target.

    @staticmethod
    def _prepare_params(params):
        params = params.copy()

        processor_param_keys = {"proc.embed_min_categories", "proc.impute_strategy", "proc.max_category_levels", "proc.skew_threshold", "use_ngram_features"}
        processor_kwargs = {k: v for k, v in params.items() if k in processor_param_keys}
        for key in processor_param_keys:
            params.pop(key, None)

        optimizer_param_keys = {"optimizer", "learning_rate", "weight_decay"}
        optimizer_kwargs = {k: v for k, v in params.items() if k in optimizer_param_keys}
        for key in optimizer_param_keys:
            params.pop(key, None)

        fit_param_keys = {"num_epochs", "epochs_wo_improve"}
        fit_kwargs = {k: v for k, v in params.items() if k in fit_param_keys}
        for key in fit_param_keys:
            params.pop(key, None)

        loss_param_keys = {"loss_function", "gamma"}
        loss_kwargs = {k: v for k, v in params.items() if k in loss_param_keys}
        for key in loss_param_keys:
            params.pop(key, None)

        return processor_kwargs, optimizer_kwargs, fit_kwargs, loss_kwargs, params

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        time_limit: float = None,
        sample_weight=None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        reporter=None,
        verbosity: int = 2,
        **kwargs,
    ):
        try_import_torch()
        import torch

        torch.set_num_threads(num_cpus)

        start_time = time.time()

        params = self._get_model_params()

        processor_kwargs, optimizer_kwargs, fit_kwargs, loss_kwargs, params = self._prepare_params(params=params)

        seed_value = params.pop(self.seed_name, self.default_random_seed)

        self._num_cpus_infer = params.pop("_num_cpus_infer", 1)
        if seed_value is not None:  # Set seeds
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)

        if sample_weight is not None:  # TODO: support # @Innixma
            logger.log(15, f"sample_weight not yet supported for {self.__class__.__name__}," " this model will ignore them in training.")

        if num_cpus is not None:
            self.num_dataloading_workers = max(1, int(num_cpus / 2.0))
        else:
            self.num_dataloading_workers = 1
        import platform

        if self.num_dataloading_workers == 1:
            self.num_dataloading_workers = 0  # TODO: verify 0 is typically faster and uses less memory than 1 in pytorch
        if platform.system() == "Darwin":
            self.num_dataloading_workers = 0  # TODO: >0 crashes on MacOS
        self.max_batch_size = params.pop("max_batch_size", 512)

        train_dataset = self._generate_dataset(X=X, y=y, train_params=processor_kwargs, is_train=True)
        if X_val is not None and y_val is not None:
            val_dataset = self._generate_dataset(X=X_val, y=y_val)
        else:
            val_dataset = None
        if X_test is not None and y_test is not None:
            test_dataset = self._generate_dataset(X=X_test, y=y_test)
        else:
            test_dataset = None

        batch_size = params.pop("batch_size", None)
        if batch_size is None:
            batch_size = min(int(2 ** (3 + np.floor(np.log10(len(X))))), self.max_batch_size, len(X))

        logger.log(
            15,
            f"Training data for {self.__class__.__name__} has: "
            f"{train_dataset.num_examples} examples, {train_dataset.num_features} features "
            f"({len(train_dataset.feature_groups['vector'])} vector, {len(train_dataset.feature_groups['embed'])} embedding)",
        )

        self.device = self._get_device(num_gpus=num_gpus)

        self._get_net(train_dataset, params=params)
        self.optimizer = self._init_optimizer(**optimizer_kwargs)

        if time_limit is not None:
            time_elapsed = time.time() - start_time
            time_limit_orig = time_limit
            time_limit = time_limit - time_elapsed

            # if 60% of time was spent preprocessing, likely not enough time to train model
            if time_limit <= time_limit_orig * 0.4:
                raise TimeLimitExceeded

        # train network
        self._train_net(
            train_dataset=train_dataset,
            loss_kwargs=loss_kwargs,
            batch_size=batch_size,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            time_limit=time_limit,
            reporter=reporter,
            verbosity=verbosity,
            **fit_kwargs,
        )

    def _get_net(self, train_dataset, params):
        from .torch_network_modules import EmbedNet

        # set network params
        params = self._set_net_defaults(train_dataset, params)
        self.model = EmbedNet(
            problem_type=self.problem_type,
            num_net_outputs=self._get_num_net_outputs(),
            quantile_levels=self.quantile_levels,
            train_dataset=train_dataset,
            device=self.device,
            **params,
        )
        self.model = self.model.to(self.device)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _train_net(
        self,
        train_dataset: TabularTorchDataset,
        loss_kwargs: dict,
        batch_size: int,
        num_epochs: int,
        epochs_wo_improve: int,
        val_dataset: TabularTorchDataset = None,
        test_dataset: TabularTorchDataset = None,
        time_limit: float = None,
        reporter=None,
        verbosity: int = 2,
    ):
        import torch

        start_time = time.time()
        logging.debug("initializing neural network...")
        self.model.init_params()
        logging.debug("initialized")
        train_dataloader = train_dataset.build_loader(batch_size, self.num_dataloading_workers, is_test=False)

        if isinstance(loss_kwargs.get("loss_function", "auto"), str) and loss_kwargs.get("loss_function", "auto") == "auto":
            loss_kwargs["loss_function"] = self._get_default_loss_function()
        if epochs_wo_improve is not None:
            early_stopping_method = SimpleES(patience=epochs_wo_improve)
        else:
            early_stopping_method = self._get_early_stopping_strategy(num_rows_train=len(train_dataset))

        ag_params = self._get_ag_params()
        generate_curves = ag_params.get("generate_curves", False)

        if generate_curves:
            scorers = ag_params.get("curve_metrics", [self.eval_metric])
            use_curve_metric_error = ag_params.get("use_error_for_curve_metrics", False)
            metric_names = [scorer.name for scorer in scorers]

            train_curves = {metric.name: [] for metric in scorers}
            val_curves = {metric.name: [] for metric in scorers}
            test_curves = {metric.name: [] for metric in scorers}

            # make copy of train_dataset to avoid interfering with train_dataloader
            curve_train_dataset = deepcopy(train_dataset)
            y_train = curve_train_dataset.get_labels()
            if y_train.ndim == 2 and y_train.shape[1] == 1:
                y_train = y_train.flatten()

            if test_dataset is not None:
                y_test = test_dataset.get_labels()
                if y_test.ndim == 2 and y_test.shape[1] == 1:
                    y_test = y_test.flatten()
            else:
                y_test = None

        if val_dataset is not None:
            y_val = val_dataset.get_labels()
            if y_val.ndim == 2 and y_val.shape[1] == 1:
                y_val = y_val.flatten()
        else:
            y_val = None

        if verbosity <= 1:
            verbose_eval = False
        else:
            verbose_eval = True

        logger.log(15, "Neural network architecture:")
        logger.log(15, str(self.model))

        io_buffer = None
        if num_epochs == 0:
            # use dummy training loop that stops immediately
            # useful for using NN just for data preprocessing / debugging
            logger.log(20, "Not training Tabular Neural Network since num_updates == 0")

            # for each batch
            for batch_idx, data_batch in enumerate(train_dataloader):
                if batch_idx > 0:
                    break
                loss = self.model.compute_loss(data_batch, **loss_kwargs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return

        # start training loop:
        logger.log(15, f"Training tabular neural network for up to {num_epochs} epochs...")
        total_updates = 0
        num_updates_per_epoch = max(round(len(train_dataset) / batch_size) + 1, 1)
        update_to_check_time = min(10, max(1, int(num_updates_per_epoch / 5)))
        do_update = True
        epoch = 0
        best_epoch = 0
        best_val_metric = -np.inf  # higher = better
        best_val_update = 0
        start_fit_time = time.time()
        if time_limit is not None:
            time_limit = time_limit - (start_fit_time - start_time)
            if time_limit <= 0:
                raise TimeLimitExceeded
        while do_update:
            time_start_epoch = time.time()
            time_cur = time_start_epoch
            total_train_loss = 0.0
            total_train_size = 0.0
            for batch_idx, data_batch in enumerate(train_dataloader):
                # forward
                loss = self.model.compute_loss(data_batch, **loss_kwargs)
                total_train_loss += loss.item()
                total_train_size += 1

                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_updates += 1

                # time limit
                if time_limit is not None:
                    time_cur_tmp = time.time()
                    time_elapsed_batch = time_cur_tmp - time_cur
                    time_cur = time_cur_tmp
                    update_cur = batch_idx + 1
                    if epoch == 0 and update_cur == update_to_check_time:
                        time_elapsed_epoch = time_cur - time_start_epoch

                        # v1 estimate is sensitive to fixed cost overhead at the start of training, such as torch initialization.
                        # v2 fixes this, but we keep both and take the min to avoid potential cases where v2 is inaccurate due to an overly slow batch.
                        estimated_time_v1 = time_elapsed_epoch / update_cur * num_updates_per_epoch  # Less accurate than v2, but never underestimates time
                        estimated_time_v2 = time_elapsed_epoch + time_elapsed_batch * (num_updates_per_epoch - update_cur)  # Less likely to overestimate time
                        estimated_time = min(estimated_time_v1, estimated_time_v2)
                        if estimated_time > time_limit:
                            logger.log(
                                30,
                                f"\tNot enough time to train first epoch. " f"(Time Required: {round(estimated_time, 2)}s, Time Left: {round(time_limit, 2)}s)",
                            )
                            raise TimeLimitExceeded
                    time_elapsed = time_cur - start_fit_time
                    if time_limit < time_elapsed:
                        if epoch == 0:
                            logger.log(
                                30,
                                f"\tNot enough time to train first epoch. Stopped on Update {total_updates} (Epoch {epoch}))",
                            )
                            raise TimeLimitExceeded
                        logger.log(15, f"\tRan out of time, stopping training early. (Stopped on Update {total_updates} (Epoch {epoch}))")
                        do_update = False
                        break

            if not do_update:
                break

            epoch += 1

            # learning curve generation
            if generate_curves:
                stop = self._generate_curves(
                    train_curves=train_curves,
                    val_curves=val_curves,
                    test_curves=test_curves,
                    scorers=scorers,
                    best_epoch=best_epoch,
                    use_curve_metric_error=use_curve_metric_error,
                    train_dataset=curve_train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                )

                if stop:
                    break

            # validation
            if val_dataset is not None:
                is_best = False
                # compute validation score
                val_metric = self.score(X=val_dataset, y=y_val, metric=self.stopping_metric, _reset_threads=False)
                if not self._assert_valid_metric(metric=val_metric, best_epoch=best_epoch):
                    break

                # update best validation
                if (val_metric >= best_val_metric) or best_epoch == 0:
                    if val_metric > best_val_metric:
                        is_best = True
                    best_val_metric = val_metric
                    io_buffer = io.BytesIO()
                    torch.save(self.model, io_buffer)  # nosec B614
                    best_epoch = epoch
                    best_val_update = total_updates
                early_stop = early_stopping_method.update(cur_round=epoch, is_best=is_best)
                if verbose_eval:
                    logger.log(
                        15,
                        f"Epoch {epoch} (Update {total_updates}).\t"
                        f"Train loss: {round(total_train_loss / total_train_size, 4)}, "
                        f"Val {self.stopping_metric.name}: {round(val_metric, 4)}, "
                        f"Best Epoch: {best_epoch}",
                    )

                if reporter is not None:
                    reporter(
                        epoch=total_updates,
                        validation_performance=val_metric,  # Higher val_metric = better
                        train_loss=total_train_loss / total_train_size,
                        eval_metric=self.eval_metric.name,
                        greater_is_better=self.eval_metric.greater_is_better,
                    )

                # no improvement
                if early_stop:
                    break

            if epoch >= num_epochs:
                break

            if time_limit is not None:
                time_elapsed = time.time() - start_fit_time
                time_epoch_average = time_elapsed / epoch
                time_left = time_limit - time_elapsed
                if time_left < time_epoch_average:
                    logger.log(20, f"\tRan out of time, stopping training early. (Stopping on epoch {epoch})")
                    break

        if epoch == 0:
            raise AssertionError("0 epochs trained!")

        if generate_curves:
            curves = {"train": train_curves}
            if val_dataset is not None:
                curves["val"] = val_curves
            if test_dataset is not None:
                curves["test"] = test_curves
            self.save_learning_curves(metrics=metric_names, curves=curves)

        # revert back to best model
        if val_dataset is not None:
            logger.log(15, f"Best model found on Epoch {best_epoch} (Update {best_val_update}). Val {self.stopping_metric.name}: {best_val_metric}")
            if io_buffer is not None:
                io_buffer.seek(0)
                self.model = torch.load(io_buffer, weights_only=False)  # nosec B614
        else:
            logger.log(15, f"Best model found on Epoch {best_epoch} (Update {best_val_update}).")
        self.params_trained["batch_size"] = batch_size
        self.params_trained["num_epochs"] = best_epoch

    def _get_early_stopping_strategy(self, num_rows_train: int):
        ag_early_stop = self._get_ag_params().get("early_stop", "default")
        if ag_early_stop is None:
            early_stopping_method = NoES()
        elif isinstance(ag_early_stop, str) and ag_early_stop == "default":
            early_stopping_method = self._get_early_stop_default()
        elif isinstance(ag_early_stop, (str, tuple, list)):
            early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=ag_early_stop)
            early_stopping_method = early_stopping_rounds[0](**early_stopping_rounds[1])
        elif isinstance(ag_early_stop, int):
            early_stopping_method = SimpleES(patience=ag_early_stop)
        else:
            raise ValueError(f"Invalid `ag.early_stop` value specified: `{ag_early_stop}`")
        return early_stopping_method

    def _get_early_stop_default(self):
        return AdaptiveES(adaptive_rate=0.5, adaptive_offset=20)

    def _get_early_stopping_rounds(self, num_rows_train, strategy="auto"):
        return get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=strategy)

    def _generate_curves(
        self,
        train_curves: dict,
        val_curves: dict,
        test_curves: dict,
        scorers: list[Scorer],
        best_epoch: int,
        use_curve_metric_error: bool,
        train_dataset: "TabularTorchDataset",
        val_dataset: "TabularTorchDataset",
        test_dataset: "TabularTorchDataset",
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> bool:
        """
        Extends learning curve dictionaries across all metrics listed in scorers by one epoch.

        Returns:
        --------
        bool:
            Whether to break out of the neural net training loop.
        """
        train_metrics = []
        val_metrics = []
        test_metrics = []

        for metric in scorers:
            train_metrics.append(self.score(X=train_dataset, y=y_train, metric=metric, _reset_threads=False))
            val_metrics += [self.score(X=val_dataset, y=y_val, metric=metric, _reset_threads=False)] if val_dataset is not None else []
            test_metrics += [self.score(X=test_dataset, y=y_test, metric=metric, _reset_threads=False)] if test_dataset is not None else []

            if use_curve_metric_error:
                train_metrics[-1] = metric.convert_score_to_error(train_metrics[-1])
                if val_dataset is not None:
                    val_metrics[-1] = metric.convert_score_to_error(val_metrics[-1])
                if test_dataset is not None:
                    test_metrics[-1] = metric.convert_score_to_error(test_metrics[-1])

            if (
                not self._assert_valid_metric(metric=train_metrics[-1], best_epoch=best_epoch)
                or (val_dataset is not None and not self._assert_valid_metric(metric=val_metrics[-1], best_epoch=best_epoch))
                or (test_dataset is not None and not self._assert_valid_metric(metric=test_metrics[-1], best_epoch=best_epoch))
            ):
                return True

        # update learning curve
        for i, metric in enumerate(scorers):
            train_curves[metric.name].append(float(train_metrics[i]))
            val_curves[metric.name] += [float(val_metrics[i])] if val_dataset is not None else []
            test_curves[metric.name] += [float(test_metrics[i])] if test_dataset is not None else []

        return False

    def _assert_valid_metric(self, metric: int | float, best_epoch: int) -> bool:
        """
        Asserts that metric calculated is valid.

        Parameters:
        -----------
        metric: int or float
            the metric calculated
        best_epoch: int
            the best epoch encountered since training started

        Returns:
        --------
        Whether the metric is valid
        """
        if np.isnan(metric):
            if best_epoch == 0:
                raise RuntimeError(
                    f"NaNs encountered in {self.__class__.__name__} training. "
                    "Features/labels may be improperly formatted, "
                    "or NN weights may have diverged."
                )
            else:
                logger.warning(f"Warning: NaNs encountered in {self.__class__.__name__} training. " "Reverting model to last checkpoint without NaNs.")
                return False
        return True

    def _predict_proba(self, X, **kwargs):
        """To align predict with abstract_model API.
        Preprocess here only refers to feature processing steps done by all AbstractModel objects,
        not tabularNN-specific preprocessing steps.
        If X is not DataFrame but instead TabularNNDataset object, we can still produce predictions,
        but cannot use preprocess in this case (needs to be already processed).
        """
        from .tabular_torch_dataset import TabularTorchDataset

        if isinstance(X, TabularTorchDataset):
            return self._predict_tabular_data(new_data=X, process=False)
        elif isinstance(X, pd.DataFrame):
            X = self.preprocess(X, **kwargs)
            return self._predict_tabular_data(new_data=X, process=True)
        else:
            raise ValueError("X must be of type pd.DataFrame or TabularTorchDataset, not type: %s" % type(X))

    def _predict_tabular_data(self, new_data, process=True):
        from .tabular_torch_dataset import TabularTorchDataset

        if process:
            new_data = self._process_test_data(new_data)
        if not isinstance(new_data, TabularTorchDataset):
            raise ValueError("new_data must be of type TabularTorchDataset if process=False")
        val_dataloader = new_data.build_loader(self.max_batch_size, self.num_dataloading_workers, is_test=True)
        preds_dataset = []
        for data_batch in val_dataloader:
            preds_batch = self.model.predict(data_batch)
            preds_dataset.append(preds_batch)
        preds_dataset = np.concatenate(preds_dataset, 0)
        return preds_dataset

    def _generate_dataset(self, X: pd.DataFrame | TabularTorchDataset, y: pd.Series, train_params: dict = {}, is_train: bool = False) -> TabularTorchDataset:
        """
        Generate TabularTorchDataset from X and y.

        Params:
        -------
        X: pd.DataFrame | TabularTorchDataset
            The X data.
        y: pd.Series
            The y data.
        params: dict
            Parameters related to processing training data.
        is_train: bool
            Whether the X and y values are training data.

        Returns:
        --------
        TabularTorchDataset containing the contents of X and y.
        """
        from .tabular_torch_dataset import TabularTorchDataset

        if is_train:
            impute_strategy = train_params["proc.impute_strategy"]
            max_category_levels = train_params["proc.max_category_levels"]
            skew_threshold = train_params["proc.skew_threshold"]
            embed_min_categories = train_params["proc.embed_min_categories"]
            use_ngram_features = train_params["use_ngram_features"]

            if isinstance(X, TabularTorchDataset):
                dataset = X
            else:
                X = self.preprocess(X)
                dataset = self._process_train_data(
                    df=X,
                    labels=y,
                    impute_strategy=impute_strategy,
                    max_category_levels=max_category_levels,
                    skew_threshold=skew_threshold,
                    embed_min_categories=embed_min_categories,
                    use_ngram_features=use_ngram_features,
                )
        else:
            if isinstance(X, TabularTorchDataset):
                dataset = X
            else:
                X = self.preprocess(X)
                dataset = self._process_test_data(df=X, labels=y)

        return dataset

    def _process_test_data(self, df, labels=None):
        """Process train or test DataFrame into a form fit for neural network models.
        Args:
            df (pd.DataFrame): Data to be processed (X)
            labels (pd.Series): labels to be processed (y)
        Returns:
            Dataset object
        """
        from .tabular_torch_dataset import TabularTorchDataset

        # sklearn processing n_quantiles warning
        warnings.filterwarnings("ignore", module="sklearn.preprocessing")
        if labels is not None and len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")
        if self.processor is None or self._types_of_features is None or self.feature_arraycol_map is None or self.feature_type_map is None:
            raise ValueError("Need to process training data before test data")
        if self.features_to_drop:
            drop_cols = [col for col in df.columns if col in self.features_to_drop]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        # self.feature_arraycol_map, self.feature_type_map have been previously set while processing training data.
        df = self.processor.transform(df)
        return TabularTorchDataset(df, self.feature_arraycol_map, self.feature_type_map, self.problem_type, labels)

    def _process_train_data(self, df, impute_strategy, max_category_levels, skew_threshold, embed_min_categories, use_ngram_features, labels):
        from .tabular_torch_dataset import TabularTorchDataset

        # sklearn processing n_quantiles warning
        warnings.filterwarnings("ignore", module="sklearn.preprocessing")
        if labels is None:
            raise ValueError("Attempting process training data without labels")
        if len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")

        # dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', values = column-names of df
        self._types_of_features, df = self._get_types_of_features(
            df, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features
        )
        logger.log(15, "Tabular Neural Network treats features as the following types:")
        logger.log(15, json.dumps(self._types_of_features, indent=4))
        logger.log(15, "\n")
        if self.processor is not None:
            logger.log(15, f"Attempting to process training data for {self.__class__.__name__}, but previously already did this.")
        self.processor = create_preprocessor(
            impute_strategy=impute_strategy,
            max_category_levels=max_category_levels,
            unique_category_str=self.unique_category_str,
            continuous_features=self._types_of_features["continuous"],
            skewed_features=self._types_of_features["skewed"],
            onehot_features=self._types_of_features["onehot"],
            embed_features=self._types_of_features["embed"],
            bool_features=self._types_of_features["bool"],
        )
        df = self.processor.fit_transform(df)
        # OrderedDict of feature-name -> list of column-indices in df corresponding to this feature
        self.feature_arraycol_map = get_feature_arraycol_map(processor=self.processor, max_category_levels=max_category_levels)
        num_array_cols = np.sum([len(self.feature_arraycol_map[key]) for key in self.feature_arraycol_map])  # should match number of columns in processed array
        if num_array_cols != df.shape[1]:
            raise ValueError(
                "Error during one-hot encoding data processing for neural network. " "Number of columns in df array does not match feature_arraycol_map."
            )

        # OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed')
        self.feature_type_map = get_feature_type_map(feature_arraycol_map=self.feature_arraycol_map, types_of_features=self._types_of_features)
        return TabularTorchDataset(df, self.feature_arraycol_map, self.feature_type_map, self.problem_type, labels)

    def _init_optimizer(self, optimizer, learning_rate, weight_decay):
        """
        Set up optimizer needed for training.
        Network must first be initialized before this.
        """
        import torch

        if optimizer == "sgd":
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adam":
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adamw":
            optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer specified: {optimizer}")
        return optimizer

    def reduce_memory_size(self, remove_fit=True, requires_save=True, **kwargs):
        super().reduce_memory_size(remove_fit=remove_fit, requires_save=requires_save, **kwargs)
        if remove_fit and requires_save:
            self.optimizer = None

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(X=X, problem_type=self.problem_type, num_classes=self.num_classes, hyperparameters=hyperparameters, **kwargs)

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        **kwargs,
    ) -> int:
        return 5 * get_approximate_df_mem_usage(X).sum()

    def _get_maximum_resources(self) -> Dict[str, Union[int, float]]:
        # torch model trains slower when utilizing virtual cores and this issue scale up when the number of cpu cores increases
        return {"num_cpus": ResourceManager.get_cpu_count(only_physical_cores=True)}

    def _get_default_resources(self):
        # only_physical_cores=True is faster in training
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = 0
        return num_cpus, num_gpus

    def save(self, path: str = None, verbose=True) -> str:
        import torch

        # Save on CPU to ensure the model can be loaded on a box without GPU
        if self.model is not None:
            self.model = self.model.to(torch.device("cpu"))
        path = super().save(path, verbose)
        # Put the model back to the device after the save
        if self.model is not None:
            self.model.to(self.device)
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        """
        Loads the model from disk to memory.
        The loaded model will be on the same device it was trained on (cuda/mps);
        if the device is not available (trained on GPU, deployed on CPU),
        then `cpu` will be used.

        Parameters
        ----------
        path : str
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            The model file is typically located in os.path.join(path, cls.model_file_name).
        reset_paths : bool, default True
            Whether to reset the self.path value of the loaded model to be equal to path.
            It is highly recommended to keep this value as True unless accessing the original self.path value is important.
            If False, the actual valid path and self.path may differ, leading to strange behaviour and potential exceptions if the model needs to load any other files at a later time.
        verbose : bool, default True
            Whether to log the location of the loaded file.

        Returns
        -------
        model : cls
            Loaded model object.
        """
        import torch

        model: TabularNeuralNetTorchModel = super().load(path=path, reset_paths=reset_paths, verbose=verbose)

        # Put the model on the same device it was train on (GPU/MPS) if it is available; otherwise use CPU
        if model.model is not None:
            original_device_type = model.device.type
            if "cuda" in original_device_type:
                # cuda: nvidia GPU
                device = torch.device(original_device_type if torch.cuda.is_available() else "cpu")
            elif "mps" in original_device_type:
                # mps: Apple Silicon
                device = torch.device(original_device_type if torch.backends.mps.is_available() else "cpu")
            else:
                device = torch.device(original_device_type)

            if verbose and (original_device_type != device.type):
                logger.log(15, f"Model is trained on {original_device_type}, but the device is not available - loading on {device.type}")

            model.device = device
            model.model = model.model.to(model.device)
            model.model.device = model.device

        # Compiled models handling
        if hasattr(model, "_compiler") and model._compiler and model._compiler.name != "native":
            model.model.eval()
            model.processor = model._compiler.load(path=model.path)
        return model

    def _get_hpo_backend(self):
        """Choose which backend(Ray or Custom) to use for hpo"""
        return RAY_BACKEND

    def get_minimum_resources(self, is_gpu_available=False):
        minimum_resources = {
            "num_cpus": 1,
        }
        if is_gpu_available:
            # Our custom implementation does not support partial GPU. No gpu usage according to nvidia-smi when the `num_gpus` passed to fit is fractional`
            minimum_resources["num_gpus"] = 1
        return minimum_resources

    @classmethod
    def _valid_compilers(cls):
        return [TabularNeuralNetTorchNativeCompiler, TabularNeuralNetTorchOnnxCompiler]

    @classmethod
    def _default_compiler(cls):
        return TabularNeuralNetTorchNativeCompiler

    def _ag_params(self) -> set:
        return {"early_stop", "generate_curves", "curve_metrics", "use_error_for_curve_metrics"}

    def _get_input_types(self, batch_size=None):
        input_types = []
        for f in self._features:
            input_types.append((f, [batch_size, 1]))
        return input_types

    def compile(self, compiler_configs=None):
        """
        Compile the trained model for faster inference.

        This completely overrides the compile() in AbstractModel, since we won't
        overwrite self.model in the compilation process.
        Instead, self.processor would be converted from sklearn ColumnTransformer
        to its alternative counterpart.
        """
        assert self.is_fit(), "The model must be fit before calling the compile method."
        if compiler_configs is None:
            compiler_configs = {}
        # Take self.max_batch_size as default batch size, instead of None in AbstractModel
        batch_size = compiler_configs.get("batch_size", self.max_batch_size)
        compiler_configs.update(batch_size=batch_size)
        super().compile(compiler_configs)

    def _compile(self, **kwargs):
        """
        Take the compiler to perform actual compilation.

        This overrides the _compile() in AbstractModel, since we won't
        overwrite self.model in the compilation process.
        Instead, self.processor would be converted from sklearn ColumnTransformer
        to TabularNeuralNetTorchOnnxTransformer.
        """
        from sklearn.compose._column_transformer import ColumnTransformer

        input_types = kwargs.get("input_types", self._get_input_types(batch_size=self.max_batch_size))
        assert isinstance(self.processor, ColumnTransformer), (
            f"unexpected processor type {type(self.processor)}, " "expecting processor type to be sklearn.compose._column_transformer.ColumnTransformer"
        )
        self.processor = self._compiler.compile(model=(self.processor, self.model), path=self.path, input_types=input_types)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression", "quantile", "softclass"]

    @classmethod
    def _class_tags(cls):
        return {
            "can_estimate_memory_usage_static": True,
            "supports_learning_curves": True,
        }

    def _more_tags(self):
        # `can_refit_full=True` because batch_size and num_epochs is communicated at end of `_fit`:
        #  self.params_trained['batch_size'] = batch_size
        #  self.params_trained['num_epochs'] = best_epoch
        return {"can_refit_full": True}
