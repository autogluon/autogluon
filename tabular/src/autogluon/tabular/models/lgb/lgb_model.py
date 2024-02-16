import gc
import logging
import os
import random
import re
import time
import warnings

import numpy as np
from pandas import DataFrame, Series

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_lightgbm
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS
from autogluon.core.models import AbstractModel
from autogluon.core.models._utils import get_early_stopping_rounds

from . import lgb_utils
from .hyperparameters.parameters import DEFAULT_NUM_BOOST_ROUND, get_lgb_objective, get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from .lgb_utils import construct_dataset, train_lgb_model

warnings.filterwarnings("ignore", category=UserWarning, message="Starting from version")  # lightGBM brew libomp warning
logger = logging.getLogger(__name__)


# TODO: Save dataset to binary and reload for HPO. This will avoid the memory spike overhead when training each model and instead it will only occur once upon saving the dataset.
class LGBModel(AbstractModel):
    """
    LightGBM model: https://lightgbm.readthedocs.io/en/latest/

    Hyperparameter options: https://lightgbm.readthedocs.io/en/latest/Parameters.html

    Extra hyperparameter options:
        ag.early_stop : int, specifies the early stopping rounds. Defaults to an adaptive strategy. Recommended to keep default.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._features_internal_map = None
        self._features_internal_list = None
        self._requires_remap = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return get_default_searchspace(problem_type=self.problem_type)

    # Use specialized LightGBM metric if available (fast), otherwise use custom func generator
    def _get_stopping_metric_internal(self):
        stopping_metric = lgb_utils.convert_ag_metric_to_lgbm(ag_metric_name=self.stopping_metric.name, problem_type=self.problem_type)
        if stopping_metric is None:
            stopping_metric = lgb_utils.func_generator(
                metric=self.stopping_metric, is_higher_better=True, needs_pred_proba=not self.stopping_metric.needs_pred, problem_type=self.problem_type
            )
            stopping_metric_name = self.stopping_metric.name
        else:
            stopping_metric_name = stopping_metric
        return stopping_metric, stopping_metric_name

    def _estimate_memory_usage(self, X: DataFrame, **kwargs) -> float:
        """
        Returns the expected peak memory usage in bytes of the LightGBM model during fit.

        The memory usage of LightGBM is primarily made up of two sources:

        1. The size of the data
        2. The size of the histogram cache
            Scales roughly by 5100*num_features*num_leaves bytes
            For 10000 features and 128 num_leaves, the histogram would be 6.5 GB.
        """
        num_classes = self.num_classes if self.num_classes else 1  # self.num_classes could be None after initialization if it's a regression problem
        data_mem_usage = get_approximate_df_mem_usage(X).sum()
        data_mem_usage_bytes = data_mem_usage * 5 + data_mem_usage / 4 * num_classes  # TODO: Extremely crude approximation, can be vastly improved

        params = self._get_model_params(convert_search_spaces_to_default=True)
        max_bins = params.get("max_bins", 255)
        num_leaves = params.get("num_leaves", 31)
        # Memory usage of histogram based on https://github.com/microsoft/LightGBM/issues/562#issuecomment-304524592
        histogram_mem_usage_bytes = 20 * max_bins * len(X.columns) * num_leaves
        histogram_mem_usage_bytes_max = params.get("histogram_pool_size", None)
        if histogram_mem_usage_bytes_max is not None:
            histogram_mem_usage_bytes_max *= 1e6  # Convert megabytes to bytes, `histogram_pool_size` is in MB.
            if histogram_mem_usage_bytes > histogram_mem_usage_bytes_max:
                histogram_mem_usage_bytes = histogram_mem_usage_bytes_max
        histogram_mem_usage_bytes *= 1.2  # Add a 20% buffer

        approx_mem_size_req = data_mem_usage_bytes + histogram_mem_usage_bytes
        return approx_mem_size_req

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_gpus=0, num_cpus=0, sample_weight=None, sample_weight_val=None, verbosity=2, **kwargs):
        try_import_lightgbm()  # raise helpful error message if LightGBM isn't installed
        start_time = time.time()
        ag_params = self._get_ag_params()
        params = self._get_model_params()

        if verbosity <= 1:
            log_period = False
        elif verbosity == 2:
            log_period = 1000
        elif verbosity == 3:
            log_period = 50
        else:
            log_period = 1

        stopping_metric, stopping_metric_name = self._get_stopping_metric_internal()

        num_boost_round = params.pop("num_boost_round", DEFAULT_NUM_BOOST_ROUND)
        dart_retrain = params.pop("dart_retrain", False)  # Whether to retrain the model to get optimal iteration if model is trained in 'dart' mode.
        if num_gpus != 0:
            if "device" not in params:
                # TODO: lightgbm must have a special install to support GPU: https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-version
                #  Before enabling GPU, we should add code to detect that GPU-enabled version is installed and that a valid GPU exists.
                #  GPU training heavily alters accuracy, often in a negative manner. We will have to be careful about when to use GPU.
                params["device"] = "gpu"
                logger.log(20, f"\tTraining {self.name} with GPU, note that this may negatively impact model quality compared to CPU training.")
        logger.log(15, f"\tFitting {num_boost_round} rounds... Hyperparameters: {params}")

        if "num_threads" not in params:
            params["num_threads"] = num_cpus
        if "objective" not in params:
            params["objective"] = get_lgb_objective(problem_type=self.problem_type)
        if self.problem_type in [MULTICLASS, SOFTCLASS] and "num_classes" not in params:
            params["num_classes"] = self.num_classes
        if "verbose" not in params:
            params["verbose"] = -1

        num_rows_train = len(X)
        dataset_train, dataset_val = self.generate_datasets(
            X=X, y=y, params=params, X_val=X_val, y_val=y_val, sample_weight=sample_weight, sample_weight_val=sample_weight_val
        )
        gc.collect()

        callbacks = []
        valid_names = []
        valid_sets = []
        if dataset_val is not None:
            from .callbacks import early_stopping_custom

            # TODO: Better solution: Track trend to early stop when score is far worse than best score, or score is trending worse over time
            early_stopping_rounds = ag_params.get("early_stop", "adaptive")
            if isinstance(early_stopping_rounds, (str, tuple, list)):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)
            if early_stopping_rounds is None:
                early_stopping_rounds = 999999
            reporter = kwargs.get("reporter", None)
            train_loss_name = self._get_train_loss_name() if reporter is not None else None
            if train_loss_name is not None:
                if "metric" not in params or params["metric"] == "":
                    params["metric"] = train_loss_name
                elif train_loss_name not in params["metric"]:
                    params["metric"] = f'{params["metric"]},{train_loss_name}'
            # early stopping callback will be added later by QuantileBooster if problem_type==QUANTILE
            early_stopping_callback_kwargs = dict(
                stopping_rounds=early_stopping_rounds,
                metrics_to_use=[("valid_set", stopping_metric_name)],
                max_diff=None,
                start_time=start_time,
                time_limit=time_limit,
                ignore_dart_warning=True,
                verbose=False,
                manual_stop_file=False,
                reporter=reporter,
                train_loss_name=train_loss_name,
            )
            callbacks += [
                # Note: Don't use self.params_aux['max_memory_usage_ratio'] here as LightGBM handles memory per iteration optimally.  # TODO: Consider using when ratio < 1.
                early_stopping_custom(**early_stopping_callback_kwargs)
            ]
            valid_names = ["valid_set"] + valid_names
            valid_sets = [dataset_val] + valid_sets
        else:
            early_stopping_callback_kwargs = None
        from lightgbm.callback import log_evaluation

        if log_period is not None:
            callbacks.append(log_evaluation(period=log_period))

        seed_val = params.pop("seed_value", 0)
        train_params = {
            "params": params,
            "train_set": dataset_train,
            "num_boost_round": num_boost_round,
            "valid_sets": valid_sets,
            "valid_names": valid_names,
            "callbacks": callbacks,
        }
        if not isinstance(stopping_metric, str):
            train_params["feval"] = stopping_metric
        else:
            if "metric" not in train_params["params"] or train_params["params"]["metric"] == "":
                train_params["params"]["metric"] = stopping_metric
            elif stopping_metric not in train_params["params"]["metric"]:
                train_params["params"]["metric"] = f'{train_params["params"]["metric"]},{stopping_metric}'
        if self.problem_type == SOFTCLASS:
            train_params["fobj"] = lgb_utils.softclass_lgbobj
        elif self.problem_type == QUANTILE:
            train_params["params"]["quantile_levels"] = self.quantile_levels
        if seed_val is not None:
            train_params["params"]["seed"] = seed_val
            random.seed(seed_val)
            np.random.seed(seed_val)

        # Train LightGBM model:
        from lightgbm.basic import LightGBMError

        with warnings.catch_warnings():
            # Filter harmless warnings introduced in lightgbm 3.0, future versions plan to remove: https://github.com/microsoft/LightGBM/issues/3379
            warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset.")
            warnings.filterwarnings("ignore", message="categorical_column in param dict is overridden.")
            try:
                self.model = train_lgb_model(early_stopping_callback_kwargs=early_stopping_callback_kwargs, **train_params)
            except LightGBMError:
                if train_params["params"].get("device", "cpu") != "gpu":
                    raise
                else:
                    logger.warning(
                        "Warning: GPU mode might not be installed for LightGBM, GPU training raised an exception. Falling back to CPU training..."
                        "Refer to LightGBM GPU documentation: https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-version"
                        "One possible method is:"
                        "\tpip uninstall lightgbm -y"
                        "\tpip install lightgbm --install-option=--gpu"
                    )
                    train_params["params"]["device"] = "cpu"
                    self.model = train_lgb_model(early_stopping_callback_kwargs=early_stopping_callback_kwargs, **train_params)
            retrain = False
            if train_params["params"].get("boosting_type", "") == "dart":
                if dataset_val is not None and dart_retrain and (self.model.best_iteration != num_boost_round):
                    retrain = True
                    if time_limit is not None:
                        time_left = time_limit + start_time - time.time()
                        if time_left < 0.5 * time_limit:
                            retrain = False
                    if retrain:
                        logger.log(15, f"Retraining LGB model to optimal iterations ('dart' mode).")
                        train_params.pop("callbacks", None)
                        train_params.pop("valid_sets", None)
                        train_params.pop("valid_names", None)
                        train_params["num_boost_round"] = self.model.best_iteration
                        self.model = train_lgb_model(**train_params)
                    else:
                        logger.log(15, f"Not enough time to retrain LGB model ('dart' mode)...")

        if dataset_val is not None and not retrain:
            self.params_trained["num_boost_round"] = self.model.best_iteration
        else:
            self.params_trained["num_boost_round"] = self.model.current_iteration()

    def _predict_proba(self, X, num_cpus=0, **kwargs):
        X = self.preprocess(X, **kwargs)

        y_pred_proba = self.model.predict(X, num_threads=num_cpus)
        if self.problem_type in [REGRESSION, QUANTILE, MULTICLASS]:
            return y_pred_proba
        elif self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif self.problem_type == SOFTCLASS:  # apply softmax
            y_pred_proba = np.exp(y_pred_proba)
            y_pred_proba = np.multiply(y_pred_proba, 1 / np.sum(y_pred_proba, axis=1)[:, np.newaxis])
            return y_pred_proba
        else:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 2:  # Should this ever happen?
                return y_pred_proba
            else:  # Should this ever happen?
                return y_pred_proba[:, 1]

    def _preprocess_nonadaptive(self, X, is_train=False, **kwargs):
        X = super()._preprocess_nonadaptive(X=X, **kwargs)

        if is_train:
            self._requires_remap = False
            for column in X.columns:
                if isinstance(column, str):
                    new_column = re.sub(r'[",:{}[\]]', "", column)
                    if new_column != column:
                        self._features_internal_map = {feature: i for i, feature in enumerate(list(X.columns))}
                        self._requires_remap = True
                        break
            if self._requires_remap:
                self._features_internal_list = np.array([self._features_internal_map[feature] for feature in list(X.columns)])
            else:
                self._features_internal_list = self._features_internal

        if self._requires_remap:
            X_new = X.copy(deep=False)
            X_new.columns = self._features_internal_list
            return X_new
        else:
            return X

    def generate_datasets(self, X: DataFrame, y: Series, params, X_val=None, y_val=None, sample_weight=None, sample_weight_val=None, save=False):
        lgb_dataset_params_keys = ["two_round"]  # Keys that are specific to lightGBM Dataset object construction.
        data_params = {key: params[key] for key in lgb_dataset_params_keys if key in params}.copy()

        X = self.preprocess(X, is_train=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        # TODO: Try creating multiple Datasets for subsets of features, then combining with Dataset.add_features_from(), this might avoid memory spike

        y_og = None
        y_val_og = None
        if self.problem_type == SOFTCLASS:
            y_og = np.array(y)
            y = None
            if X_val is not None:
                y_val_og = np.array(y_val)
                y_val = None

        # X, W_train = self.convert_to_weight(X=X)
        dataset_train = construct_dataset(
            x=X, y=y, location=os.path.join("self.path", "datasets", "train"), params=data_params, save=save, weight=sample_weight
        )
        # dataset_train = construct_dataset_lowest_memory(X=X, y=y, location=self.path + 'datasets/train', params=data_params)
        if X_val is not None:
            # X_val, W_val = self.convert_to_weight(X=X_val)
            dataset_val = construct_dataset(
                x=X_val,
                y=y_val,
                location=os.path.join(self.path, "datasets", "val"),
                reference=dataset_train,
                params=data_params,
                save=save,
                weight=sample_weight_val,
            )
            # dataset_val = construct_dataset_lowest_memory(X=X_val, y=y_val, location=self.path + 'datasets/val', reference=dataset_train, params=data_params)
        else:
            dataset_val = None
        if self.problem_type == SOFTCLASS:
            if y_og is not None:
                dataset_train.softlabels = y_og
            if y_val_og is not None:
                dataset_val.softlabels = y_val_og
        return dataset_train, dataset_val

    def _get_train_loss_name(self):
        if self.problem_type == BINARY:
            train_loss_name = "binary_logloss"
        elif self.problem_type == MULTICLASS:
            train_loss_name = "multi_logloss"
        elif self.problem_type == REGRESSION:
            train_loss_name = "l2"
        else:
            raise ValueError(f"unknown problem_type for LGBModel: {self.problem_type}")
        return train_loss_name

    def _get_early_stopping_rounds(self, num_rows_train, strategy="auto"):
        return get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=strategy)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _is_gpu_lgbm_installed(self):
        # Taken from https://github.com/microsoft/LightGBM/issues/3939
        try_import_lightgbm()
        import lightgbm

        try:
            data = np.random.rand(50, 2)
            label = np.random.randint(2, size=50)
            train_data = lightgbm.Dataset(data, label=label)
            params = {"device": "gpu"}
            gbm = lightgbm.train(params, train_set=train_data, verbose=-1)
            return True
        except Exception as e:
            return False

    def get_minimum_resources(self, is_gpu_available=False):
        minimum_resources = {
            "num_cpus": 1,
        }
        if is_gpu_available and self._is_gpu_lgbm_installed():
            minimum_resources["num_gpus"] = 0.5
        return minimum_resources

    def _get_default_resources(self):
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus

    @property
    def _features(self):
        return self._features_internal_list

    def _ag_params(self) -> set:
        return {"early_stop"}

    def _more_tags(self):
        # `can_refit_full=True` because num_boost_round is communicated at end of `_fit`
        return {"can_refit_full": True}
