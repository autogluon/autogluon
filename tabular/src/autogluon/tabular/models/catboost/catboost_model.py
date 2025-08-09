from __future__ import annotations

import logging
import math
import os
import time
from types import MappingProxyType

import numpy as np
import pandas as pd

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_catboost
from autogluon.core.constants import MULTICLASS, PROBLEM_TYPES_CLASSIFICATION, QUANTILE, REGRESSION, SOFTCLASS
from autogluon.core.models import AbstractModel
from autogluon.core.models._utils import get_early_stopping_rounds
from autogluon.core.utils.exceptions import TimeLimitExceeded

from .callbacks import EarlyStoppingCallback, MemoryCheckCallback, TimeCheckCallback
from .catboost_utils import CATBOOST_EVAL_METRIC_TO_LOSS_FUNCTION, get_catboost_metric_from_ag_metric
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace

logger = logging.getLogger(__name__)


# TODO: Consider having CatBoost variant that converts all categoricals to numerical as done in RFModel, was showing improved results in some problems.
class CatBoostModel(AbstractModel):
    """
    CatBoost model: https://catboost.ai/

    Hyperparameter options: https://catboost.ai/en/docs/references/training-parameters
    """
    ag_key = "CAT"
    ag_name = "CatBoost"
    ag_priority = 70
    ag_priority_by_problem_type = MappingProxyType({
        SOFTCLASS: 60
    })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._category_features = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        # Set 'allow_writing_files' to True in order to keep log files created by catboost during training (these will be saved in the directory where AutoGluon stores this model)
        self._set_default_param_value("allow_writing_files", False)  # Disables creation of catboost logging files during training by default
        if self.problem_type != SOFTCLASS:  # TODO: remove this after catboost 0.24
            default_eval_metric = get_catboost_metric_from_ag_metric(self.stopping_metric, self.problem_type, self.quantile_levels)
            self._set_default_param_value("eval_metric", default_eval_metric)

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type, num_classes=self.num_classes)

    def _preprocess_nonadaptive(self, X, **kwargs):
        X = super()._preprocess_nonadaptive(X, **kwargs)
        if self._category_features is None:
            self._category_features = list(X.select_dtypes(include="category").columns)
        if self._category_features:
            X = X.copy()
            for category in self._category_features:
                current_categories = X[category].cat.categories
                if "__NaN__" in current_categories:
                    X[category] = X[category].fillna("__NaN__")
                else:
                    X[category] = X[category].cat.add_categories("__NaN__").fillna("__NaN__")
        return X

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(X=X, problem_type=self.problem_type, num_classes=self.num_classes, hyperparameters=hyperparameters, **kwargs)

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict = None,
        num_classes: int = 1,
        **kwargs,
    ) -> int:
        """
        Returns the expected peak memory usage in bytes of the CatBoost model during fit.

        The memory usage of CatBoost is primarily made up of two sources:

        1. The size of the data
        2. The size of the histogram cache
            Scales roughly by 5080*num_features*2^depth bytes
            For 10000 features and 6 depth, the histogram would be 3.2 GB.
        """
        if hyperparameters is None:
            hyperparameters = {}
        num_classes = num_classes if num_classes else 1  # self.num_classes could be None after initialization if it's a regression problem
        data_mem_usage = get_approximate_df_mem_usage(X).sum()
        data_mem_usage_bytes = data_mem_usage * 5 + data_mem_usage / 4 * num_classes  # TODO: Extremely crude approximation, can be vastly improved

        border_count = hyperparameters.get("border_count", 254)
        depth = hyperparameters.get("depth", 6)

        # if depth < 7, treat it as 1 step larger for histogram size estimate
        #  this fixes cases where otherwise histogram size appears to be off by around a factor of 2 for depth=6
        histogram_effective_depth = max(min(depth+1, 7), depth)

        # Formula based on manual testing, aligns with LightGBM histogram sizes
        histogram_mem_usage_bytes = 24 * math.pow(2, histogram_effective_depth) * len(X.columns) * border_count
        histogram_mem_usage_bytes *= 1.2  # Add a 20% buffer

        baseline_memory_bytes = 4e8  # 400 MB baseline memory

        approx_mem_size_req = data_mem_usage_bytes + histogram_mem_usage_bytes + baseline_memory_bytes
        return approx_mem_size_req

    def _get_random_seed_from_hyperparameters(self, hyperparameters: dict) -> int | None | str:
        return hyperparameters.get("random_seed", "N/A")

    # TODO: Use Pool in preprocess, optimize bagging to do Pool.split() to avoid re-computing pool for each fold! Requires stateful + y
    #  Pool is much more memory efficient, avoids copying data twice in memory
    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_gpus=0, num_cpus=-1, sample_weight=None, sample_weight_val=None, **kwargs):
        time_start = time.time()
        try_import_catboost()
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        ag_params = self._get_ag_params()
        params = self._get_model_params()
        params["random_seed"] = self.random_seed

        params["thread_count"] = num_cpus
        if self.problem_type == SOFTCLASS:
            # FIXME: This is extremely slow due to unoptimized metric / objective sent to CatBoost
            from .catboost_softclass_utils import SoftclassCustomMetric, SoftclassObjective

            params.setdefault("loss_function",  SoftclassObjective.SoftLogLossObjective())
            params["eval_metric"] = SoftclassCustomMetric.SoftLogLossMetric()
        elif self.problem_type in [REGRESSION, QUANTILE]:
            # Choose appropriate loss_function that is as close as possible to the eval_metric
            params.setdefault(
                "loss_function",
                CATBOOST_EVAL_METRIC_TO_LOSS_FUNCTION.get(params["eval_metric"], params["eval_metric"])
            )

        model_type = CatBoostClassifier if self.problem_type in PROBLEM_TYPES_CLASSIFICATION else CatBoostRegressor
        num_rows_train = len(X)
        num_cols_train = len(X.columns)
        num_classes = self.num_classes if self.num_classes else 1  # self.num_classes could be None after initialization if it's a regression problem

        X = self.preprocess(X)
        cat_features = list(X.select_dtypes(include="category").columns)
        X = Pool(data=X, label=y, cat_features=cat_features, weight=sample_weight)

        if X_val is None:
            eval_set = None
            early_stopping_rounds = None
        else:
            X_val = self.preprocess(X_val)
            X_val = Pool(data=X_val, label=y_val, cat_features=cat_features, weight=sample_weight_val)
            eval_set = X_val
            early_stopping_rounds = ag_params.get("early_stop", "adaptive")
            if isinstance(early_stopping_rounds, (str, tuple, list)):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)

        if params.get("allow_writing_files", False):
            if "train_dir" not in params:
                try:
                    # TODO: What if path is in S3?
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                except:
                    pass
                else:
                    params["train_dir"] = os.path.join(self.path, "catboost_info")

        # TODO: Add more control over these params (specifically early_stopping_rounds)
        verbosity = kwargs.get("verbosity", 2)
        if verbosity <= 1:
            verbose = False
        elif verbosity == 2:
            verbose = False
        elif verbosity == 3:
            verbose = 20
        else:
            verbose = True

        num_features = len(self._features)

        if num_gpus != 0:
            if "task_type" not in params:
                params["task_type"] = "GPU"
                logger.log(20, f"\tTraining {self.name} with GPU, note that this may negatively impact model quality compared to CPU training.")
                # TODO: Confirm if GPU is used in HPO (Probably not)
                # TODO: Adjust max_bins to 254?

        if params.get("task_type", None) == "GPU":
            if "colsample_bylevel" in params:
                params.pop("colsample_bylevel")
                logger.log(30, f"\t'colsample_bylevel' is not supported on GPU, using default value (Default = 1).")
            if "rsm" in params:
                params.pop("rsm")
                logger.log(30, f"\t'rsm' is not supported on GPU, using default value (Default = 1).")

        if self.problem_type == MULTICLASS and "rsm" not in params and "colsample_bylevel" not in params and num_features > 1000:
            # Subsample columns to speed up training
            if params.get("task_type", None) != "GPU":  # RSM does not work on GPU
                params["colsample_bylevel"] = max(min(1.0, 1000 / num_features), 0.05)
                logger.log(
                    30,
                    f'\tMany features detected ({num_features}), dynamically setting \'colsample_bylevel\' to {params["colsample_bylevel"]} to speed up training (Default = 1).',
                )
                logger.log(30, f"\tTo disable this functionality, explicitly specify 'colsample_bylevel' in the model hyperparameters.")
            else:
                params["colsample_bylevel"] = 1.0
                logger.log(30, f"\t'colsample_bylevel' is not supported on GPU, using default value (Default = 1).")

        logger.log(15, f"\tCatboost model hyperparameters: {params}")

        extra_fit_kwargs = dict()
        if params.get("task_type", None) != "GPU":
            callbacks = []
            if early_stopping_rounds is not None:
                callbacks.append(EarlyStoppingCallback(stopping_rounds=early_stopping_rounds, eval_metric=params["eval_metric"]))

            if num_rows_train * num_cols_train * num_classes > 5_000_000:
                # The data is large enough to potentially cause memory issues during training, so monitor memory usage via callback.
                callbacks.append(MemoryCheckCallback())
            if time_limit is not None:
                time_cur = time.time()
                time_left = time_limit - (time_cur - time_start)
                if time_left <= time_limit * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                    raise TimeLimitExceeded
                callbacks.append(TimeCheckCallback(time_start=time_cur, time_limit=time_left))
            extra_fit_kwargs["callbacks"] = callbacks
        else:
            logger.log(30, f"\tWarning: CatBoost on GPU is experimental. If you encounter issues, use CPU for training CatBoost instead.")
            if time_limit is not None:
                params["iterations"] = self._estimate_iter_in_time_gpu(
                    X=X,
                    eval_set=eval_set,
                    time_limit=time_limit,
                    verbose=verbose,
                    params=params,
                    num_rows_train=num_rows_train,
                    time_start=time_start,
                    model_type=model_type,
                )
            if early_stopping_rounds is not None:
                if isinstance(early_stopping_rounds, int):
                    extra_fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
                elif isinstance(early_stopping_rounds, tuple):
                    extra_fit_kwargs["early_stopping_rounds"] = 50
        self.model = model_type(**params)

        # TODO: Custom metrics don't seem to work anymore
        # TODO: Custom metrics not supported in GPU mode
        # TODO: Callbacks not supported in GPU mode
        fit_final_kwargs = dict(
            eval_set=eval_set,
            verbose=verbose,
            **extra_fit_kwargs,
        )

        if eval_set is not None:
            fit_final_kwargs["use_best_model"] = True

        self.model.fit(X, **fit_final_kwargs)

        self.params_trained["iterations"] = self.model.tree_count_

    # FIXME: This logic is a hack made to maintain compatibility with GPU CatBoost.
    #  GPU CatBoost does not support callbacks or custom metrics.
    #  Since we use callbacks to check memory and training time in CPU mode, we need a way to estimate these things prior to training for GPU mode.
    #  This method will train a model on a toy number of iterations to estimate memory and training time.
    #  It will return an updated iterations to train on that will avoid running OOM and running over time limit.
    #  Remove this logic once CatBoost fixes GPU support for callbacks and custom metrics.
    def _estimate_iter_in_time_gpu(self, *, X, eval_set, time_limit, verbose, params, num_rows_train, time_start, model_type):
        import math
        import pickle
        import sys

        modifier = min(1.0, 10000 / num_rows_train)
        num_sample_iter_max = max(round(modifier * 50), 2)
        time_left_start = time_limit - (time.time() - time_start)
        if time_left_start <= time_limit * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
            raise TimeLimitExceeded
        default_iters = params["iterations"]
        params_init = params.copy()
        num_sample_iter = min(num_sample_iter_max, params_init["iterations"])
        params_init["iterations"] = num_sample_iter
        sample_model = model_type(
            **params_init,
        )
        sample_model.fit(
            X,
            eval_set=eval_set,
            use_best_model=True,
            verbose=verbose,
        )

        time_left_end = time_limit - (time.time() - time_start)
        time_taken_per_iter = (time_left_start - time_left_end) / num_sample_iter
        estimated_iters_in_time = round(time_left_end / time_taken_per_iter)

        available_mem = ResourceManager.get_available_virtual_mem()
        if self.problem_type == SOFTCLASS:
            model_size_bytes = 1  # skip memory check
        else:
            model_size_bytes = sys.getsizeof(pickle.dumps(sample_model))

        max_memory_proportion = 0.3
        mem_usage_per_iter = model_size_bytes / num_sample_iter
        max_memory_iters = math.floor(available_mem * max_memory_proportion / mem_usage_per_iter)

        final_iters = min(default_iters, min(max_memory_iters, estimated_iters_in_time))
        return final_iters

    def _predict_proba(self, X, **kwargs):
        if self.problem_type != SOFTCLASS:
            return super()._predict_proba(X, **kwargs)
        # For SOFTCLASS problems, manually transform predictions into probabilities via softmax
        X = self.preprocess(X, **kwargs)
        y_pred_proba = self.model.predict(X, prediction_type="RawFormulaVal")
        y_pred_proba = np.exp(y_pred_proba)
        y_pred_proba = np.multiply(y_pred_proba, 1 / np.sum(y_pred_proba, axis=1)[:, np.newaxis])
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
        return y_pred_proba

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_early_stopping_rounds(self, num_rows_train, strategy="auto"):
        return get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=strategy)

    def _ag_params(self) -> set:
        return {"early_stop"}

    def _validate_fit_memory_usage(self, mem_error_threshold: float = 1, mem_warning_threshold: float = 0.75, mem_size_threshold: int = 1e9, **kwargs):
        return super()._validate_fit_memory_usage(
            mem_error_threshold=mem_error_threshold, mem_warning_threshold=mem_warning_threshold, mem_size_threshold=mem_size_threshold, **kwargs
        )

    def get_minimum_resources(self, is_gpu_available=False):
        minimum_resources = {
            "num_cpus": 1,
        }
        if is_gpu_available:
            # Our custom implementation does not support partial GPU. No gpu usage according to nvidia-smi when the `num_gpus` passed to fit is fractional`
            minimum_resources["num_gpus"] = 0.5
        return minimum_resources

    def _get_default_resources(self):
        # only_physical_cores=True is faster in training
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = 0
        return num_cpus, num_gpus

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression", "quantile", "softclass"]

    @classmethod
    def _class_tags(cls):
        return {
            "can_estimate_memory_usage_static": True,
        }

    def _more_tags(self):
        # `can_refit_full=True` because iterations is communicated at end of `_fit`
        return {"can_refit_full": True}
