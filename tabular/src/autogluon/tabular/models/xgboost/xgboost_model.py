import logging
import math
import os
import time

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT
from autogluon.common.utils.lite import disable_if_lite_mode
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_xgboost
from autogluon.core.constants import MULTICLASS, PROBLEM_TYPES_CLASSIFICATION, REGRESSION, SOFTCLASS
from autogluon.core.models import AbstractModel
from autogluon.core.models._utils import get_early_stopping_rounds

from . import xgboost_utils
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace

logger = logging.getLogger(__name__)


class XGBoostModel(AbstractModel):
    """
    XGBoost model: https://xgboost.readthedocs.io/en/latest/

    Hyperparameter options: https://xgboost.readthedocs.io/en/latest/parameter.html
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ohe: bool = True
        self._ohe_generator = None
        self._xgb_model_type = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type, num_classes=self.num_classes)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return get_default_searchspace(problem_type=self.problem_type, num_classes=self.num_classes)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    # Use specialized XGBoost metric if available (fast), otherwise use custom func generator
    def get_eval_metric(self):
        eval_metric = xgboost_utils.convert_ag_metric_to_xgbm(ag_metric_name=self.stopping_metric.name, problem_type=self.problem_type)
        if eval_metric is None:
            eval_metric = xgboost_utils.func_generator(metric=self.stopping_metric, problem_type=self.problem_type)
        return eval_metric

    def _preprocess(self, X, is_train=False, max_category_levels=None, **kwargs):
        X = super()._preprocess(X=X, **kwargs)

        if is_train:
            if self._ohe:
                self._ohe_generator = xgboost_utils.OheFeatureGenerator(max_levels=max_category_levels)
                self._ohe_generator.fit(X)

        if self._ohe:
            X = self._ohe_generator.transform(X)

        return X

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_gpus=0, num_cpus=None, sample_weight=None, sample_weight_val=None, verbosity=2, **kwargs):
        # TODO: utilize sample_weight_val in early-stopping if provided
        start_time = time.time()
        ag_params = self._get_ag_params()
        params = self._get_model_params()
        if num_cpus:
            params["n_jobs"] = num_cpus
        max_category_levels = params.pop("proc.max_category_levels", 100)
        enable_categorical = params.get("enable_categorical", False)
        if enable_categorical:
            """Skip one-hot-encoding and pass categoricals directly to XGBoost"""
            self._ohe = False
        else:
            """One-hot-encode categorical features"""
            self._ohe = True

        if verbosity <= 2:
            verbose = False
            log_period = None
        elif verbosity == 3:
            verbose = True
            log_period = 50
        else:
            verbose = True
            log_period = 1

        X = self.preprocess(X, is_train=True, max_category_levels=max_category_levels)
        num_rows_train = X.shape[0]

        eval_set = []
        if "eval_metric" not in params:
            eval_metric = self.get_eval_metric()
            if eval_metric is not None:
                params["eval_metric"] = eval_metric

        if X_val is None:
            early_stopping_rounds = None
            eval_set = None
        else:
            X_val = self.preprocess(X_val, is_train=False)
            eval_set.append((X_val, y_val))
            early_stopping_rounds = ag_params.get("early_stop", "adaptive")
            if isinstance(early_stopping_rounds, (str, tuple, list)):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)

        if num_gpus != 0:
            params["tree_method"] = "gpu_hist"
            if "gpu_id" not in params:
                params["gpu_id"] = 0
        elif "tree_method" not in params:
            params["tree_method"] = "hist"

        try_import_xgboost()
        from xgboost.callback import EvaluationMonitor

        from .callbacks import EarlyStoppingCustom

        if eval_set is not None and "callbacks" not in params:
            callbacks = []
            if log_period is not None:
                callbacks.append(EvaluationMonitor(period=log_period))
            callbacks.append(EarlyStoppingCustom(early_stopping_rounds, start_time=start_time, time_limit=time_limit, verbose=verbose))
            params["callbacks"] = callbacks

        from xgboost import XGBClassifier, XGBRegressor

        model_type = XGBClassifier if self.problem_type in PROBLEM_TYPES_CLASSIFICATION else XGBRegressor
        self.model = model_type(**params)
        self.model.fit(X=X, y=y, eval_set=eval_set, verbose=False, sample_weight=sample_weight)

        bst = self.model.get_booster()
        # TODO: Investigate speed-ups from GPU inference
        # bst.set_param({"predictor": "gpu_predictor"})

        self.params_trained["n_estimators"] = bst.best_ntree_limit
        # Don't save the callback or eval_metric objects
        self.model.set_params(callbacks=None, eval_metric=None)

    def _predict_proba(self, X, num_cpus=-1, **kwargs):
        X = self.preprocess(X, **kwargs)
        self.model.set_params(n_jobs=num_cpus)

        if self.problem_type == REGRESSION:
            return self.model.predict(X)

        y_pred_proba = self.model.predict_proba(X)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _get_early_stopping_rounds(self, num_rows_train, strategy="auto"):
        return get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=strategy)

    def _get_num_classes(self, y):
        if self.problem_type == MULTICLASS:
            if self.num_classes is not None:
                num_classes = self.num_classes
            else:
                num_classes = 10  # Guess if not given, can do better by looking at y
        elif self.problem_type == SOFTCLASS:  # TODO: delete this elif if it's unnecessary.
            num_classes = y.shape[1]
        else:
            num_classes = 1
        return num_classes

    def _ag_params(self) -> set:
        return {"early_stop"}

    def _estimate_memory_usage(self, X, **kwargs):
        """
        Returns the expected peak memory usage in bytes of the XGBoost model during fit.

        The memory usage of XGBoost is primarily made up of two sources:

        1. The size of the data
        2. The size of the histogram cache
            Scales roughly by 5120*num_features*2^max_depth bytes
            For 10000 features and 6 max_depth, the histogram would be 3.2 GB.
        """
        num_classes = self.num_classes if self.num_classes else 1  # self.num_classes could be None after initialization if it's a regression problem
        data_mem_usage = get_approximate_df_mem_usage(X).sum()
        data_mem_usage_bytes = data_mem_usage * 7 + data_mem_usage / 4 * num_classes  # TODO: Extremely crude approximation, can be vastly improved

        params = self._get_model_params()
        max_bin = params.get("max_bin", 256)
        max_depth = params.get("max_depth", 6)
        # Formula based on manual testing, aligns with LightGBM histogram sizes
        #  This approximation is less accurate than it is for LightGBM and CatBoost.
        #  Note that max_depth didn't appear to reduce memory usage below 6, and it was unclear if it increased memory usage above 6.
        if max_depth < 7:
            depth_modifier = math.pow(2, 6)
        elif max_depth < 9:
            depth_modifier = math.pow(2, max_depth)
        else:
            depth_modifier = math.pow(2, max_depth - 1)
        histogram_mem_usage_bytes = 20 * depth_modifier * len(X.columns) * max_bin
        histogram_mem_usage_bytes *= 1.2  # Add a 20% buffer

        approx_mem_size_req = data_mem_usage_bytes + histogram_mem_usage_bytes
        return approx_mem_size_req

    def _validate_fit_memory_usage(self, mem_error_threshold: float = 1.0, mem_warning_threshold: float = 0.75, mem_size_threshold: int = 1e9, **kwargs):
        return super()._validate_fit_memory_usage(
            mem_error_threshold=mem_error_threshold, mem_warning_threshold=mem_warning_threshold, mem_size_threshold=mem_size_threshold, **kwargs
        )

    def get_minimum_resources(self, is_gpu_available=False):
        minimum_resources = {
            "num_cpus": 1,
        }
        if is_gpu_available:
            minimum_resources["num_gpus"] = 0.5
        return minimum_resources

    @disable_if_lite_mode(ret=(1, 0))
    def _get_default_resources(self):
        # logical=False is faster in training
        num_cpus = ResourceManager.get_cpu_count_psutil(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus

    def save(self, path: str = None, verbose=True) -> str:
        _model = self.model
        self.model = None
        if _model is not None:
            self._xgb_model_type = _model.__class__
        path = super().save(path=path, verbose=verbose)
        if _model is not None:
            # Halves disk usage compared to .json / .pkl
            _model.save_model(os.path.join(path, "xgb.ubj"))
        self.model = _model
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._xgb_model_type is not None:
            model.model = model._xgb_model_type()
            # Much faster to load using .ubj than .json (10x+ speedup)
            model.model.load_model(os.path.join(path, "xgb.ubj"))
            model._xgb_model_type = None
        return model

    def _more_tags(self):
        # `can_refit_full=True` because n_estimators is communicated at end of `_fit`:
        #  self.params_trained['n_estimators'] = bst.best_ntree_limit
        return {"can_refit_full": True}
