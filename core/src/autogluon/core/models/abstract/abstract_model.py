from __future__ import annotations

import copy
import gc
import inspect
import logging
import math
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.utils.distribute_utils import DistributedContext
from autogluon.common.utils.lite import disable_if_lite_mode
from autogluon.common.utils.log_utils import DuplicateFilter
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager, get_resource_manager
from autogluon.common.utils.try_import import try_import_ray
from autogluon.common.utils.utils import setup_outputdir

from ... import metrics
from ...calibrate.temperature_scaling import apply_temperature_scaling
from ...constants import (
    AG_ARG_PREFIX,
    AG_ARGS_FIT,
    BINARY,
    MULTICLASS,
    OBJECTIVES_TO_NORMALIZE,
    QUANTILE,
    REFIT_FULL_SUFFIX,
    REGRESSION,
    SOFTCLASS,
)
from ...data.label_cleaner import LabelCleaner
from ...hpo.constants import CUSTOM_BACKEND, RAY_BACKEND
from ...hpo.exceptions import EmptySearchSpace
from ...hpo.executors import HpoExecutor, HpoExecutorFactory
from ...metrics import Scorer
from ...utils import (
    compute_permutation_feature_importance,
    compute_weighted_metric,
    get_pred_from_proba,
    infer_eval_metric,
    infer_problem_type,
    normalize_pred_probas,
)
from ...utils.exceptions import NotEnoughMemoryError, NoValidFeatures, TimeLimitExceeded
from ...utils.loaders import load_pkl
from ...utils.savers import save_json, save_pkl
from ...utils.time import sample_df_for_time_func, time_func
from ._tags import _DEFAULT_CLASS_TAGS, _DEFAULT_TAGS
from .model_trial import model_trial, skip_hpo

logger = logging.getLogger(__name__)
dup_filter = DuplicateFilter()
logger.addFilter(dup_filter)


class AbstractModel:
    """
    Abstract model implementation from which all AutoGluon models inherit.

    Parameters
    ----------
    path : str, default = None
        Directory location to store all outputs.
        If None, a new unique time-stamped directory is chosen.
    name : str, default = None
        Name of the subdirectory inside path where model will be saved.
        The final model directory will be os.path.join(path, name)
        If None, defaults to the model's class name: self.__class__.__name__
    problem_type : str, default = None
        Type of prediction problem, i.e. is this a binary/multiclass classification or regression problem (options: 'binary', 'multiclass', 'regression').
        If None, will attempt to infer the problem type based on training data labels during training.
    eval_metric : :class:`autogluon.core.metrics.Scorer` or str, default = None
        Metric by which predictions will be ultimately evaluated on test data.
        This only impacts `model.score()`, as eval_metric is not used during training.

        If `eval_metric = None`, it is automatically chosen based on `problem_type`.
        Defaults to 'accuracy' for binary and multiclass classification and 'root_mean_squared_error' for regression.
        Otherwise, options for classification:
            ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
            'roc_auc', 'roc_auc_ovo_macro', 'average_precision', 'precision', 'precision_macro', 'precision_micro',
            'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss', 'pac_score']
        Options for regression:
            ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'r2']
        Options for quantile regression:
            ['pinball_loss']
        For more information on these options, see `sklearn.metrics`: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

        You can also pass your own evaluation function here as long as it follows formatting of the functions defined in folder `autogluon.core.metrics`.
    hyperparameters : dict, default = None
        Hyperparameters that will be used by the model (can be search spaces instead of fixed values).
        If None, model defaults are used. This is identical to passing an empty dictionary.
    """

    model_file_name = "model.pkl"
    model_info_name = "info.pkl"
    model_info_json_name = "info.json"

    def __init__(
        self,
        path: str = None,
        name: str = None,
        problem_type: str = None,
        eval_metric: Union[str, metrics.Scorer] = None,
        hyperparameters: dict = None,
    ):
        if name is None:
            self.name = self.__class__.__name__
            logger.log(20, f"Warning: No name was specified for model, defaulting to class name: {self.name}")
        else:
            self.name = name  # TODO: v0.1 Consider setting to self._name and having self.name be a property so self.name can't be set outside of self.rename()

        self.path_root = path
        if self.path_root is None:
            path_suffix = self.name
            # TODO: Would be ideal to not create dir, but still track that it is unique. However, this isn't possible to do without a global list of used dirs or using UUID.
            path_cur = setup_outputdir(path=None, create_dir=True, path_suffix=path_suffix)
            self.path_root = path_cur.rsplit(self.path_suffix, 1)[0]
            logger.log(20, f"Warning: No path was specified for model, defaulting to: {self.path_root}")

        self.path = self.create_contexts(os.path.join(self.path_root, self.path_suffix))  # TODO: Make this path a function for consistency.

        self.num_classes = None
        self.quantile_levels = None
        self.model = None
        self.problem_type = problem_type

        # whether to calibrate predictions via conformal methods
        self.conformalize = None

        if eval_metric is not None:
            self.eval_metric = metrics.get_metric(eval_metric, self.problem_type, "eval_metric")  # Note: we require higher values = better performance
        else:
            self.eval_metric = None
        self.stopping_metric = None
        self.normalize_pred_probas = None

        self.features = None  # External features, do not use internally
        self.feature_metadata = None  # External feature metadata, do not use internally
        self._features_internal = None  # Internal features, safe to use internally via the `_features` property
        self._feature_metadata = None  # Internal feature metadata, safe to use internally
        self._is_features_in_same_as_ex = None  # Whether self.features == self._features_internal

        self.fit_time = None  # Time taken to fit in seconds (Training data)
        self.predict_time = None  # Time taken to predict in seconds (Validation data)
        self.predict_1_time = None  # Time taken to predict 1 row of data in seconds (with batch size `predict_1_batch_size` in params_aux)
        self.compile_time = None  # Time taken to compile the model in seconds
        self.val_score = None  # Score with eval_metric (Validation data)

        self._user_params, self._user_params_aux = self._init_user_params(params=hyperparameters)

        self.params = {}
        self.params_aux = {}
        self.params_trained = dict()
        self.nondefault_params = []
        self._is_initialized = False
        self._is_fit_metadata_registered = False
        self._fit_metadata = dict()

        self._compiler = None

    @classmethod
    def _init_user_params(
        cls, params: Optional[Dict[str, Any]] = None, ag_args_fit: str = AG_ARGS_FIT, ag_arg_prefix: str = AG_ARG_PREFIX
    ) -> (Dict[str, Any], Dict[str, Any]):
        """
        Given the user-specified hyperparameters, split into `params` and `params_aux`.

        Parameters
        ----------
        params : Optional[Dict[str, Any]], default = None
            The model hyperparameters dictionary
        ag_args_fit : str, default = "ag_args_fit"
            The params key to look for that contains params_aux.
            If the key is present, the value is used for params_aux and popped from params.
            If no such key is found, then initialize params_aux as an empty dictionary.
        ag_arg_prefix : str, default = "ag."
            The key prefix to look for that indicates a parameter is intended for params_aux.
            If None, this logic is skipped.
            If a key starts with this prefix, it is popped from params and added to params_aux with the prefix removed.
            For example:
                input:  params={'ag.foo': 2, 'abc': 7}, params_aux={'bar': 3}, and ag_arg_prefix='.ag',
                output: params={'abc': 7}, params_aux={'bar': 3, 'foo': 2}
            In cases where the key is specified multiple times, the value of the key with the prefix will always take priority.
            A warning will be logged if a key is present multiple times.
            For example, given the most complex scenario:
                input:  params={'ag.foo': 1, 'foo': 2, 'ag_args_fit': {'ag.foo': 3, 'foo': 4}}
                output: params={'foo': 2}, params_aux={'foo': 1}

        Returns
        -------
        params, params_aux : (Dict[str, Any], Dict[str, Any])
            params will contain the native model hyperparameters
            params_aux will contain special auxiliary hyperparameters
        """
        params = copy.deepcopy(params) if params is not None else dict()
        assert isinstance(params, dict), f"Invalid dtype of params! Expected dict, but got {type(params)}"
        for k in params.keys():
            if not isinstance(k, str):
                logger.warning(
                    f"Warning: Specified {cls.__name__} hyperparameter key is not of type str: {k} (type={type(k)}). "
                    f"There might be a bug in your configuration."
                )

        params_aux = params.pop(ag_args_fit, dict())
        if params_aux is None:
            params_aux = dict()
        assert isinstance(params_aux, dict), f"Invalid dtype of params_aux! Expected dict, but got {type(params_aux)}"
        if ag_arg_prefix is not None:
            param_aux_keys = list(params_aux.keys())
            for k in param_aux_keys:
                if isinstance(k, str) and k.startswith(ag_arg_prefix):
                    k_no_prefix = k[len(ag_arg_prefix) :]
                    if k_no_prefix in params_aux:
                        logger.warning(
                            f'Warning: {cls.__name__} hyperparameter "{k}" is present '
                            f'in `ag_args_fit` as both "{k}" and "{k_no_prefix}". '
                            f'Will use "{k}" and ignore "{k_no_prefix}".'
                        )
                    params_aux[k_no_prefix] = params_aux.pop(k)
            param_keys = list(params.keys())
            for k in param_keys:
                if isinstance(k, str) and k.startswith(ag_arg_prefix):
                    k_no_prefix = k[len(ag_arg_prefix) :]
                    if k_no_prefix in params_aux:
                        logger.warning(
                            f'Warning: {cls.__name__} hyperparameter "{k}" is present '
                            f"in both `ag_args_fit` and `hyperparameters`. "
                            f"Will use `hyperparameters` value."
                        )
                    params_aux[k_no_prefix] = params.pop(k)
        return params, params_aux

    def _init_params(self):
        """Initializes model hyperparameters"""
        hyperparameters = self._user_params
        self._set_default_params()
        self.nondefault_params = []
        if hyperparameters is not None:
            self.params.update(hyperparameters)
            self.nondefault_params = list(hyperparameters.keys())[:]  # These are hyperparameters that user has specified.
        self.params_trained = dict()

    def _init_params_aux(self):
        """
        Initializes auxiliary hyperparameters.
        These parameters are generally not model specific and can have a wide variety of effects.
        For documentation on some of the available options and their defaults, refer to `self._get_default_auxiliary_params`.
        """
        hyperparameters_aux = self._user_params_aux
        self._set_default_auxiliary_params()
        if hyperparameters_aux is not None:
            self.params_aux.update(hyperparameters_aux)

    @property
    def path_suffix(self) -> str:
        return self.name

    def is_valid(self) -> bool:
        """
        Returns True if the model is capable of inference on new data (if normal model) or has produced out-of-fold predictions (if bagged model)
        This indicates whether the model can be used as a base model to fit a stack ensemble model.
        """
        return self.is_fit()

    def is_initialized(self) -> bool:
        """
        Returns True if the model is initialized.
        This indicates whether the model has inferred various information such as problem_type and num_classes.
        A model is automatically initialized when `.fit` or `.hyperparameter_tune` are called.
        """
        return self._is_initialized

    def can_infer(self) -> bool:
        """Returns True if the model is capable of inference on new data."""
        return self.is_valid()

    def is_fit(self) -> bool:
        """Returns True if the model has been fit."""
        return self.model is not None

    def can_fit(self) -> bool:
        """Returns True if the model can be fit."""
        return not self.is_fit()

    def can_predict_proba(self) -> bool:
        """Returns True if the model can predict probabilities."""
        # TODO: v1.0: Enforce this by raising if `predict_proba` called when this is False.
        return self.can_infer() and self.problem_type in [BINARY, MULTICLASS, SOFTCLASS]

    # TODO: v0.1 update to be aligned with _set_default_auxiliary_params(), add _get_default_params()
    def _set_default_params(self):
        pass

    def _set_default_auxiliary_params(self):
        """
        Sets the default aux parameters of the model.
        This method should not be extended by inheriting models, instead extend _get_default_auxiliary_params.
        """
        # TODO: Consider adding to get_info() output
        default_auxiliary_params = self._get_default_auxiliary_params()
        for key, value in default_auxiliary_params.items():
            self._set_default_param_value(key, value, params=self.params_aux)

    # TODO: v0.1 consider adding documentation to each model highlighting which feature dtypes are valid
    def _get_default_auxiliary_params(self) -> dict:
        """
        Dictionary of auxiliary parameters that dictate various model-agnostic logic, such as:
            Which column dtypes are filtered out of the input data, or how much memory the model is allowed to use.
        """
        default_auxiliary_params = dict(
            max_memory_usage_ratio=1.0,  # Ratio of memory usage allowed by the model. Values > 1.0 have an increased risk of causing OOM errors. Used in memory checks during model training to avoid OOM errors.
            # TODO: Add more params
            # max_memory_usage=None,
            # max_disk_usage=None,
            max_time_limit_ratio=1.0,  # ratio of given time_limit to use during fit(). If time_limit == 10 and max_time_limit_ratio=0.3, time_limit would be changed to 3.
            max_time_limit=None,  # max time_limit value during fit(). If the provided time_limit is greater than this value, it will be replaced by max_time_limit. Occurs after max_time_limit_ratio is applied.
            min_time_limit=0,  # min time_limit value during fit(). If the provided time_limit is less than this value, it will be replaced by min_time_limit. Occurs after max_time_limit is applied.
            # drop_unique=True,  # Whether to drop features that have only 1 unique value
            # num_cpus=None,
            # num_gpus=None,
            # ignore_hpo=False,
            # max_early_stopping_rounds=None,
            # TODO: add option for only top-k ngrams
            valid_raw_types=None,  # If a feature's raw type is not in this list, it is pruned.
            valid_special_types=None,  # If a feature has a special type not in this list, it is pruned.
            ignored_type_group_special=None,  # List, drops any features in `self.feature_metadata.type_group_map_special[type]` for type in `ignored_type_group_special`. | Currently undocumented in task.
            ignored_type_group_raw=None,  # List, drops any features in `self.feature_metadata.type_group_map_raw[type]` for type in `ignored_type_group_raw`. | Currently undocumented in task.
            # Kwargs for `autogluon.tabular.features.feature_metadata.FeatureMetadata.get_features()`.
            #  Overrides valid_raw_types, valid_special_types, ignored_type_group_special and ignored_type_group_raw. | Currently undocumented in task.
            get_features_kwargs=None,
            # TODO: v0.1 Document get_features_kwargs_extra in task.fit
            get_features_kwargs_extra=None,  # If not None, applies an additional feature filter to the result of get_feature_kwargs. This should be reserved for users and be None by default. | Currently undocumented in task.
            predict_1_batch_size=None,  # If not None, calculates `self.predict_1_time` at end of fit call by predicting on this many rows of data.
            temperature_scalar=None,  # Temperature scaling parameter that is set post-fit if calibrate=True during TabularPredictor.fit() on the model with the best validation score and eval_metric="log_loss".
        )
        return default_auxiliary_params

    def _set_default_param_value(self, param_name, param_value, params=None):
        if params is None:
            params = self.params
        if param_name not in params:
            params[param_name] = param_value

    def _get_default_searchspace(self) -> dict:
        """
        Get the default hyperparameter searchspace of the model.
        See `autogluon.common.space` for available space classes.
        Returns
        -------
        dict of hyperparameter search spaces.
        """
        return {}

    def _get_search_space(self):
        """Sets up default search space for HPO. Each hyperparameter which user did not specify is converted from
        default fixed value to default search space.
        """
        def_search_space = self._get_default_searchspace().copy()
        # Note: when subclassing AbstractModel, you must define or import get_default_searchspace() from the appropriate location.
        for key in self.nondefault_params:  # delete all user-specified hyperparams from the default search space
            def_search_space.pop(key, None)
        params = self._get_params()
        params.update(def_search_space)
        return params

    # TODO: v0.1 Change this to update path_root only, path change to property
    def set_contexts(self, path_context):
        self.path = self.create_contexts(path_context)
        self.path_root = self.path.rsplit(self.path_suffix, 1)[0]

    @staticmethod
    def create_contexts(path_context: str) -> str:
        path = path_context
        return path

    def rename(self, name: str):
        """Renames the model and updates self.path to reflect the updated name."""
        if self.name is not None and len(self.name) > 0:
            self.path = os.path.join(os.path.dirname(self.path), name)
        else:
            self.path = os.path.join(self.path, name)
        self.name = name

    def preprocess(self, X, preprocess_nonadaptive: bool = True, preprocess_stateful: bool = True, **kwargs):
        """
        Preprocesses the input data into internal form ready for fitting or inference.
        It is not recommended to override this method, as it is closely tied to multi-layer stacking logic. Instead, override `_preprocess`.
        """
        if preprocess_nonadaptive:
            X = self._preprocess_nonadaptive(X, **kwargs)
        if preprocess_stateful:
            X = self._preprocess(X, **kwargs)
        return X

    # TODO: Remove kwargs?
    def _preprocess(self, X: pd.DataFrame, **kwargs):
        """
        Data transformation logic should be added here.

        Input data should not be trusted to be in a clean and ideal form, while the output should be in an ideal form for training/inference.
        Examples of logic that should be added here include missing value handling, rescaling of features (if neural network), etc.
        If implementing a new model, it is recommended to refer to existing model implementations and experiment using toy datasets.

        In bagged ensembles, preprocessing code that lives in `_preprocess` will be executed on each child model once per inference call.
        If preprocessing code could produce different output depending on the child model that processes the input data, then it must live here.
        When in doubt, put preprocessing code here instead of in `_preprocess_nonadaptive`.
        """
        if not self._is_features_in_same_as_ex:
            X = X[self._features]
        return X

    # TODO: Remove kwargs?
    def _preprocess_nonadaptive(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Note: This method is intended for advanced users. It is usually sufficient to implement all preprocessing in `_preprocess` and leave this method untouched.
            The potential benefit of implementing preprocessing in this method is an inference speedup when used in a bagged ensemble.
        Data transformation logic that is non-stateful or ignores internal data values beyond feature dtypes should be added here.
        In bagged ensembles, preprocessing code that lives here will be executed only once per inference call regardless of the number of child models.
        If preprocessing code will produce the same output regardless of which child model processes the input data, then it should live here to avoid redundant repeated processing for each child.
        This means this method cannot be used for data normalization. Refer to `_preprocess` instead.
        """
        # TODO: In online-inference this becomes expensive, add option to remove it (only safe in controlled environment where it is already known features are present
        if list(X.columns) != self.features:
            X = X[self.features]
        return X

    def _preprocess_set_features(self, X: pd.DataFrame, feature_metadata: FeatureMetadata = None):
        """
        Infers self.features and self.feature_metadata from X.

        If no valid features were found, a NoValidFeatures exception is raised.
        """
        if self.features is None:
            self.features = list(X.columns)
        # TODO: Consider changing how this works or where it is done
        if feature_metadata is None:
            feature_metadata = FeatureMetadata.from_df(X)
        else:
            feature_metadata = copy.deepcopy(feature_metadata)
        get_features_kwargs = self.params_aux.get("get_features_kwargs", None)
        if get_features_kwargs is not None:
            valid_features = feature_metadata.get_features(**get_features_kwargs)
        else:
            valid_raw_types = self.params_aux.get("valid_raw_types", None)
            valid_special_types = self.params_aux.get("valid_special_types", None)
            ignored_type_group_raw = self.params_aux.get("ignored_type_group_raw", None)
            ignored_type_group_special = self.params_aux.get("ignored_type_group_special", None)
            valid_features = feature_metadata.get_features(
                valid_raw_types=valid_raw_types,
                valid_special_types=valid_special_types,
                invalid_raw_types=ignored_type_group_raw,
                invalid_special_types=ignored_type_group_special,
            )
        get_features_kwargs_extra = self.params_aux.get("get_features_kwargs_extra", None)
        if get_features_kwargs_extra is not None:
            valid_features_extra = feature_metadata.get_features(**get_features_kwargs_extra)
            valid_features = [feature for feature in valid_features if feature in valid_features_extra]
        dropped_features = [feature for feature in self.features if feature not in valid_features]
        logger.log(10, f"\tDropped {len(dropped_features)} of {len(self.features)} features.")
        self.features = [feature for feature in self.features if feature in valid_features]
        self.feature_metadata = feature_metadata.keep_features(self.features)
        error_if_no_features = self.params_aux.get("error_if_no_features", True)
        if error_if_no_features and not self.features:
            raise NoValidFeatures
        # TODO: If unique_counts == 2 (including NaN), then treat as boolean
        if self.params_aux.get("drop_unique", True):
            # TODO: Could this be optimized to be faster? This might be a bit slow for large data.
            unique_counts = X[self.features].nunique(axis=0, dropna=False)
            columns_to_drop = list(unique_counts[unique_counts < 2].index)
            features_to_drop_internal = columns_to_drop
            if not features_to_drop_internal:
                features_to_drop_internal = None
        else:
            features_to_drop_internal = None
        if features_to_drop_internal is not None:
            logger.log(10, f"\tDropped {len(features_to_drop_internal)} of {len(self.features)} internal features: {features_to_drop_internal}")
            self._features_internal = [feature for feature in self.features if feature not in features_to_drop_internal]
            self._feature_metadata = self.feature_metadata.keep_features(self._features_internal)
            self._is_features_in_same_as_ex = False
        else:
            self._features_internal = self.features
            self._feature_metadata = self.feature_metadata
            self._is_features_in_same_as_ex = True
        if error_if_no_features and not self._features_internal:
            raise NoValidFeatures

    def _preprocess_fit_args(self, **kwargs) -> dict:
        sample_weight = kwargs.get("sample_weight", None)
        if sample_weight is not None and isinstance(sample_weight, str):
            raise ValueError("In model.fit(), sample_weight should be array of sample weight values, not string.")
        time_limit = kwargs.get("time_limit", None)
        time_limit_og = time_limit
        max_time_limit_ratio = self.params_aux.get("max_time_limit_ratio", 1)
        if time_limit is not None:
            time_limit *= max_time_limit_ratio
        max_time_limit = self.params_aux.get("max_time_limit", None)
        if max_time_limit is not None:
            if time_limit is None:
                time_limit = max_time_limit
            else:
                time_limit = min(time_limit, max_time_limit)
        min_time_limit = self.params_aux.get("min_time_limit", 0)
        if min_time_limit is None:
            time_limit = min_time_limit
        elif time_limit is not None:
            time_limit = max(time_limit, min_time_limit)
        kwargs["time_limit"] = time_limit
        if time_limit_og != time_limit:
            time_limit_og_str = f"{time_limit_og:.2f}s" if time_limit_og is not None else "None"
            time_limit_str = f"{time_limit:.2f}s" if time_limit is not None else "None"
            logger.log(
                20,
                f"\tTime limit adjusted due to model hyperparameters: "
                f"{time_limit_og_str} -> {time_limit_str} "
                f"(ag.max_time_limit={max_time_limit}, "
                f"ag.max_time_limit_ratio={max_time_limit_ratio}, "
                f"ag.min_time_limit={min_time_limit})",
            )
        kwargs = self._preprocess_fit_resources(**kwargs)
        return kwargs

    def initialize(self, **kwargs) -> dict:
        if not self._is_initialized:
            self._initialize(**kwargs)
            self._is_initialized = True

        kwargs.pop("feature_metadata", None)
        kwargs.pop("num_classes", None)
        return kwargs

    def _initialize(self, X=None, y=None, feature_metadata=None, num_classes=None, **kwargs):
        if num_classes is not None:
            self.num_classes = num_classes
        if y is not None:
            if self.problem_type is None:
                self.problem_type = infer_problem_type(y=y)
            if self.num_classes is None:
                label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y)
                self.num_classes = label_cleaner.num_classes

        self._init_params_aux()

        self._init_misc(X=X, y=y, feature_metadata=feature_metadata, num_classes=num_classes, **kwargs)

        if X is not None:
            self._preprocess_set_features(X=X, feature_metadata=feature_metadata)

        self._init_params()

    def _init_misc(self, **kwargs):
        """Initialize parameters that depend on self.params_aux being initialized"""
        if self.eval_metric is None:
            self.eval_metric = infer_eval_metric(problem_type=self.problem_type)
            logger.log(
                20,
                f"Model {self.name}'s eval_metric inferred to be '{self.eval_metric.name}' because problem_type='{self.problem_type}' and eval_metric was not specified during init.",
            )
        self.eval_metric = metrics.get_metric(self.eval_metric, self.problem_type, "eval_metric")  # Note: we require higher values = better performance

        self.stopping_metric = self.params_aux.get("stopping_metric", self._get_default_stopping_metric())
        self.stopping_metric = metrics.get_metric(self.stopping_metric, self.problem_type, "stopping_metric")

        self.quantile_levels = self.params_aux.get("quantile_levels", None)

        if self.eval_metric.name in OBJECTIVES_TO_NORMALIZE:
            self.normalize_pred_probas = True
            logger.debug(f"{self.name} predicted probabilities will be transformed to never =0 since eval_metric='{self.eval_metric.name}'")
        else:
            self.normalize_pred_probas = False

    def _process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble(
        self, system_resource, user_specified_total_resource, user_specified_ensemble_resource, resource_type, k_fold
    ):
        if user_specified_total_resource == "auto":
            user_specified_total_resource = math.inf

        # retrieve model level requirement when self is bagged model
        user_specified_model_level_resource = self._get_child_aux_val(key=resource_type, default=None)
        if user_specified_model_level_resource is not None:
            assert user_specified_model_level_resource <= system_resource, f"Specified {resource_type} per model base is more than the total: {system_resource}"
        user_specified_lower_level_resource = user_specified_ensemble_resource
        if user_specified_ensemble_resource is not None:
            if user_specified_model_level_resource is not None:
                user_specified_lower_level_resource = min(
                    user_specified_model_level_resource * k_fold, user_specified_ensemble_resource, system_resource, user_specified_total_resource
                )
        else:
            if user_specified_model_level_resource is not None:
                user_specified_lower_level_resource = min(user_specified_model_level_resource * k_fold, system_resource, user_specified_total_resource)
        return user_specified_lower_level_resource

    def _calculate_total_resources(
        self, silent: bool = False, total_resources: Optional[Dict[str, Union[int, float]]] = None, parallel_hpo: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Process user-specified total resources.
        Sanity checks will be done to user-specified total resources to make sure it's legit.
        When user-specified resources are not defined, will instead look at model's default resource requirements.

        Will set the calculated total resources in kwargs and return it
        """
        resource_manager = get_resource_manager()
        system_num_cpus = resource_manager.get_cpu_count()
        system_num_gpus = resource_manager.get_gpu_count_all()
        if total_resources is None:
            total_resources = {}
        num_cpus = total_resources.get("num_cpus", "auto")
        num_gpus = total_resources.get("num_gpus", "auto")
        default_num_cpus, default_num_gpus = self._get_default_resources()
        # This could be resource requirement for bagged model or individual model
        user_specified_lower_level_num_cpus = self._user_params_aux.get("num_cpus", None)
        user_specified_lower_level_num_gpus = self._user_params_aux.get("num_gpus", None)
        if user_specified_lower_level_num_cpus is not None:
            assert (
                user_specified_lower_level_num_cpus <= system_num_cpus
            ), f"Specified num_cpus per {self.__class__.__name__} is more than the total: {system_num_cpus}"
        if user_specified_lower_level_num_gpus is not None:
            assert (
                user_specified_lower_level_num_gpus <= system_num_cpus
            ), f"Specified num_gpus per {self.__class__.__name__} is more than the total: {system_num_cpus}"
        k_fold = kwargs.get("k_fold", None)
        if k_fold is not None and k_fold > 0:
            # bagged model will look ag_args_ensemble and ag_args_fit internally to determine resources
            # pass all resources here by default
            default_num_cpus = system_num_cpus
            default_num_gpus = system_num_gpus if default_num_gpus > 0 else 0
            user_specified_lower_level_num_cpus = self._process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble(
                system_resource=system_num_cpus,
                user_specified_total_resource=num_cpus,
                user_specified_ensemble_resource=user_specified_lower_level_num_cpus,
                resource_type="num_cpus",
                k_fold=k_fold,
            )
            user_specified_lower_level_num_gpus = self._process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble(
                system_resource=system_num_gpus,
                user_specified_total_resource=num_gpus,
                user_specified_ensemble_resource=user_specified_lower_level_num_gpus,
                resource_type="num_gpus",
                k_fold=k_fold,
            )
        if num_cpus != "auto" and num_cpus > system_num_cpus:
            logger.warning(f"Specified total num_cpus: {num_cpus}, but only {system_num_cpus} are available. Will use {system_num_cpus} instead")
            num_cpus = system_num_cpus
        if num_gpus != "auto" and num_gpus > system_num_gpus:
            logger.warning(f"Specified total num_gpus: {num_gpus}, but only {system_num_gpus} are available. Will use {system_num_gpus} instead")
            num_gpus = system_num_gpus
        if num_cpus == "auto":
            if user_specified_lower_level_num_cpus is not None:
                if not parallel_hpo:
                    num_cpus = user_specified_lower_level_num_cpus
                else:
                    num_cpus = system_num_cpus
            else:
                if not parallel_hpo:
                    num_cpus = default_num_cpus
                else:
                    num_cpus = system_num_cpus
        else:
            if not parallel_hpo:
                if user_specified_lower_level_num_cpus is not None:
                    assert (
                        user_specified_lower_level_num_cpus <= num_cpus
                    ), f"Specified num_cpus per {self.__class__.__name__} is more than the total specified: {num_cpus}"
                    num_cpus = user_specified_lower_level_num_cpus
        if num_gpus == "auto":
            if user_specified_lower_level_num_gpus is not None:
                if not parallel_hpo:
                    num_gpus = user_specified_lower_level_num_gpus
                else:
                    num_gpus = system_num_gpus if user_specified_lower_level_num_gpus > 0 else 0
            else:
                if not parallel_hpo:
                    num_gpus = default_num_gpus
                else:
                    num_gpus = system_num_gpus if default_num_gpus > 0 else 0
        else:
            if not parallel_hpo:
                if user_specified_lower_level_num_gpus is not None:
                    assert (
                        user_specified_lower_level_num_gpus <= num_gpus
                    ), f"Specified num_gpus per {self.__class__.__name__} is more than the total specified: {num_gpus}"
                    num_gpus = user_specified_lower_level_num_gpus

        minimum_model_resources = self.get_minimum_resources(is_gpu_available=(num_gpus > 0))
        minimum_model_num_cpus = minimum_model_resources.get("num_cpus", 1)
        minimum_model_num_gpus = minimum_model_resources.get("num_gpus", 0)

        assert system_num_cpus >= num_cpus
        assert system_num_gpus >= num_gpus

        assert (
            system_num_cpus >= minimum_model_num_cpus
        ), f"The total system num_cpus={system_num_cpus} is less than minimum num_cpus={minimum_model_num_cpus} to fit {self.__class__.__name__}. Consider using a machine with more CPUs."
        assert (
            system_num_gpus >= minimum_model_num_gpus
        ), f"The total system num_gpus={system_num_gpus} is less than minimum num_gpus={minimum_model_num_gpus} to fit {self.__class__.__name__}. Consider using a machine with more GPUs."

        assert (
            num_cpus >= minimum_model_num_cpus
        ), f"Specified num_cpus={num_cpus} per {self.__class__.__name__} is less than minimum num_cpus={minimum_model_num_cpus}"
        assert (
            num_gpus >= minimum_model_num_gpus
        ), f"Specified num_gpus={num_gpus} per {self.__class__.__name__} is less than minimum num_gpus={minimum_model_num_gpus}"

        kwargs["num_cpus"] = num_cpus
        kwargs["num_gpus"] = num_gpus
        if not silent:
            logger.log(15, f"\tFitting {self.name} with 'num_gpus': {kwargs['num_gpus']}, 'num_cpus': {kwargs['num_cpus']}")

        return kwargs

    def _preprocess_fit_resources(
        self, silent: bool = False, total_resources: Optional[Dict[str, Union[int, float]]] = None, parallel_hpo: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        This function should be called to process user-specified total resources.
        Sanity checks will be done to user-specified total resources to make sure it's legit.
        When user-specified resources are not defined, will instead look at model's default resource requirements.

        When kwargs contains `num_cpus` and `num_gpus` means this total resources has been calculated by previous layers(i.e. bagged model to model base).
        Will respect this value and check if there's specific maximum resource requirements and enforce those

        Will set the calculated resources in kwargs and return it
        """
        if "num_cpus" in kwargs and "num_gpus" in kwargs:
            # This value will only be passed by autogluon through previous layers(i.e. bagged model to model base).
            # We respect this value with highest priority
            # They should always be set to valid values
            enforced_num_cpus = kwargs.get("num_cpus", None)
            enforced_num_gpus = kwargs.get("num_gpus", None)
            assert enforced_num_cpus is not None and enforced_num_cpus != "auto" and enforced_num_gpus is not None and enforced_num_gpus != "auto"
            # The logic below is needed because ray cluster is running some process in the backend even when it's ready to be used
            # Trying to use all cores on the machine could lead to resource contention situation
            # TODO: remove this logic if ray team can identify what's going on underneath and how to workaround
            max_resources = self._get_maximum_resources()
            max_num_cpus = max_resources.get("num_cpus", None)
            max_num_gpus = max_resources.get("num_gpus", None)
            if max_num_gpus is not None:
                enforced_num_gpus = min(max_num_gpus, enforced_num_gpus)
            if DistributedContext.is_distributed_mode():
                minimum_model_resources = self.get_minimum_resources(is_gpu_available=(enforced_num_gpus > 0))
                minimum_model_num_cpus = minimum_model_resources.get("num_cpus", 1)
                enforced_num_cpus = max(minimum_model_num_cpus, enforced_num_cpus - 2)  # leave some cpu resources for process running by cluster nodes
            if max_num_cpus is not None:
                enforced_num_cpus = min(max_num_cpus, enforced_num_cpus)
            kwargs["num_cpus"] = enforced_num_cpus
            kwargs["num_gpus"] = enforced_num_gpus
            return kwargs

        return self._calculate_total_resources(silent=silent, total_resources=total_resources, parallel_hpo=parallel_hpo, **kwargs)

    def _register_fit_metadata(self, **kwargs):
        """
        Used to track properties of the inputs received during fit, such as if validation data was present.
        """
        if not self._is_fit_metadata_registered:
            self._fit_metadata = self._compute_fit_metadata(**kwargs)
            self._is_fit_metadata_registered = True

    def _compute_fit_metadata(self, X_val: pd.DataFrame = None, X_unlabeled: pd.DataFrame = None, **kwargs) -> dict:
        fit_metadata = dict(val_in_fit=X_val is not None, unlabeled_in_fit=X_unlabeled is not None)
        return fit_metadata

    def get_fit_metadata(self) -> dict:
        """
        Returns dictionary of metadata related to model fit that isn't related to hyperparameters.
        Must be called after model has been fit.
        """
        assert self._is_fit_metadata_registered, "fit_metadata must be registered before calling get_fit_metadata()!"
        fit_metadata = dict()
        fit_metadata.update(self._fit_metadata)
        fit_metadata["predict_1_batch_size"] = self._get_child_aux_val(key="predict_1_batch_size", default=None)
        return fit_metadata

    def _get_child_aux_val(self, key: str, default=None):
        """
        Get aux val of child model (or self if no child)
        This is necessary to get a parameter value that is constant across all children without having to load the children after fitting.
        """
        assert self.is_initialized(), "Model must be initialized before calling self._get_child_aux_val!"
        return self.params_aux.get(key, default)

    def fit(self, **kwargs):
        """
        Fit model to predict values in y based on X.

        Models should not override the `fit` method, but instead override the `_fit` method which has the same arguments.

        Parameters
        ----------
        X : DataFrame
            The training data features.
        y : Series
            The training data ground truth labels.
        X_val : DataFrame, default = None
            The validation data features.
            If None, early stopping via validation score will be disabled.
        y_val : Series, default = None
            The validation data ground truth labels.
            If None, early stopping via validation score will be disabled.
        X_unlabeled : DataFrame, default = None
            Unlabeled data features.
            Models may optionally implement logic which leverages unlabeled data to improve model accuracy.
        time_limit : float, default = None
            Time limit in seconds to adhere to when fitting model.
            Ideally, model should early stop during fit to avoid going over the time limit if specified.
        sample_weight : Series, default = None
            The training data sample weights.
            Models may optionally leverage sample weights during fit.
            If None, model decides. Typically, models assume uniform sample weight.
        sample_weight_val : Series, default = None
            The validation data sample weights.
            If None, model decides. Typically, models assume uniform sample weight.
        num_cpus : int, default = 'auto'
            How many CPUs to use during fit.
            This is counted in virtual cores, not in physical cores.
            If 'auto', model decides.
        num_gpus : int, default = 'auto'
            How many GPUs to use during fit.
            If 'auto', model decides.
        feature_metadata : :class:`autogluon.common.features.feature_metadata.FeatureMetadata`, default = None
            Contains feature type information that can be used to identify special features such as text ngrams and datetime as well as which features are numerical vs categorical.
            If None, feature_metadata is inferred during fit.
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            verbosity 4: logs every training iteration, and logs the most detailed information.
            verbosity 3: logs training iterations periodically, and logs more detailed information.
            verbosity 2: logs only important information.
            verbosity 1: logs only warnings and exceptions.
            verbosity 0: logs only exceptions.
        **kwargs :
            Any additional fit arguments a model supports.
        """
        kwargs = self.initialize(
            **kwargs
        )  # FIXME: This might have to go before self._preprocess_fit_args, but then time_limit might be incorrect in **kwargs init to initialize
        kwargs = self._preprocess_fit_args(**kwargs)
        if "time_limit" in kwargs and kwargs["time_limit"] is not None and kwargs["time_limit"] <= 0:
            logger.warning(f'\tWarning: Model has no time left to train, skipping model... (Time Left = {kwargs["time_limit"]:.1f}s)')
            raise TimeLimitExceeded

        self._register_fit_metadata(**kwargs)
        self.validate_fit_resources(**kwargs)
        self._validate_fit_memory_usage(**kwargs)
        out = self._fit(**kwargs)
        if out is None:
            out = self
        out = out._post_fit(**kwargs)
        return out

    def _post_fit(self, **kwargs):
        """
        Logic to perform at the end of `self.fit(...)`
        This should be focused around computing and saving metadata that is only possible post-fit.
        Parameters are identical to those passed to `self._fit(...)`.

        Returns
        -------
        Returns self
        """
        predict_1_batch_size = self.params_aux.get("predict_1_batch_size", None)
        if self.predict_1_time is None and predict_1_batch_size is not None and "X" in kwargs and kwargs["X"] is not None:
            X_1 = sample_df_for_time_func(df=kwargs["X"], sample_size=predict_1_batch_size)
            self.predict_1_time = time_func(f=self.predict, args=[X_1]) / len(X_1)
        return self

    def get_features(self) -> List[str]:
        assert self.is_fit(), "The model must be fit before calling the get_features method."
        if self.feature_metadata:
            return self.feature_metadata.get_features()
        else:
            return self.features

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        X_unlabeled: pd.DataFrame = None,
        time_limit: float = None,
        sample_weight: pd.Series = None,
        sample_weight_val: pd.Series = None,
        num_cpus: int = None,
        num_gpus: int = None,
        verbosity: int = 2,
        **kwargs,
    ):
        """
        Fit model to predict values in y based on X.

        Models should override this method with their custom model fit logic.
        X should not be assumed to be in a state ready for fitting to the inner model, and models may require special preprocessing in this method.
        It is very important that `X = self.preprocess(X)` is called within `_fit`, or else `predict` and `predict_proba` may not work as intended.
        It is also important that `_preprocess` is overwritten to properly clean the data.
        Examples of logic that should be handled by a model include missing value handling, rescaling of features (if neural network), etc.
        If implementing a new model, it is recommended to refer to existing model implementations and experiment using toy datasets.

        Refer to `fit` method for documentation.
        """

        X = self.preprocess(X)
        self.model = self.model.fit(X, y)

    def _apply_temperature_scaling(self, y_pred_proba: np.ndarray) -> np.ndarray:
        return apply_temperature_scaling(
            y_pred_proba=y_pred_proba,
            temperature_scalar=self.params_aux.get("temperature_scalar"),
            problem_type=self.problem_type,
        )

    def _apply_conformalization(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Return conformalized quantile predictions
        This is applicable only to quantile regression problems,
        and the given predictions (y_pred) are adjusted by adding quantile-level constants.
        """
        y_pred += self.conformalize
        return y_pred

    def predict(self, X, **kwargs) -> np.ndarray:
        """
        Returns class predictions of X.
        For binary and multiclass problems, this returns the predicted class labels as a Series.
        For regression problems, this returns the predicted values as a Series.
        """
        y_pred_proba = self.predict_proba(X, **kwargs)
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        return y_pred

    def predict_proba(self, X, normalize=None, **kwargs) -> np.ndarray:
        """
        Returns class prediction probabilities of X.
        For binary problems, this returns the positive class label probability as a Series.
        For multiclass problems, this returns the class label probabilities of each class as a DataFrame.
        For regression problems, this returns the predicted values as a Series.
        """
        if normalize is None:
            normalize = self.normalize_pred_probas
        y_pred_proba = self._predict_proba(X=X, **kwargs)
        if normalize:
            y_pred_proba = normalize_pred_probas(y_pred_proba, self.problem_type)
        y_pred_proba = y_pred_proba.astype(np.float32)

        if self.params_aux.get("temperature_scalar", None) is not None:
            y_pred_proba = self._apply_temperature_scaling(y_pred_proba)
        elif self.conformalize is not None:
            y_pred_proba = self._apply_conformalization(y_pred_proba)
        return y_pred_proba

    def predict_from_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Convert prediction probabilities to predictions.

        Parameters
        ----------
        y_pred_proba : np.ndarray
            The prediction probabilities to be converted to predictions.

        Returns
        -------
        y_pred : np.ndarray
            The predictions obtained from `y_pred_proba`.

        Examples
        --------
        >>> y_pred = predictor.predict(X)
        >>> y_pred_proba = predictor.predict_proba(X)
        >>>
        >>> # Identical to y_pred
        >>> y_pred_from_proba = predictor.predict_from_proba(y_pred_proba)
        """
        return get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type in [REGRESSION, QUANTILE]:
            y_pred = self.model.predict(X)
            return y_pred

        y_pred_proba = self.model.predict_proba(X)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _convert_proba_to_unified_form(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Ensures that y_pred_proba is in a consistent form across all models.
        For binary classification, converts y_pred_proba to a 1 dimensional array of prediction probabilities of the positive class.
        For multiclass and softclass classification, keeps y_pred_proba as a 2 dimensional array of prediction probabilities for each class.
        For regression, converts y_pred_proba to a 1 dimensional array of predictions.
        """
        if self.problem_type == REGRESSION:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            else:
                return y_pred_proba[:, 1]
        elif self.problem_type == BINARY:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif y_pred_proba.shape[1] > 2:  # Multiclass, Softclass
            return y_pred_proba
        else:  # Unknown problem type
            raise AssertionError(f'Unknown y_pred_proba format for `problem_type="{self.problem_type}"`.')

    def score(self, X, y, metric=None, sample_weight=None, **kwargs) -> np.ndarray:
        if metric is None:
            metric = self.eval_metric

        if metric.needs_pred or metric.needs_quantile:
            y_pred = self.predict(X=X, **kwargs)
        else:
            y_pred = self.predict_proba(X=X, **kwargs)
        return compute_weighted_metric(y, y_pred, metric, sample_weight, quantile_levels=self.quantile_levels)

    def score_with_y_pred_proba(self, y, y_pred_proba: np.ndarray, metric=None, sample_weight=None) -> np.ndarray:
        if metric is None:
            metric = self.eval_metric
        if metric.needs_pred:
            y_pred = self.predict_from_proba(y_pred_proba=y_pred_proba)
        else:
            y_pred = y_pred_proba
        return compute_weighted_metric(y, y_pred, metric, sample_weight, quantile_levels=self.quantile_levels)

    def save(self, path: str = None, verbose: bool = True) -> str:
        """
        Saves the model to disk.

        Parameters
        ----------
        path : str, default None
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            If None, self.path is used.
            The final model file is typically saved to os.path.join(path, self.model_file_name).
        verbose : bool, default True
            Whether to log the location of the saved file.

        Returns
        -------
        path : str
            Path to the saved model, minus the file name.
            Use this value to load the model from disk via cls.load(path), cls being the class of the model object, such as model = RFModel.load(path)
        """
        if path is None:
            path = self.path
        file_path = os.path.join(path, self.model_file_name)
        _model = self.model
        if self.model is not None:
            if self._compiler is None:
                self._compiler = self._get_compiler()
                if self._compiler is not None and not self._compiler.save_in_pkl:
                    self._compiler.save(model=self.model, path=path)
            if self._compiler is not None and not self._compiler.save_in_pkl:
                self.model = None  # Don't save model in pkl
        save_pkl.save(path=file_path, object=self, verbose=verbose)
        self.model = _model
        return path

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """
        Loads the model from disk to memory.

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
        file_path = os.path.join(path, cls.model_file_name)
        model = load_pkl.load(path=file_path, verbose=verbose)
        if reset_paths:
            model.set_contexts(path)
        if hasattr(model, "_compiler"):
            if model._compiler is not None and not model._compiler.save_in_pkl:
                model.model = model._compiler.load(path=path)
        return model

    # TODO: v1.0: Add docs
    def compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str] = None,
        silent: bool = False,
        importance_as_list: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute feature importance via permutation shuffling.

        Parameters
        ----------
        X
        y
        features
        silent
        importance_as_list
        kwargs

        Returns
        -------
        pd.DataFrame of feature importance
        """
        if self.features is not None:
            X = X[self.features]

        if not features:
            features = self.features
        else:
            features = list(features)

        # NOTE: Needed as bagged models 'features' attribute is not the same as childrens' 'features' attributes
        banned_features = [feature for feature in features if feature not in self.get_features()]
        features_to_check = [feature for feature in features if feature not in banned_features]

        if features_to_check:
            fi_df = self._compute_permutation_importance(X=X, y=y, features=features_to_check, silent=silent, importance_as_list=importance_as_list, **kwargs)
            n = fi_df.iloc[0]["n"] if len(fi_df) > 0 else 1
        else:
            fi_df = None
            n = kwargs.get("num_shuffle_sets", 1)

        if importance_as_list:
            banned_importance = [0] * n
            results_banned = pd.Series(data=[banned_importance for _ in range(len(banned_features))], index=banned_features, dtype="object")
        else:
            banned_importance = 0
            results_banned = pd.Series(data=[banned_importance for _ in range(len(banned_features))], index=banned_features, dtype="float64")

        results_banned_df = results_banned.to_frame(name="importance")
        results_banned_df["stddev"] = 0
        results_banned_df["n"] = n
        results_banned_df["n"] = results_banned_df["n"].astype("int64")
        if fi_df is not None:
            fi_df = pd.concat([fi_df, results_banned_df])
        else:
            fi_df = results_banned_df
        fi_df = fi_df.sort_values(ascending=False, by="importance")

        return fi_df

    # Compute feature importance via permutation importance
    # Note: Expensive to compute
    #  Time to compute is O(predict_time*num_features)
    def _compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
        eval_metric: Scorer = None,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        if eval_metric is None:
            eval_metric = self.eval_metric
        transform_func = self.preprocess
        if eval_metric.needs_pred:
            predict_func = self.predict
        else:
            predict_func = self.predict_proba
        transform_func_kwargs = dict(preprocess_stateful=False)
        predict_func_kwargs = dict(preprocess_nonadaptive=False)

        return compute_permutation_feature_importance(
            X=X,
            y=y,
            features=features,
            eval_metric=self.eval_metric,
            predict_func=predict_func,
            predict_func_kwargs=predict_func_kwargs,
            transform_func=transform_func,
            transform_func_kwargs=transform_func_kwargs,
            silent=silent,
            **kwargs,
        )

    def can_compile(self, compiler_configs: dict = None) -> bool:
        """
        Verify whether the model can be compiled with the compiler configuration.

        Parameters
        ----------
        compiler_configs : dict, default=None
            Model specific compiler options.
            This can be useful to specify the compiler backend for a specific model,
            e.g. {"RandomForest": {"compiler": "onnx"}}
        """
        if not self.is_fit():
            return False
        compiler = compiler_configs.get("compiler", "native")
        compiler_fallback_to_native = compiler_configs.get("compiler_fallback_to_native", False)

        compilers = self._valid_compilers()
        compiler_names = {c.name: c for c in compilers}
        if compiler is not None and compiler not in compiler_names:
            return False
        compiler_cls = compiler_names[compiler]
        if not compiler_cls.can_compile():
            if not compiler_fallback_to_native:
                return False
        return True

    def compile(self, compiler_configs: dict = None):
        """
        Compile the trained model for faster inference.

        NOTE:
        - The model is assumed to be fitted before compilation.
        - If save_in_pkl attribute of the compiler is False, self.model would be set to None.

        Parameters
        ----------
        compiler_configs : dict, default=None
            Model specific compiler options.
            This can be useful to specify the compiler backend for a specific model,
            e.g. {"RandomForest": {"compiler": "onnx"}}
        """
        assert self.is_fit(), "The model must be fit before calling the compile method."
        if compiler_configs is None:
            compiler_configs = {}
        compiler = compiler_configs.get("compiler", "native")
        batch_size = compiler_configs.get("batch_size", None)
        compiler_fallback_to_native = compiler_configs.get("compiler_fallback_to_native", False)

        self._compiler = self._get_compiler(compiler=compiler, compiler_fallback_to_native=compiler_fallback_to_native)
        if self._compiler is not None:
            input_types = self._get_input_types(batch_size=batch_size)
            self._compile(input_types=input_types)

    def _compile(self, **kwargs):
        """Take the compiler to perform actual compilation."""
        input_types = kwargs.get("input_types", self._get_input_types(batch_size=None))
        self.model = self._compiler.compile(model=self.model, path=self.path, input_types=input_types)

    # FIXME: This won't work for all models, and self._features is not
    # a trustworthy variable for final input shape
    def _get_input_types(self, batch_size=None) -> list:
        """
        Get input types as a list of tuples, containing shape and dtype.
        This can be useful for building the input_types argument for
        model compilation. This method can be overloaded in derived classes,
        in order to satisfy class-specific requirements.

        Parameters
        ----------
        batch_size : int, default=None
            The batch size for all returned input types.

        Returns
        -------
        List of (shape: Tuple[int], dtype: Any)
        shape: Tuple[int]
            A tuple that describes input
        dtype: Any, default=np.float32
            The element type in numpy dtype.
        """
        return [((batch_size, len(self._features)), np.float32)]

    def _default_compiler(self):
        """The default compiler for the underlining model."""
        return None

    def _valid_compilers(self) -> list:
        """A list of supported compilers for the underlining model."""
        return []

    def _get_compiler(self, compiler: str = None, compiler_fallback_to_native=False):
        """
        Verify whether the dependencies of the compiler class can be satisfied,
        and return the specified compiler from _valid_compilers.

        Parameters
        ----------
        compiler : str, default=None
            The specific compiler for model compilation.
        compiler_fallback_to_native : bool, default=False
            If this is True, the method would return native compiler when
            dependencies of the specified compiler is not installed. The fallback
            strategy won't be used by default.
        """
        compilers = self._valid_compilers()
        compiler_names = {c.name: c for c in compilers}
        if compiler is not None and compiler not in compiler_names:
            raise AssertionError(f"Unknown compiler: {compiler}. Valid compilers: {compiler_names}")
        if compiler is None:
            return self._default_compiler()
        compiler_cls = compiler_names[compiler]
        if not compiler_cls.can_compile():
            if not compiler_fallback_to_native:
                raise AssertionError(
                    f"Specified compiler ({compiler}) is unable to compile" ' (potentially lacking dependencies) and "compiler_fallback_to_native==False"'
                )
            compiler_cls = self._default_compiler()
        return compiler_cls

    def get_compiler_name(self) -> str:
        assert self.is_fit(), "The model must be fit before calling the get_compiler_name method."
        if self._compiler is not None:
            return self._compiler.name
        else:
            return "native"

    def get_trained_params(self) -> dict:
        """
        Returns the hyperparameters of the trained model.
        If the model early stopped, this will contain the epoch/iteration the model uses during inference, instead of the epoch/iteration specified during fit.
        This is used for generating a model template to refit on all of the data (no validation set).
        """
        trained_params = self.params.copy()
        trained_params.update(self.params_trained)
        return trained_params

    def convert_to_refit_full_via_copy(self):
        """
        Creates a new refit_full variant of the model, but instead of training it simply copies `self`.
        This method is for compatibility with models that have not implemented refit_full support as a fallback.
        """
        __name = self.name
        self.rename(self.name + REFIT_FULL_SUFFIX)
        __path_refit = self.path
        self.save(path=self.path, verbose=False)
        self.rename(__name)
        return self.load(path=__path_refit, verbose=False)

    def get_params(self) -> dict:
        """Get params of the model at the time of initialization"""
        name = self.name
        path = self.path_root
        problem_type = self.problem_type
        eval_metric = self.eval_metric
        hyperparameters = self._user_params.copy()
        if self._user_params_aux:
            hyperparameters[AG_ARGS_FIT] = self._user_params_aux.copy()

        args = dict(
            path=path,
            name=name,
            problem_type=problem_type,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
        )

        return args

    def convert_to_template(self):
        """
        After calling this function, returned model should be able to be fit as if it was new, as well as deep-copied.
        The model name and path will be identical to the original, and must be renamed prior to training to avoid overwriting the original model files if they exist.
        """

        params = self.get_params()
        template = self.__class__(**params)

        return template

    def convert_to_refit_full_template(self):
        """
        After calling this function, returned model should be able to be fit without X_val, y_val using the iterations trained by the original model.

        Increase max_memory_usage_ratio by 25% to reduce the chance that the refit model will trigger NotEnoughMemoryError and skip training.
        This can happen without the 25% increase since the refit model generally will use more training data and thus require more memory.
        """
        params = copy.deepcopy(self.get_params())

        if "hyperparameters" not in params:
            params["hyperparameters"] = dict()

        if AG_ARGS_FIT not in params["hyperparameters"]:
            params["hyperparameters"][AG_ARGS_FIT] = dict()

        # Increase memory limit by 25% to avoid memory restrictions during fit
        params["hyperparameters"][AG_ARGS_FIT]["max_memory_usage_ratio"] = params["hyperparameters"][AG_ARGS_FIT].get("max_memory_usage_ratio", 1.0) * 1.25

        params["hyperparameters"].update(self.params_trained)
        params["name"] = params["name"] + REFIT_FULL_SUFFIX
        template = self.__class__(**params)

        return template

    def hyperparameter_tune(self, hyperparameter_tune_kwargs="auto", hpo_executor: HpoExecutor = None, time_limit: float = None, **kwargs):
        """
        Perform hyperparameter tuning of the model, fitting multiple variants of the model based on the search space provided in `hyperparameters` during init.

        Parameters
        ----------
        hyperparameter_tune_kwargs : str or dict, default='auto'
            Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run).
            Valid keys:
                'num_trials': Number of hpo trials you want to perform.
                'scheduler': Scheduler used by hpo experiment.
                    Valid values:
                        'local': Local FIFO scheduler. Sequential if Custom backend and parallel if Ray Tune backend.
                'searcher': Search algorithm used by hpo experiment.
                    Valid values:
                        'auto': Random search.
                        'random': Random search.
                        'bayes': Bayes Optimization. Only supported by Ray Tune backend.
            Valid preset values:
                'auto': Uses the 'random' preset.
                'random': Performs HPO via random search using local scheduler.
            The 'searcher' key is required when providing a dict.
        hpo_executor : HpoExecutor, default None
            Executor to perform HPO experiment. This implements the interface for different HPO backends.
            For more info, please refer to `HpoExecutor` under `core/hpo/executors.py`
        time_limit : float, default None
            In general, this is the time limit in seconds to run HPO for.
            In reality, this is the time limit in seconds budget to fully train all trials executed by HPO.
            For example, BaggedEnsemble will only use a fraction of the time limit during HPO because it needs the remaining time later to fit all of the folds of the trials.
        **kwargs :
            Same kwargs you would pass to fit call, such as:
                X
                y
                X_val
                y_val
                feature_metadata
                sample_weight
                sample_weight_val

        Returns
        -------
        Tuple of (hpo_results: Dict[str, dict], hpo_info: Any)
        hpo_results: Dict[str, dict]
            A dictionary of trial model names to a dictionary containing:
                path: str
                    Absolute path to the trained model artifact. Used to load the model.
                val_score: float
                    val_score of the model
                trial: int
                    Trial number of the model, starting at 0.
                hyperparameters: dict
                    Hyperparameter config of the model trial.
        hpo_info: Any
            Advanced output with scheduler specific logic, primarily for debugging.
            In case of Ray Tune backend, this will be an Analysis object: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.ExperimentAnalysis.html
        """
        # if hpo_executor is not None, ensemble has already created the hpo_executor
        if hpo_executor is None:
            hpo_executor = self._get_default_hpo_executor()
            default_num_trials = kwargs.pop("default_num_trials", None)
            hpo_executor.initialize(hyperparameter_tune_kwargs, default_num_trials=default_num_trials, time_limit=time_limit)
        kwargs = self.initialize(time_limit=time_limit, **kwargs)
        self._register_fit_metadata(**kwargs)
        self._validate_fit_memory_usage(**kwargs)
        kwargs = self._preprocess_fit_resources(parallel_hpo=hpo_executor.executor_type == "ray", **kwargs)
        self.validate_fit_resources(**kwargs)
        hpo_executor.register_resources(self, **kwargs)
        return self._hyperparameter_tune(hpo_executor=hpo_executor, **kwargs)

    def _hyperparameter_tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hpo_executor: HpoExecutor,
        **kwargs,
    ):
        """
        Hyperparameter tune the model.

        This usually does not need to be overwritten by models.
        """
        # verbosity = kwargs.get('verbosity', 2)
        time_start = time.time()
        logger.log(15, "Starting generic AbstractModel hyperparameter tuning for %s model..." % self.name)
        search_space = self._get_search_space()

        try:
            hpo_executor.validate_search_space(search_space, self.name)
        except EmptySearchSpace:
            return skip_hpo(self, X=X, y=y, X_val=X_val, y_val=y_val, **kwargs)

        directory = self.path
        os.makedirs(directory, exist_ok=True)
        data_path = directory
        if DistributedContext.is_distributed_mode():
            data_path = DistributedContext.get_util_path()
        train_path, val_path = hpo_executor.prepare_data(X=X, y=y, X_val=X_val, y_val=y_val, path_prefix=data_path)

        model_cls = self.__class__
        init_params = self.get_params()
        # We set soft time limit to avoid trials being terminated directly by ray tune
        trial_soft_time_limit = None
        if hpo_executor.time_limit is not None:
            trial_soft_time_limit = max(hpo_executor.time_limit * 0.9, hpo_executor.time_limit - 5)  # 5 seconds max for buffer

        fit_kwargs = dict()
        fit_kwargs["feature_metadata"] = self.feature_metadata
        fit_kwargs["num_classes"] = self.num_classes
        fit_kwargs["sample_weight"] = kwargs.get("sample_weight", None)
        fit_kwargs["sample_weight_val"] = kwargs.get("sample_weight_val", None)
        train_fn_kwargs = dict(
            model_cls=model_cls,
            init_params=init_params,
            time_start=time_start,
            time_limit=trial_soft_time_limit,
            fit_kwargs=fit_kwargs,
            train_path=train_path,
            val_path=val_path,
            hpo_executor=hpo_executor,
        )
        model_estimate_memory_usage = None
        if self.estimate_memory_usage is not None:
            model_estimate_memory_usage = self.estimate_memory_usage(X=X, **kwargs)
        minimum_resources = self.get_minimum_resources(is_gpu_available=(hpo_executor.resources.get("num_gpus", 0) > 0))
        hpo_executor.execute(
            model_trial=model_trial,
            train_fn_kwargs=train_fn_kwargs,
            directory=directory,
            minimum_cpu_per_trial=minimum_resources.get("num_cpus", 1),
            minimum_gpu_per_trial=minimum_resources.get("num_gpus", 0),
            model_estimate_memory_usage=model_estimate_memory_usage,
            adapter_type="tabular",
            tune_config_kwargs={"chdir_to_trial_dir": False},
        )

        hpo_results = hpo_executor.get_hpo_results(
            model_name=self.name,
            model_path_root=self.path_root,
            time_start=time_start,
        )

        # cleanup artifacts
        for data_file in [train_path, val_path]:
            try:
                os.remove(data_file)
            except FileNotFoundError:
                pass

        return hpo_results

    def _get_hpo_backend(self) -> str:
        """Choose which backend("ray" or "custom") to use for hpo"""
        if DistributedContext.is_distributed_mode():
            return RAY_BACKEND
        return CUSTOM_BACKEND

    def _get_default_hpo_executor(self) -> HpoExecutor:
        backend = self._get_model_base()._get_hpo_backend()  # If ensemble, will use the base model to determine backend
        if backend == RAY_BACKEND:
            try:
                try_import_ray()
            except Exception as e:
                warning_msg = f"Will use custom hpo logic because ray import failed. Reason: {str(e)}"
                dup_filter.attach_filter_targets(warning_msg)
                logger.warning(warning_msg)
                backend = CUSTOM_BACKEND
        hpo_executor = HpoExecutorFactory.get_hpo_executor(backend)()
        return hpo_executor

    @property
    def _path_v2(self) -> str:
        """Path as a property, replace old path logic with this eventually"""
        return self.path_root + self.path_suffix

    # Resets metrics for the model
    def reset_metrics(self):
        self.fit_time = None
        self.predict_time = None
        self.compile_time = None
        self.val_score = None
        self.params_trained = dict()

    # TODO: Experimental, currently unused
    #  Has not been tested on Windows
    #  Does not work if model is located in S3
    #  Does not work if called before model was saved to disk (Will output 0)
    def get_disk_size(self) -> int:
        # Taken from https://stackoverflow.com/a/1392549
        from pathlib import Path

        model_path = Path(self.path)
        model_disk_size = sum(f.stat().st_size for f in model_path.glob("**/*") if f.is_file())
        return model_disk_size

    def get_memory_size(self, allow_exception: bool = False) -> int | None:
        """
        Pickled the model object (self) and returns the size in bytes.
        Will raise an exception if `self` cannot be pickled.

        Note: This will temporarily double the memory usage of the model, as both the original and the pickled version will exist in memory.
        This can lead to an out-of-memory error if the model is larger than the remaining available memory.

        Parameters
        ----------
        allow_exception: bool, default = False
            If True and an exception occurs during the memory size calculation, will return None instead of raising the exception.
            For example, if a model failed during fit and had a messy internal state, and then `get_memory_size` was called,
            it may still contain a non-serializable object. By setting `allow_exception=True`, we avoid crashing in this scenario.
            For example: "AttributeError: Can't pickle local object 'func_generator.<locals>.custom_metric'"

        Returns
        -------
        memory_size: int | None
            The memory size in bytes of the pickled model object.
            None if an exception occurred and `allow_exception=True`.
        """
        if allow_exception:
            try:
                return self._get_memory_size()
            except:
                return None
        else:
            return self._get_memory_size()

    def _get_memory_size(self) -> int:
        gc.collect()  # Try to avoid OOM error
        return sys.getsizeof(pickle.dumps(self, protocol=4))

    def estimate_memory_usage(self, **kwargs) -> int:
        """
        Estimates the memory usage of the model while training.
        Returns
        -------
            int: number of bytes will be used during training
        """
        assert self.is_initialized(), "Only estimate memory usage after the model is initialized."
        return self._estimate_memory_usage(**kwargs)

    def validate_fit_resources(self, num_cpus="auto", num_gpus="auto", total_resources=None, **kwargs):
        """
        Verifies that the provided num_cpus and num_gpus (or defaults if not provided) are sufficient to train the model.
        Raises an AssertionError if not sufficient.
        """
        resources = self._preprocess_fit_resources(num_cpus=num_cpus, num_gpus=num_gpus, total_resources=total_resources, silent=True)
        self._validate_fit_resources(**resources)

    def _validate_fit_resources(self, **resources):
        res_min = self.get_minimum_resources()
        for resource_name in res_min:
            if resource_name not in resources:
                raise AssertionError(f"Model requires {res_min[resource_name]} {resource_name} to fit, but no available amount was defined.")
            elif res_min[resource_name] > resources[resource_name]:
                raise AssertionError(f"Model requires {res_min[resource_name]} {resource_name} to fit, but {resources[resource_name]} are available.")
        total_resources = resources.get("total_resources", None)
        if total_resources is None:
            total_resources = {}
        for resource_name, resource_value in total_resources.items():
            if resources[resource_name] > resource_value:
                raise AssertionError(f"Specified {resources[resource_name]} {resource_name} to fit, but only {resource_value} are available in total.")

    def get_minimum_resources(self, is_gpu_available: bool = False) -> Dict[str, Union[int, float]]:
        """
        Parameters
        ----------
        is_gpu_available : bool, default = False
            Whether gpu is available in the system.
            Model that can be trained both on cpu and gpu can decide the minimum resources based on this.

        Returns a dictionary of minimum resource requirements to fit the model.
        Subclass should consider overriding this method if it requires more resources to train.
        If a resource is not part of the output dictionary, it is considered unnecessary.
        Valid keys: 'num_cpus', 'num_gpus'.
        """
        return {
            "num_cpus": 1,
        }

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        """
        Estimates the peak memory usage during model fitting.
        This method simply provides a default implementation. Each model should consider implementing custom memory estimation logic.

        Parameters
        ----------
        X : pd.DataFrame,
            The training data intended to fit the model with.
        **kwargs : dict,
            The `.fit` kwargs.
            Can optionally be used by custom implementations to better estimate memory usage.
            To best understand what kwargs are available, enter a debugger and put a breakpoint in this method to manually inspect the keys.

        Returns
        -------
        The estimated peak memory usage in bytes during model fit.
        """
        return 4 * get_approximate_df_mem_usage(X).sum()

    @disable_if_lite_mode()
    def _validate_fit_memory_usage(self, mem_error_threshold: float = 0.9, mem_warning_threshold: float = 0.75, mem_size_threshold: int = None, **kwargs):
        """
        Asserts that enough memory is available to fit the model

        If not enough memory, will raise NotEnoughMemoryError
        Memory thresholds depend on the `params_aux` hyperparameter `max_memory_usage_ratio`, which generally defaults to 1.
        if `max_memory_usage_ratio=None`, all memory checks are skipped.

        Parameters
        ----------
        mem_error_threshold : float, default = 0.9
            A multiplier to max_memory_usage_ratio to get the max_memory_usage_error_ratio
            If expected memory usage is >max_memory_usage_error_ratio, raise NotEnoughMemoryError
        mem_warning_threshold : float, default = 0.75
            A multiplier to max_memory_usage_ratio to get the max_memory_usage_warning_ratio
            If expected memory usage is >max_memory_usage_error_ratio, raise NotEnoughMemoryError
        mem_size_threshold : int, default = None
            If not None, skips checking available memory if the expected model size is less than `mem_size_threshold` bytes.
            This is used to speed-up training by avoiding the check in cases where the machine almost certainly has sufficient memory.
        **kwargs : dict,
            Fit time kwargs, including X, y, X_val, and y_val.
            Can be used to customize estimation of memory usage.
        """
        max_memory_usage_ratio = self.params_aux["max_memory_usage_ratio"]
        if max_memory_usage_ratio is None:
            return  # Skip memory check

        approx_mem_size_req = self.estimate_memory_usage(**kwargs)
        if mem_size_threshold is not None and approx_mem_size_req < (mem_size_threshold * min(max_memory_usage_ratio, 1)):
            return  # Model is smaller than the min threshold to check available mem

        available_mem = ResourceManager.get_available_virtual_mem()

        # The expected memory usage percentage of the model during fit
        expected_memory_usage_ratio = approx_mem_size_req / available_mem

        # The minimum `max_memory_usage_ratio` values required to avoid an error/warning
        min_error_memory_ratio = expected_memory_usage_ratio / mem_error_threshold
        min_warning_memory_ratio = expected_memory_usage_ratio / mem_warning_threshold

        # The max allowed `expected_memory_usage_ratio` values to avoid an error/warning
        max_memory_usage_error_ratio = mem_error_threshold * max_memory_usage_ratio
        max_memory_usage_warning_ratio = mem_warning_threshold * max_memory_usage_ratio

        log_ag_args_fit_example = '`predictor.fit(..., ag_args_fit={"ag.max_memory_usage_ratio": VALUE})`'
        log_ag_args_fit_example = f"\n\t\tTo set the same value for all models, do the following when calling predictor.fit: {log_ag_args_fit_example}"

        log_user_guideline = (
            f"Estimated to require {approx_mem_size_req / 1e9:.3f} GB "
            f"out of {available_mem / 1e9:.3f} GB available memory ({expected_memory_usage_ratio*100:.3f}%)... "
            f"({max_memory_usage_error_ratio*100:.3f}% of avail memory is the max safe size)"
        )
        if expected_memory_usage_ratio > max_memory_usage_error_ratio:
            log_user_guideline += (
                f'\n\tTo force training the model, specify the model hyperparameter "ag.max_memory_usage_ratio" to a larger value '
                f"(currently {max_memory_usage_ratio}, set to >={min_error_memory_ratio + 0.05:.2f} to avoid the error)"
                f"{log_ag_args_fit_example}"
            )
            if min_error_memory_ratio >= 1:
                log_user_guideline += (
                    f'\n\t\tSetting "ag.max_memory_usage_ratio" to values above 1 may result in out-of-memory errors. '
                    f"You may consider using a machine with more memory as a safer alternative."
                )
            logger.warning(f"\tWarning: Not enough memory to safely train model. {log_user_guideline}")
            raise NotEnoughMemoryError
        elif expected_memory_usage_ratio > max_memory_usage_warning_ratio:
            log_user_guideline += (
                f'\n\tTo avoid this warning, specify the model hyperparameter "ag.max_memory_usage_ratio" to a larger value '
                f"(currently {max_memory_usage_ratio}, set to >={min_warning_memory_ratio + 0.05:.2f} to avoid the warning)"
                f"{log_ag_args_fit_example}"
            )
            if min_warning_memory_ratio >= 1:
                log_user_guideline += (
                    f'\n\t\tSetting "ag.max_memory_usage_ratio" to values above 1 may result in out-of-memory errors. '
                    f"You may consider using a machine with more memory as a safer alternative."
                )
            logger.warning(f"\tWarning: Potentially not enough memory to safely train model. {log_user_guideline}")

    def reduce_memory_size(self, remove_fit: bool = True, remove_info: bool = False, requires_save: bool = True, **kwargs):
        """
        Removes non-essential objects from the model to reduce memory and disk footprint.
        If `remove_fit=True`, enables the removal of variables which are required for fitting the model. If the model is already fully trained, then it is safe to remove these.
        If `remove_info=True`, enables the removal of variables which are used during model.get_info(). The values will be None when calling model.get_info().
        If `requires_save=True`, enables the removal of variables which are part of the model.pkl object, requiring an overwrite of the model to disk if it was previously persisted.

        It is not necessary for models to implement this.
        """
        pass

    def delete_from_disk(self, silent: bool = False):
        """
        Deletes the model from disk.

        WARNING: This will DELETE ALL FILES in the self.path directory, regardless if they were created by AutoGluon or not.
        DO NOT STORE FILES INSIDE OF THE MODEL DIRECTORY THAT ARE UNRELATED TO AUTOGLUON.
        """
        if not silent:
            logger.log(30, f"Deleting model {self.name}. All files under {self.path} will be removed.")
        import shutil
        from pathlib import Path

        model_path = Path(self.path)
        # TODO: Report errors?
        shutil.rmtree(path=model_path, ignore_errors=True)

    def get_info(self) -> dict:
        """
        Returns a dictionary of numerous fields describing the model.
        """
        info = {
            "name": self.name,
            "model_type": type(self).__name__,
            "problem_type": self.problem_type,
            "eval_metric": self.eval_metric.name,
            "stopping_metric": self.stopping_metric.name if self.stopping_metric is not None else None,
            "fit_time": self.fit_time,
            "num_classes": self.num_classes,
            "quantile_levels": self.quantile_levels,
            "predict_time": self.predict_time,
            "val_score": self.val_score,
            "hyperparameters": self.params,
            "hyperparameters_fit": self.params_trained,  # TODO: Explain in docs that this is for hyperparameters that differ in final model from original hyperparameters, such as epochs (from early stopping)
            "hyperparameters_nondefault": self.nondefault_params,
            AG_ARGS_FIT: self.params_aux,
            "num_features": len(self.features) if self.features else None,
            "features": self.features,
            "feature_metadata": self.feature_metadata,
            # 'disk_size': self.get_disk_size(),
            "memory_size": self.get_memory_size(allow_exception=True),  # Memory usage of model in bytes
            "compile_time": self.compile_time if hasattr(self, "compile_time") else None,
            "is_initialized": self.is_initialized(),
            "is_fit": self.is_fit(),
            "is_valid": self.is_valid(),
            "can_infer": self.can_infer(),
        }
        return info

    @classmethod
    def load_info(cls, path: str, load_model_if_required: bool = True) -> dict:
        load_path = os.path.join(path, cls.model_info_name)
        try:
            return load_pkl.load(path=load_path)
        except:
            if load_model_if_required:
                model = cls.load(path=path, reset_paths=True)
                return model.get_info()
            else:
                raise

    def save_info(self) -> dict:
        info = self.get_info()

        save_pkl.save(path=os.path.join(self.path, self.model_info_name), object=info)
        json_path = os.path.join(self.path, self.model_info_json_name)
        save_json.save(path=json_path, obj=info)
        return info

    def _get_maximum_resources(self) -> Dict[str, Union[int, float]]:
        """
        Get the maximum resources allowed to use for this model.
        This can be useful when model not scale well with resources, i.e. cpu cores.
        Return empty dict if no maximum resources needed

        Return
        ------
        Dict[str, Union[int, float]]
            key, name of the resource, i.e. `num_cpus`, `num_gpus`
            value, maximum amount of resources
        """
        return {}

    def _get_default_resources(self) -> Tuple[int, int]:
        """
        Determines the default resource usage of the model during fit.

        Models may want to override this if they depend heavily on GPUs, as the default sets num_gpus to 0.
        """
        num_cpus = ResourceManager.get_cpu_count()
        num_gpus = 0
        return num_cpus, num_gpus

    # TODO: v0.1 Add reference link to all valid keys and their usage or keep full docs here and reference elsewhere?
    @classmethod
    def _get_default_ag_args(cls) -> dict:
        """
        Dictionary of customization options related to meta properties of the model such as its name, the order it is trained, and the problem types it is valid for.
        """
        return {}

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """
        [Advanced] Dictionary of customization options related to meta properties of the model ensemble this model will be a child in.
        Refer to hyperparameters of ensemble models for valid options.
        """
        return {}

    def _get_default_stopping_metric(self) -> Scorer:
        """
        Returns the default stopping metric to use for early stopping.
        This is used if stopping_metric was not explicitly specified.
        Models may wish to override this in case a more suitable stopping_metric exists for a given eval_metric.
        """
        if self.eval_metric.name == "roc_auc":
            stopping_metric = "log_loss"
        else:
            stopping_metric = self.eval_metric
        stopping_metric = metrics.get_metric(stopping_metric, self.problem_type, "stopping_metric")
        return stopping_metric

    # TODO: v1.0 Move params_aux to params, separate logic as in _get_ag_params, keep `ag.` prefix for ag_args_fit params
    #  This will allow to hyperparameter tune ag_args_fit hyperparameters.
    #  Also delete `self.params_aux` entirely, make it a method instead.
    def _get_params(self) -> dict:
        """Gets all params."""
        return self.params.copy()

    def _get_ag_params(self) -> dict:
        """
        Gets params that are not passed to the inner model, but are used by the wrapper.
        These params should exist in `self.params_aux`.
        """
        ag_param_names = self._ag_params()
        if ag_param_names:
            return {key: val for key, val in self.params_aux.items() if key in ag_param_names}
        else:
            return dict()

    def _get_model_params(self) -> dict:
        """Gets params that are passed to the inner model."""
        return self._get_params()

    # TODO: Add documentation for valid args for each model. Currently only `early_stop`
    def _ag_params(self) -> set:
        """
        Set of params that are not passed to self.model, but are used by the wrapper.
        For developers, this is purely optional and is just for convenience to logically distinguish between model specific parameters and added AutoGluon functionality.
        The goal is to have common parameter names for useful functionality shared between models,
        even if the functionality is not natively available as a parameter in the model itself or under a different name.

        Below are common patterns / options to make available. Their actual usage and options in a particular model should be documented in the model itself, as it has flexibility to differ.

        Possible params:

        early_stop : int, str, or tuple
            generic name for early stopping logic. Typically can be an int or a str preset/strategy.
            Also possible to pass tuple of (class, kwargs) to construct a custom early stopping object.
                Refer to `autogluon.core.utils.early_stopping` for examples.

        """
        return set()

    @property
    def _features(self) -> List[str]:
        return self._features_internal

    def _get_tags(self) -> dict:
        """
        Tags are key-value pairs assigned to an object.
        These can be accessed after initializing an object.
        Tags are used for identifying if an object supports certain functionality.
        """
        # first get class tags, which are overwritten by any object tags
        collected_tags = self._get_class_tags()
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, "_more_tags"):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags

    @classmethod
    def _get_class_tags(cls) -> dict:
        """
        Class tags are tags assigned to a class that are fixed.
        These can be accessed prior to initializing an object.
        Tags are used for identifying if an object supports certain functionality.
        """
        collected_tags = {}
        for base_class in reversed(inspect.getmro(cls)):
            if hasattr(base_class, "_class_tags"):
                # need the if because mixins might not have _class_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._class_tags()
                collected_tags.update(more_tags)
        return collected_tags

    @classmethod
    def _class_tags(cls) -> dict:
        """
        [Advanced] Optional tags used to communicate model capabilities to AutoML systems, such as if the model supports text features.
        """
        return _DEFAULT_CLASS_TAGS

    def _more_tags(self) -> dict:
        return _DEFAULT_TAGS

    def _get_model_base(self):
        return self
