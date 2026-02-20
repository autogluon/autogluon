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
from abc import ABC, abstractmethod
from pathlib import Path
from types import MappingProxyType
from typing import Any, Type

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.space import Space
from autogluon.common.utils.distribute_utils import DistributedContext
from autogluon.common.utils.lite import disable_if_lite_mode
from autogluon.common.utils.log_utils import DuplicateFilter
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager, get_resource_manager
from autogluon.common.utils.try_import import try_import_ray
from autogluon.common.utils.utils import setup_outputdir
from autogluon.features.generators.abstract import AbstractFeatureGenerator, estimate_feature_metadata_after_generators
from autogluon.features.generators.bulk import BulkFeatureGenerator
from autogluon.features.registry._ag_feature_generator_registry import ag_feature_generator_registry
from autogluon.features.registry.parse_custom_generator import resolve_fg_class

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
from ...metrics import Scorer, compute_metric
from ...utils import (
    compute_permutation_feature_importance,
    get_pred_from_proba,
    infer_eval_metric,
    infer_problem_type,
    normalize_pred_probas,
)
from ...utils.exceptions import NotEnoughMemoryError, NoValidFeatures, TimeLimitExceeded
from ...utils.loaders import load_json, load_pkl
from ...utils.savers import save_json, save_pkl
from ...utils.time import sample_df_for_time_func, time_func
from ._tags import _DEFAULT_CLASS_TAGS, _DEFAULT_TAGS
from .model_trial import model_trial, skip_hpo

logger = logging.getLogger(__name__)
dup_filter = DuplicateFilter()
logger.addFilter(dup_filter)


class Taggable(ABC):
    @classmethod
    def _class_tags(cls) -> dict:
        return _DEFAULT_CLASS_TAGS

    def _more_tags(self) -> dict:
        return _DEFAULT_TAGS

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


# TODO: refactor this class as a clean interface HPO works with. The methods below are not
# an exhaustive set of all methods the HPO module needs!
class Tunable(ABC):
    def estimate_memory_usage(self, *args, **kwargs) -> float | None:
        """Return the estimated memory usage of the model. None if memory usage cannot be
        estimated.
        """
        return None

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
        }

    # TODO: remove. this is needed by hpo to determine if the model is an ensemble.
    @abstractmethod
    def _get_model_base(self) -> "Tunable":
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return a clean copy of constructor parameters that can be used to
        clone the current model.
        """
        pass

    @abstractmethod
    def hyperparameter_tune(self, *args, **kwargs) -> tuple:
        pass


class ModelBase(Taggable, ABC):
    @abstractmethod
    def __init__(
        self,
        path: str | None = None,
        name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ):
        self.name: str
        self.path: str

    @abstractmethod
    def rename(self, name: str) -> None:
        pass

    @abstractmethod
    def get_info(self, *args, **kwargs) -> dict[str, Any]:
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> Self:
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def save(self, path: str | None = None, verbose: bool = True) -> str:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, reset_paths: bool = True) -> Self:
        pass


# TODO: move to tabular, rename AbstractTabularModel
class AbstractModel(ModelBase, Tunable):
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
            ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovr',
            'average_precision', 'precision', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall',
            'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss', 'pac_score', 'quadratic_kappa']
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

    ag_key: str | None = None  # set to string value for subclasses for use in AutoGluon
    ag_name: str | None = None  # set to string value for subclasses for use in AutoGluon
    ag_priority: int = 0  # set to int value for subclasses for use in AutoGluon
    ag_priority_by_problem_type: dict[str, int] = MappingProxyType(
        {}
    )  # if not set, we fall back to ag_priority. Use MappingProxyType to avoid mutation.
    seed_name: str | None = (
        None  # the name of the hyperparameter that controls the model's random seed, or None if no random seed exists.
    )
    seed_name_alt: list[str] = []  # alternative names for the random seed hyperparameter.

    model_file_name = "model.pkl"
    model_info_name = "info.pkl"
    model_info_json_name = "info.json"
    learning_curve_file_name = "curves.json"

    default_random_seed: int | None = 0

    def __init__(
        self,
        path: str | None = None,
        name: str | None = None,
        problem_type: str | None = None,
        eval_metric: str | metrics.Scorer | None = None,
        hyperparameters: dict | None = None,
    ):
        if name is None:
            self.name: str = self.__class__.__name__
            logger.log(20, f"Warning: No name was specified for model, defaulting to class name: {self.name}")
        else:
            self.name: str = name  # TODO: v0.1 Consider setting to self._name and having self.name be a property so self.name can't be set outside of self.rename()

        self.path_root: str = path
        if self.path_root is None:
            path_suffix = self.name
            # TODO: Would be ideal to not create dir, but still track that it is unique. However, this isn't possible to do without a global list of used dirs or using UUID.
            path_cur = setup_outputdir(path=None, create_dir=True, path_suffix=path_suffix)
            self.path_root = path_cur.rsplit(self.path_suffix, 1)[0]
            logger.log(20, f"Warning: No path was specified for model, defaulting to: {self.path_root}")

        self.path: str = self.create_contexts(
            os.path.join(self.path_root, self.path_suffix)
        )  # TODO: Make this path a function for consistency.

        self.num_classes: int | None = None
        self.quantile_levels: list[float] | None = None
        self.model = None
        self.problem_type: str = problem_type

        # whether to calibrate predictions via conformal methods
        self.conformalize: bool | None = None
        self.label_cleaner: LabelCleaner | None = None

        if eval_metric is not None:
            self.eval_metric: Scorer | None = metrics.get_metric(
                eval_metric, self.problem_type, "eval_metric"
            )  # Note: we require higher values = better performance
        else:
            self.eval_metric: Scorer | None = None
        self.stopping_metric: Scorer | None = None
        self.normalize_pred_probas: bool | None = None

        self.features: list[str] | None = None  # External features, do not use internally
        self.feature_metadata: FeatureMetadata | None = None  # External feature metadata, do not use internally
        self._features_internal: list[str] | None = (
            None  # Internal features, safe to use internally via the `_features` property
        )
        self._features_internal_to_align: list[str] | None = (
            None  # Intermediate internal features, only used for ensuring consistent column order
        )
        self._feature_metadata: FeatureMetadata | None = None  # Internal feature metadata, safe to use internally
        self._is_features_in_same_as_ex: bool | None = None  # Whether self.features == self._features_internal

        self.fit_time: float | None = None  # Time taken to fit in seconds (Training data)
        self.predict_time: float | None = None  # Time taken to predict in seconds (Validation data)
        self._predict_n_size: int | None = None  # Batch size used to calculate predict_time
        self.predict_1_time: float | None = (
            None  # Time taken to predict 1 row of data in seconds (with batch size `predict_1_batch_size` in params_aux)
        )
        self.compile_time: float | None = None  # Time taken to compile the model in seconds
        self.val_score: float | None = None  # Score with eval_metric (Validation data)
        self._memory_usage_estimate: float | None = None  # Peak training memory usage estimate in bytes

        self._user_params, self._user_params_aux = self._init_user_params(params=hyperparameters)

        self.params: dict = {}
        self.params_aux: dict = {}
        self.params_trained = dict()
        self.nondefault_params: list[str] = []
        self._is_initialized: bool = False
        self._is_fit_metadata_registered: bool = False
        self._fit_metadata: dict = dict()
        self.saved_learning_curves: bool = False

        self._compiler = None

        # None is a valid value, "NOTSET" indicates `.init_random_seed` was not called yet.
        self.random_seed: int | None | str = "NOTSET"
        # Model specific preprocessing: NOTSET indicates init is missing, None indicates no preprocessing
        self._model_specific_feature_generators: BulkFeatureGenerator | None | str = "NOTSET"

    @classmethod
    def _init_user_params(
        cls, params: dict[str, Any] | None = None, ag_args_fit: str = AG_ARGS_FIT, ag_arg_prefix: str = AG_ARG_PREFIX
    ) -> (dict[str, Any], dict[str, Any]):
        """
        Given the user-specified hyperparameters, split into `params` and `params_aux`.

        Parameters
        ----------
        params : dict[str, Any], default = None
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
        params, params_aux : (dict[str, Any], dict[str, Any])
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
            self.nondefault_params = list(hyperparameters.keys())[
                :
            ]  # These are hyperparameters that user has specified.
        self.params_trained = dict()
        self._validate_params()

    def _init_params_aux(self):
        """
        Initializes auxiliary hyperparameters.
        These parameters are generally not model specific and can have a wide variety of effects.
        For documentation on some of the available options and their defaults, refer to `self._get_default_auxiliary_params`.
        """
        self.params_aux = self._get_params_aux()
        self._validate_params_aux()

    def _get_params_aux(self) -> dict:
        hyperparameters_aux = self._user_params_aux
        default_auxiliary_params = self._get_default_auxiliary_params()
        if hyperparameters_aux is not None:
            default_auxiliary_params.update(hyperparameters_aux)
        return default_auxiliary_params

    # TODO: Consider validating before fit call to avoid executing a ray task when it will immediately fail this check in distributed mode
    # TODO: Consider avoiding logging `Fitting model: xyz...` if this fails for particular error types.
    def _validate_params(self):
        """
        Verify correctness of self.params
        """
        pass

    def _validate_params_aux(self):
        """
        Verify correctness of self.params_aux
        """
        if "num_cpus" in self.params_aux:
            num_cpus = self.params_aux["num_cpus"]
            if num_cpus is not None and not isinstance(num_cpus, int):
                raise TypeError(f"`num_cpus` must be an int or None. Found: {type(num_cpus)} | Value: {num_cpus}")

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

    def can_estimate_memory_usage_static(self) -> bool:
        """
        True if `estimate_memory_usage_static` is implemented for this model.
        If False, calling `estimate_memory_usage_static` will raise a NotImplementedError.
        """
        return self._get_class_tags().get("can_estimate_memory_usage_static", False)

    def can_estimate_memory_usage_static_child(self) -> bool:
        """
        True if `estimate_memory_usage_static` is implemented for this model's child.
        If False, calling `estimate_memory_usage_static_child` will raise a NotImplementedError.
        """
        return self.can_estimate_memory_usage_static()

    def can_estimate_memory_usage_static_lite(self) -> bool:
        """
        True if `estimate_memory_usage_static_lite` is implemented for this model.
        If False, calling `estimate_memory_usage_static_lite` will raise a NotImplementedError.
        """
        return self._get_class_tags().get("can_estimate_memory_usage_static_lite", False)

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
            X = self._preprocess_model_specific(X, **kwargs)
            X = self._preprocess_align_features(X, **kwargs)
            X = self._preprocess(X, **kwargs)

        return X

    def _preprocess_align_features(self, X: pd.DataFrame, **kwargs):
        if not self._is_features_in_same_as_ex:
            X = X[self._features_internal_to_align]
        return X

    # TODO: support preprocessing methods that require y_train
    def _preprocess_model_specific(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """General model-specific data-transformation logic.

        This is the place to add and configure data transformations that can be enabled
        through AutoGluon or passing FeatureGenerator classes. This is different to
        model-agnostic preprocessing from the general `_feature_generator_kwargs`,
        as this logic is called each time the model is fit (that is for each fold).

        A general rule of thumb is to add here any data transformation that
        conditions on the training samples (e.g. PCA).
        """

        if self._model_specific_feature_generators == "NOTSET":
            self._model_specific_feature_generators = self.get_preprocessor()
            if self._model_specific_feature_generators is None:
                return X

            X = self._model_specific_feature_generators.fit_transform(
                X,
                feature_metadata_in=self._feature_metadata,
                problem_type=self.problem_type,
                **kwargs,
            )

            self._preprocess_set_features_internal(
                X=X, feature_metadata=self._model_specific_feature_generators.feature_metadata
            )
            return X

        if self._model_specific_feature_generators is None:
            return X

        return self._model_specific_feature_generators.transform(X)

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
            feature_metadata = self._infer_feature_metadata(X=X)
        else:
            feature_metadata = copy.deepcopy(feature_metadata)
        feature_metadata = self._update_feature_metadata(X=X, feature_metadata=feature_metadata)

        valid_features = self._get_valid_features(feature_metadata=feature_metadata)
        dropped_features = [feature for feature in self.features if feature not in valid_features]
        if dropped_features:
            logger.log(10, f"\tDropped {len(dropped_features)} of {len(self.features)} features.")
        self.features = [feature for feature in self.features if feature in valid_features]
        self.feature_metadata = feature_metadata.keep_features(self.features)
        error_if_no_features = self.params_aux.get("error_if_no_features", True)
        if error_if_no_features and not self.features:
            raise NoValidFeatures(f"No valid features exist to fit {self.name}")
        # TODO: If unique_counts == 2 (including NaN), then treat as boolean
        #  FIXME: v1.3: Need to do this on a per-fold basis
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
            logger.log(
                10,
                f"\tDropped {len(features_to_drop_internal)} of {len(self.features)} internal features: {features_to_drop_internal}",
            )
            self._features_internal = [
                feature for feature in self.features if feature not in features_to_drop_internal
            ]
            self._feature_metadata = self.feature_metadata.keep_features(self._features_internal)
            self._is_features_in_same_as_ex = False
        else:
            self._features_internal = self.features
            self._feature_metadata = self.feature_metadata
            self._is_features_in_same_as_ex = True
        self._features_internal_to_align = self._features_internal
        if error_if_no_features and not self._features_internal:
            raise NoValidFeatures(
                f"No valid features exist after dropping features with only a single value to fit {self.name}"
            )

    def _preprocess_set_features_internal(self, X: pd.DataFrame, feature_metadata: FeatureMetadata = None):
        """Update self._features and self._feature_metadata from X.

        If no valid internal features were found, a NoValidFeatures exception is raised.
        """
        logger.log(10, "\tUpdating internal feature metadata.")

        if (self.features is None) or (self.feature_metadata is None):
            raise ValueError(
                "self.features and self.feature_metadata must be set before calling _preprocess_set_features_internal"
            )
        if feature_metadata is None:
            feature_metadata = self._infer_feature_metadata(X=X)
        else:
            feature_metadata = copy.deepcopy(feature_metadata)
        feature_metadata = self._update_feature_metadata(X=X, feature_metadata=feature_metadata)

        valid_features = self._get_valid_features(feature_metadata=feature_metadata)
        features = list(X.columns)
        if features != valid_features:
            logger.log(10, f"\tDropped {len(features) - len(valid_features)} of {len(features)} internal features")

        # Set internal features
        self._features_internal = valid_features
        self._feature_metadata = feature_metadata.keep_features(valid_features)
        self._is_features_in_same_as_ex = (self._features_internal == self.features) and (
            self._feature_metadata == self.feature_metadata
        )
        self._features_internal_to_align = self._features_internal

        error_if_no_features = self.params_aux.get("error_if_no_features", True)
        if error_if_no_features and not self._features_internal:
            raise NoValidFeatures(f"No valid internal features exist to fit {self.name}")

    def _get_valid_features(self, feature_metadata: FeatureMetadata = None) -> list[str]:
        """Infer the valid features to use based on feature_metadata, self.params_aux,
        and get_features_kwargs_extra.
        """
        # TODO: Consider changing how this works or where it is done
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

        return valid_features

    def _update_feature_metadata(self, X: pd.DataFrame, feature_metadata: FeatureMetadata) -> FeatureMetadata:
        """
        [Advanced] Method that performs updates to feature_metadata during initialization.
        Primarily present for use in stacker models.
        """
        return feature_metadata

    def _infer_feature_metadata(self, X: pd.DataFrame) -> FeatureMetadata:
        return FeatureMetadata.from_df(X)

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
        kwargs.pop("random_seed", None)
        return kwargs

    @classmethod
    def _infer_problem_type(cls, *, y: pd.Series, silent: bool = True) -> str:
        """Infer the problem_type based on y train"""
        return infer_problem_type(y=y, silent=silent)

    @classmethod
    def _infer_num_classes(cls, *, y: pd.Series, problem_type: str = None) -> int | None:
        """Infer num_classes based on y train"""
        if problem_type is None:
            problem_type = cls._infer_problem_type(y=y, silent=True)
        label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
        return label_cleaner.num_classes

    def _initialize(self, X=None, y=None, feature_metadata=None, num_classes=None, label_cleaner=None, **kwargs):
        if num_classes is not None:
            self.num_classes = num_classes
        if y is not None:
            if self.problem_type is None:
                self.problem_type = self._infer_problem_type(y=y)
            if self.num_classes is None:
                self.num_classes = self._infer_num_classes(y=y, problem_type=self.problem_type)
        self.label_cleaner = label_cleaner

        self._init_params_aux()

        self._init_misc(X=X, y=y, feature_metadata=feature_metadata, num_classes=num_classes, **kwargs)

        self._init_params()

        self.params = self.init_random_seed(random_seed=kwargs.get("random_seed", "auto"), hyperparameters=self.params)

        if X is not None:
            self._preprocess_set_features(X=X, feature_metadata=feature_metadata)

    def _init_misc(self, **kwargs):
        """Initialize parameters that depend on self.params_aux being initialized"""
        if self.eval_metric is None:
            self.eval_metric = infer_eval_metric(problem_type=self.problem_type)
            logger.log(
                20,
                f"Model {self.name}'s eval_metric inferred to be '{self.eval_metric.name}' because problem_type='{self.problem_type}' and eval_metric was not specified during init.",
            )
        self.eval_metric = metrics.get_metric(
            self.eval_metric, self.problem_type, "eval_metric"
        )  # Note: we require higher values = better performance

        self.stopping_metric = self.params_aux.get("stopping_metric", self._get_default_stopping_metric())
        self.stopping_metric = metrics.get_metric(self.stopping_metric, self.problem_type, "stopping_metric")
        self.quantile_levels = self.params_aux.get("quantile_levels", None)

        if self.eval_metric.name in OBJECTIVES_TO_NORMALIZE:
            self.normalize_pred_probas = True
            logger.debug(
                f"{self.name} predicted probabilities will be transformed to never =0 since eval_metric='{self.eval_metric.name}'"
            )
        else:
            self.normalize_pred_probas = False

    def _process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble(
        self, system_resource, user_specified_total_resource, user_specified_ensemble_resource, resource_type, k_fold
    ):
        if user_specified_total_resource == "auto":
            user_specified_total_resource = math.inf

        # retrieve model level requirement when self is bagged model
        user_specified_model_level_resource = self._get_child_aux_val(key=resource_type, default=None)
        if user_specified_model_level_resource is not None and not isinstance(
            user_specified_model_level_resource, (int, float)
        ):
            raise TypeError(
                f"{resource_type} must be int or float. Found: {type(user_specified_model_level_resource)} | Value: {user_specified_model_level_resource}"
            )
        if user_specified_model_level_resource is not None:
            assert user_specified_model_level_resource <= system_resource, (
                f"Specified {resource_type} per model base is more than the total: {system_resource}"
            )
        user_specified_lower_level_resource = user_specified_ensemble_resource
        if user_specified_ensemble_resource is not None:
            if user_specified_model_level_resource is not None:
                user_specified_lower_level_resource = min(
                    user_specified_model_level_resource * k_fold,
                    user_specified_ensemble_resource,
                    system_resource,
                    user_specified_total_resource,
                )
        else:
            if user_specified_model_level_resource is not None:
                user_specified_lower_level_resource = min(
                    user_specified_model_level_resource * k_fold, system_resource, user_specified_total_resource
                )
        return user_specified_lower_level_resource

    def _calculate_total_resources(
        self,
        silent: bool = False,
        total_resources: dict[str, int | float] | None = None,
        parallel_hpo: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Process user-specified total resources.
        Sanity checks will be done to user-specified total resources to make sure it's legit.
        When user-specified resources are not defined, will instead look at model's default resource requirements.

        Will set the calculated total resources in kwargs and return it
        """
        resource_manager = get_resource_manager()
        system_num_cpus = resource_manager.get_cpu_count()
        system_num_gpus = resource_manager.get_gpu_count()
        if total_resources is None:
            total_resources = {}
        num_cpus = total_resources.get("num_cpus", "auto")
        num_gpus = total_resources.get("num_gpus", "auto")
        default_num_cpus, default_num_gpus = self._get_default_resources()
        # This could be resource requirement for bagged model or individual model
        user_specified_lower_level_num_cpus = self._user_params_aux.get("num_cpus", None)
        user_specified_lower_level_num_gpus = self._user_params_aux.get("num_gpus", None)
        if user_specified_lower_level_num_cpus is not None:
            assert user_specified_lower_level_num_cpus <= system_num_cpus, (
                f"Specified num_cpus per {self.__class__.__name__} is more than the total: {system_num_cpus}"
            )
        if user_specified_lower_level_num_gpus is not None:
            assert user_specified_lower_level_num_gpus <= system_num_gpus, (
                f"Specified num_gpus per {self.__class__.__name__} is more than the total: {system_num_gpus}"
            )
        k_fold = kwargs.get("k_fold", None)
        k_fold = 1 if self.params.get("use_child_oof", False) else k_fold
        if k_fold is not None and k_fold > 0:
            # bagged model will look ag_args_ensemble and ag_args_fit internally to determine resources
            # pass all resources here by default
            default_num_cpus = system_num_cpus
            default_num_gpus = system_num_gpus if default_num_gpus > 0 else 0
            user_specified_lower_level_num_cpus = (
                self._process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble(
                    system_resource=system_num_cpus,
                    user_specified_total_resource=num_cpus,
                    user_specified_ensemble_resource=user_specified_lower_level_num_cpus,
                    resource_type="num_cpus",
                    k_fold=k_fold,
                )
            )
            user_specified_lower_level_num_gpus = (
                self._process_user_provided_resource_requirement_to_calculate_total_resource_when_ensemble(
                    system_resource=system_num_gpus,
                    user_specified_total_resource=num_gpus,
                    user_specified_ensemble_resource=user_specified_lower_level_num_gpus,
                    resource_type="num_gpus",
                    k_fold=k_fold,
                )
            )
        if num_cpus != "auto" and num_cpus > system_num_cpus:
            logger.warning(
                f"Specified total num_cpus: {num_cpus}, but only {system_num_cpus} are available. Will use {system_num_cpus} instead"
            )
            num_cpus = system_num_cpus
        if num_gpus != "auto" and num_gpus > system_num_gpus:
            logger.warning(
                f"Specified total num_gpus: {num_gpus}, but only {system_num_gpus} are available. Will use {system_num_gpus} instead"
            )
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
                    assert user_specified_lower_level_num_cpus <= num_cpus, (
                        f"Specified num_cpus per {self.__class__.__name__} is more than the total specified: {num_cpus}"
                    )
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
                    assert user_specified_lower_level_num_gpus <= num_gpus, (
                        f"Specified num_gpus per {self.__class__.__name__} is more than the total specified: {num_gpus}"
                    )
                    num_gpus = user_specified_lower_level_num_gpus

        minimum_model_resources = self.get_minimum_resources(is_gpu_available=(num_gpus > 0))
        minimum_model_num_cpus = minimum_model_resources.get("num_cpus", 1)
        minimum_model_num_gpus = minimum_model_resources.get("num_gpus", 0)

        maximum_model_resources = self._get_maximum_resources()
        maximum_model_num_cpus = maximum_model_resources.get("num_cpus", None)
        maximum_model_num_gpus = maximum_model_resources.get("num_gpus", None)

        if maximum_model_num_cpus is not None and maximum_model_num_cpus < num_cpus:
            num_cpus = maximum_model_num_cpus
        if maximum_model_num_gpus is not None and maximum_model_num_gpus < num_gpus:
            num_gpus = maximum_model_num_gpus

        assert system_num_cpus >= num_cpus
        assert system_num_gpus >= num_gpus

        assert system_num_cpus >= minimum_model_num_cpus, (
            f"The total system num_cpus={system_num_cpus} is less than minimum num_cpus={minimum_model_num_cpus} to fit {self.__class__.__name__}. Consider using a machine with more CPUs."
        )
        assert system_num_gpus >= minimum_model_num_gpus, (
            f"The total system num_gpus={system_num_gpus} is less than minimum num_gpus={minimum_model_num_gpus} to fit {self.__class__.__name__}. Consider using a machine with more GPUs."
        )

        assert num_cpus >= minimum_model_num_cpus, (
            f"Specified num_cpus={num_cpus} per {self.__class__.__name__} is less than minimum num_cpus={minimum_model_num_cpus}"
        )
        assert num_gpus >= minimum_model_num_gpus, (
            f"Specified num_gpus={num_gpus} per {self.__class__.__name__} is less than minimum num_gpus={minimum_model_num_gpus}"
        )

        if not isinstance(num_cpus, int):
            raise TypeError(f"`num_cpus` must be an int. Found: {type(num_cpus)} | Value: {num_cpus}")

        kwargs["num_cpus"] = num_cpus
        kwargs["num_gpus"] = num_gpus
        if not silent:
            logger.log(
                15, f"\tFitting {self.name} with 'num_gpus': {kwargs['num_gpus']}, 'num_cpus': {kwargs['num_cpus']}"
            )

        return kwargs

    def _preprocess_fit_resources(
        self,
        silent: bool = False,
        total_resources: dict[str, int | float] | None = None,
        parallel_hpo: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
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
            assert (
                enforced_num_cpus is not None
                and enforced_num_cpus != "auto"
                and enforced_num_gpus is not None
                and enforced_num_gpus != "auto"
            )
            # The logic below is needed because ray cluster is running some process in the backend even when it's ready to be used
            # Trying to use all cores on the machine could lead to resource contention situation
            # TODO: remove this logic if ray team can identify what's going on underneath and how to workaround
            max_resources = self._get_maximum_resources()
            max_num_cpus = max_resources.get("num_cpus", None)
            max_num_gpus = max_resources.get("num_gpus", None)
            if max_num_gpus is not None:
                enforced_num_gpus = min(max_num_gpus, enforced_num_gpus)
            if DistributedContext.is_distributed_mode() and (not DistributedContext.is_shared_network_file_system()):
                minimum_model_resources = self.get_minimum_resources(is_gpu_available=(enforced_num_gpus > 0))
                minimum_model_num_cpus = minimum_model_resources.get("num_cpus", 1)
                enforced_num_cpus = max(
                    minimum_model_num_cpus, enforced_num_cpus - 2
                )  # leave some cpu resources for process running by cluster nodes
            if max_num_cpus is not None:
                enforced_num_cpus = min(max_num_cpus, enforced_num_cpus)
            kwargs["num_cpus"] = enforced_num_cpus
            kwargs["num_gpus"] = enforced_num_gpus
            return kwargs

        return self._calculate_total_resources(
            silent=silent, total_resources=total_resources, parallel_hpo=parallel_hpo, **kwargs
        )

    def _register_fit_metadata(self, **kwargs):
        """
        Used to track properties of the inputs received during fit, such as if validation data was present.
        """
        if not self._is_fit_metadata_registered:
            self._fit_metadata = self._compute_fit_metadata(**kwargs)
            self._is_fit_metadata_registered = True

    def _compute_fit_metadata(
        self,
        X: pd.DataFrame = None,
        X_val: pd.DataFrame = None,
        X_unlabeled: pd.DataFrame = None,
        num_cpus: int = None,
        num_gpus: int = None,
        **kwargs,
    ) -> dict:
        fit_metadata = dict(
            num_samples=len(X) if X is not None else None,
            val_in_fit=X_val is not None,
            unlabeled_in_fit=X_unlabeled is not None,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )
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

    def fit(
        self,
        *,
        log_resources: bool = False,
        log_resources_prefix: str | None = None,
        **kwargs,
    ):
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
        X_test : DataFrame, default = None
            The test data features. Note: Not used for training, but for tracking test performance.
            If None, early stopping via validation score will be disabled.
        y_test : Series, default = None
            The test data ground truth labels. Note: Not used for training, but for tracking test performance.
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
        random_seed : int | None | str, default = "auto"
            The random seed value provided by AutoGluon that can be used to control the randomness of the model (e.g.,
            init, training, etc.). Note, this parameter is not passed to `._fit` but used in `_initialize`!
            If "auto", the model will use a default random seed of 0.
            When using a bagged model, this value differs per fold model. The first fold model uses `model_random_seed`,
            the second uses `model_random_seed + 1`, and the last uses `model_random_seed+n_splits-1` where `n_splits`.
            The start value `model_random_seed` can be set via `ag_args_ensemble` in the model's hyperparameters.
        log_resources : bool, default = False
            If True, will log information about the number of CPUs, GPUs, and memory usage during fit.
        log_resources_prefix : str | None, default = None
            If specified, will be prepended to the log generated when `log_resources=True`.
        **kwargs :
            Any additional fit arguments a model supports.
        """
        time_start = time.time()
        kwargs = self.initialize(
            **kwargs
        )  # FIXME: This might have to go before self._preprocess_fit_args, but then time_limit might be incorrect in **kwargs init to initialize
        kwargs = self._preprocess_fit_args(**kwargs)

        self._register_fit_metadata(**kwargs)
        self.validate_fit_resources(**kwargs)
        approx_mem_size_req, available_mem = self._validate_fit_memory_usage(**kwargs)
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - time_start
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f"\tWarning: Model has no time left to train, skipping model... (Time Left = {kwargs['time_limit']:.1f}s)"
                )
                raise TimeLimitExceeded
        self.validate_fit_args(**kwargs)
        if log_resources:
            num_cpus = kwargs.get("num_cpus", None)
            num_gpus = kwargs.get("num_gpus", None)
            approx_mem_size_req_gb = approx_mem_size_req / (1024**3) if approx_mem_size_req is not None else None
            available_mem_gb = available_mem / (1024**3) if available_mem is not None else None
            if log_resources_prefix is None:
                log_resources_prefix = ""
            msg = f"\t{log_resources_prefix}Fitting with cpus={num_cpus}, gpus={num_gpus}"
            if approx_mem_size_req_gb is not None and available_mem_gb is not None:
                msg_mem = f", mem={approx_mem_size_req_gb:.1f}/{available_mem_gb:.1f} GB"
                msg += msg_mem
            logger.log(20, msg)
        reset_torch_threads = self._get_class_tags().get("reset_torch_threads", False)
        reset_torch_cudnn_deterministic = self._get_class_tags().get("reset_torch_cudnn_deterministic", False)

        torch_threads_og = None
        torch_cudnn_deterministic_og = None

        # --- Snapshot original values ----------------------------------------------
        if reset_torch_threads or reset_torch_cudnn_deterministic:
            try:
                import torch
            except ImportError:
                # torch missing -> nothing to restore
                pass
            else:
                if reset_torch_threads:
                    torch_threads_og = torch.get_num_threads()

                if reset_torch_cudnn_deterministic:
                    torch_cudnn_deterministic_og = torch.backends.cudnn.deterministic
        try:
            out = self._fit(**kwargs)
            if out is None:
                out = self
            out = out._post_fit(**kwargs)
        finally:
            # Always executed even if _fit or _post_fit raise
            if (torch_threads_og is not None) or (torch_cudnn_deterministic_og is not None):
                try:
                    import torch
                except ImportError:
                    pass
                else:
                    if torch_threads_og is not None:
                        if torch.get_num_threads() != torch_threads_og:
                            torch.set_num_threads(torch_threads_og)

                    if torch_cudnn_deterministic_og is not None:
                        cudnn = torch.backends.cudnn
                        if cudnn.deterministic != torch_cudnn_deterministic_og:
                            cudnn.deterministic = torch_cudnn_deterministic_og
        return out

    # FIXME: Simply log a message that the model is being skipped instead of logging a traceback.
    def validate_fit_args(self, X: pd.DataFrame, feature_metadata: FeatureMetadata | None = None, **kwargs):
        """
        Verifies if the fit arguments satisfy the model's constraints.
        Raises an exception if constraints are not satisfied.

        Checks for:
            ag.problem_types
            ag.max_rows
            ag.max_features
            ag.max_classes
            ag.ignore_constraints
        """
        if self.is_initialized():
            ag_params = self._get_ag_params()
        else:
            ag_params = self._get_ag_params(params_aux=self._get_params_aux())

        problem_types: list[str] | None = ag_params.get("problem_types", None)
        max_classes: int | None = ag_params.get("max_classes", None)
        max_rows: int | None = ag_params.get("max_rows", None)
        max_features: int | None = ag_params.get("max_features", None)
        ignore_constraints: bool = ag_params.get("ignore_constraints", False)

        if ignore_constraints:
            # skip all validation checks
            logger.log(15, f"\t`ag.ignore_constraints=True`, skipping sanity checks for model...")
            return

        if problem_types is not None:
            if self.problem_type not in problem_types:
                raise AssertionError(
                    f"ag.problem_types={problem_types} for model '{self.name}', "
                    f"but found '{self.problem_type}' problem_type."
                )
            assert self.problem_type in problem_types
        if max_classes is not None:
            if self.num_classes is not None and self.num_classes > max_classes:
                raise AssertionError(
                    f"ag.max_classes={max_classes} for model '{self.name}', but found {self.num_classes} classes."
                )
        if max_rows is not None:
            n_rows = X.shape[0]
            if n_rows > max_rows:
                raise AssertionError(f"ag.max_rows={max_rows} for model '{self.name}', but found {n_rows} rows.")
        if max_features is not None:
            n_features = X.shape[1]

            if feature_metadata is None:
                # Fallback to using self._feature_metadata if not provided
                feature_metadata = self._feature_metadata

            if feature_metadata is not None:
                feature_generator = self.get_preprocessor()
                if feature_generator is not None:
                    # TODO: Can be faster if can calculate new_feature_metadata w/o fitting feature generator
                    new_feature_metadata = self._estimate_dtypes_after_preprocessing_cheap(
                        X=X,
                        y=kwargs["y"],
                        feature_generator=feature_generator,
                    )
                    n_features = len(new_feature_metadata.get_features())

            if n_features > max_features:
                raise AssertionError(
                    f"ag.max_features={max_features} for model '{self.name}', but found {n_features} features."
                )

    def _post_fit(self, **kwargs):
        """
        Logic to perform at the end of `self.fit(...)`
        This should be focused around computing and saving metadata that is only possible post-fit.
        Parameters are identical to those passed to `self._fit(...)`.

        Returns
        -------
        Returns self
        """
        if self._get_ag_params().get("max_rows", None) is not None:
            # ensures that an exception is not raised on refit
            if "ag.max_rows" not in self.params_trained:
                self.params_trained["ag.max_rows"] = None

        compiler_configs = self.params_aux.get("compile", None)
        if compiler_configs is not None:
            compile_model = True
            if isinstance(compiler_configs, bool):
                if compiler_configs:
                    compiler_configs = None
                else:
                    compile_model = False
            if compile_model:
                self.compile(compiler_configs=compiler_configs)
        predict_1_batch_size = self.params_aux.get("predict_1_batch_size", None)
        if (
            self.predict_1_time is None
            and predict_1_batch_size is not None
            and "X" in kwargs
            and kwargs["X"] is not None
        ):
            X_1 = sample_df_for_time_func(df=kwargs["X"], sample_size=predict_1_batch_size)
            self.predict_1_time = time_func(f=self.predict, args=[X_1]) / len(X_1)
        return self

    def get_features(self) -> list[str]:
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

        X = self.preprocess(X=X, y=y)
        self.model = self.model.fit(X, y)

    # TODO: add model-tag to check if the model can work with `None` random seed?
    # TODO: add check that int seed is smaller than `int(np.iinfo(np.int32).max)`?
    def init_random_seed(self, random_seed: int | None | str, hyperparameters: dict | None = None):
        """Initialize the random seed used by the model by setting `self.random_seed`.

        The random seed can be used to control the randomness of the model (e.g., init, training, etc.).
        By default, AutoGluon's random_seed is 0 to ensure reproducibility. Following convention,
        a random seed can be either an integer or None.

        When using a bagged model, this value differs per fold model. The first fold model uses `model_random_seed`,
        the second uses `model_random_seed + 1`, and the last uses `model_random_seed+n_splits-1` where `n_splits`.
        The start value `model_random_seed` can be set via `ag_args_ensemble` in the model's hyperparameters.

        Parameters
        ----------
        random_seed:
            The random seed passed to `fit`. If "auto", the model will use a default random seed of 0.
            Otherwise, it will set the model's random seed to the provided value.
        hyperparameters
            The hyperparameters of the model, which may or may not contain a random_seed.
            If the hyperparameters contain a random_seed, it will be used to set the model's random seed and
            thus override the random_seed provided in `random_seed`.
        """
        # Set default random seed
        if random_seed == "auto":
            random_seed = self.default_random_seed

        # Overwrite random seed based on hyperparameters, if available
        if hyperparameters is not None:
            hp_rs, seed_name = self._get_random_seed_from_hyperparameters(hyperparameters=hyperparameters)
            if not isinstance(hp_rs, str) and seed_name is not None:
                hyperparameters = hyperparameters.copy()
                random_seed = hyperparameters.pop(seed_name)
                assert random_seed == hp_rs

        if self.seed_name is not None:
            if hyperparameters is None:
                hyperparameters = {}
            else:
                hyperparameters = hyperparameters.copy()
            hyperparameters[self.seed_name] = random_seed
            self.random_seed = hyperparameters[self.seed_name]
        else:
            self.random_seed = random_seed

        return hyperparameters

    def _get_random_seed_from_hyperparameters(self, hyperparameters: dict) -> tuple[int | None | str, str | None]:
        """Extract the random seed from the hyperparameters if available.

        A model implementation may override this method to extract the random seed from the hyperparameters such that
        it is used to init the model's random seed. Otherwise, we default to not being able to extract a random seed
        and use the random seed provided by AutoGluon.

        Parameters
        ----------
        hyperparameters:
            The hyperparameters that may contain a random seed.

        Returns
        -------
        random_seed : int | None | str
            The random seed extracted from the hyperparameters, or "N/A" if not available.
        seed_name: str | None
            The key of the extracted random_seed value, or None if not available.
        """
        if self.seed_name is not None:
            if self.seed_name in hyperparameters:
                return hyperparameters[self.seed_name], self.seed_name
            else:
                for seed_name in self.seed_name_alt:
                    if seed_name in hyperparameters:
                        return hyperparameters[seed_name], seed_name
        return "N/A", None

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
        For binary and multiclass problems, this returns the predicted class labels as a 1d numpy array.
        For regression problems, this returns the predicted values as a 1d numpy array.
        """
        y_pred_proba = self.predict_proba(X, **kwargs)
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        return y_pred

    def predict_proba(
        self, X: pd.DataFrame, *, normalize: bool | None = None, record_time: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Returns class prediction probabilities of X.
        For binary problems, this returns the positive class label probability as a 1d numpy array.
        For multiclass problems, this returns the class label probabilities of each class as a 2d numpy array.
        For regression problems, this returns the predicted values as a 1d numpy array.

        Parameters
        ----------
        X
            The data used for prediction.
        normalize: bool | None, default = None
            Whether to normalize the predictions prior to returning.
            If None, will default to `self.normalize_pred_probas`.
        record_time: bool, default = False
            If True, will record the time taken for prediction in `self.predict_time` and the number of rows of X in `self.predict_n_size`.
        kwargs
            Keyword arguments to pass into `self._predict_proba`.

        Returns
        -------
        y_pred_proba : np.ndarray
            The prediction probabilities
        """
        time_start = time.time() if record_time else None

        max_batch_size: int | None = self.params_aux.get("max_batch_size", None)
        if max_batch_size is not None and max_batch_size < len(X):
            y_pred_proba = self._predict_proba_batch(X=X, max_batch_size=max_batch_size, normalize=normalize, **kwargs)
        else:
            y_pred_proba = self._predict_proba_internal(X=X, normalize=normalize, **kwargs)

        if self.params_aux.get("temperature_scalar", None) is not None:
            y_pred_proba = self._apply_temperature_scaling(y_pred_proba)
        elif self.conformalize is not None:
            y_pred_proba = self._apply_conformalization(y_pred_proba)
        if record_time:
            self.predict_time = time.time() - time_start
            self.record_predict_info(X=X)
        return y_pred_proba

    def _predict_proba_batch(
        self,
        X: pd.DataFrame,
        max_batch_size: int,
        **kwargs,
    ) -> np.ndarray:
        assert max_batch_size > 0

        len_X = len(X)
        chunks: list[np.ndarray] = []
        for start in range(0, len_X, max_batch_size):
            stop = min(start + max_batch_size, len_X)
            X_batch = X.iloc[start:stop]  # preserves row order and index
            proba_batch = self._predict_proba_internal(X=X_batch, **kwargs)
            chunks.append(proba_batch)

        # Concatenate along the first axis so the result matches the unbatched call
        y_pred_proba = np.concatenate(chunks, axis=0)
        return y_pred_proba

    def _predict_proba_internal(self, X, *, normalize: bool | None = None, **kwargs):
        if normalize is None:
            normalize = self.normalize_pred_probas
        y_pred_proba = self._predict_proba(X=X, **kwargs)
        if normalize:
            y_pred_proba = normalize_pred_probas(y_pred_proba, self.problem_type)
        y_pred_proba = y_pred_proba.astype(np.float32)
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

        if self.problem_type == REGRESSION:
            return self.model.predict(X)
        elif self.problem_type == QUANTILE:
            y_pred = self.model.predict(X)
            return y_pred.reshape([-1, len(self.quantile_levels)])

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

    def score(
        self,
        X,
        y: np.ndarray,
        metric: Scorer = None,
        sample_weight: np.ndarray = None,
        as_error: bool = False,
        **kwargs,
    ) -> float:
        if metric is None:
            metric = self.eval_metric

        if metric.needs_pred or metric.needs_quantile:
            y_pred = self.predict(X=X, **kwargs)
            y_pred_proba = None
        else:
            y_pred = None
            y_pred_proba = self.predict_proba(X=X, **kwargs)

        return compute_metric(
            y=y,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            metric=metric,
            weights=sample_weight,
            as_error=as_error,
            quantile_levels=self.quantile_levels,
        )

    def score_with_y_pred_proba(
        self,
        y: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: Scorer = None,
        sample_weight: np.ndarray = None,
        as_error: bool = False,
    ) -> float:
        if metric is None:
            metric = self.eval_metric
        if metric.needs_pred or metric.needs_quantile:
            y_pred = self.predict_from_proba(y_pred_proba=y_pred_proba)
            y_pred_proba = None
        else:
            y_pred = None
        return compute_metric(
            y=y,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            metric=metric,
            weights=sample_weight,
            as_error=as_error,
            quantile_levels=self.quantile_levels,
        )

    def save(self, path: str | None = None, verbose: bool = True) -> str:
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

    def save_learning_curves(
        self, metrics: str | list[str], curves: dict[dict[str, list[float]]], path: str = None
    ) -> str:
        """
        Saves learning curves to disk.

        Outputted Curve Format:
            out = [
                metrics,
                [
                    [ # log_loss
                        [0.693147, 0.690162, ...], # train
                        [0.693147, 0.690162, ...], # val
                        [0.693147, 0.690162, ...], # test
                    ],
                    [ # accuracy
                        [0.693147, 0.690162, ...], # train
                        [0.693147, 0.690162, ...], # val
                        [0.693147, 0.690162, ...], # test
                    ],
                    [ # f1
                        [0.693147, 0.690162, ...], # train
                        [0.693147, 0.690162, ...], # val
                        [0.693147, 0.690162, ...], # test
                    ],
                ]
            ]

        Parameters
        ----------
        metrics : str or list(str)
            List of all evaluation metrics computed at each iteration of the curve
        curves : dict[dict[str : list[float]]]
            Dictionary of evaluation sets and their learning curve dictionaries.
            Each learning curve dictionary contains evaluation metrics computed at each iteration.
            e.g.
                curves = {
                        "train": {
                            'logloss': [0.693147, 0.690162, ...],
                            'accuracy': [0.500000, 0.400000, ...],
                            'f1': [0.693147, 0.690162, ...]
                        },
                        "val": {...},
                        "test": {...},
                    }

        path : str, default None
            Path where the learning curves are saved, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            If None, self.path is used.
            The final curve file is typically saved to os.path.join(path, curves.json).

        Returns
        -------
        path : str
            Path to the saved curves, minus the file name.
        """
        if not self._get_class_tags().get("supports_learning_curves", False):
            raise AssertionError(f"Learning Curves are not supported for model: {self.name}")

        if path is None:
            path = self.path
        if not isinstance(metrics, list):
            metrics = [metrics]
        if len(metrics) == 0:
            raise ValueError("At least one metric must be specified to save generated learning curves.")

        os.makedirs(path, exist_ok=True)
        out = self._make_learning_curves(metrics=metrics, curves=curves)
        file_path = os.path.join(path, self.learning_curve_file_name)
        save_json.save(file_path, out)
        self.saved_learning_curves = True
        return file_path

    def _make_learning_curves(
        self, metrics: str | list[str], curves: dict[dict[str, list[float]]]
    ) -> list[list[str], list[str], list[list[float]]]:
        """
        Parameters
        ----------
        metrics : str or list(str)
            List of all evaluation metrics computed at each iteration of the curve
        curves : dict[dict[str : list[float]]]
            Dictionary of evaluation sets and their learning curve dictionaries.
            Each learning curve dictionary contains evaluation metrics computed at each iteration.
            See Abstract Model's save_learning_curves method for a sample curves input.

        Returns
        -------
        list[list[str], list[str], list[list[float]]]: The generated learning curve artifact.
            if eval set names includes: train, val, or test
            these sets will be placed first in the above order.
        """

        # ensure main eval sets first: train, val, test
        items = []
        order = ["train", "val", "test"]
        for eval_set in order:
            if eval_set in curves:
                items.append((eval_set, curves[eval_set]))
                del curves[eval_set]

        items.extend(curves.items())
        eval_sets, curves = list(zip(*items))

        data = []
        for metric in metrics:
            data.append([c[metric] for c in curves])

        return [eval_sets, metrics, data]

    @classmethod
    def load_learning_curves(cls, path: str) -> list:
        """
        Loads the learning_curve data from disk to memory.

        Parameters
        ----------
        path : str
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            The model file is typically located in os.path.join(path, cls.model_file_name).

        Returns
        -------
        learning_curves : list
            Loaded learning curve data.
        """
        if not cls._get_class_tags().get("supports_learning_curves", False):
            raise AssertionError("Attempted to load learning curves from model without learning curve support")

        file = os.path.join(path, cls.learning_curve_file_name)

        if not os.path.exists(file):
            raise FileNotFoundError(
                f"Could not find learning curve file at {file}"
                + "\nDid you call predictor.fit() with an appropriate learning_curves parameter?"
            )

        return load_json.load(file)

    # TODO: v1.0: Add docs
    def compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str] = None,
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
            fi_df = self._compute_permutation_importance(
                X=X, y=y, features=features_to_check, silent=silent, importance_as_list=importance_as_list, **kwargs
            )
            n = fi_df.iloc[0]["n"] if len(fi_df) > 0 else 1
        else:
            fi_df = None
            n = kwargs.get("num_shuffle_sets", 1)

        if importance_as_list:
            banned_importance = [0] * n
            results_banned = pd.Series(
                data=[banned_importance for _ in range(len(banned_features))], index=banned_features, dtype="object"
            )
        else:
            banned_importance = 0
            results_banned = pd.Series(
                data=[banned_importance for _ in range(len(banned_features))], index=banned_features, dtype="float64"
            )

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
        features: list[str],
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
        List of (shape: tuple[int], dtype: Any)
        shape: tuple[int]
            A tuple that describes input
        dtype: Any, default=np.float32
            The element type in numpy dtype.
        """
        return [((batch_size, len(self._features)), np.float32)]

    @classmethod
    def _default_compiler(cls):
        """The default compiler for the underlining model."""
        return None

    @classmethod
    def _valid_compilers(cls) -> list:
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
                    f"Specified compiler ({compiler}) is unable to compile"
                    ' (potentially lacking dependencies) and "compiler_fallback_to_native==False"'
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
        hyperparameters = self.get_hyperparameters_init()

        args = dict(
            path=path,
            name=name,
            problem_type=problem_type,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
        )

        return args

    def get_hyperparameters_init(self) -> dict:
        """

        Returns
        -------
        hyperparameters: dict
            The dictionary of user specified hyperparameters for the model.

        """
        hyperparameters = self._user_params.copy()
        if self._user_params_aux:
            hyperparameters[AG_ARGS_FIT] = self._user_params_aux.copy()
        return hyperparameters

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
        params["hyperparameters"][AG_ARGS_FIT]["max_memory_usage_ratio"] = (
            params["hyperparameters"][AG_ARGS_FIT].get("max_memory_usage_ratio", 1.0) * 1.25
        )

        params["hyperparameters"].update(self.params_trained)
        params["name"] = params["name"] + REFIT_FULL_SUFFIX
        template = self.__class__(**params)

        return template

    def hyperparameter_tune(
        self, hyperparameter_tune_kwargs="auto", hpo_executor: HpoExecutor = None, time_limit: float = None, **kwargs
    ):
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
        Tuple of (hpo_results: dict[str, dict], hpo_info: Any)
        hpo_results: dict[str, dict]
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
            hpo_executor.initialize(
                hyperparameter_tune_kwargs, default_num_trials=default_num_trials, time_limit=time_limit
            )
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
            trial_soft_time_limit = max(
                hpo_executor.time_limit * 0.9, hpo_executor.time_limit - 5
            )  # 5 seconds max for buffer

        fit_kwargs = dict()
        fit_kwargs["feature_metadata"] = self.feature_metadata
        fit_kwargs["num_classes"] = self.num_classes
        fit_kwargs["sample_weight"] = kwargs.get("sample_weight", None)
        fit_kwargs["sample_weight_val"] = kwargs.get("sample_weight_val", None)
        fit_kwargs["verbosity"] = kwargs.get("verbosity", 2)
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
        minimum_resources = self.get_minimum_resources(
            is_gpu_available=(hpo_executor.resources.get("num_gpus", 0) > 0)
        )
        # This explicitly tells ray.Tune to not change the working directory
        # to the trial directory, giving access to paths relative to
        # the original working directory.
        os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
        hpo_executor.execute(
            model_trial=model_trial,
            train_fn_kwargs=train_fn_kwargs,
            directory=directory,
            minimum_cpu_per_trial=minimum_resources.get("num_cpus", 1),
            minimum_gpu_per_trial=minimum_resources.get("num_gpus", 0),
            model_estimate_memory_usage=model_estimate_memory_usage,
            adapter_type="tabular",
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
        backend = (
            self._get_model_base()._get_hpo_backend()
        )  # If ensemble, will use the base model to determine backend
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
    def disk_usage(self) -> int:
        # Taken from https://stackoverflow.com/a/1392549
        from pathlib import Path

        model_path = Path(self.path)
        model_disk_usage = sum(f.stat().st_size for f in model_path.glob("**/*") if f.is_file())
        return model_disk_usage

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
            except Exception:
                return None
        else:
            return self._get_memory_size()

    def _get_memory_size(self) -> int:
        gc.collect()  # Try to avoid OOM error
        return sys.getsizeof(pickle.dumps(self, protocol=4))

    # TODO: Refine this
    def _estimate_dtypes_after_preprocessing_cheap(
        self,
        X: pd.DataFrame,
        y,
        feature_generator: AbstractFeatureGenerator,
    ) -> FeatureMetadata:
        sample_size = 1000
        from autogluon.core.utils.utils import generate_train_test_split

        if X.shape[0] > sample_size:
            X_sample, _, y_sample, _ = generate_train_test_split(
                X=X,
                y=y,
                train_size=sample_size,
                problem_type=self.problem_type,
            )
        else:
            X_sample = X
            y_sample = y

        _ = feature_generator.fit_transform(
            X=X_sample,
            y=y_sample,
            feature_metadata_in=self._feature_metadata,
            problem_type=self.problem_type,
        )
        new_feature_metadata = feature_generator.feature_metadata
        return new_feature_metadata

    def estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        """
        Estimates the peak memory usage of the model while training.

        Parameters
        ----------
        X: pd.DataFrame
            The training data features

        Returns
        -------
        int: estimated peak memory usage in bytes during training
        """
        assert self.is_initialized(), "Only estimate memory usage after the model is initialized."

        feature_generator = self.get_preprocessor()
        if feature_generator is not None:
            if self.can_estimate_memory_usage_static_lite():
                # TODO: Can be faster if can calculate new_feature_metadata w/o fitting feature generator
                new_feature_metadata = self._estimate_dtypes_after_preprocessing_cheap(
                    X=X,
                    y=kwargs["y"],
                    feature_generator=feature_generator,
                )
                hyperparameters = self._get_model_params()
                memory_usage_estimate = self.estimate_memory_usage_static_lite(
                    num_samples=len(X),
                    num_features=len(new_feature_metadata.get_features()),
                    hyperparameters=hyperparameters,
                    problem_type=self.problem_type,
                    num_classes=self.num_classes,
                )
                self._memory_usage_estimate = memory_usage_estimate
                return memory_usage_estimate
            else:
                # FIXME: This is expensive
                X_transformed = feature_generator.fit_transform(
                    X=X,
                    y=kwargs["y"],
                    feature_metadata_in=self._feature_metadata,
                    problem_type=self.problem_type,
                )
                memory_usage_estimate = self._estimate_memory_usage(X=X_transformed, **kwargs)
                self._memory_usage_estimate = memory_usage_estimate
                return memory_usage_estimate

        memory_usage_estimate = self._estimate_memory_usage(X=X, **kwargs)
        self._memory_usage_estimate = memory_usage_estimate
        return memory_usage_estimate

    # FIXME: Update args, maybe use feature metadata instead?
    @classmethod
    def estimate_memory_usage_static_lite(
        cls,
        num_samples: int,
        num_features: int,
        num_bytes_per_cell: float = 4,
        hyperparameters: dict = None,
        num_classes: int = 1,
        **kwargs,
    ) -> int:
        return cls._estimate_memory_usage_static_lite(
            num_samples=num_samples,
            num_features=num_features,
            num_bytes_per_cell=num_bytes_per_cell,
            hyperparameters=hyperparameters,
            num_classes=num_classes,
            **kwargs,
        )

    @classmethod
    def estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        y: pd.Series = None,
        hyperparameters: dict = None,
        problem_type: str = "infer",
        num_classes: int | None | str = "infer",
        **kwargs,
    ) -> int:
        """
        Estimates the peak memory usage of the model while training, without having to initialize the model.

        Parameters
        ----------
        X: pd.DataFrame
            The training data features
        y: pd.Series, optional
            The training data ground truth. Must be specified if problem_type or num_classes is unspecified.
        hyperparameters: dict, optional
            The model hyperparameters
        problem_type: str, default = "infer"
            The problem_type. If "infer" will infer based on y.
        num_classes
            The num_classes. If "infer" will infer based on y.
        **kwargs
            Other optional key-word fit arguments that could impact memory usage for the model.

        Returns
        -------
        int: estimated peak memory usage in bytes during training
        """
        if problem_type == "infer":
            problem_type = cls._infer_problem_type(y=y)
        if isinstance(num_classes, str) and num_classes == "infer":
            num_classes = cls._infer_num_classes(y=y, problem_type=problem_type)
        if hyperparameters is None:
            hyperparameters = {}
        hyperparameters = cls._get_model_params_static(
            hyperparameters=hyperparameters, convert_search_spaces_to_default=True
        )
        return cls._estimate_memory_usage_static(
            X=X, y=y, hyperparameters=hyperparameters, problem_type=problem_type, num_classes=num_classes, **kwargs
        )

    def estimate_memory_usage_child(self, X: pd.DataFrame, **kwargs) -> int:
        """
        Estimates the peak memory usage of the child model while training.

        If the model is not a bagged model (aka has no children), then will return its personal memory usage estimate.

        Parameters
        ----------
        X: pd.DataFrame
            The training data features
        **kwargs

        Returns
        -------
        int: estimated peak memory usage in bytes during training of the child
        """
        return self.estimate_memory_usage(**kwargs)

    def estimate_memory_usage_static_child(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series = None,
        hyperparameters: dict = None,
        problem_type: str = "infer",
        num_classes: int | None | str = "infer",
        **kwargs,
    ) -> int:
        """
        Estimates the peak memory usage of the child model while training, without having to initialize the model.

        Note that this method itself is not static, because the child model must be present
        as a variable in the model to call its static memory estimate method.

        To obtain the child memory estimate in a fully static manner, instead directly call the child's `estimate_memory_usage_static` method.

        Parameters
        ----------
        X: pd.DataFrame
            The training data features
        y: pd.Series, optional
            The training data ground truth. Must be specified if problem_type or num_classes is unspecified.
        hyperparameters: dict, optional
            The model hyperparameters
        problem_type: str, default = "infer"
            The problem_type. If "infer" will infer based on y.
        num_classes
            The num_classes. If "infer" will infer based on y.
        **kwargs
            Other optional key-word fit arguments that could impact memory usage for the model.

        Returns
        -------
        int: estimated peak memory usage in bytes during training of the child
        """
        return self.estimate_memory_usage_static(
            X=X, y=y, hyperparameters=hyperparameters, problem_type=problem_type, num_classes=num_classes, **kwargs
        )

    def validate_fit_resources(self, num_cpus="auto", num_gpus="auto", total_resources=None, **kwargs):
        """
        Verifies that the provided num_cpus and num_gpus (or defaults if not provided) are sufficient to train the model.
        Raises an AssertionError if not sufficient.
        """
        resources = self._preprocess_fit_resources(
            num_cpus=num_cpus, num_gpus=num_gpus, total_resources=total_resources, silent=True
        )
        self._validate_fit_resources(**resources)

    def _validate_fit_resources(self, **resources):
        res_min = self.get_minimum_resources()
        for resource_name in res_min:
            if resource_name not in resources:
                raise AssertionError(
                    f"Model requires {res_min[resource_name]} {resource_name} to fit, but no available amount was defined."
                )
            elif res_min[resource_name] > resources[resource_name]:
                raise AssertionError(
                    f"Model requires {res_min[resource_name]} {resource_name} to fit, but {resources[resource_name]} are available."
                )
        total_resources = resources.get("total_resources", None)
        if total_resources is None:
            total_resources = {}
        for resource_name, resource_value in total_resources.items():
            if resources[resource_name] > resource_value:
                raise AssertionError(
                    f"Specified {resources[resource_name]} {resource_name} to fit, but only {resource_value} are available in total."
                )

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
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

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict = None,
        num_classes: int = 1,
        **kwargs,
    ) -> int:
        raise NotImplementedError

    @disable_if_lite_mode(ret=(None, None))
    def _validate_fit_memory_usage(
        self,
        mem_error_threshold: float = 0.9,
        mem_warning_threshold: float = 0.75,
        mem_size_threshold: int = None,
        approx_mem_size_req: int = None,
        available_mem: int = None,
        **kwargs,
    ) -> tuple[int | None, int | None]:
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
        approx_mem_size_req: int, default = None
            If specified, will use this value as the overall memory usage estimate instead of calculating within the method.
        available_mem: int, default = None
            If specified, will use this value as the available memory instead of calculating within the method.
        **kwargs : dict,
            Fit time kwargs, including X, y, X_val, and y_val.
            Can be used to customize estimation of memory usage.

        Returns
        -------
        approx_mem_size_req: int | None
            The estimated memory requirement of the model, in bytes
            If None, approx_mem_size_req was not calculated.
        available_mem: int | None
            The available memory of the system, in bytes
            If None, available_mem was not calculated.
        """
        max_memory_usage_ratio = self.params_aux["max_memory_usage_ratio"]
        if max_memory_usage_ratio is None:
            return approx_mem_size_req, available_mem  # Skip memory check

        if approx_mem_size_req is None:
            approx_mem_size_req = self.estimate_memory_usage(**kwargs)
        if mem_size_threshold is not None and approx_mem_size_req < (
            mem_size_threshold * min(max_memory_usage_ratio, 1)
        ):
            return approx_mem_size_req, available_mem  # Model is smaller than the min threshold to check available mem

        if available_mem is None:
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
            f"Estimated to require {approx_mem_size_req / (1024**3):.3f} GB "
            f"out of {available_mem / (1024**3):.3f} GB available memory ({expected_memory_usage_ratio * 100:.3f}%)... "
            f"({max_memory_usage_error_ratio * 100:.3f}% of avail memory is the max safe size)"
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

        return approx_mem_size_req, available_mem

    def reduce_memory_size(
        self, remove_fit: bool = True, remove_info: bool = False, requires_save: bool = True, **kwargs
    ):
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

    def get_info(self, include_feature_metadata: bool = True) -> dict:
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
            "hyperparameters_user": self.get_hyperparameters_init(),
            "hyperparameters_fit": self.params_trained,  # TODO: Explain in docs that this is for hyperparameters that differ in final model from original hyperparameters, such as epochs (from early stopping)
            "hyperparameters_nondefault": self.nondefault_params,
            AG_ARGS_FIT: self.get_params_aux_info(),
            "num_features": len(self.features) if self.features else None,
            "features": self.features,
            "feature_metadata": self.feature_metadata,
            # 'disk_usage': self.disk_usage(),
            "memory_size": self.get_memory_size(allow_exception=True),  # Memory usage of model in bytes
            "compile_time": self.compile_time if hasattr(self, "compile_time") else None,
            "is_initialized": self.is_initialized(),
            "is_fit": self.is_fit(),
            "is_valid": self.is_valid(),
            "can_infer": self.can_infer(),
            "has_learning_curves": self.saved_learning_curves,
        }
        if self._is_fit_metadata_registered:
            info.update(self._fit_metadata)
        if not include_feature_metadata:
            info.pop("feature_metadata")
        return info

    def get_params_aux_info(self) -> dict:
        """
        Converts learning curve scorer objects into their name strings.

        Returns:
        --------
        params_aux dictionary with changed curve_metrics field, if applicable.
        """
        if self.params_aux.get("curve_metrics", None) is not None:
            params_aux = self.params_aux.copy()
            params_aux["curve_metrics"] = [metric.name for metric in params_aux["curve_metrics"]]
            return params_aux

        return self.params_aux

    @classmethod
    def load_info(cls, path: str, load_model_if_required: bool = True) -> dict:
        load_path = os.path.join(path, cls.model_info_name)
        if Path(load_path).exists():
            return load_pkl.load(path=load_path)
        else:
            if load_model_if_required:
                model = cls.load(path=path, reset_paths=True)
                return model.get_info()
            else:
                raise AssertionError(
                    f"No info file exists in '{load_path}', and `load_model_if_required={load_model_if_required}"
                )

    def save_info(self) -> dict:
        info = self.get_info()

        save_pkl.save(path=os.path.join(self.path, self.model_info_name), object=info)
        json_path = os.path.join(self.path, self.model_info_json_name)
        save_json.save(path=json_path, obj=info)
        return info

    @property
    def predict_n_size(self) -> int | None:
        """
        The number of rows in the data used when calculating `self.predict_time`.
        """
        return self._predict_n_size

    @property
    def predict_n_time_per_row(self) -> float | None:
        """
        The time in seconds required to predict 1 row of data given a batch size of `self.predict_n_size`.
        Returns None if either `self.predict_time` or `self.predict_n_size` are None.
        """
        if self.predict_time is None or self.predict_n_size is None:
            return None
        return self.predict_time / self.predict_n_size

    def record_predict_info(self, X: pd.DataFrame):
        """
        Records the necessary information to compute `self.predict_n_time_per_row`.

        Parameters
        ----------
        X: pd.DataFrame
            The data used to predict on when calculating `self.predict_time`.
        """
        self._predict_n_size = len(X)

    # TODO: Move out of AbstractModel
    def _init_preprocessor(
        self,
        preprocessor_cls: Type[AbstractFeatureGenerator] | str,
        init_params: dict | None,
    ) -> AbstractFeatureGenerator:
        if isinstance(preprocessor_cls, str):
            preprocessor_cls = resolve_fg_class(
                name=preprocessor_cls,
                registry=ag_feature_generator_registry.key_to_cls_map(),
            )
        if init_params is None:
            init_params = {}
        _init_params = dict(
            verbosity=0,
            random_state=self.random_seed,
            target_type=self.problem_type,
        )
        _init_params.update(**init_params)
        return preprocessor_cls(
            **_init_params,
        )

    # TODO: Move out of AbstractModel
    def _recursive_init_preprocessors(self, prep_param: tuple | list[list | tuple]):
        if isinstance(prep_param, list):
            if len(prep_param) == 0:
                param_type = "list"
            elif len(prep_param) == 2:
                if isinstance(prep_param[0], (str, AbstractFeatureGenerator)):
                    param_type = "generator"
                else:
                    param_type = "list"
            else:
                param_type = "list"
        elif isinstance(prep_param, tuple):
            param_type = "generator"
        else:
            raise ValueError(f"Invalid value for prep_param: {prep_param}")

        if param_type == "list":
            out = []
            for i, p in enumerate(prep_param):
                out.append(self._recursive_init_preprocessors(p))
            return out
        elif param_type == "generator":
            assert len(prep_param) == 2
            preprocessor_cls = prep_param[0]
            init_params = prep_param[1]
            return self._init_preprocessor(
                preprocessor_cls=preprocessor_cls,
                init_params=init_params,
            )
        else:
            raise ValueError(f"Invalid value for prep_param: {prep_param}")

    def get_preprocessor(self, ag_params: dict | None = None) -> AbstractFeatureGenerator | None:
        if ag_params is None:
            ag_params: dict | None = self._get_ag_params().get("model_specific_feature_generator_kwargs", None)
        if ag_params is None:
            return None
        prep_params = ag_params.get("feature_generators", None)
        init_kwargs = ag_params.get("init_kwargs", None)
        passthrough_types = ag_params.get("passthrough_types", None)
        if init_kwargs is None:
            init_kwargs = {}
        if prep_params is None:
            return None
        if not prep_params:
            return None

        preprocessors = self._recursive_init_preprocessors(prep_param=prep_params)
        if len(preprocessors) == 0:
            return None
        if len(preprocessors) == 1 and isinstance(preprocessors[0], AbstractFeatureGenerator):
            return preprocessors[0]
        else:
            kwargs = dict(
                # TODO: "false_recursive" technically can slow down inference, but need to optimize `True` first
                #  Refer to `Bioresponse` dataset where setting to `True` -> 200s fit time vs `false_recursive` -> 1s fit time
                remove_unused_features="false_recursive",
                post_drop_duplicates=True,
                passthrough=True,
                passthrough_types=passthrough_types,
                verbosity=0,
            )

            kwargs.update(init_kwargs)

            preprocessor = BulkFeatureGenerator(generators=preprocessors, **kwargs)
            return preprocessor

    def _get_maximum_resources(self) -> dict[str, int | float]:
        """
        Get the maximum resources allowed to use for this model.
        This can be useful when model not scale well with resources, i.e. cpu cores.
        Return empty dict if no maximum resources needed

        Return
        ------
        dict[str, int | float]
            key, name of the resource, i.e. `num_cpus`, `num_gpus`
            value, maximum amount of resources
        """
        return {}

    def _get_default_resources(self) -> tuple[int, float]:
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
        supported_problem_types = cls.supported_problem_types()
        if supported_problem_types is not None:
            return {"problem_types": supported_problem_types}
        return {}

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """
        [Advanced] Dictionary of customization options related to meta properties of the model ensemble this model will be a child in.
        Refer to hyperparameters of ensemble models for valid options.
        """
        return {}

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        """
        Returns the list of supported problem types.
        If None is returned, then the model has not specified the supported problem types, and it is unknown which problem types are valid.
            In this case, all problem types are considered supported and the model will never be filtered out based on problem type.
        """
        return None

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

    def _get_ag_params(self, params_aux: dict | None = None) -> dict:
        """
        Gets params that are not passed to the inner model, but are used by the wrapper.
        These params should exist in `self.params_aux`.
        """
        if params_aux is None:
            params_aux = self.params_aux
        ag_param_names = self._ag_params()
        ag_param_names_common = self._ag_params_common()
        ag_param_names = ag_param_names.union(ag_param_names_common)
        if ag_param_names:
            return {key: val for key, val in params_aux.items() if key in ag_param_names}
        else:
            return dict()

    def _get_model_params(self, convert_search_spaces_to_default: bool = False) -> dict:
        """
        Gets params that are passed to the inner model.

        Parameters
        ----------
        convert_search_spaces_to_default: bool, default = False
            If True, search spaces are converted to the default value.
            This is useful when having to estimate memory usage estimates prior to doing hyperparameter tuning.

        Returns
        -------
        hyperparameters: dict
            Dictionary of model hyperparameters.
        """
        params = self._get_params()
        return self._get_model_params_static(
            hyperparameters=params, convert_search_spaces_to_default=convert_search_spaces_to_default
        )

    @classmethod
    def _get_model_params_static(cls, hyperparameters: dict, convert_search_spaces_to_default: bool = False) -> dict:
        """
        Gets params that are passed to the inner model.
        This is the static version of `_get_model_params`.
        This method can be called prior to initializing the model.

        Parameters
        ----------
        convert_search_spaces_to_default: bool, default = False
            If True, search spaces are converted to the default value.
            This is useful when having to estimate memory usage estimates prior to doing hyperparameter tuning.

        Returns
        -------
        hyperparameters: dict
            Dictionary of model hyperparameters.
        """
        hyperparameters = hyperparameters.copy()
        if convert_search_spaces_to_default:
            for param, val in hyperparameters.items():
                if isinstance(val, Space):
                    hyperparameters[param] = val.default
        return hyperparameters

    # TODO: Add documentation for valid args for each model. Currently only `early_stop`
    def _ag_params(self) -> set[str]:
        """
        Set of params that are not passed to self.model, but are used by the wrapper.
        For developers, this is purely optional and is just for convenience to logically distinguish between model specific parameters and added AutoGluon functionality.
        The goal is to have common parameter names for useful functionality shared between models,
        even if the functionality is not natively available as a parameter in the model itself or under a different name.

        Below are common patterns / options to make available. Their actual usage and options in a particular model should be documented in the model itself, as it has flexibility to differ.

        Possible params:

        generate_curves : bool
            boolean flag determining if learning curves should be saved to disk for iterative learners.

        curve_metrics : list(...)
            list of metrics to be evaluated at each iteration of the learning curves
            (only used if generate_curves is True)

        use_error_for_curve_metrics : bool
            boolean flag determining if learning curve metrics should be displayed in error format (see Scorer class)

        early_stop : int, str, or tuple
            generic name for early stopping logic. Typically can be an int or a str preset/strategy.
            Also possible to pass tuple of (class, kwargs) to construct a custom early stopping object.
                Refer to `autogluon.core.utils.early_stopping` for examples.

        """
        return set()

    @classmethod
    def _ag_params_common(cls) -> set[str]:
        """
        Set of params that are not passed to self.model, but are used by the wrapper.

        These params are available to all models without requiring special handling in the model.
        They are in addition to the params specified in `_ag_params`

        max_rows: int
            If specified, raises an AssertionError at fit time if len(X) > max_rows
        max_features: int
            If specified, raises an AssertionError at fit time if len(X.columns) > max_rows
        max_classes: int
            If specified, raises an AssertionError at fit time if self.num_classes > max_classes
        problem_types: list[str]
            If specified, raises an AssertionError at fit time if self.problem_type not in problem_types
        ignore_constraints: bool
            If True, ignores the values of `max_rows`, `max_features`, `max_classes` and `problem_types`.

        """
        return {
            "max_rows",
            "max_features",
            "max_classes",
            "problem_types",
            "ignore_constraints",
            "model_specific_feature_generator_kwargs",
            "prep_params",
            "prep_params.passthrough_types",
        }

    @property
    def _features(self) -> list[str]:
        return self._features_internal

    def _get_model_base(self):
        return self

    @property
    def fit_num_cpus(self) -> int:
        """Number of CPUs used when this model was fit"""
        return self.get_fit_metadata()["num_cpus"]

    @property
    def fit_num_gpus(self) -> float:
        """Number of GPUs used when this model was fit"""
        return self.get_fit_metadata()["num_gpus"]

    @property
    def fit_num_cpus_child(self) -> int:
        """Number of CPUs used for fitting one model (i.e. a child model)"""
        return self.fit_num_cpus

    @property
    def fit_num_gpus_child(self) -> float:
        """Number of GPUs used for fitting one model (i.e. a child model)"""
        return self.fit_num_gpus

    @classmethod
    def get_ag_priority(cls, problem_type: str | None = None) -> int:
        """
        Returns the AutoGluon fit priority,
        defined by `cls.ag_priority` and `cls.ag_priority_by_problem_type`.
        """
        if problem_type is None:
            return cls.ag_priority
        else:
            return cls.ag_priority_by_problem_type.get(problem_type, cls.ag_priority)

    @classmethod
    def _class_tags(cls) -> dict:
        return {"supports_learning_curves": False}
