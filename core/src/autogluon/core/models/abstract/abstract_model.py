import copy
import gc
import inspect
import logging
import os
import pickle
from autogluon.core.utils import try_import
import sys
import time
from typing import Dict, Union

import numpy as np
import pandas as pd
import scipy

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.utils import setup_outputdir, disable_if_lite_mode
from autogluon.common.utils.log_utils import DuplicateFilter

from .model_trial import model_trial, skip_hpo
from ._tags import _DEFAULT_TAGS
from ... import metrics, Space
from ...constants import AG_ARGS_FIT, BINARY, REGRESSION, QUANTILE, REFIT_FULL_SUFFIX, OBJECTIVES_TO_NORMALIZE
from ...data.label_cleaner import LabelCleaner, LabelCleanerMulticlassToBinary
from ...hpo.exceptions import EmptySearchSpace
from ...hpo.constants import RAY_BACKEND, CUSTOM_BACKEND
from ...hpo.executors import HpoExecutor, HpoExecutorFactory
from ...scheduler import LocalSequentialScheduler
from ...utils import get_cpu_count, get_pred_from_proba, normalize_pred_probas, infer_eval_metric, infer_problem_type, \
    compute_permutation_feature_importance, compute_weighted_metric
from ...utils.exceptions import TimeLimitExceeded, NoValidFeatures, NotEnoughMemoryError
from ...utils.loaders import load_pkl
from ...utils.savers import save_json, save_pkl
from ...utils.time import sample_df_for_time_func, time_func
from ...utils.try_import import try_import_ray


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
        The final model directory will be path+name+os.path.sep()
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

    model_file_name = 'model.pkl'
    model_info_name = 'info.pkl'
    model_info_json_name = 'info.json'

    def __init__(self,
                 path: str = None,
                 name: str = None,
                 problem_type: str = None,
                 eval_metric: Union[str, metrics.Scorer] = None,
                 hyperparameters=None):

        if name is None:
            self.name = self.__class__.__name__
            logger.log(20, f'Warning: No name was specified for model, defaulting to class name: {self.name}')
        else:
            self.name = name  # TODO: v0.1 Consider setting to self._name and having self.name be a property so self.name can't be set outside of self.rename()

        self.path_root = path
        if self.path_root is None:
            path_suffix = self.name
            if len(self.name) > 0:
                if self.name[0] != os.path.sep:
                    path_suffix = os.path.sep + path_suffix
            # TODO: Would be ideal to not create dir, but still track that it is unique. However, this isn't possible to do without a global list of used dirs or using UUID.
            path_cur = setup_outputdir(path=None, create_dir=True, path_suffix=path_suffix)
            self.path_root = path_cur.rsplit(self.path_suffix, 1)[0]
            logger.log(20, f'Warning: No path was specified for model, defaulting to: {self.path_root}')
        self.path = self.create_contexts(self.path_root + self.path_suffix)  # TODO: Make this path a function for consistency.

        self.num_classes = None
        self.model = None
        self.problem_type = problem_type

        # whether to calibrate predictions via conformal methods
        self.conformalize = None

        if eval_metric is not None:
            self.eval_metric = metrics.get_metric(eval_metric, self.problem_type, 'eval_metric')  # Note: we require higher values = better performance
        else:
            self.eval_metric = None
        self.normalize_pred_probas = None

        self.features = None  # External features, do not use internally
        self.feature_metadata = None  # External feature metadata, do not use internally
        self._features_internal = None  # Internal features, safe to use internally via the `_features` property
        self._feature_metadata = None  # Internal feature metadata, safe to use internally
        self._is_features_in_same_as_ex = None  # Whether self.features == self._features_internal

        self.fit_time = None  # Time taken to fit in seconds (Training data)
        self.predict_time = None  # Time taken to predict in seconds (Validation data)
        self.predict_1_time = None  # Time taken to predict 1 row of data in seconds (with batch size `predict_1_batch_size` in params_aux)
        self.val_score = None  # Score with eval_metric (Validation data)

        self.params = {}
        self.params_aux = {}

        if hyperparameters is not None:
            hyperparameters = hyperparameters.copy()
        if hyperparameters is not None and AG_ARGS_FIT in hyperparameters:
            self._user_params_aux = hyperparameters.pop(AG_ARGS_FIT)  # TODO: Delete after initialization?
        else:
            self._user_params_aux = None
        if self._user_params_aux is None:
            self._user_params_aux = dict()
        self._user_params = hyperparameters  # TODO: Delete after initialization?
        if self._user_params is None:
            self._user_params = dict()

        self.params_trained = dict()
        self._is_initialized = False
        self._is_fit_metadata_registered = False
        self._fit_metadata = dict()

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
    def path_suffix(self):
        return self.name + os.path.sep

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
        See `autogluon.core.space` for available space classes.
        Returns
        -------
        dict of hyperparameter search spaces.
        """
        return {}

    def _get_search_space(self):
        """ Sets up default search space for HPO. Each hyperparameter which user did not specify is converted from
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
    def create_contexts(path_context):
        path = path_context
        return path

    def rename(self, name: str):
        """Renames the model and updates self.path to reflect the updated name."""
        self.path = self.path[:-len(self.name) - 1] + name + os.path.sep
        self.name = name

    def preprocess(self, X, preprocess_nonadaptive=True, preprocess_stateful=True, **kwargs):
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
        get_features_kwargs = self.params_aux.get('get_features_kwargs', None)
        if get_features_kwargs is not None:
            valid_features = feature_metadata.get_features(**get_features_kwargs)
        else:
            valid_raw_types = self.params_aux.get('valid_raw_types', None)
            valid_special_types = self.params_aux.get('valid_special_types', None)
            ignored_type_group_raw = self.params_aux.get('ignored_type_group_raw', None)
            ignored_type_group_special = self.params_aux.get('ignored_type_group_special', None)
            valid_features = feature_metadata.get_features(
                valid_raw_types=valid_raw_types,
                valid_special_types=valid_special_types,
                invalid_raw_types=ignored_type_group_raw,
                invalid_special_types=ignored_type_group_special
            )
        get_features_kwargs_extra = self.params_aux.get('get_features_kwargs_extra', None)
        if get_features_kwargs_extra is not None:
            valid_features_extra = feature_metadata.get_features(**get_features_kwargs_extra)
            valid_features = [feature for feature in valid_features if feature in valid_features_extra]
        dropped_features = [feature for feature in self.features if feature not in valid_features]
        logger.log(10, f'\tDropped {len(dropped_features)} of {len(self.features)} features.')
        self.features = [feature for feature in self.features if feature in valid_features]
        self.feature_metadata = feature_metadata.keep_features(self.features)
        error_if_no_features = self.params_aux.get('error_if_no_features', True)
        if error_if_no_features and not self.features:
            raise NoValidFeatures
        # TODO: If unique_counts == 2 (including NaN), then treat as boolean
        if self.params_aux.get('drop_unique', True):
            # TODO: Could this be optimized to be faster? This might be a bit slow for large data.
            unique_counts = X[self.features].nunique(axis=0, dropna=False)
            columns_to_drop = list(unique_counts[unique_counts < 2].index)
            features_to_drop_internal = columns_to_drop
            if not features_to_drop_internal:
                features_to_drop_internal = None
        else:
            features_to_drop_internal = None
        if features_to_drop_internal is not None:
            logger.log(10, f'\tDropped {len(features_to_drop_internal)} of {len(self.features)} internal features: {features_to_drop_internal}')
            self._features_internal = [feature for feature in self.features if feature not in features_to_drop_internal]
            self._feature_metadata = self.feature_metadata.keep_features(self._features_internal)
            self._is_features_in_same_as_ex = False
        else:
            self._features_internal = self.features
            self._feature_metadata = self.feature_metadata
            self._is_features_in_same_as_ex = True
        if error_if_no_features and not self._features_internal:
            raise NoValidFeatures

    def _preprocess_fit_args(self, **kwargs):
        sample_weight = kwargs.get('sample_weight', None)
        if sample_weight is not None and isinstance(sample_weight, str):
            raise ValueError("In model.fit(), sample_weight should be array of sample weight values, not string.")
        time_limit = kwargs.get('time_limit', None)
        max_time_limit_ratio = self.params_aux.get('max_time_limit_ratio', 1)
        if time_limit is not None:
            time_limit *= max_time_limit_ratio
        max_time_limit = self.params_aux.get('max_time_limit', None)
        if max_time_limit is not None:
            if time_limit is None:
                time_limit = max_time_limit
            else:
                time_limit = min(time_limit, max_time_limit)
        min_time_limit = self.params_aux.get('min_time_limit', 0)
        if min_time_limit is None:
            time_limit = min_time_limit
        elif time_limit is not None:
            time_limit = max(time_limit, min_time_limit)
        kwargs['time_limit'] = time_limit
        kwargs = self._preprocess_fit_resources(**kwargs)
        return kwargs

    def initialize(self, **kwargs):
        if not self._is_initialized:
            self._initialize(**kwargs)
            self._is_initialized = True

        kwargs.pop('feature_metadata', None)
        kwargs.pop('num_classes', None)
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

        self._init_misc(
            X=X,
            y=y,
            feature_metadata=feature_metadata,
            num_classes=num_classes,
            **kwargs
        )

        if X is not None:
            self._preprocess_set_features(X=X, feature_metadata=feature_metadata)

        self._init_params()

    def _init_misc(self, **kwargs):
        """Initialize parameters that depend on self.params_aux being initialized"""
        if self.eval_metric is None:
            self.eval_metric = infer_eval_metric(problem_type=self.problem_type)
            logger.log(20, f"Model {self.name}'s eval_metric inferred to be '{self.eval_metric.name}' because problem_type='{self.problem_type}' and eval_metric was not specified during init.")
        self.eval_metric = metrics.get_metric(self.eval_metric, self.problem_type, 'eval_metric')  # Note: we require higher values = better performance

        self.stopping_metric = self.params_aux.get('stopping_metric', self._get_default_stopping_metric())
        self.stopping_metric = metrics.get_metric(self.stopping_metric, self.problem_type, 'stopping_metric')

        self.quantile_levels = self.params_aux.get('quantile_levels', None)

        if self.eval_metric.name in OBJECTIVES_TO_NORMALIZE:
            self.normalize_pred_probas = True
            logger.debug(f"{self.name} predicted probabilities will be transformed to never =0 since eval_metric='{self.eval_metric.name}'")
        else:
            self.normalize_pred_probas = False

    def _preprocess_fit_resources(self, silent=False, **kwargs):
        default_num_cpus, default_num_gpus = self._get_default_resources()
        num_cpus = self.params_aux.get('num_cpus', 'auto')
        num_gpus = self.params_aux.get('num_gpus', 'auto')
        kwargs['num_cpus'] = kwargs.get('num_cpus', num_cpus)
        kwargs['num_gpus'] = kwargs.get('num_gpus', num_gpus)
        if kwargs['num_cpus'] == 'auto':
            kwargs['num_cpus'] = default_num_cpus
        if kwargs['num_gpus'] == 'auto':
            kwargs['num_gpus'] = default_num_gpus
        if not silent:
            logger.log(15, f"\tFitting {self.name} with 'num_gpus': {kwargs['num_gpus']}, 'num_cpus': {kwargs['num_cpus']}")
        return kwargs

    def _register_fit_metadata(self, **kwargs):
        """
        Used to track properties of the inputs received during fit, such as if validation data was present.
        """
        if not self._is_fit_metadata_registered:
            self._fit_metadata = self._compute_fit_metadata(**kwargs)
            self._is_fit_metadata_registered = True

    def _compute_fit_metadata(self, X_val=None, X_unlabeled=None, **kwargs):
        fit_metadata = dict(
            val_in_fit=X_val is not None,
            unlabeled_in_fit=X_unlabeled is not None
        )
        return fit_metadata

    def get_fit_metadata(self) -> dict:
        """
        Returns dictionary of metadata related to model fit that isn't related to hyperparameters.
        Must be called after model has been fit.
        """
        assert self._is_fit_metadata_registered, 'fit_metadata must be registered before calling get_fit_metadata()!'
        fit_metadata = dict()
        fit_metadata.update(self._fit_metadata)
        fit_metadata['predict_1_batch_size'] = self._get_child_aux_val(key='predict_1_batch_size', default=None)
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
        kwargs = self.initialize(**kwargs)  # FIXME: This might have to go before self._preprocess_fit_args, but then time_limit might be incorrect in **kwargs init to initialize
        kwargs = self._preprocess_fit_args(**kwargs)
        if 'time_limit' in kwargs and kwargs['time_limit'] is not None and kwargs['time_limit'] <= 0:
            logger.warning(f'\tWarning: Model has no time left to train, skipping model... (Time Left = {round(kwargs["time_limit"], 1)}s)')
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
        predict_1_batch_size = self.params_aux.get('predict_1_batch_size', None)
        if self.predict_1_time is None and predict_1_batch_size is not None and 'X' in kwargs and kwargs['X'] is not None:
            X_1 = sample_df_for_time_func(df=kwargs['X'], sample_size=predict_1_batch_size)
            self.predict_1_time = time_func(f=self.predict, args=[X_1]) / len(X_1)
        return self

    def get_features(self):
        assert self.is_fit(), "The model must be fit before calling the get_features method."
        if self.feature_metadata:
            return self.feature_metadata.get_features()
        else:
            return self.features

    def _fit(self,
             X,
             y,
             X_val=None,
             y_val=None,
             X_unlabeled=None,
             time_limit=None,
             sample_weight=None,
             sample_weight_val=None,
             num_cpus=None,
             num_gpus=None,
             verbosity=2,
             **kwargs):
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

    def _apply_temperature_scaling(self, y_pred_proba):
        # TODO: This is expensive to convert at inference time, try to avoid in future
        if self.problem_type == BINARY:
            y_pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_pred_proba)

        logits = np.log(y_pred_proba)
        y_pred_proba = scipy.special.softmax(logits / self.params_aux.get("temperature_scalar"), axis=1)
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

        if self.problem_type == BINARY:
            y_pred_proba = y_pred_proba[:, 1]

        return y_pred_proba

    def _apply_conformalization(self, y_pred):
        """
        Return conformalized quantile predictions
        This is applicable only to quantile regression problems,
        and the given predictions (y_pred) are adjusted by adding quantile-level constants.
        """
        y_pred += self.conformalize
        return y_pred

    def predict(self, X, **kwargs):
        """
        Returns class predictions of X.
        For binary and multiclass problems, this returns the predicted class labels as a Series.
        For regression problems, this returns the predicted values as a Series.
        """
        y_pred_proba = self.predict_proba(X, **kwargs)
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        return y_pred

    def predict_proba(self, X, normalize=None, **kwargs):
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

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)

        if self.problem_type in [REGRESSION, QUANTILE]:
            y_pred = self.model.predict(X)
            return y_pred

        y_pred_proba = self.model.predict_proba(X)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _convert_proba_to_unified_form(self, y_pred_proba):
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

    def score(self, X, y, metric=None, sample_weight=None, **kwargs):
        if metric is None:
            metric = self.eval_metric

        if metric.needs_pred or metric.needs_quantile:
            y_pred = self.predict(X=X, **kwargs)
        else:
            y_pred = self.predict_proba(X=X, **kwargs)
        return compute_weighted_metric(y, y_pred, metric, sample_weight, quantile_levels=self.quantile_levels)

    def score_with_y_pred_proba(self, y, y_pred_proba, metric=None, sample_weight=None):
        if metric is None:
            metric = self.eval_metric
        if metric.needs_pred:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        else:
            y_pred = y_pred_proba
        return compute_weighted_metric(y, y_pred, metric, sample_weight, quantile_levels=self.quantile_levels)

    def save(self, path: str = None, verbose=True) -> str:
        """
        Saves the model to disk.

        Parameters
        ----------
        path : str, default None
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            If None, self.path is used.
            The final model file is typically saved to path + self.model_file_name.
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
        file_path = path + self.model_file_name
        save_pkl.save(path=file_path, object=self, verbose=verbose)
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        """
        Loads the model from disk to memory.

        Parameters
        ----------
        path : str
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            The model file is typically located in path + cls.model_file_name.
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
        file_path = path + cls.model_file_name
        model = load_pkl.load(path=file_path, verbose=verbose)
        if reset_paths:
            model.set_contexts(path)
        return model

    def compute_feature_importance(self,
                                   X,
                                   y,
                                   features=None,
                                   silent=False,
                                   importance_as_list=False,
                                   **kwargs) -> pd.DataFrame:
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
            n = fi_df.iloc[0]['n'] if len(fi_df) > 0 else 1
        else:
            fi_df = None
            n = kwargs.get('num_shuffle_sets', 1)

        if importance_as_list:
            banned_importance = [0] * n
            results_banned = pd.Series(data=[banned_importance for _ in range(len(banned_features))], index=banned_features, dtype='object')
        else:
            banned_importance = 0
            results_banned = pd.Series(data=[banned_importance for _ in range(len(banned_features))], index=banned_features, dtype='float64')

        results_banned_df = results_banned.to_frame(name='importance')
        results_banned_df['stddev'] = 0
        results_banned_df['n'] = n
        results_banned_df['n'] = results_banned_df['n'].astype('int64')
        if fi_df is not None:
            fi_df = pd.concat([fi_df, results_banned_df])
        else:
            fi_df = results_banned_df
        fi_df = fi_df.sort_values(ascending=False, by='importance')

        return fi_df

    # Compute feature importance via permutation importance
    # Note: Expensive to compute
    #  Time to compute is O(predict_time*num_features)
    def _compute_permutation_importance(self, X, y, features: list, eval_metric=None, silent=False, **kwargs) -> pd.DataFrame:
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
            X=X, y=y, features=features, eval_metric=self.eval_metric, predict_func=predict_func, predict_func_kwargs=predict_func_kwargs,
            transform_func=transform_func, transform_func_kwargs=transform_func_kwargs, silent=silent, **kwargs
        )

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
        """After calling this function, returned model should be able to be fit without X_val, y_val using the iterations trained by the original model."""
        params = copy.deepcopy(self.get_params())
        params['hyperparameters'].update(self.params_trained)
        params['name'] = params['name'] + REFIT_FULL_SUFFIX
        template = self.__class__(**params)

        return template

    def hyperparameter_tune(self,
                            hyperparameter_tune_kwargs='auto',
                            hpo_executor: HpoExecutor = None,
                            time_limit: float = None,
                            **kwargs):
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
            In case of Ray Tune backend, this will be an Analysis object: https://docs.ray.io/en/latest/tune/api_docs/analysis.html
        """
        # if hpo_executor is not None, ensemble has already created the hpo_executor
        if hpo_executor is None:
            hpo_executor = self._get_default_hpo_executor()
            default_num_trials = kwargs.pop('default_num_trials', None)
            hpo_executor.initialize(hyperparameter_tune_kwargs, default_num_trials=default_num_trials, time_limit=time_limit)
        kwargs = self.initialize(time_limit=time_limit, **kwargs)
        self._register_fit_metadata(**kwargs)
        self._validate_fit_memory_usage(**kwargs)
        hpo_executor.register_resources(self)
        return self._hyperparameter_tune(hpo_executor=hpo_executor, **kwargs)

    def _hyperparameter_tune(self, X, y, X_val, y_val, hpo_executor, **kwargs):
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

        # Use absolute path here because ray tune will change the working directory
        self.set_contexts(os.path.abspath(self.path) + os.path.sep)
        directory = self.path  # also create model directory if it doesn't exist
        # TODO: This will break on S3. Use tabular/utils/savers for datasets, add new function
        dataset_train_filename = 'dataset_train.pkl'
        train_path = os.path.join(directory, dataset_train_filename)
        save_pkl.save(path=train_path, object=(X, y))

        dataset_val_filename = 'dataset_val.pkl'
        val_path = os.path.join(directory, dataset_val_filename)
        save_pkl.save(path=val_path, object=(X_val, y_val))

        model_cls = self.__class__
        init_params = self.get_params()
        # We set soft time limit to avoid trials being terminated directly by ray tune
        trial_soft_time_limit = max(hpo_executor.time_limit * 0.9, hpo_executor.time_limit - 5)  # 5 seconds max for buffer

        fit_kwargs = dict()
        fit_kwargs['feature_metadata'] = self.feature_metadata
        fit_kwargs['num_classes'] = self.num_classes
        fit_kwargs['sample_weight'] = kwargs.get('sample_weight', None)
        fit_kwargs['sample_weight_val'] = kwargs.get('sample_weight_val', None)
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
        minimum_resources = self.get_minimum_resources()
        hpo_executor.execute(
            model_trial=model_trial,
            train_fn_kwargs=train_fn_kwargs,
            directory=directory,
            minimum_cpu_per_trial=minimum_resources.get('num_cpus', 1),
            minimum_gpu_per_trial=minimum_resources.get('num_gpus', 0),
            model_estimate_memory_usage=model_estimate_memory_usage,
            adapter_type='tabular',
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
    
    def _get_hpo_backend(self):
        """Choose which backend(Ray or Custom) to use for hpo"""
        return CUSTOM_BACKEND

    def _get_default_hpo_executor(self) -> HpoExecutor:
        backend = self._get_model_base()._get_hpo_backend()  # If ensemble, will use the base model to determine backend
        if backend == RAY_BACKEND:
            try:
                try_import_ray()
            except Exception as e:
                warning_msg = f'Will use custom hpo logic because ray import failed. Reason: {str(e)}'
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
        model_disk_size = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file())
        return model_disk_size

    # TODO: This results in a doubling of memory usage of the model to calculate its size.
    #  If the model takes ~40%+ of memory, this may result in an OOM error.
    #  This is generally not an issue because the model already needed to do this when being saved to disk, so the error would have been triggered earlier.
    #  Consider using Pympler package for memory efficiency: https://pympler.readthedocs.io/en/latest/asizeof.html#asizeof
    def get_memory_size(self) -> int:
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

    def validate_fit_resources(self, num_cpus='auto', num_gpus='auto', **kwargs):
        """
        Verifies that the provided num_cpus and num_gpus (or defaults if not provided) are sufficient to train the model.
        Raises an AssertionError if not sufficient.
        """
        resources = self._preprocess_fit_resources(num_cpus=num_cpus, num_gpus=num_gpus, silent=True)
        self._validate_fit_resources(**resources)

    def _validate_fit_resources(self, **resources):
        res_min = self.get_minimum_resources()
        for resource_name in res_min:
            if resource_name not in resources:
                raise AssertionError(f'Model requires {res_min[resource_name]} {resource_name} to fit, but no available amount was defined.')
            elif res_min[resource_name] > resources[resource_name]:
                raise AssertionError(f'Model requires {res_min[resource_name]} {resource_name} to fit, but {resources[resource_name]} are available.')

    def get_minimum_resources(self) -> Dict[str, int]:
        """
        Returns a dictionary of minimum resource requirements to fit the model.
        Subclass should consider overriding this method if it requires more resources to train.
        If a resource is not part of the output dictionary, it is considered unnecessary.
        Valid keys: 'num_cpus', 'num_gpus'.
        """
        return {
            'num_cpus': 1,
        }

    def _estimate_memory_usage(self, X, **kwargs) -> int:
        """
        This method simply provides a default implementation. Each model should consider implementing custom memory estimate logic.
        """
        return 4 * get_approximate_df_mem_usage(X).sum()

    @disable_if_lite_mode()
    def _validate_fit_memory_usage(self, **kwargs):
        import psutil
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        approx_mem_size_req = self.estimate_memory_usage(**kwargs)
        available_mem = psutil.virtual_memory().available
        ratio = approx_mem_size_req / available_mem
        if ratio > (0.9 * max_memory_usage_ratio):
            logger.warning('\tWarning: Not enough memory to safely train model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))
            raise NotEnoughMemoryError
        elif ratio > (0.6 * max_memory_usage_ratio):
            logger.warning('\tWarning: Potentially not enough memory to safely train model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))

    # Removes non-essential objects from the model to reduce memory and disk footprint.
    # If `remove_fit=True`, enables the removal of variables which are required for fitting the model. If the model is already fully trained, then it is safe to remove these.
    # If `remove_info=True`, enables the removal of variables which are used during model.get_info(). The values will be None when calling model.get_info().
    # If `requires_save=True`, enables the removal of variables which are part of the model.pkl object, requiring an overwrite of the model to disk if it was previously persisted.
    def reduce_memory_size(self, remove_fit=True, remove_info=False, requires_save=True, **kwargs):
        """
        Removes non-essential objects from the model to reduce memory and disk footprint.
        If `remove_fit=True`, enables the removal of variables which are required for fitting the model. If the model is already fully trained, then it is safe to remove these.
        If `remove_info=True`, enables the removal of variables which are used during model.get_info(). The values will be None when calling model.get_info().
        If `requires_save=True`, enables the removal of variables which are part of the model.pkl object, requiring an overwrite of the model to disk if it was previously persisted.

        It is not necessary for models to implement this.
        """
        pass

    def delete_from_disk(self, silent=False):
        """
        Deletes the model from disk.

        WARNING: This will DELETE ALL FILES in the self.path directory, regardless if they were created by AutoGluon or not.
        DO NOT STORE FILES INSIDE OF THE MODEL DIRECTORY THAT ARE UNRELATED TO AUTOGLUON.
        """
        if not silent:
            logger.log(30, f'Deleting model {self.name}. All files under {self.path} will be removed.')
        from pathlib import Path
        import shutil
        model_path = Path(self.path)
        # TODO: Report errors?
        shutil.rmtree(path=model_path, ignore_errors=True)

    def get_info(self) -> dict:
        """
        Returns a dictionary of numerous fields describing the model.
        """
        info = {
            'name': self.name,
            'model_type': type(self).__name__,
            'problem_type': self.problem_type,
            'eval_metric': self.eval_metric.name,
            'stopping_metric': self.stopping_metric.name,
            'fit_time': self.fit_time,
            'num_classes': self.num_classes,
            'quantile_levels': self.quantile_levels,
            'predict_time': self.predict_time,
            'val_score': self.val_score,
            'hyperparameters': self.params,
            'hyperparameters_fit': self.params_trained,  # TODO: Explain in docs that this is for hyperparameters that differ in final model from original hyperparameters, such as epochs (from early stopping)
            'hyperparameters_nondefault': self.nondefault_params,
            AG_ARGS_FIT: self.params_aux,
            'num_features': len(self.features) if self.features else None,
            'features': self.features,
            'feature_metadata': self.feature_metadata,
            # 'disk_size': self.get_disk_size(),
            'memory_size': self.get_memory_size(),  # Memory usage of model in bytes
        }
        return info

    @classmethod
    def load_info(cls, path, load_model_if_required=True) -> dict:
        load_path = path + cls.model_info_name
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

        save_pkl.save(path=self.path + self.model_info_name, object=info)
        json_path = self.path + self.model_info_json_name
        save_json.save(path=json_path, obj=info)
        return info

    def _get_default_resources(self):
        """
        Determines the default resource usage of the model during fit.

        Models may want to override this if they depend heavily on GPUs, as the default sets num_gpus to 0.
        """
        num_cpus = get_cpu_count()
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

    def _get_default_stopping_metric(self):
        """
        Returns the default stopping metric to use for early stopping.
        This is used if stopping_metric was not explicitly specified.
        Models may wish to override this in case a more suitable stopping_metric exists for a given eval_metric.
        """
        if self.eval_metric.name == 'roc_auc':
            stopping_metric = 'log_loss'
        else:
            stopping_metric = self.eval_metric
        stopping_metric = metrics.get_metric(stopping_metric, self.problem_type, 'stopping_metric')
        return stopping_metric

    def _get_params(self) -> dict:
        """Gets all params."""
        return self.params.copy()

    def _get_ag_params(self) -> dict:
        """Gets params that are not passed to the inner model, but are used by the wrapper."""
        ag_param_names = self._ag_params()
        if ag_param_names:
            return {key: val for key, val in self.params.items() if key in ag_param_names}
        else:
            return dict()

    def _get_model_params(self) -> dict:
        """Gets params that are passed to the inner model."""
        ag_param_names = self._ag_params()
        if ag_param_names:
            return {key: val for key, val in self.params.items() if key not in ag_param_names}
        else:
            return self._get_params()

    # TODO: Add documentation for valid args for each model. Currently only `ag.early_stop`
    def _ag_params(self) -> set:
        """
        Set of params that are not passed to self.model, but are used by the wrapper.
        For developers, this is purely optional and is just for convenience to logically distinguish between model specific parameters and added AutoGluon functionality.
        The goal is to have common parameter names for useful functionality shared between models,
        even if the functionality is not natively available as a parameter in the model itself or under a different name.

        Below are common patterns / options to make available. Their actual usage and options in a particular model should be documented in the model itself, as it has flexibility to differ.

        Possible params:

        ag.early_stop : int, str, or tuple
            generic name for early stopping logic. Typically can be an int or a str preset/strategy.
            Also possible to pass tuple of (class, kwargs) to construct a custom early stopping object.
                Refer to `autogluon.core.utils.early_stopping` for examples.

        """
        return set()

    @property
    def _features(self):
        return self._features_internal

    def _get_tags(self):
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, '_more_tags'):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags

    def _more_tags(self):
        return _DEFAULT_TAGS
    
    def _get_model_base(self):
        return self
