import logging
import os
import time
import psutil
import numpy as np

from autogluon.common.features.types import R_OBJECT
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.constants import PROBLEM_TYPES_CLASSIFICATION, MULTICLASS, SOFTCLASS
from autogluon.core.models import AbstractModel
from autogluon.core.models._utils import get_early_stopping_rounds
from autogluon.core.utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded
from autogluon.core.utils import try_import_catboost

from .callbacks import EarlyStoppingCallback, MemoryCheckCallback, TimeCheckCallback
from .catboost_utils import construct_custom_catboost_metric
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace

logger = logging.getLogger(__name__)


# TODO: Consider having CatBoost variant that converts all categoricals to numerical as done in RFModel, was showing improved results in some problems.
class CatBoostModel(AbstractModel):
    """
    CatBoost model: https://catboost.ai/

    Hyperparameter options: https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._category_features = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        self._set_default_param_value('random_seed', 0)  # Remove randomness for reproducibility
        # Set 'allow_writing_files' to True in order to keep log files created by catboost during training (these will be saved in the directory where AutoGluon stores this model)
        self._set_default_param_value('allow_writing_files', False)  # Disables creation of catboost logging files during training by default
        if self.problem_type != SOFTCLASS:  # TODO: remove this after catboost 0.24
            self._set_default_param_value('eval_metric', construct_custom_catboost_metric(self.stopping_metric, True, not self.stopping_metric.needs_pred, self.problem_type))

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type, num_classes=self.num_classes)

    def _preprocess_nonadaptive(self, X, **kwargs):
        X = super()._preprocess_nonadaptive(X, **kwargs)
        if self._category_features is None:
            self._category_features = list(X.select_dtypes(include='category').columns)
        if self._category_features:
            X = X.copy()
            for category in self._category_features:
                current_categories = X[category].cat.categories
                if '__NaN__' in current_categories:
                    X[category] = X[category].fillna('__NaN__')
                else:
                    X[category] = X[category].cat.add_categories('__NaN__').fillna('__NaN__')
        return X

    def _estimate_memory_usage(self, X, **kwargs):
        num_classes = self.num_classes if self.num_classes else 1  # self.num_classes could be None after initalization if it's a regression problem
        data_mem_uasge = get_approximate_df_mem_usage(X).sum()
        approx_mem_size_req = data_mem_uasge * 7 + data_mem_uasge / 4 * num_classes  # TODO: Extremely crude approximation, can be vastly improved
        return approx_mem_size_req

    # TODO: Use Pool in preprocess, optimize bagging to do Pool.split() to avoid re-computing pool for each fold! Requires stateful + y
    #  Pool is much more memory efficient, avoids copying data twice in memory
    def _fit(self,
             X,
             y,
             X_val=None,
             y_val=None,
             time_limit=None,
             num_gpus=0,
             num_cpus=-1,
             sample_weight=None,
             sample_weight_val=None,
             **kwargs):
        time_start = time.time()
        try_import_catboost()
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
        ag_params = self._get_ag_params()
        params = self._get_model_params()
        params['thread_count'] = num_cpus
        if self.problem_type == SOFTCLASS:
            # FIXME: This is extremely slow due to unoptimized metric / objective sent to CatBoost
            from .catboost_softclass_utils import SoftclassCustomMetric, SoftclassObjective
            params['loss_function'] = SoftclassObjective.SoftLogLossObjective()
            params['eval_metric'] = SoftclassCustomMetric.SoftLogLossMetric()

        model_type = CatBoostClassifier if self.problem_type in PROBLEM_TYPES_CLASSIFICATION else CatBoostRegressor
        num_rows_train = len(X)
        num_cols_train = len(X.columns)
        num_classes = self.num_classes if self.num_classes else 1  # self.num_classes could be None after initalization if it's a regression problem

        X = self.preprocess(X)
        cat_features = list(X.select_dtypes(include='category').columns)
        X = Pool(data=X, label=y, cat_features=cat_features, weight=sample_weight)

        if X_val is None:
            eval_set = None
            early_stopping_rounds = None
        else:
            X_val = self.preprocess(X_val)
            X_val = Pool(data=X_val, label=y_val, cat_features=cat_features, weight=sample_weight_val)
            eval_set = X_val
            early_stopping_rounds = ag_params.get('ag.early_stop', 'adaptive')
            if isinstance(early_stopping_rounds, (str, tuple, list)):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)

        if params.get('allow_writing_files', False):
            if 'train_dir' not in params:
                try:
                    # TODO: What if path is in S3?
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                except:
                    pass
                else:
                    params['train_dir'] = self.path + 'catboost_info'

        # TODO: Add more control over these params (specifically early_stopping_rounds)
        verbosity = kwargs.get('verbosity', 2)
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
            if 'task_type' not in params:
                params['task_type'] = 'GPU'
                logger.log(20, f'\tTraining {self.name} with GPU, note that this may negatively impact model quality compared to CPU training.')
                # TODO: Confirm if GPU is used in HPO (Probably not)
                # TODO: Adjust max_bins to 254?

        if params.get('task_type', None) == 'GPU':
            if 'colsample_bylevel' in params:
                params.pop('colsample_bylevel')
                logger.log(30, f'\t\'colsample_bylevel\' is not supported on GPU, using default value (Default = 1).')
            if 'rsm' in params:
                params.pop('rsm')
                logger.log(30, f'\t\'rsm\' is not supported on GPU, using default value (Default = 1).')

        if self.problem_type == MULTICLASS and 'rsm' not in params and 'colsample_bylevel' not in params and num_features > 1000:
            # Subsample columns to speed up training
            if params.get('task_type', None) != 'GPU':  # RSM does not work on GPU
                params['colsample_bylevel'] = max(min(1.0, 1000 / num_features), 0.05)
                logger.log(30, f'\tMany features detected ({num_features}), dynamically setting \'colsample_bylevel\' to {params["colsample_bylevel"]} to speed up training (Default = 1).')
                logger.log(30, f'\tTo disable this functionality, explicitly specify \'colsample_bylevel\' in the model hyperparameters.')
            else:
                params['colsample_bylevel'] = 1.0
                logger.log(30, f'\t\'colsample_bylevel\' is not supported on GPU, using default value (Default = 1).')

        logger.log(15, f'\tCatboost model hyperparameters: {params}')

        self.model = model_type(**params)

        callbacks = []
        if early_stopping_rounds is not None:
            callbacks.append(EarlyStoppingCallback(stopping_rounds=early_stopping_rounds, eval_metric=params['eval_metric']))

        if num_rows_train * num_cols_train * num_classes > 5_000_000:
            # The data is large enough to potentially cause memory issues during training, so monitor memory usage via callback.
            callbacks.append(MemoryCheckCallback())
        if time_limit is not None:
            time_cur = time.time()
            time_left = time_limit - (time_cur - time_start)
            if time_left <= time_limit * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                raise TimeLimitExceeded
            callbacks.append(TimeCheckCallback(time_start=time_cur, time_limit=time_left))

        # TODO: Custom metrics don't seem to work anymore
        # TODO: Custom metrics not supported in GPU mode
        # TODO: Callbacks not supported in GPU mode
        fit_final_kwargs = dict(
            eval_set=eval_set,
            verbose=verbose,
            use_best_model=True,
            # early_stopping_rounds=early_stopping_rounds,  # Disabled in favor of ES callback
            callbacks=callbacks,
        )
        self.model.fit(X, **fit_final_kwargs)

        self.params_trained['iterations'] = self.model.tree_count_

    def _predict_proba(self, X, **kwargs):
        if self.problem_type != SOFTCLASS:
            return super()._predict_proba(X, **kwargs)
        # For SOFTCLASS problems, manually transform predictions into probabilities via softmax
        X = self.preprocess(X, **kwargs)
        y_pred_proba = self.model.predict(X, prediction_type='RawFormulaVal')
        y_pred_proba = np.exp(y_pred_proba)
        y_pred_proba = np.multiply(y_pred_proba, 1/np.sum(y_pred_proba, axis=1)[:, np.newaxis])
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:,1]
        return y_pred_proba

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_OBJECT],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_early_stopping_rounds(self, num_rows_train, strategy='auto'):
        return get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=strategy)

    def _ag_params(self) -> set:
        return {'ag.early_stop'}

    def _validate_fit_memory_usage(self, **kwargs):
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        approx_mem_size_req = self.estimate_memory_usage(**kwargs)
        if approx_mem_size_req > 1e9:  # > 1 GB
            available_mem = psutil.virtual_memory().available
            ratio = approx_mem_size_req / available_mem
            if ratio > (1 * max_memory_usage_ratio):
                logger.warning('\tWarning: Not enough memory to safely train CatBoost model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))
                raise NotEnoughMemoryError
            elif ratio > (0.2 * max_memory_usage_ratio):
                logger.warning('\tWarning: Potentially not enough memory to safely train CatBoost model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))

    def _get_default_resources(self):
        # psutil.cpu_count(logical=False) is faster in training than psutil.cpu_count()
        num_cpus = psutil.cpu_count(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus
