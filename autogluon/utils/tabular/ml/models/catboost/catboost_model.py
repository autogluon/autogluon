import logging
import math
import os
import pickle
import sys
import time

import psutil

from .catboost_utils import construct_custom_catboost_metric
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from ..abstract.abstract_model import AbstractModel
from ...constants import PROBLEM_TYPES_CLASSIFICATION, MULTICLASS
from ....utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded
from .....try_import import try_import_catboost

logger = logging.getLogger(__name__)


# TODO: Catboost crashes on multiclass problems where only two classes have significant member count.
#  Question: Do we turn these into binary classification and then convert to multiclass output in Learner? This would make the most sense.
# TODO: Consider having Catboost variant that converts all categoricals to numerical as done in RFModel, was showing improved results in some problems.
class CatboostModel(AbstractModel):
    def __init__(self, path: str, name: str, problem_type: str, objective_func, stopping_metric=None, num_classes=None, hyperparameters=None, features=None, debug=0, **kwargs):
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, stopping_metric=stopping_metric, num_classes=num_classes, hyperparameters=hyperparameters, features=features, debug=debug, **kwargs)
        try_import_catboost()
        from catboost import CatBoostClassifier, CatBoostRegressor
        self.model_type = CatBoostClassifier if problem_type in PROBLEM_TYPES_CLASSIFICATION else CatBoostRegressor
        if isinstance(self.params['eval_metric'], str):
            self.metric_name = self.params['eval_metric']
        else:
            self.metric_name = type(self.params['eval_metric']).__name__

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        self._set_default_param_value('random_seed', 0)  # Remove randomness for reproducibility
        self._set_default_param_value('eval_metric', construct_custom_catboost_metric(self.stopping_metric, True, not self.stopping_metric_needs_y_pred, self.problem_type))
        # Set 'allow_writing_files' to True in order to keep log files created by catboost during training (these will be saved in the directory where AutoGluon stores this model)
        self._set_default_param_value('allow_writing_files', False)  # Disables creation of catboost logging files during training by default

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type, num_classes=self.num_classes)

    def preprocess(self, X):
        X = super().preprocess(X)
        categoricals = list(X.select_dtypes(include='category').columns)
        if categoricals:
            X = X.copy()
            for category in categoricals:
                current_categories = X[category].cat.categories
                if '__NaN__' in current_categories:
                    X[category] = X[category].fillna('__NaN__')
                else:
                    X[category] = X[category].cat.add_categories('__NaN__').fillna('__NaN__')
        return X

    # TODO: Use Pool in preprocess, optimize bagging to do Pool.split() to avoid re-computing pool for each fold! Requires stateful + y
    #  Pool is much more memory efficient, avoids copying data twice in memory
    def fit(self, X_train, Y_train, X_test=None, Y_test=None, time_limit=None, **kwargs):
        from catboost import Pool
        num_rows_train = len(X_train)
        num_cols_train = len(X_train.columns)
        if self.problem_type == MULTICLASS:
            if self.num_classes is not None:
                num_classes = self.num_classes
            else:
                num_classes = 10  # Guess if not given, can do better by looking at y_train
        else:
            num_classes = 1

        # TODO: Add ignore_memory_limits param to disable NotEnoughMemoryError Exceptions
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        approx_mem_size_req = num_rows_train * num_cols_train * num_classes / 2  # TODO: Extremely crude approximation, can be vastly improved
        if approx_mem_size_req > 1e9:  # > 1 GB
            available_mem = psutil.virtual_memory().available
            ratio = approx_mem_size_req / available_mem
            if ratio > (1 * max_memory_usage_ratio):
                logger.warning('\tWarning: Not enough memory to safely train CatBoost model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))
                raise NotEnoughMemoryError
            elif ratio > (0.2 * max_memory_usage_ratio):
                logger.warning('\tWarning: Potentially not enough memory to safely train CatBoost model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))

        start_time = time.time()
        X_train = self.preprocess(X_train)
        cat_features = list(X_train.select_dtypes(include='category').columns)
        X_train = Pool(data=X_train, label=Y_train, cat_features=cat_features)

        if X_test is not None:
            X_test = self.preprocess(X_test)
            X_test = Pool(data=X_test, label=Y_test, cat_features=cat_features)
            eval_set = X_test
            if num_rows_train <= 10000:
                modifier = 1
            else:
                modifier = 10000/num_rows_train
            early_stopping_rounds = max(round(modifier*150), 10)
            num_sample_iter_max = max(round(modifier*50), 2)
        else:
            eval_set = None
            early_stopping_rounds = None
            num_sample_iter_max = 50

        invalid_params = ['num_threads', 'num_gpus']
        for invalid in invalid_params:
            if invalid in self.params:
                self.params.pop(invalid)
        train_dir = None
        if 'allow_writing_files' in self.params and self.params['allow_writing_files']:
            if 'train_dir' not in self.params:
                try:
                    # TODO: What if path is in S3?
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                except:
                    pass
                else:
                    train_dir = self.path + 'catboost_info'
        logger.log(15, f'\tCatboost model hyperparameters: {self.params}')

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

        init_model = None
        init_model_tree_count = None
        init_model_best_iteration = None
        init_model_best_score = None

        params = self.params.copy()
        num_features = len(self.features)
        if self.problem_type == MULTICLASS and 'rsm' not in params and 'colsample_bylevel' not in params and num_features > 1000:
            if time_limit:
                # Reduce sample iterations to avoid taking unreasonable amounts of time
                num_sample_iter_max = max(round(num_sample_iter_max/2), 2)
            # Subsample columns to speed up training
            params['colsample_bylevel'] = max(min(1.0, 1000 / num_features), 0.05)
            logger.log(30, f'\tMany features detected ({num_features}), dynamically setting \'colsample_bylevel\' to {params["colsample_bylevel"]} to speed up training (Default = 1).')
            logger.log(30, f'\tTo disable this functionality, explicitly specify \'colsample_bylevel\' in the model hyperparameters.')

        if time_limit:
            time_left_start = time_limit - (time.time() - start_time)
            if time_left_start <= time_limit * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                raise TimeLimitExceeded
            params_init = params.copy()
            num_sample_iter = min(num_sample_iter_max, params_init['iterations'])
            params_init['iterations'] = num_sample_iter
            if train_dir is not None:
                params_init['train_dir'] = train_dir
            self.model = self.model_type(
                **params_init,
            )
            self.model.fit(
                X_train,
                eval_set=eval_set,
                use_best_model=True,
                verbose=verbose,
                # early_stopping_rounds=early_stopping_rounds,
            )

            init_model_tree_count = self.model.tree_count_
            init_model_best_iteration = self.model.get_best_iteration()
            init_model_best_score = self.model.get_best_score()['validation'][self.metric_name]

            time_left_end = time_limit - (time.time() - start_time)
            time_taken_per_iter = (time_left_start - time_left_end) / num_sample_iter
            estimated_iters_in_time = round(time_left_end / time_taken_per_iter)
            init_model = self.model

            params_final = params.copy()

            # TODO: This only handles memory with time_limits specified, but not with time_limits=None, handle when time_limits=None
            available_mem = psutil.virtual_memory().available
            model_size_bytes = sys.getsizeof(pickle.dumps(self.model))

            max_memory_proportion = 0.3 * max_memory_usage_ratio
            mem_usage_per_iter = model_size_bytes / num_sample_iter
            max_memory_iters = math.floor(available_mem * max_memory_proportion / mem_usage_per_iter)

            params_final['iterations'] = min(params['iterations'] - num_sample_iter, estimated_iters_in_time)
            if params_final['iterations'] > max_memory_iters - num_sample_iter:
                if max_memory_iters - num_sample_iter <= 500:
                    logger.warning('\tWarning: CatBoost will be early stopped due to lack of memory, increase memory to enable full quality models, max training iterations changed to %s from %s' % (max_memory_iters - num_sample_iter, params_final['iterations']))
                params_final['iterations'] = max_memory_iters - num_sample_iter
        else:
            params_final = params.copy()

        if train_dir is not None:
            params_final['train_dir'] = train_dir
        if params_final['iterations'] > 0:
            self.model = self.model_type(
                **params_final,
            )

            # TODO: Strangely, this performs different if clone init_model is sent in than if trained for same total number of iterations. May be able to optimize catboost models further with this
            self.model.fit(
                X_train,
                eval_set=eval_set,
                verbose=verbose,
                early_stopping_rounds=early_stopping_rounds,
                # use_best_model=True,
                init_model=init_model,
            )

            if init_model is not None:
                final_model_best_score = self.model.get_best_score()['validation'][self.metric_name]
                if self.stopping_metric._optimum > final_model_best_score:
                    if final_model_best_score > init_model_best_score:
                        best_iteration = init_model_tree_count + self.model.get_best_iteration()
                    else:
                        best_iteration = init_model_best_iteration
                else:
                    if final_model_best_score < init_model_best_score:
                        best_iteration = init_model_tree_count + self.model.get_best_iteration()
                    else:
                        best_iteration = init_model_best_iteration

                self.model.shrink(ntree_start=0, ntree_end=best_iteration+1)

        self.params_trained['iterations'] = self.model.tree_count_

    def get_model_feature_importance(self):
        importance_df = self.model.get_feature_importance(prettified=True)
        importance_df['Importances'] = importance_df['Importances'] / 100
        importance_series = importance_df.set_index('Feature Id')['Importances']
        importance_dict = importance_series.to_dict()
        return importance_dict
