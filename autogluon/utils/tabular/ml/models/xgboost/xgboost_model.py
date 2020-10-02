import os
import time
import logging

from . import xgboost_utils
from .callbacks import early_stop_custom
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from ..abstract.abstract_model import AbstractModel
from ...constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS, PROBLEM_TYPES_CLASSIFICATION

logger = logging.getLogger(__name__)


class XGBoostModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ohe_generator = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type, num_classes=self.num_classes)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return get_default_searchspace(problem_type=self.problem_type, num_classes=self.num_classes)

    # Use specialized XGBoost metric if available (fast), otherwise use custom func generator
    def get_eval_metric(self):
        eval_metric = xgboost_utils.convert_ag_metric_to_xgbm(ag_metric_name=self.stopping_metric.name, problem_type=self.problem_type)
        if eval_metric is None:
            eval_metric = xgboost_utils.func_generator(metric=self.stopping_metric, is_higher_better=True, needs_pred_proba=not self.stopping_metric_needs_y_pred, problem_type=self.problem_type)
        return eval_metric

    def preprocess(self, X, is_train=False):
        X = super().preprocess(X=X)

        if self._ohe_generator is None:
            self._ohe_generator = xgboost_utils.OheFeatureGenerator()

        if is_train:
            self._ohe_generator.fit(X)

        X = self._ohe_generator.transform(X)

        return X

    def _fit(self, X_train, y_train, X_val=None, y_val=None, time_limit=None, **kwargs):
        start_time = time.time()
        params = self.params.copy()
        params['n_jobs'] = -1
        
        verbosity = kwargs.get('verbosity', 2)
        if verbosity <= 2:
            verbose = False
        elif verbosity >= 3:
            verbose = True
        
        X_train = self.preprocess(X_train, is_train=True)
        num_rows_train = X_train.shape[0]

        eval_set = []
        eval_metric = self.get_eval_metric()

        if X_val is None:
            early_stopping_rounds = 150
            eval_set.append((X_train, y_train))  # TODO: if the train dataset is large, use sample of train dataset for validation
        else:
            modifier = 1 if num_rows_train <= 10000 else 10000 / num_rows_train
            early_stopping_rounds = max(round(modifier * 150), 10)
            X_val = self.preprocess(X_val, is_train=False)
            eval_set.append((X_val, y_val))

        from xgboost import XGBClassifier, XGBRegressor
        model_type = XGBClassifier if self.problem_type in PROBLEM_TYPES_CLASSIFICATION else XGBRegressor
        self.model = model_type(**params)
        self.model.fit(
            X=X_train,
            y=y_train,
            eval_set=eval_set,
            eval_metric=eval_metric,
            verbose=verbose,
            callbacks=[early_stop_custom(early_stopping_rounds, start_time=start_time, time_limit=time_limit, verbose=verbose)]
        )

        bst = self.model.get_booster()
        self.params_trained['best_iteration'] = bst.best_iteration
        self.params_trained['best_ntree_limit'] = bst.best_ntree_limit

    def get_model_feature_importance(self):
        original_feature_names: list = self._ohe_generator.get_original_feature_names()
        feature_names = self._ohe_generator.get_feature_names()
        importances = self.model.feature_importances_.tolist()

        importance_dict = {}
        for original_feature in original_feature_names:
            importance_dict[original_feature] = 0
            for feature, value in zip(feature_names, importances):
                if feature in self._ohe_generator.othercols:
                    importance_dict[feature] = value
                else:
                    feature = '_'.join(feature.split('_')[:-1])
                    if feature == original_feature:
                        importance_dict[feature] += value

        return importance_dict
