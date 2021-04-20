import os
import time
import logging

from autogluon.core.constants import REGRESSION, PROBLEM_TYPES_CLASSIFICATION
from autogluon.core.features.types import R_OBJECT
from autogluon.core.models import AbstractModel
from autogluon.core.models._utils import get_early_stopping_rounds
from autogluon.core.utils import try_import_xgboost

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
        self._ohe_generator = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type, num_classes=self.num_classes)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return get_default_searchspace(problem_type=self.problem_type, num_classes=self.num_classes)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_OBJECT],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    # Use specialized XGBoost metric if available (fast), otherwise use custom func generator
    def get_eval_metric(self):
        eval_metric = xgboost_utils.convert_ag_metric_to_xgbm(ag_metric_name=self.stopping_metric.name, problem_type=self.problem_type)
        if eval_metric is None:
            eval_metric = xgboost_utils.func_generator(metric=self.stopping_metric, is_higher_better=True, needs_pred_proba=not self.stopping_metric.needs_pred, problem_type=self.problem_type)
        return eval_metric

    def _preprocess(self, X, is_train=False, max_category_levels=None, **kwargs):
        X = super()._preprocess(X=X, **kwargs)

        if self._ohe_generator is None:
            self._ohe_generator = xgboost_utils.OheFeatureGenerator(max_levels=max_category_levels)

        if is_train:
            self._ohe_generator.fit(X)

        X = self._ohe_generator.transform(X)

        return X

    def _fit(self,
             X,
             y,
             X_val=None,
             y_val=None,
             time_limit=None,
             num_gpus=0,
             sample_weight=None,
             sample_weight_val=None,
             verbosity=2,
             **kwargs):
        # TODO: utilize sample_weight_val in early-stopping if provided
        start_time = time.time()
        ag_params = self._get_ag_params()
        params = self._get_model_params()
        max_category_levels = params.pop('proc.max_category_levels', 100)

        if verbosity <= 2:
            verbose = False
            verbose_eval = None
        elif verbosity == 3:
            verbose = True
            verbose_eval = 50
        else:
            verbose = True
            verbose_eval = 1

        X = self.preprocess(X, is_train=True, max_category_levels=max_category_levels)
        num_rows_train = X.shape[0]

        eval_set = []
        eval_metric = self.get_eval_metric()

        if X_val is None:
            early_stopping_rounds = None
            eval_set.append((X, y))  # TODO: if the train dataset is large, use sample of train dataset for validation
        else:
            X_val = self.preprocess(X_val, is_train=False)
            eval_set.append((X_val, y_val))
            early_stopping_rounds = ag_params.get('ag.early_stop', 'auto')
            if isinstance(early_stopping_rounds, str):
                early_stopping_rounds = self._get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=early_stopping_rounds)

        if num_gpus != 0:
            params['tree_method'] = 'gpu_hist'
            if 'gpu_id' not in params:
                params['gpu_id'] = 0

        try_import_xgboost()
        from .callbacks import EarlyStoppingCustom
        from xgboost.callback import EvaluationMonitor
        callbacks = []
        if verbose_eval is not None:
            callbacks.append(EvaluationMonitor(period=verbose_eval))
        # TODO: disable early stopping during refit_full
        callbacks.append(EarlyStoppingCustom(early_stopping_rounds, start_time=start_time, time_limit=time_limit, verbose=verbose))

        from xgboost import XGBClassifier, XGBRegressor
        model_type = XGBClassifier if self.problem_type in PROBLEM_TYPES_CLASSIFICATION else XGBRegressor
        self.model = model_type(**params)
        self.model.fit(
            X=X,
            y=y,
            eval_set=eval_set,
            eval_metric=eval_metric,
            verbose=False,
            callbacks=callbacks,
            sample_weight=sample_weight
        )

        bst = self.model.get_booster()
        # TODO: Investigate speed-ups from GPU inference
        # bst.set_param({"predictor": "gpu_predictor"})

        self.params_trained['n_estimators'] = bst.best_ntree_limit
        self._best_ntree_limit = bst.best_ntree_limit

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)

        if self.problem_type == REGRESSION:
            return self.model.predict(X, ntree_limit=self._best_ntree_limit)

        y_pred_proba = self.model.predict_proba(X, ntree_limit=self._best_ntree_limit)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _get_early_stopping_rounds(self, num_rows_train, strategy='auto'):
        return get_early_stopping_rounds(num_rows_train=num_rows_train, strategy=strategy)

    def _ag_params(self) -> set:
        return {'ag.early_stop'}
