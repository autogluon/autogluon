import logging
import time

import numpy as np
import psutil
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from .knn_utils import FAISSNeighborsClassifier, FAISSNeighborsRegressor

from ..abstract import model_trial
from ..abstract.abstract_model import AbstractModel
from ...constants import REGRESSION
from ....utils.exceptions import NotEnoughMemoryError

logger = logging.getLogger(__name__)


# TODO: Normalize data!
class KNNModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_type = self._get_model_type()

    def _get_model_type(self):
        if self.problem_type == REGRESSION:
            return KNeighborsRegressor
        else:
            return KNeighborsClassifier

    def preprocess(self, X):
        X = super().preprocess(X).fillna(0)
        X = X.to_numpy(dtype=np.float32)
        return X

    def _set_default_params(self):
        default_params = {
            'weights': 'uniform',
            'n_jobs': -1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _set_default_auxiliary_params(self):
        default_auxiliary_params = dict(
            ignored_feature_types_special=['text_ngram', 'text_special'],
            ignored_feature_types_raw=['category', 'object'],  # TODO: Eventually use category features
        )
        for key, value in default_auxiliary_params.items():
            self._set_default_param_value(key, value, params=self.params_aux)
        super()._set_default_auxiliary_params()

    # TODO: Enable HPO for KNN
    def _get_default_searchspace(self):
        spaces = {}
        return spaces

    def _fit(self, X_train, y_train, **kwargs):
        X_train = self.preprocess(X_train)
        self._validate_fit_memory_usage(X_train=X_train)
        self.model = self._model_type(**self.params).fit(X_train, y_train)

    def _validate_fit_memory_usage(self, X_train):
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        model_size_bytes = 4 * X_train.shape[0] * X_train.shape[1]  # Assuming float32 types
        expected_final_model_size_bytes = model_size_bytes * 3.6  # Roughly what can be expected of the final KNN model in memory size
        if expected_final_model_size_bytes > 10000000:  # Only worth checking if expected model size is >10MB
            available_mem = psutil.virtual_memory().available
            model_memory_ratio = expected_final_model_size_bytes / available_mem
            if model_memory_ratio > (0.30 * max_memory_usage_ratio):
                logger.warning(f'\tWarning: Model is expected to require {model_memory_ratio * 100} percent of available memory...')
            if model_memory_ratio > (0.40 * max_memory_usage_ratio):
                raise NotEnoughMemoryError  # don't train full model to avoid OOM error

    def hyperparameter_tune(self, X_train, y_train, X_val, y_val, scheduler_options=None, **kwargs):
        fit_model_args = dict(X_train=X_train, y_train=y_train, **kwargs)
        predict_proba_args = dict(X=X_val)
        model_trial.fit_and_save_model(model=self, params=dict(), fit_args=fit_model_args, predict_proba_args=predict_proba_args, y_val=y_val, time_start=time.time(), time_limit=None)
        hpo_results = {'total_time': self.fit_time}
        hpo_model_performances = {self.name: self.val_score}
        hpo_models = {self.name: self.path}
        return hpo_models, hpo_model_performances, hpo_results


class FAISSModel(KNNModel):
    def _get_model_type(self):
        if self.problem_type == REGRESSION:
            return FAISSNeighborsRegressor
        else:
            return FAISSNeighborsClassifier

    def _set_default_params(self):
        default_params = {
            'index_factory_string': 'Flat',
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()
