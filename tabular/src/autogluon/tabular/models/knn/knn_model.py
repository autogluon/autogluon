import logging

import numpy as np
import psutil
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from .knn_utils import FAISSNeighborsClassifier, FAISSNeighborsRegressor

from ..abstract.model_trial import skip_hpo
from ..abstract.abstract_model import AbstractModel
from ...constants import REGRESSION
from autogluon.core.utils.exceptions import NotEnoughMemoryError
from ...features.feature_metadata import R_CATEGORY, R_OBJECT, S_TEXT_NGRAM, S_TEXT_SPECIAL, S_DATETIME_AS_INT

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

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    def _set_default_params(self):
        default_params = {
            'weights': 'uniform',
            'n_jobs': -1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_CATEGORY, R_OBJECT],  # TODO: Eventually use category features
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_SPECIAL, S_DATETIME_AS_INT],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

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
            if model_memory_ratio > (0.15 * max_memory_usage_ratio):
                logger.warning(f'\tWarning: Model is expected to require {round(model_memory_ratio * 100, 2)}% of available memory...')
            if model_memory_ratio > (0.20 * max_memory_usage_ratio):
                raise NotEnoughMemoryError  # don't train full model to avoid OOM error

    # TODO: Add HPO
    def hyperparameter_tune(self, **kwargs):
        return skip_hpo(self, **kwargs)


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
