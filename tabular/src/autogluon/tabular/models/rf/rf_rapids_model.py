import logging
import math
import pickle
import sys
import time

import numpy as np
import psutil

from autogluon.core.constants import REGRESSION, SOFTCLASS

from .rf_model import RFModel

logger = logging.getLogger(__name__)


class RFRapidsModel(RFModel):
    """
    RAPIDS Random Forest model : https://rapids.ai/start.html

    NOTE: This code is experimental, it is recommend to not use this unless you are a developer.
    This was tested on rapids-0.18 via:

    conda create -n rapids-0.18 -c rapidsai -c nvidia -c conda-forge -c defaults rapids-blazing=0.18 python=3.7 cudatoolkit=10.1 -y
    conda activate rapids-0.18
    pip install --pre autogluon.tabular[all]
    """
    def _get_model_type(self):
        from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
        if self.problem_type in [REGRESSION, SOFTCLASS]:
            return RandomForestRegressor
        else:
            return RandomForestClassifier

    def _set_default_params(self):
        default_params = {
            'n_estimators': 300,
            'max_depth': 99,
            'random_state': 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Improve memory safety
    # TODO: Significantly less accurate than RFModel with same hyperparameters.
    #  Refer to https://github.com/rapidsai/cuml/issues/2518
    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X)
        self.model = self._get_model_type()(**self.params)
        self.model = self.model.fit(X, y)
        self.params_trained['n_estimators'] = self.model.n_estimators
