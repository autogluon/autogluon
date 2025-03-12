from abc import abstractmethod

import pandas as pd

from autogluon.common.utils.try_import import try_import_interpret
from autogluon.core.constants import REGRESSION
from autogluon.core.models import AbstractModel
import warnings


class EBMModel(AbstractModel):

    ag_key = "EBM"
    ag_name = "EBMModel"
    ag_priority = 20
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model(self):
        try_import_interpret()
        from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

        if self.problem_type == REGRESSION:
            return ExplainableBoostingRegressor
        else:
            return ExplainableBoostingClassifier

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        model_cls = self.get_model()
        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        self.model = model_cls(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model.fit(X, y)

