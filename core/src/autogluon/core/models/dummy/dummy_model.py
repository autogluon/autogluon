from __future__ import annotations

import pandas as pd

from ...constants import BINARY, MULTICLASS, QUANTILE, REGRESSION
from .. import AbstractModel
from ._dummy_quantile_regressor import DummyQuantileRegressor


class DummyModel(AbstractModel):
    """
    A dummy model that ignores input features and predicts only a constant value.
    Useful for tests and calculating worst-case performance.
    """
    ag_key = "DUMMY"
    ag_name = "Dummy"

    def _get_model_type(self):
        from sklearn.dummy import DummyClassifier, DummyRegressor

        if self.problem_type == REGRESSION:
            return DummyRegressor
        elif self.problem_type == QUANTILE:
            return DummyQuantileRegressor
        elif self.problem_type in [BINARY, MULTICLASS]:
            return DummyClassifier
        else:
            raise ValueError(f"DummyModel does not support problem_type={self.problem_type}")

    def preprocess(self, X: pd.DataFrame, **kwargs):
        return X

    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X)
        model_cls = self._get_model_type()
        model_params = self._get_model_params()
        if "raise" in model_params:
            raise_msg = model_params.get("raise_msg", "")
            raise model_params["raise"](raise_msg)
        if model_cls == DummyQuantileRegressor:
            self.model = model_cls(quantile_levels=self.quantile_levels)
        else:
            self.model = model_cls()
        self.model.fit(X=X, y=y)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression", "quantile"]
