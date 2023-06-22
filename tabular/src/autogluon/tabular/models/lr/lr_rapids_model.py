import logging

import numpy as np

from autogluon.common.utils.try_import import try_import_rapids_cuml
from autogluon.core.constants import REGRESSION

from .._utils.rapids_utils import RapidsModelMixin
from .hyperparameters.parameters import get_param_baseline
from .lr_model import LinearModel

logger = logging.getLogger(__name__)


# FIXME: If rapids is installed, normal CPU LinearModel crashes.
class LinearRapidsModel(RapidsModelMixin, LinearModel):
    """
    RAPIDS Linear model : https://rapids.ai/start.html

    NOTE: This code is experimental, it is recommend to not use this unless you are a developer.
    This was tested on rapids-21.06 via:

    conda create -n rapids-21.06 -c rapidsai -c nvidia -c conda-forge rapids=21.06 python=3.8 cudatoolkit=11.2
    conda activate rapids-21.06
    pip install --pre autogluon.tabular[all]
    """

    def _get_model_type(self):
        penalty = self.params.get("penalty", "L2")
        try_import_rapids_cuml()
        from cuml.linear_model import Lasso, LogisticRegression, Ridge

        if self.problem_type == REGRESSION:
            if penalty == "L2":
                model_type = Ridge
            elif penalty == "L1":
                model_type = Lasso
            else:
                raise AssertionError(f'Unknown value for penalty "{penalty}" - supported types are ["L1", "L2"]')
        else:
            model_type = LogisticRegression
        return model_type

    def _set_default_params(self):
        default_params = {"fit_intercept": True, "max_iter": 10000}
        if self.problem_type != REGRESSION:
            default_params.update({"solver": "qn"})
        default_params.update(get_param_baseline())
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X=X, **kwargs)
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        return X

    def _fit(self, X, y, **kwargs):
        kwargs.pop("sample_weight", None)  # sample_weight is not supported
        super()._fit(X=X, y=y, **kwargs)
