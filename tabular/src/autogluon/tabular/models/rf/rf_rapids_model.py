import logging

from autogluon.common.utils.try_import import try_import_rapids_cuml
from autogluon.core.constants import REGRESSION, SOFTCLASS

from .._utils.rapids_utils import RapidsModelMixin
from .rf_model import RFModel

logger = logging.getLogger(__name__)


# TODO: Improve memory safety
# TODO: Respect time limit
# TODO: Depending on max_depth parameter, RFRapidsModel is slower than RFModel.
#  A lower max_depth (e.g., 16) results in a RFRapidsModel that is faster than RFModel,
#  but a higher max_depth (e.g., approximating unlimited depth)
#  results in a RFRapidsModel that is significantly slower than RFModel.
#  Refer to https://github.com/rapidsai/cuml/issues/1977
class RFRapidsModel(RapidsModelMixin, RFModel):
    """
    RAPIDS Random Forest model : https://rapids.ai/start.html

    NOTE: This code is experimental, it is recommend to not use this unless you are a developer.
    This was tested on rapids-21.06 via:

    conda create -n rapids-21.06 -c rapidsai -c nvidia -c conda-forge rapids=21.06 python=3.8 cudatoolkit=11.2
    conda activate rapids-21.06
    pip install --pre autogluon.tabular[all]
    """

    def _get_model_type(self):
        try_import_rapids_cuml()
        from cuml.ensemble import RandomForestClassifier, RandomForestRegressor

        if self.problem_type in [REGRESSION, SOFTCLASS]:
            return RandomForestRegressor
        else:
            return RandomForestClassifier

    def _set_default_params(self):
        default_params = {
            "n_estimators": 300,
            "max_depth": 99,  # RAPIDS does not allow unlimited depth, so this approximates it.
            "random_state": 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X, y=y)
        self.model = self._get_model_type()(**self._get_model_params())
        self.model = self.model.fit(X, y)
        self.params_trained["n_estimators"] = self.model.n_estimators
