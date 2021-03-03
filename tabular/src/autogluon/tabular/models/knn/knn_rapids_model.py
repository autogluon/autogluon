import logging

from autogluon.core.constants import REGRESSION

from .knn_model import KNNModel

logger = logging.getLogger(__name__)


class KNNRapidsModel(KNNModel):
    """
    RAPIDS KNearestNeighbors model : https://rapids.ai/start.html

    NOTE: This code is experimental, it is recommend to not use this unless you are a developer.
    This was tested on rapids-0.18 via:

    conda create -n rapids-0.18 -c rapidsai -c nvidia -c conda-forge -c defaults rapids-blazing=0.18 python=3.7 cudatoolkit=10.1 -y
    conda activate rapids-0.18
    pip install --pre autogluon.tabular[all]
    """
    def _get_model_type(self):
        from cuml.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if self.problem_type == REGRESSION:
            return KNeighborsRegressor
        else:
            return KNeighborsClassifier

    def _set_default_params(self):
        default_params = {'weights': 'uniform'}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
