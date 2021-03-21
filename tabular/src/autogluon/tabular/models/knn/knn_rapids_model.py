import logging

from autogluon.core.constants import REGRESSION
from autogluon.core.utils.try_import import try_import_rapids_cuml

from .knn_model import KNNModel

logger = logging.getLogger(__name__)


# FIXME: Benchmarks show that CPU KNN can be trained in ~3 seconds with 0.2 second validation time for CoverType on automlbenchmark (m5.2xlarge)
#  This is over 100 seconds validation time on CPU with rapids installed, investigate how it was so fast on CPU.
#  "2021_02_26/autogluon_hpo_auto.openml_s_271.1h8c.aws.20210228T000327/aws.openml_s_271.1h8c.covertype.0.autogluon_hpo_auto/"
#  Noticed: different input data types, investigate locally with openml dataset version and dtypes.
# TODO: Given this is so fast, consider doing rapid feature pruning
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
        try_import_rapids_cuml()
        from cuml.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if self.problem_type == REGRESSION:
            return KNeighborsRegressor
        else:
            return KNeighborsClassifier

    def _set_default_params(self):
        default_params = {'weights': 'uniform'}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def _get_default_ag_args_ensemble(cls) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble()
        extra_ag_args_ensemble = {'use_child_oof': False}
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _more_tags(self):
        return {'valid_oof': False}
