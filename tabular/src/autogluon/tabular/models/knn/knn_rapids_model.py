import logging

from autogluon.common.utils.try_import import try_import_rapids_cuml
from autogluon.core.constants import REGRESSION

from .._utils.rapids_utils import RapidsModelMixin
from .knn_model import KNNModel

logger = logging.getLogger(__name__)


# FIXME: Benchmarks show that CPU KNN can be trained in ~3 seconds with 0.2 second validation time for CoverType on automlbenchmark (m5.2xlarge)
#  This is over 100 seconds validation time on CPU with rapids installed, investigate how it was so fast on CPU.
#  "2021_02_26/autogluon_hpo_auto.openml_s_271.1h8c.aws.20210228T000327/aws.openml_s_271.1h8c.covertype.0.autogluon_hpo_auto/"
#  Noticed: different input data types, investigate locally with openml dataset version and dtypes.
# TODO: Given this is so fast, consider doing rapid feature pruning
class KNNRapidsModel(RapidsModelMixin, KNNModel):
    """
    RAPIDS KNearestNeighbors model : https://rapids.ai/start.html

    NOTE: This code is experimental, it is recommend to not use this unless you are a developer.
    This was tested on rapids-21.06 via:

    conda create -n rapids-21.06 -c rapidsai -c nvidia -c conda-forge rapids=21.06 python=3.8 cudatoolkit=11.2
    conda activate rapids-21.06
    pip install --pre autogluon.tabular[all]
    """

    def _get_model_type(self):
        try_import_rapids_cuml()
        from cuml.neighbors import KNeighborsClassifier, KNeighborsRegressor

        if self.problem_type == REGRESSION:
            return KNeighborsRegressor
        else:
            return KNeighborsClassifier
