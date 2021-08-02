from ..abstract_gluonts.abstract_gluonts_model import AbstractGluonTSModel
from autogluon.core.utils import warning_filter
with warning_filter():
    from gluonts.model.deepar import DeepAREstimator
import logging

logger = logging.getLogger(__name__)

"""
DeepAR model from Gluon-TS
"""
class DeepARModel(AbstractGluonTSModel):

    def __init__(self, path: str, freq: str, prediction_length: int, name: str = "DeepAR",
                 eval_metric: str = None, hyperparameters=None, model=None, **kwargs):
        super().__init__(path=path,
                         freq=freq,
                         prediction_length=prediction_length,
                         hyperparameters=hyperparameters,
                         name=name,
                         eval_metric=eval_metric,
                         model=model,
                         **kwargs)

    def create_model(self):
        with warning_filter():
            self.model = DeepAREstimator.from_hyperparameters(**self.params)