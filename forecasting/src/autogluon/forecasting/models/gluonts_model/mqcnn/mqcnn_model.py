from ..abstract_gluonts.abstract_gluonts_model import AbstractGluonTSModel
from gluonts.model.seq2seq import MQCNNEstimator
import logging

logger = logging.getLogger(__name__)

"""
MQCNN model from Gluon-TS
"""
class MQCNNModel(AbstractGluonTSModel):

    def __init__(self, path: str, freq: str, prediction_length: int, name: str = "MQCNN",
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
        self.model = MQCNNEstimator.from_hyperparameters(**self.params)
