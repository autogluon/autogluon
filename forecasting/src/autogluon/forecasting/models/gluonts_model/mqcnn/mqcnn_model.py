import logging

from autogluon.core.utils import warning_filter
import mxnet as mx
with warning_filter():
    from gluonts.model.seq2seq import MQCNNEstimator
    from gluonts.mx.context import get_mxnet_context

from ..abstract_gluonts.abstract_gluonts_model import AbstractGluonTSModel

logger = logging.getLogger(__name__)


class MQCNNModel(AbstractGluonTSModel):
    """MQCNN model from Gluon-TS"""
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
        if get_mxnet_context() != mx.context.cpu():
            self.params["hybridize"] = False

        with warning_filter():
            self.model = MQCNNEstimator.from_hyperparameters(**self.params)
