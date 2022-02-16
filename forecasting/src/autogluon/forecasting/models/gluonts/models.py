import mxnet as mx

from autogluon.core.utils import warning_filter

with warning_filter():
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.model.seq2seq import MQCNNEstimator
    from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.mx.context import get_mxnet_context
    from gluonts.nursery.autogluon_tabular import TabularEstimator

from .abstract_gluonts import AbstractGluonTSModel


# TODO: add main hyperparameters and documentation for each model
class DeepARModel(AbstractGluonTSModel):
    """DeepAR model from Gluon-TS"""

    def _get_estimator(self):
        with warning_filter():
            return DeepAREstimator.from_hyperparameters(**self.params)


class MQCNNModel(AbstractGluonTSModel):
    """MQCNN model from Gluon-TS"""

    def _get_estimator(self):
        if get_mxnet_context() != mx.context.cpu():
            self.params["hybridize"] = False

        with warning_filter():
            return MQCNNEstimator.from_hyperparameters(**self.params)


class SimpleFeedForwardModel(AbstractGluonTSModel):
    """SimpleFeedForward model from Gluon-TS"""

    def _get_estimator(self):
        with warning_filter():
            return SimpleFeedForwardEstimator.from_hyperparameters(**self.params)


class AutoTabularModel(AbstractGluonTSModel):
    """Autotabular model from Gluon-TS"""

    # TODO: AutoTabular model is experimental, may need its own logic for
    # TODO: handling time limit and training data. See PR #1538.
    def _get_estimator(self):
        return TabularEstimator(
            freq=self.params["freq"],
            prediction_length=self.params["prediction_length"],
            time_limit=self.params["time_limit"],
            last_k_for_val=self.params["prediction_length"],
        )
