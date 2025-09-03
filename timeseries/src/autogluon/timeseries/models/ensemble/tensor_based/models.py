from .abstract import TensorBasedTimeSeriesEnsembleModel
from .regressor import SimpleAverageEnsembleRegressor


class TensorBasedSimpleAverageEnsemble(TensorBasedTimeSeriesEnsembleModel):
    _default_model_name = "TensorBasedSimpleAverageEnsemble"
    _regressor_type = SimpleAverageEnsembleRegressor
