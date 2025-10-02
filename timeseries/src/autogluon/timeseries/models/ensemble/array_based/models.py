from .abstract import ArrayBasedTimeSeriesEnsembleModel
from .regressor import SimpleAverageEnsembleRegressor


class ArrayBasedSimpleAverageEnsemble(ArrayBasedTimeSeriesEnsembleModel):
    _default_model_name = "ArrayBasedSimpleAverageEnsemble"
    _regressor_type = SimpleAverageEnsembleRegressor
