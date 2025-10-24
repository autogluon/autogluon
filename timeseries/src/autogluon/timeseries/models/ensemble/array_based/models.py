from .abstract import ArrayBasedTimeSeriesEnsembleModel
from .regressor import MedianEnsembleRegressor


class MedianEnsemble(ArrayBasedTimeSeriesEnsembleModel):
    _regressor_type = MedianEnsembleRegressor
