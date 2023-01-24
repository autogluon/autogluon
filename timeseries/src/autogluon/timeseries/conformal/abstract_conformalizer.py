from typing import List, Optional
from autogluon.timeseries import TimeSeriesDataFrame


class AbstractConformalizer:
    def __init__(self, prediction_length: int, quantile_levels: List[float], target_column: str = "target"):
        self.prediction_length = prediction_length
        self.quantile_levels = quantile_levels
        self.target_column = target_column
        self._is_fit = False

    def fit(self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame):
        raise NotImplementedError

    def transform(
        self, predictions: TimeSeriesDataFrame, data_past: Optional[TimeSeriesDataFrame] = None
    ) -> TimeSeriesDataFrame:
        raise NotImplementedError

    def is_fit(self) -> bool:
        return self._is_fit

    def __repr__(self):
        return self.__class__.__name__
