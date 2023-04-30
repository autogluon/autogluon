from typing import List
from autogluon.timeseries import TimeSeriesDataFrame

from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame

from .abstract_transformer import AbstractTransformer


class PipelineTransformer(AbstractTransformer):
    """Combination of multiple transforms as a single operation."""

    def __init__(self, transformations: List[AbstractTransformer], target: str = "target", copy: bool = True):
        super().__init__(target=target, copy=copy)
        assert all(isinstance(t, AbstractTransformer) for t in transformations)
        self.transformations = transformations

    def _fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        for t in self.transformations:
            data = t.fit_transform(data)
        return data

    def _inverse_transform_predictions(self, predictions: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        for t in reversed(self.transformations):
            predictions = t.inverse_transform_predictions(predictions)
        return predictions
