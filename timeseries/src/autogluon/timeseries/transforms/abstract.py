from autogluon.timeseries.dataset import TimeSeriesDataFrame


class AbstractTargetTransform:
    """Base class for time series transformations applied to the target column."""

    def __init__(self, target: str = "target", **kwargs):
        self.target = target

    def fit(self, data: TimeSeriesDataFrame) -> "AbstractTargetTransform":
        raise NotImplementedError

    def transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        raise NotImplementedError

    def fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        return self.fit(data=data).transform(data=data)

    def inverse_transform(self, predictions: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        raise NotImplementedError
