from autogluon.timeseries import TimeSeriesDataFrame


class AbstractTransform:
    def __init__(self, target: str = "target"):
        self.target = target
        self._is_fit = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def fit(self, data: TimeSeriesDataFrame) -> "AbstractTransform":
        self._fit(data)
        self._is_fit = True
        return self

    def _fit(self, data: TimeSeriesDataFrame) -> None:
        raise NotImplementedError

    def transform(self, data: TimeSeriesDataFrame, inplace: bool = False) -> TimeSeriesDataFrame:
        if not self._is_fit:
            raise ValueError(f"{self.name} must be fit before calling transform")
        if not inplace:
            data = data.copy()
        return self._transform(data)

    def _transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        raise NotImplementedError

    def fit_transform(self, data: TimeSeriesDataFrame, inplace: bool = False) -> TimeSeriesDataFrame:
        return self.fit(data).transform(data, inplace=inplace)

    def inverse_transform_predictions(
        self, predictions: TimeSeriesDataFrame, inplace: bool = False
    ) -> TimeSeriesDataFrame:
        if not self._is_fit:
            raise ValueError(f"{self.name} must be fit before calling transform")
        if not inplace:
            predictions = predictions.copy()
        return self._inverse_transform_predictions(predictions)

    def _inverse_transform_predictions(self, predictions: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        raise NotImplementedError
