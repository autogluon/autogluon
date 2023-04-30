from typing import Optional
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


class AbstractTransformer:
    """Abstract class for time series transformations.

    Subclasses should implement methods ``_fit_transform`` and ``_inverse_transform_predictions``.

    Example usage inside a forecasting model::

        def _fit(self, train_data):
            train_data = self.transform.fit_transform(train_data)
            val_data = self.transform.fit_transform(val_data)
            self._fit_internal(train_data, val_data)

        def predict(self, data):
            data = self.transform.fit_transform(data)
            predictions = self._predict_internal(data)
            return self.transform.inverse_transform_predictions(predictions)

    Parameters
    ----------
    target : str, default = "target"
        Name of the target column.
    copy : bool, default = True
        If True, data will be copied before transforming to avoid in-place modifications.
    """

    def __init__(self, target: str = "target", copy: bool = True):
        self.target = target
        self.copy = copy
        # Keep track of the item_ids to ensure that inverse_transform_predictions is
        # applied to the same time series as seen during fit_transform
        self._fit_item_ids: Optional[pd.Series] = None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.name

    @property
    def _is_fit(self) -> bool:
        return self._fit_item_ids is not None

    def _validate_inputs(self, data: TimeSeriesDataFrame, prediction_mode: bool = False):
        assert isinstance(data, TimeSeriesDataFrame)
        assert isinstance(data.index, pd.MultiIndex)
        if prediction_mode:
            # Only mean and quantile columns are allowed
            assert "mean" in data.columns
            for col in data.columns:
                if col != "mean":
                    assert str(float(col)) == col
        else:
            assert self.target in data.columns

    def fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Fit the transformation to the data, then transform the data."""
        self._validate_inputs(data)
        if self.copy:
            data = data.copy()
        transformed = self._fit_transform(data)
        self._fit_item_ids = data.item_ids
        return transformed

    def _fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        raise NotImplementedError

    def inverse_transform_predictions(self, predictions: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Apply inverse transformation to the predictions to make them compatible with the original data."""
        self._validate_inputs(predictions, prediction_mode=True)
        if not self._is_fit:
            raise AssertionError(f"{self.name} must be fit before calling inverse_transform_predictions")
        if not predictions.item_ids.equals(self._fit_item_ids):
            raise AssertionError(f"{self.name} was fit on data with different item_ids")
        if self.copy:
            predictions = predictions.copy()
        return self._inverse_transform_predictions(predictions)

    def _inverse_transform_predictions(self, predictions: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        raise NotImplementedError
