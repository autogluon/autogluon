import itertools
from typing import Optional, Tuple

import pandas as pd
import pytest

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

from ..common import get_data_frame_with_item_index


class ConcreteTimeSeriesModel(AbstractTimeSeriesModel):
    """A dummy model that predicts 42s, implemented according to the custom model
    tutorial [1].
    
    References
    ----------
    .. [1] https://auto.gluon.ai/dev/tutorials/timeseries/advanced/forecasting-custom-model.html#implement-the-custom-model
    """
    _supports_known_covariates: bool = True
    _supports_past_covariates: bool = True
    _supports_static_features: bool = True

    def preprocess(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        is_train: bool = False,
        **kwargs,
    ) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        """Method that implements model-specific preprocessing logic.

        This method is called on all data that is passed to `_fit` and `_predict` methods.
        """
        data = data.fill_missing_values(method="constant", value=42.0)
        return data, known_covariates

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
        **kwargs,
    ) -> None:
        # _fit depends on _get_model_params() to provide parameters for the inner model
        _ = self._get_model_params()

        # let's do some work
        self.dummy_learned_parameters = train_data.groupby(level=0).mean().to_dict()

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Predict future target given the historic time series data and the future values of known_covariates."""
        assert self.dummy_learned_parameters is not None

        return TimeSeriesDataFrame(
            pd.DataFrame(
                index=self.get_forecast_horizon_index(data), 
                columns=["mean"] + self.quantile_levels
            ).fillna(42.0)
        )


class TestConcreteModelsInitialization:
    
    @pytest.fixture(
        scope="class", 
        params=list(
            itertools.product(
                [["A", "B"], ["C", "D"], ["A"], [0, 1, 2, 3]],
                [10, 45],
                ["h", "T", "d"],
            )
        ),
    )
    def train_data(self, request):
        index, data_length, freq = request.param
        return get_data_frame_with_item_index(index, data_length=data_length, freq=freq)
    
    def test_models_can_be_initialized(self, train_data, temp_model_path):
        model = ConcreteTimeSeriesModel(
            path=temp_model_path, freq=train_data.freq, prediction_length=24
        )
        assert isinstance(model, AbstractTimeSeriesModel)
