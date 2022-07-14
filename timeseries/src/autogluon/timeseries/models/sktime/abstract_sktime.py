import logging
import re
import warnings
from typing import Optional, List, Dict, Any, Type

import numpy as np
import pandas as pd
import sktime
from sktime.forecasting.base import BaseForecaster
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from autogluon.common.utils.log_utils import set_logger_verbosity

from ... import TimeSeriesDataFrame
from ...utils.seasonality import get_seasonality
from ..abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)
skt_logger = logging.getLogger(sktime.__name__)


class AbstractSktimeModel(AbstractTimeSeriesModel):
    """Abstract class wrapping ``sktime`` estimators for use in autogluon.timeseries.

    Parameters
    ----------
    path: str
        directory to store model artifacts.
    freq: str
        string representation (compatible with GluonTS frequency strings) for the data provided.
        For example, "1D" for daily data, "1H" for hourly data, etc.
    prediction_length: int
        Number of time steps ahead (length of the forecast horizon) the model will be optimized
        to predict. At inference time, this will be the number of time steps the model will
        predict.
    name: str
        Name of the model. Also, name of subdirectory inside path where model will be saved.
    eval_metric: str
        objective function the model will be scored on, will use mean_wQuantileLoss by default.
    hyperparameters:
        various hyperparameters that will be used by model (can be search spaces instead of
        fixed values). See *Other Parameters* in each inheriting model's documentation for
        possible values.
    """

    sktime_model_path = "sktime"
    sktime_forecaster_class: Type[BaseForecaster] = None
    sktime_allowed_init_args: List[str] = []

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: str = None,
        hyperparameters: Dict[str, Any] = None,
        **kwargs,  # noqa
    ):
        name = name or re.sub(r"Model$", "", self.__class__.__name__)
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        self.skt_forecaster: Optional[BaseForecaster] = None
        self._fit_index: Optional[pd.Index] = None

    def _get_skt_forecaster(self, sp: Optional[int] = None) -> BaseForecaster:
        """Return the sktime forecaster object for the model"""
        params = self._get_model_params().copy()
        if sp is not None and sp > 1:
            params["sp"] = sp
            params["seasonal"] = "add"

        return self.sktime_forecaster_class(
            **{  # noqa
                k: v for k, v in params.items() if k in self.sktime_allowed_init_args
            }
        )

    def _to_skt_data_frame(self, data: TimeSeriesDataFrame) -> pd.DataFrame:
        """Convert time series data frame's DateTimeIndex to PeriodIndex for use in
        sktime, and cast to pandas DataFrame.
        """
        return pd.DataFrame(
            data=data.values,
            index=pd.MultiIndex.from_arrays(
                [
                    data.index.get_level_values(0),
                    data.index.get_level_values(1).to_period(freq=data.freq),  # noqa
                ]
            ),
        )

    def _to_time_series_data_frame(
        self, data: pd.DataFrame, freq: Optional[str] = None
    ) -> TimeSeriesDataFrame:
        df = data.copy(deep=False)
        df.set_index(
            [
                df.index.get_level_values(0),
                df.index.get_level_values(1).to_timestamp(freq=freq),  # noqa
            ],
            inplace=True,
        )
        return TimeSeriesDataFrame(df)

    def _fh(self):
        return np.arange(1, self.prediction_length + 1)

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        verbosity = kwargs.get("verbosity", 2)
        set_logger_verbosity(verbosity, logger=logger)
        skt_logger.setLevel(logging.ERROR if verbosity <= 3 else logging.INFO)

        self._check_fit_params()

        min_length = min(len(train_data.loc[i]) for i in train_data.iter_items())
        period = get_seasonality(train_data.freq)
        self.skt_forecaster = self._get_skt_forecaster(
            sp=period if min_length > 2 * period else None
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)

            self.skt_forecaster.fit(
                self._to_skt_data_frame(train_data[[self.target]]), fh=self._fh()
            )

        self._fit_index = train_data.index.copy()

    def predict(
        self, data: TimeSeriesDataFrame, quantile_levels: List[float] = None, **kwargs
    ) -> TimeSeriesDataFrame:
        self._check_predict_inputs(data, quantile_levels=quantile_levels, **kwargs)

        if not self.skt_forecaster:
            raise ValueError("No sktime forecaster found. Please fit the model first.")

        # TODO: reconsider when to refit the model. currently we refit whenever train and
        #  test indices are not identical
        if not self._fit_index.equals(data.index):
            logger.warning(
                f"Different set of items than those provided during training were provided for "
                f"prediction. The model {self.name} will be re-trained on newly provided data"
            )
            self._fit(data)

        mean_predictions = self.skt_forecaster.predict(fh=self._fh())
        mean_predictions.columns = ["mean"]
        quantile_predictions = self.skt_forecaster.predict_quantiles(
            fh=self._fh(), alpha=quantile_levels or self.quantile_levels
        )
        quantile_predictions.columns = [
            str(q) for q in quantile_predictions.columns.levels[1]  # noqa
        ]

        predictions = pd.concat([mean_predictions, quantile_predictions], axis=1)
        return self._to_time_series_data_frame(predictions, freq=data.freq)
