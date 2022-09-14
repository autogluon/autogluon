import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
import sktime
from sktime.forecasting.base import BaseForecaster
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from autogluon.common.utils.log_utils import set_logger_verbosity

from ...dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from ...utils.hashing import hash_ts_dataframe_items
from ...utils.seasonality import get_seasonality
from ..abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)
sktime_logger = logging.getLogger(sktime.__name__)


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
        self.sktime_forecaster: Optional[BaseForecaster] = None
        self._fit_hash: Optional[pd.Series] = None

    def _get_sktime_forecaster_init_args(self, min_length: int, inferred_period: int = 1):
        """Get arguments that will be passed to the sktime forecaster at initialization."""
        return self._get_model_params().copy()

    def _get_sktime_forecaster(self, sktime_forecaster_init_args: dict) -> BaseForecaster:
        """Create an sktime forecaster object for the model with given args."""
        unused_args = [k for k in sktime_forecaster_init_args.keys() if k not in self.sktime_allowed_init_args]
        if len(unused_args) > 0:
            logger.warning(
                f"{self.name} ignores following arguments: {unused_args}. "
                f"See `{self.name}.sktime_allowed_init_args` for the list of allowed arguments."
            )

        return self.sktime_forecaster_class(
            **{k: v for k, v in sktime_forecaster_init_args.items() if k in self.sktime_allowed_init_args}  # noqa
        )

    def _to_sktime_data_frame(self, data: TimeSeriesDataFrame) -> pd.DataFrame:
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

    def _to_time_series_data_frame(self, data: pd.DataFrame, freq: Optional[str] = None) -> TimeSeriesDataFrame:
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
        sktime_logger.setLevel(logging.ERROR if verbosity <= 3 else logging.INFO)

        self._check_fit_params()

        min_length = train_data.num_timesteps_per_item().min()
        inferred_period = get_seasonality(train_data.freq)
        sktime_forecaster_init_args = self._get_sktime_forecaster_init_args(
            min_length=min_length, inferred_period=inferred_period
        )
        self.sktime_forecaster = self._get_sktime_forecaster(sktime_forecaster_init_args)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)

            self.sktime_forecaster.fit(self._to_sktime_data_frame(train_data[[self.target]]), fh=self._fh())

        self._fit_hash = hash_ts_dataframe_items(train_data)

    def predict(self, data: TimeSeriesDataFrame, quantile_levels: List[float] = None, **kwargs) -> TimeSeriesDataFrame:
        self._check_predict_inputs(data, quantile_levels=quantile_levels, **kwargs)

        if not self.sktime_forecaster:
            raise ValueError("No sktime forecaster found. Please fit the model first.")

        # sktime trains one local model for each time series (item), so we need to refit if a different set of items is
        # given for prediction. We check that train and pred items match using hash based on `timestamp` and `target`
        # fields (`item_id` can be different as long as `timestamp` and `target` fields match)
        data_hash = hash_ts_dataframe_items(data)
        if len(self._fit_hash) != len(data_hash) or (self._fit_hash.values != data_hash.values).any():
            logger.warning(
                f"Different set of items than those provided during training were provided for "
                f"prediction. The model {self.name} will be re-trained on newly provided data"
            )
            self._fit(data)

        with warnings.catch_warnings():
            # Models in statsmodels may run into numerical issues for some datasets - ignore these
            warnings.simplefilter("ignore", category=RuntimeWarning)

            mean_predictions = self.sktime_forecaster.predict(fh=self._fh())
            quantile_predictions = self.sktime_forecaster.predict_quantiles(
                fh=self._fh(), alpha=quantile_levels or self.quantile_levels
            )

        mean_predictions.columns = ["mean"]
        quantile_predictions.columns = [str(q) for q in quantile_predictions.columns.levels[1]]  # noqa

        predictions = pd.concat([mean_predictions, quantile_predictions], axis=1)
        predictions_df = self._to_time_series_data_frame(predictions, freq=data.freq)
        # Make sure item_id matches `data` (in case trainining and prediction data use different `item_id`s)
        fit_item_id_to_pred_item_id = dict(zip(self._fit_hash.index, data_hash.index))
        pred_item_id = predictions_df.index.get_level_values(ITEMID).map(fit_item_id_to_pred_item_id.get)
        pred_timestamp = predictions_df.index.get_level_values(TIMESTAMP)
        predictions_df.index = pd.MultiIndex.from_arrays([pred_item_id, pred_timestamp], names=(ITEMID, TIMESTAMP))
        return predictions_df
