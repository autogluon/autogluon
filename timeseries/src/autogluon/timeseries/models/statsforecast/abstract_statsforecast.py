import logging
import re
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.hashing import hash_ts_dataframe_items
from autogluon.timeseries.utils.seasonality import get_seasonality

logger = logging.getLogger(__name__)


class AbstractStatsForecastModel(AbstractTimeSeriesModel):
    """Wrapper for StatsForecast models.

    Cached predictions are stored inside the model to speed up validation & ensemble training downstream.

    Attributes
    ----------
    allowed_local_model_args : List[str]
        List of allowed arguments that can be passed to the underlying model.
        Arguments not in this list will be filtered out and not passed to the underlying model.
    """

    allowed_local_model_args: List[str] = []

    def get_model_type(self) -> Type:
        raise NotImplementedError

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
        # TODO: Replace with 'num_cpus' argument passed to fit (after predictor API is changed)
        n_jobs = hyperparameters.get("n_jobs", -1)
        if isinstance(n_jobs, float) and 0 < n_jobs <= 1:
            self.n_jobs = max(int(cpu_count() * n_jobs), 1)
        elif isinstance(n_jobs, int):
            self.n_jobs = n_jobs
        else:
            raise ValueError(f"n_jobs must be a float between 0 and 1 or an integer (received n_jobs = {n_jobs})")
        self._local_model_args: Dict[str, Any] = None
        self._cached_predictions: Dict[str, pd.DataFrame] = {}

    def _fit(self, train_data, time_limit=None, verbosity=2, **kwargs) -> None:
        """Prepare hyperparameters that will be passed to the underlying model.

        As for all local models, actual fitting + predictions are delegated to the ``predict`` method.
        """
        # TODO: Find a way to ensure that SF models respect time_limit
        # Fitting usually takes >= 20 seconds
        if time_limit is not None and time_limit < 20:
            raise TimeLimitExceeded

        unused_local_model_args = []
        raw_local_model_args = self._get_model_params().copy()
        raw_local_model_args.pop("n_jobs", None)
        local_model_args = {}
        for key, value in raw_local_model_args.items():
            if key in self.allowed_local_model_args:
                local_model_args[key] = value
            else:
                unused_local_model_args.append(key)

        if len(unused_local_model_args):
            logger.warning(
                f"{self.name} ignores following hyperparameters: {unused_local_model_args}. "
                f"See the docstring of {self.name} for the list of supported hyperparameters."
            )

        seasonal_period = local_model_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = get_seasonality(train_data.freq)
        local_model_args["season_length"] = seasonal_period

        self.freq = train_data.freq
        self._local_model_args = self._update_local_model_args(local_model_args)
        logger.debug(f"{self.name} is a local model, so the model will be fit at prediction time.")
        return self

    def _update_local_model_args(self, local_model_args: dict) -> dict:
        """Update arguments passed to the local model (e.g., overriding defaults)"""
        return local_model_args

    def _to_statsforecast_dataframe(self, data: TimeSeriesDataFrame) -> pd.DataFrame:
        target = data[[self.target]]
        return target.reset_index().rename({ITEMID: "unique_id", TIMESTAMP: "ds", self.target: "y"}, axis=1)

    def _fit_and_cache_predictions(self, data: TimeSeriesDataFrame):
        """Make predictions for time series in data that are not cached yet."""
        # TODO: Improve prediction caching logic -> save predictions to a separate file, like in Tabular?
        from statsforecast import StatsForecast
        from statsforecast.models import SeasonalNaive

        data_hash = hash_ts_dataframe_items(data)
        items_to_fit = [item_id for item_id, ts_hash in data_hash.items() if ts_hash not in self._cached_predictions]
        if len(items_to_fit) > 0:
            logger.debug(f"{self.name} received {len(items_to_fit)} new items to predict, generating predictions")
            data_to_fit = pd.DataFrame(data).query("item_id in @items_to_fit")

            model_type = self.get_model_type()
            model = model_type(**self._local_model_args)

            sf = StatsForecast(
                models=[model],
                fallback_model=SeasonalNaive(season_length=self._local_model_args["season_length"]),
                sort_df=False,
                freq=self.freq,
                n_jobs=self.n_jobs,
            )

            # StatsForecast generates probabilistic forecasts in lo/hi confidence region boundaries
            # We chose the columns that correspond to the desired quantile_levels
            model_name = str(model)
            new_column_names = {"unique_id": ITEMID, "ds": TIMESTAMP, model_name: "mean"}
            levels = []
            for q in self.quantile_levels:
                level = round(abs(q - 0.5) * 200, 1)
                suffix = "lo" if q < 0.5 else "hi"
                levels.append(level)
                new_column_names[f"{model_name}-{suffix}-{level}"] = str(q)
            levels = sorted(list(set(levels)))
            chosen_columns = list(new_column_names.values())

            raw_predictions = sf.forecast(
                df=self._to_statsforecast_dataframe(data_to_fit),
                h=self.prediction_length,
                level=levels,
            ).reset_index()
            predictions = raw_predictions.rename(new_column_names, axis=1)[chosen_columns].set_index(TIMESTAMP)
            item_ids = predictions.pop(ITEMID)

            for item_id, preds in predictions.groupby(item_ids, sort=False):
                self._cached_predictions[data_hash.loc[item_id]] = preds
            # Make sure cached predictions can be reused by other models
            self.save()

    def predict(self, data: TimeSeriesDataFrame, **kwargs) -> TimeSeriesDataFrame:
        if self.freq != data.freq:
            raise RuntimeError(
                f"{self.name} has frequency '{self.freq}', which doesn't match the frequency "
                f"of the dataset '{data.freq}'."
            )

        self._fit_and_cache_predictions(data)
        item_id_to_prediction = {}
        for item_id, ts_hash in hash_ts_dataframe_items(data).items():
            item_id_to_prediction[item_id] = self._cached_predictions[ts_hash]
        predictions_df = pd.concat(item_id_to_prediction)
        predictions_df.index.rename([ITEMID, TIMESTAMP], inplace=True)
        return TimeSeriesDataFrame(predictions_df)
