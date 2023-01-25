import logging
import re
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from joblib import Parallel, delayed

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.hashing import hash_ts_dataframe_items
from autogluon.timeseries.utils.seasonality import get_seasonality
from autogluon.timeseries.utils.warning_filters import statsmodels_joblib_warning_filter

logger = logging.getLogger(__name__)


class AbstractLocalModel(AbstractTimeSeriesModel):
    allowed_local_model_args: List[str] = []
    # Use 50% of the cores since some models rely on parallel ops and are actually slower if n_jobs=-1
    DEFAULT_N_JOBS: Union[float, int] = 0.5

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
        if hyperparameters is None:
            hyperparameters = {}
        # TODO: Replace with 'num_cpus' argument passed to fit (after predictor API is changed)
        n_jobs = hyperparameters.get("n_jobs", self.DEFAULT_N_JOBS)
        if isinstance(n_jobs, float) and 0 < n_jobs <= 1:
            self.n_jobs = max(int(cpu_count() * n_jobs), 1)
        elif isinstance(n_jobs, int):
            self.n_jobs = n_jobs
        else:
            raise ValueError(f"n_jobs must be a float between 0 and 1 or an integer (received n_jobs = {n_jobs})")
        self._local_model_args: Dict[str, Any] = None
        self._cached_predictions: Dict[str, pd.DataFrame] = {}

    def _fit(self, train_data: TimeSeriesDataFrame, time_limit: int = None, **kwargs):
        self._check_fit_params()
        # Initialize parameters passed to each local model
        raw_local_model_args = self._get_model_params().copy()
        raw_local_model_args.pop("n_jobs", None)

        unused_local_model_args = []
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

        if "seasonal_period" not in local_model_args or local_model_args["seasonal_period"] is None:
            local_model_args["seasonal_period"] = get_seasonality(train_data.freq)
        self.freq = train_data.freq

        self._local_model_args = self._update_local_model_args(local_model_args=local_model_args, data=train_data)

        logger.debug(f"{self.name} is a local model, so the model will be fit at prediction time.")
        return self

    def _update_local_model_args(
        self, local_model_args: Dict[str, Any], data: TimeSeriesDataFrame, **kwargs
    ) -> Dict[str, Any]:
        return local_model_args

    def predict(self, data: TimeSeriesDataFrame, quantile_levels: List[float] = None, **kwargs) -> TimeSeriesDataFrame:
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        if self.freq != data.freq:
            raise RuntimeError(
                f"{self.name} has frequency '{self.freq}', which doesn't match the frequency "
                f"of the dataset '{data.freq}'."
            )

        self._fit_and_cache_predictions(data, quantile_levels=quantile_levels)
        item_id_to_prediction = {}
        for item_id, ts_hash in hash_ts_dataframe_items(data).items():
            item_id_to_prediction[item_id] = self._cached_predictions[ts_hash]
        predictions_df = pd.concat(item_id_to_prediction)
        predictions_df.index.rename([ITEMID, TIMESTAMP], inplace=True)
        return TimeSeriesDataFrame(predictions_df)

    def _fit_and_cache_predictions(self, data: TimeSeriesDataFrame, quantile_levels: List[float]):
        data_hash = hash_ts_dataframe_items(data)
        items_to_fit = [item_id for item_id, ts_hash in data_hash.items() if ts_hash not in self._cached_predictions]
        if len(items_to_fit) > 0:
            logger.debug(f"{self.name} received {len(items_to_fit)} new items to predict, generating predictions")
            target_series = data[self.target]
            time_series_to_fit = (target_series.loc[item_id] for item_id in items_to_fit)
            with statsmodels_joblib_warning_filter():
                predictions = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._predict_with_local_model)(
                        time_series=ts,
                        prediction_length=self.prediction_length,
                        freq=self.freq,
                        quantile_levels=quantile_levels,
                        local_model_args=self._local_model_args.copy(),
                    )
                    for ts in time_series_to_fit
                )
            for item_id, preds in zip(items_to_fit, predictions):
                self._cached_predictions[data_hash.loc[item_id]] = preds
            # Make sure cached predictions can be reused by other models
            self.save()

    @staticmethod
    def _predict_with_local_model(
        time_series: pd.Series,
        freq: str,
        prediction_length: int,
        quantile_levels: List[float],
        local_model_args: dict,
        **kwargs,
    ) -> pd.DataFrame:
        raise NotImplementedError
