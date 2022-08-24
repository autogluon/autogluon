import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from ...utils.hashing import hash_ts_dataframe_items
from ...utils.warning_filters import statsmodels_warning_filter
from ..abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


@dataclass
class ModelFitSummary:
    """Stores the parameters of a fitted Statsmodels time series model.

    ModelFitSummary doesn't store intermediate results and therefore takes up very little disk space when pickled. In
    contrast, StatsmodelsFittedModel stores the intermediate results and therefore can often take several gigabytes
    when pickled.

    ModelFitSummary is consumed by AbstractStatsmodelsModel._predict_local_model to generate predictions.
    """

    model_name: str
    model_init_args: Dict[str, Any]
    parameters: Dict[str, Any]


class AbstractStatsmodelsModel(AbstractTimeSeriesModel):
    statsmodels_allowed_init_args: List[str] = []
    statsmodels_allowed_fit_args: List[str] = []
    quantile_method_name: str = "pred_int"

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
        self._fitted_models: Dict[str, ModelFitSummary] = {}

    def _get_default_model_init_and_fit_args(self) -> Tuple[dict, dict]:
        default_model_init_args = {}
        default_model_fit_args = {}
        unused_args = []
        for key, value in self._get_model_params().items():
            if key in self.statsmodels_allowed_init_args:
                default_model_init_args[key] = value
            elif key in self.statsmodels_allowed_fit_args:
                default_model_fit_args[key] = value
            else:
                unused_args.append(key)
        if len(unused_args) > 0:
            logger.warning(
                f" {self.name} ignores the following hyperparameters: {unused_args}. "
                f"See the docstring of {self.name} for the list of supported hyperparameters."
            )
        return default_model_init_args, default_model_fit_args

    def _fit_local_model(
        self, timeseries: TimeSeriesDataFrame, model_init_args: dict, model_fit_args: dict
    ) -> ModelFitSummary:
        """Fit a single local model to the time series."""
        raise NotImplementedError

    def _fit(self, train_data: TimeSeriesDataFrame, time_limit: int = None, **kwargs):
        train_hash = hash_ts_dataframe_items(train_data)
        items_to_fit = [item_id for item_id, ts_hash in train_hash.iteritems() if ts_hash not in self._fitted_models]
        default_model_init_args, default_model_fit_args = self._get_default_model_init_and_fit_args()
        # TODO: Parallelize fitting
        with statsmodels_warning_filter():
            for item_id in items_to_fit:
                self._fitted_models[train_hash.loc[item_id]] = self._fit_local_model(
                    timeseries=train_data.loc[item_id],
                    default_model_init_args=default_model_init_args,
                    default_model_fit_args=default_model_fit_args,
                )

    def predict(self, data: TimeSeriesDataFrame, quantile_levels: List[float] = None, **kwargs) -> TimeSeriesDataFrame:
        self._check_predict_inputs(data, quantile_levels=quantile_levels, **kwargs)
        if quantile_levels is None:
            quantile_levels = self.quantile_levels
        # Make sure that we fitted a local model to each time series in data
        self._fit(train_data=data)

        data_hash = hash_ts_dataframe_items(data)
        # TODO: Parallelize prediction
        predictions_per_item = {}
        with statsmodels_warning_filter():
            for item_id, ts_hash in data_hash.iteritems():
                predictions_per_item[item_id] = self._predict_using_fit_summary(
                    fit_summary=self._fitted_models[ts_hash],
                    timeseries=data.loc[item_id],
                    quantile_levels=quantile_levels,
                )
        result_df = pd.concat(predictions_per_item)
        result_df.index.rename([ITEMID, TIMESTAMP], inplace=True)
        return TimeSeriesDataFrame(result_df)

    def _predict_using_fit_summary(
        self,
        fit_summary: ModelFitSummary,
        timeseries: TimeSeriesDataFrame,
        quantile_levels: List[float],
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _get_predictions_from_fitted_model(self, fitted_model, cutoff, quantile_levels):
        start = cutoff + pd.tseries.frequencies.to_offset(self.freq)
        end = cutoff + self.prediction_length * pd.tseries.frequencies.to_offset(self.freq)
        predictions = fitted_model.get_prediction(start=start, end=end)
        results = [predictions.predicted_mean.rename("mean")]
        for q in quantile_levels:
            if q < 0.5:
                coverage = 2 * q
                column_index = 0
            else:
                coverage = 2 * (1 - q)
                column_index = 1
            quantile_pred = getattr(predictions, self.quantile_method_name)(alpha=coverage)
            # Select lower bound of the confidence interval if q < 0.5, upper bound otherwise
            results.append(quantile_pred.iloc[:, column_index].rename(str(q)))
        return pd.concat(results, axis=1)
