import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gluonts.time_feature import TimeFeature, get_lags_for_frequency, time_features_from_frequency_str
from joblib import Parallel, delayed

import autogluon.core as ag
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.tabular import TabularPredictor
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

TS_METRIC_TO_TABULAR_METRIC = {
    "mean_wQuantileLoss": "pinball_loss",
    "MAPE": "mean_absolute_percentage_error",
    "sMAPE": "mean_absolute_percentage_error",
    "MASE": "mean_absolute_error",
    "RMSE": "root_mean_squared_error",
    "MSE": "mean_squared_error",
}

logger = logging.getLogger(__name__)


class TabularModel(AbstractTimeSeriesModel):
    """Uses TabularPredictor to forecast future time series values one step at a time.

    The forecasting is converted to a tabular problem using the following features:

    - lag features (observed time series values) based on ``freq`` of the data
    - time features (e.g., day of the week) based on the timestamp of the measurement
    """

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
        name = name or re.sub(r"Model$", "", self.__class__.__name__)  # TODO: look name up from presets
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        self._lag_indices: List[int] = None
        self._time_features: List[TimeFeature] = None
        self._available_features: pd.Index = None
        self._drop_median_prediction = False

        # TODO: Should we always produce quantiles here?
        if self.eval_metric == "mean_wQuantileLoss":
            # Make sure that median is included in the quantiles - used for autoregressive generation
            if 0.5 not in self.quantile_levels:
                self._drop_median_prediction = True
                self.quantile_levels = sorted(self.quantile_levels + [0.5])
            self.tabular_predictor = TabularPredictor(
                label=self.target,
                problem_type=ag.constants.QUANTILE,
                quantile_levels=self.quantile_levels,
            )
        else:
            self.tabular_predictor = TabularPredictor(
                label=self.target,
                problem_type=ag.constants.REGRESSION,
                eval_metric=TS_METRIC_TO_TABULAR_METRIC[eval_metric],
            )

    def _get_features_dataframe(
        self,
        data: TimeSeriesDataFrame,
        last_k_values: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate a feature matrix used by TabularPredictor.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Dataframe containing features derived from time index & past time series values, as well as the target.
        last_k_values: int, optional
            If provided, features will be generated only for the last `last_k_values` timesteps of each time series.
        """
        dataframe_per_item = []
        if last_k_values is None:
            selected_slice = slice(None, None)
        else:
            selected_slice = slice(-last_k_values, None)

        # TODO: Parallelize with joblib
        for item_id in data.item_ids:
            series = data.loc[item_id][self.target]
            time_feature_columns = {
                feature.__class__.__name__: feature(series.index[selected_slice]) for feature in self._time_features
            }
            lag_columns = {f"lag_{idx}": series.shift(idx).values.ravel()[selected_slice] for idx in self._lag_indices}
            columns = {**time_feature_columns, **lag_columns, "target": series.values.ravel()[selected_slice]}
            df = pd.DataFrame(columns, index=series.index[selected_slice])
            dataframe_per_item.append(df.iloc[selected_slice])

        return pd.concat(dataframe_per_item, ignore_index=True)

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        if self.tabular_predictor._learner.is_fit:
            raise AssertionError(f"{self.name} has already been fit!")
        self._lag_indices = get_lags_for_frequency(train_data.freq)
        self._time_features = time_features_from_frequency_str(train_data.freq)

        train_data, _ = self._normalize_targets(train_data)
        train_df = self._get_features_dataframe(train_data)
        # Remove features that are completely missing in the training set
        train_df.dropna(axis=1, how="all", inplace=True)
        self._available_features = train_df.columns

        if val_data is not None:
            val_data, _ = self._normalize_targets(val_data)
            val_df = self._get_features_dataframe(val_data, last_k_values=self.prediction_length)
            val_df = val_df[self._available_features]
        else:
            logger.warning(
                f"No val_data was provided to {self.name}. "
                "TabularPredictor will generate a validation set without respecting the temporal ordering."
            )
            val_df = None

        # TODO: Select suitable presets for TabularPredictor
        # TODO: Set verbosity
        self.tabular_predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            time_limit=time_limit,
        )

    def predict(self, data: TimeSeriesDataFrame, quantile_levels: List[float] = None, **kwargs) -> TimeSeriesDataFrame:
        data, scale_per_item = self._normalize_targets(data)

        last_observed_timestamp = data.slice_by_timestep(-1, None).index.get_level_values(TIMESTAMP)
        offset = pd.tseries.frequencies.get_offset(data.freq)
        item_ids = data.item_ids
        nan_array = np.full(len(item_ids), fill_value=np.nan)

        predictions_list = []
        full_df = data

        # Autoregressively generate predictions
        for idx in range(1, self.prediction_length + 1):
            next_timestamp = last_observed_timestamp + idx * offset
            next_index = pd.MultiIndex.from_arrays([item_ids, next_timestamp])
            next_step_dummy = pd.DataFrame(nan_array, index=next_index, columns=[self.target])
            # Target for the next timestep (to be predicted) is set to NaN
            full_df_with_dummy = pd.concat([full_df, next_step_dummy])
            features = self._get_features_dataframe(full_df_with_dummy, last_k_values=1)
            preds = self.tabular_predictor.predict(features[self._available_features])
            if self.eval_metric == "mean_wQuantileLoss":
                next_values = preds[0.5].values
                preds.rename(str, axis=1, inplace=True)
                preds.insert(0, "mean", next_values)
            else:
                next_values = preds.values
                preds = preds.to_frame(name="mean")
                for q in self.quantile_levels:
                    preds[str(q)] = np.nan

            preds.index = next_index
            predictions_list.append(preds)

            # Use predictions for the next timestep as if they were actually observed
            next_values_with_index = pd.DataFrame(next_values, index=next_index, columns=[self.target])
            full_df = pd.concat([full_df, next_values_with_index])
            data = pd.concat([data, next_values_with_index])

        predictions = pd.concat(predictions_list).sort_index()
        if self._drop_median_prediction:
            predictions.drop("0.5", axis=1, inplace=True)
        predictions = self._rescale_targets(predictions, scale_per_item)
        return predictions

    def _normalize_targets(self, data: TimeSeriesDataFrame, min_scale=1e-5) -> Tuple[TimeSeriesDataFrame, pd.Series]:
        """Normalize data such that each the average absolute value of each time series is equal to 1."""
        scale_per_item = data.abs().groupby(ITEMID, sort=False)[self.target].mean().clip(lower=min_scale)
        normalized_data = data.copy()
        for col in normalized_data.columns:
            normalized_data[col] = normalized_data[col] / scale_per_item
        return normalized_data, scale_per_item

    def _rescale_targets(self, normalized_data: TimeSeriesDataFrame, scale_per_item: pd.Series) -> TimeSeriesDataFrame:
        """Scale all columns in the normalized dataframe back to original scale (inplace)."""
        data = normalized_data
        for col in data.columns:
            data[col] = data[col] * scale_per_item
        return data
