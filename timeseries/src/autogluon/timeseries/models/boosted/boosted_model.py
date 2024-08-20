import time
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from autogluon.core.models import AbstractModel
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe


class BoostedRegressionModel(AbstractTimeSeriesModel):
    """Fit a regression model on the covariates and apply a forecasting model to the residuals."""

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
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        time_series_model_cls = self._get_time_series_model_cls(
            hyperparameters.pop("time_series_model", "SeasonalNaive")
        )
        # TODO: What paths / names to set for the base models?
        self.time_series_model = time_series_model_cls(
            path=f"{path}/TS",
            freq=freq,
            prediction_length=prediction_length,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters.pop("time_series_model_hyperparameters", {}),
            **kwargs,
        )
        tabular_model_cls = self._get_tabular_model_cls(model_name=hyperparameters.pop("tabular_model", "GBM"))
        self.tabular_model = tabular_model_cls(
            path=f"{path}/TAB",
            eval_metric=self.eval_metric.equivalent_tabular_regression_metric,
            hyperparameters=hyperparameters.pop("tabular_model_hyperparameters", {}),
            problem_type="regression",
        )

    def _get_time_series_model_cls(self, model_name: str) -> Type[AbstractTimeSeriesModel]:
        from autogluon.timeseries.models.presets import MODEL_TYPES

        if model_name not in MODEL_TYPES:
            raise ValueError(
                f"Time series model {model_name} is not supported. "
                f"Available time series models: {sorted(MODEL_TYPES)}"
            )
        return MODEL_TYPES[model_name]

    def _get_tabular_model_cls(self, model_name: str) -> Type[AbstractModel]:
        from autogluon.tabular.trainer.model_presets.presets import MODEL_TYPES

        if model_name not in MODEL_TYPES:
            raise ValueError(
                f"Tabular model {model_name} is not supported. " f"Available tabular models: {sorted(MODEL_TYPES)}"
            )
        return MODEL_TYPES[model_name]

    def _get_tabular_df(self, data: TimeSeriesDataFrame, static_features: Optional[pd.DataFrame] = None):
        tabular_df = pd.DataFrame(data).reset_index().drop(columns=["timestamp"] + self.metadata.past_covariates)
        if static_features is not None:
            tabular_df = pd.merge(data, static_features, on="item_id")
        return tabular_df

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> None:
        # TODO: Move scaler to autogluon.timeseries.utils.scaler
        from utils import StandardScaler

        self._scaler = StandardScaler(target=self.target)

        train_data = self._scaler.fit_transform(train_data)
        if val_data is not None:
            self._scaler.transform(val_data)
        train_df = self._get_tabular_df(train_data, static_features=train_data.static_features)
        X_train = train_df.drop(columns=[self.target])
        y_train = train_df[self.target]
        self.tabular_model.fit(X=X_train, y=y_train)

        y_pred_train = self.tabular_model.predict(X_train)
        train_residuals = train_data.assign(**{self.target: train_data[self.target] - y_pred_train})
        if val_data is not None and self.time_series_model_get_tags()["can_use_val_data"]:
            y_pred_val = self._tabular_predict(val_data, static_features=val_data.static_features)
            val_residuals = val_data.assign(**{self.target: val_data[self.target] - y_pred_val})
        else:
            val_residuals = None
        self.time_series_model.fit(train_data=train_residuals, val_data=val_residuals)

    def _tabular_predict(
        self,
        data: TimeSeriesDataFrame,
        static_features: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        # TODO: How to deal with past covariates?
        X = self._get_tabular_df(data, static_features=static_features)
        if self.target in X.columns:
            X = X.drop(columns=[self.target])
        return self.tabular_model.predict(X)

    def _predict(
        self,
        data: Union[TimeSeriesDataFrame, Dict[str, TimeSeriesDataFrame]],
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        data = self._scaler.fit_transform(data)
        y_pred_past = self._tabular_predict(data, static_features=data.static_features)
        if known_covariates is None:
            future_index = get_forecast_horizon_index_ts_dataframe(
                data, prediction_length=self.prediction_length, freq=self.freq
            )
            known_covariates = pd.DataFrame(columns=[self.target], index=future_index, dtype="float32")
        y_pred_future = self._tabular_predict(known_covariates, static_features=data.static_features)

        past_residuals = data.assign(**{self.target: data[self.target] - y_pred_past})
        forecast = self.time_series_model.predict(past_residuals, known_covariates=known_covariates)
        for col in forecast.columns:
            forecast[col] += y_pred_future
        forecast = self._scaler.inverse_transform(forecast)
        return forecast
