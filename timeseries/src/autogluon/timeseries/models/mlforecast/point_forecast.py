from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import autogluon.core as ag
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame

from .base import BaseMLForecastModel


class PointRecursiveTabularModel(BaseMLForecastModel):
    """Predict time series values one by one using TabularPredictor.

    Based on the `mlforecast`<https://github.com/Nixtla/mlforecast>_ library.


    Other Parameters
    ----------------
    lags : List[int], default = None
        Lags of the target that will be used as features for predictions. If None, will be determined automatically
        based on the frequency of the data.
    date_features : List[Union[str, Callable]], default = None
        Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input.
        If None, will be determined automatically based on the frequency of the data.
    differences : List[int], default = None
        Differences to take of the target before computing the features. These are restored at the forecasting step.
        If None, will be set to ``[seasonal_period]``, where seasonal_period is determined based on the data frequency.
    standardize : bool, default = True
        If True, time series values will be divided by the standard deviation.
    tabular_hyperparameters : Dict[Dict[str, Any]], optional
        Hyperparameters dictionary passed to ``TabularPredictor.fit``. Contains the names of models that should be fit.
        Defaults to ``{"GBM": {}}``.
    tabular_fit_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to ``TabularPredictor.fit``. Defaults to an empty dict.
    max_num_samples : int, default = 1_000_000
        If given, training and validation datasets will contain at most this many rows.

    """

    # TODO: Find good tabular presets
    # TODO: Use sample_weight to align metrics with Tabular
    # TODO: Add lag_transforms

    TIMESERIES_METRIC_TO_TABULAR_METRIC = {
        "MASE": "mean_absolute_error",
        "MAPE": "mean_absolute_percentage_error",
        "sMAPE": "mean_absolute_percentage_error",
        "mean_wQuantileLoss": "mean_absolute_error",
        "MSE": "mean_squared_error",
        "RMSE": "root_mean_squared_error",
    }

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

    def _get_estimator(self, predictor_init_kwargs: dict, predictor_fit_kwargs: dict) -> BaseEstimator:
        from .utils import TabularRegressor

        predictor_init_kwargs["problem_type"] = ag.constants.REGRESSION
        predictor_init_kwargs["eval_metric"] = self.TIMESERIES_METRIC_TO_TABULAR_METRIC[self.eval_metric]
        return TabularRegressor(
            predictor_init_kwargs=predictor_init_kwargs,
            predictor_fit_kwargs=predictor_fit_kwargs,
        )

    def _after_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.residuals_std = ((self.mlf.models_["mean"].predict(X_train) - y_train).values ** 2).mean()

    def _add_gaussian_quantiles(self, predictions: pd.DataFrame, scale_per_item: pd.Series) -> pd.DataFrame:
        """Add quantile predictions on top of the point forecast.

        We assume that the residuals N steps into the future follow Gaussian distribution with zero mean and scale

            std(observed_time_series) * sqrt(1 + N) * residuals_std

        where residuals_std is the standard deviation of the residuals estimated on the internal validation set.

        Parameters
        ----------
        predictions : pd.DataFrame
            Point forecast of the future values.
        scale_per_item : pd.Series
            Scale of each time series.
        """
        from scipy.stats import norm

        num_items = int(len(predictions) / self.prediction_length)
        sqrt_h = np.sqrt(np.arange(1, self.prediction_length + 1))
        # Series where normal_scale_per_timestep.loc[item_id].loc[N] = sqrt(1 + N) for N in range(prediction_length)
        normal_scale_per_timestep = pd.Series(np.tile(sqrt_h, num_items), index=predictions.index)

        std_per_timestep = self.residuals_std * scale_per_item * normal_scale_per_timestep
        for q in self.quantile_levels:
            predictions[str(q)] = predictions["mean"] + norm.ppf(q) * std_per_timestep
        return predictions

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame = None,
    ) -> TimeSeriesDataFrame:
        predictions = self._predict_point_forecast(data, known_covariates)
        _, scale_per_item = self._get_mean_and_scale_per_item(data.item_ids)
        return self._add_gaussian_quantiles(predictions, scale_per_item)
