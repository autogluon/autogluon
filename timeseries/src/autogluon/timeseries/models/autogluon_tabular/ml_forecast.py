import logging
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator

from autogluon.tabular import TabularPredictor
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
from autogluon.timeseries.utils.seasonality import get_seasonality
from autogluon.timeseries.utils.warning_filters import statsmodels_warning_filter


logger = logging.getLogger(__name__)


class TabularEstimator(BaseEstimator):
    """Scikit-learn compatible interface for TabularPredictor."""
    label_column_name = "y"

    def __init__(self, predictor_init_kwargs: Optional[dict] = None, predictor_fit_kwargs: Optional[dict] = None):
        self.predictor_init_kwargs = predictor_init_kwargs if predictor_init_kwargs is not None else {}
        self.predictor_fit_kwargs = predictor_fit_kwargs if predictor_fit_kwargs is not None else {}

    def get_params(self, deep: bool = True):
        return {
            "predictor_init_kwargs": self.predictor_init_kwargs,
            "predictor_fit_kwargs": self.predictor_fit_kwargs,
        }

    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame)
        df = pd.concat([X, y.rename(self.label_column_name).to_frame()], axis=1)
        self.predictor = TabularPredictor(label=self.label_column_name, **self.predictor_init_kwargs)
        self.predictor.fit(df, **self.predictor_fit_kwargs)
        return self

    def predict(self, X):
        assert isinstance(X, pd.DataFrame)
        return self.predictor.predict(X)


class RecurrentTabularModel(AbstractTimeSeriesModel):
    """Predict time series values one by one using TabularPredictor.

    Other Parameters
    ----------------
    lags : List[int], default = None
        Lags of the target that will be used as features for predictions. If None, will be determined automatically
        based on the frequency of the data.
    differences : List[int], default = None
        Differences to take of the target before computing the features. These are restored at the forecasting step.
        If None, will be determined automatically based on the frequency of the data.

    """
    # TODO: Parse model kwargs
    # TODO: Add static features
    # TODO: Add time features
    # TODO: Add covariates
    # TODO: Add transforms
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from mlforecast import MLForecast
        from mlforecast.target_transforms import Differences
        from gluonts.time_feature import get_lags_for_frequency

        self.mlf = MLForecast(
            models={},
            lags=get_lags_for_frequency(self.freq),
            freq=self.freq,
            target_transforms=[Differences([get_seasonality(self.freq)])]
        )

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        d_train, d_val = train_data.train_test_split(self.prediction_length)
        tuning_data = self._prepare_data(d_val, last_k_values=self.prediction_length)

        estimator = TabularEstimator(
            predictor_init_kwargs={
                "problem_type": "regression",
                "eval_metric": "mean_absolute_error",
                "verbosity": 0,
            },
            predictor_fit_kwargs={
                "time_limit": time_limit,
                "hyperparameters": {"GBM": {}},
                "tuning_data": tuning_data,
            },
        )

        self.mlf.models = {"mean": estimator}

        X_train, y_train = self._prepare_data(d_train, return_X_y=True)
        with statsmodels_warning_filter():
            self.mlf.fit_models(X_train, y_train)

    def _to_mlforecast_df(self, data: TimeSeriesDataFrame) -> pd.DataFrame:
        return pd.DataFrame(data)[[self.target]].reset_index().rename(
            columns={"item_id": "unique_id", "timestamp": "ds", self.target: "y"}
        )

    def _prepare_data(self, data: TimeSeriesDataFrame, last_k_values: int = None, return_X_y: bool = False) -> pd.DataFrame:
        df = self._to_mlforecast_df(data)
        features = self.mlf.preprocess(df, dropna=False)
        if last_k_values is not None:
            features = features.groupby("unique_id", sort=False).tail(last_k_values)
        features.dropna(subset=self.mlf.ts.target_col, inplace=True)
        if return_X_y:
            return features[self.mlf.ts.features_order_], features[self.mlf.ts.target_col]
        else:
            return features[self.mlf.ts.features_order_ + [self.mlf.ts.target_col]]

    def predict(self, data: TimeSeriesDataFrame, known_covariates: TimeSeriesDataFrame = None, **kwargs) -> TimeSeriesDataFrame:
        with statsmodels_warning_filter():
            raw_predictions = self.mlf.predict(horizon=self.prediction_length, new_data=self._to_mlforecast_df(data))

        predictions = raw_predictions.rename(columns={"unique_id": ITEMID, "ds": TIMESTAMP})
        for q in self.quantile_levels:
            predictions[str(q)] = predictions["mean"]
        forecast_index = get_forecast_horizon_index_ts_dataframe(data, self.prediction_length)
        predictions = TimeSeriesDataFrame(predictions).reindex(forecast_index)
        return predictions
