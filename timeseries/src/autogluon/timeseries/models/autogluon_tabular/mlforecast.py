import logging
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import autogluon.core as ag
from autogluon.tabular import TabularPredictor
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.transformations import AbstractTransformer, Detrender, PipelineTransformer, StdScaler
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
from autogluon.timeseries.utils.seasonality import get_seasonality
from autogluon.timeseries.utils.warning_filters import statsmodels_warning_filter

logger = logging.getLogger(__name__)


class TabularEstimator(BaseEstimator):
    """Scikit-learn compatible interface for TabularPredictor."""

    _label_column_name = "y"

    def __init__(self, predictor_init_kwargs: Optional[dict] = None, predictor_fit_kwargs: Optional[dict] = None):
        self.predictor_init_kwargs = predictor_init_kwargs if predictor_init_kwargs is not None else {}
        self.predictor_fit_kwargs = predictor_fit_kwargs if predictor_fit_kwargs is not None else {}

    def get_params(self, deep: bool = True) -> dict:
        return {
            "predictor_init_kwargs": self.predictor_init_kwargs,
            "predictor_fit_kwargs": self.predictor_fit_kwargs,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabularEstimator":
        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)
        df = pd.concat([X, y.rename(self._label_column_name).to_frame()], axis=1)
        self.predictor = TabularPredictor(label=self._label_column_name, **self.predictor_init_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.predictor.fit(df, **self.predictor_fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert isinstance(X, pd.DataFrame)
        return self.predictor.predict(X)


class RecursiveTabularModel(AbstractTimeSeriesModel):
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
    detrend : bool, default = True
        If True, a linear trend will be removed from each time series before training the model. This is necessary
        since tree-based models cannot extrapolate outside of the time series values seen during training.
    scale : bool, default = True
        If True, time series values will be divided by the standard deviation.
    tabular_hyperparameters : Dict[Dict[str, Any]], optional
        Hyperparameters dictionary passed to `TabularPredictor.fit`. Contains the names of models that should be fit.
        Defaults to ``{"GBM": {}}``.

    """

    # TODO: Find good tabular presets
    # TODO: Add transforms - std scaling / detrending
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

    default_tabular_hyperparameters = {
        "GBM": {},
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
        from mlforecast import MLForecast

        self.mlf: Optional[MLForecast] = None
        self.transformer: Optional[PipelineTransformer] = None
        self.quantile_adjustments: Dict[str, float] = {}

    def _setup_transformer(self, model_params: dict) -> None:
        transformations = []
        # Avoid copying data multiple times
        if model_params.get("detrend", True):
            transformations.append(Detrender(target=self.target, copy=False))
        if model_params.get("scale", True):
            transformations.append(StdScaler(target=self.target, copy=False))
        self.transformer = PipelineTransformer(transformations, target=self.target, copy=True)

    @staticmethod
    def _get_date_features(freq: str) -> List[Callable]:
        # TODO: Use categorical variables for date features
        from gluonts.time_feature import time_features_from_frequency_str

        return time_features_from_frequency_str(freq)

    def _get_mlforecast_init_args(self, train_data: TimeSeriesDataFrame, model_params: dict) -> dict:
        from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str
        from mlforecast.target_transforms import Differences

        lags = model_params.get("lags")
        if lags is None:
            lags = get_lags_for_frequency(self.freq)

        date_features = model_params.get("date_features")
        if date_features is None:
            date_features = self._get_date_features(self.freq)

        differences = model_params.get("differences")
        if differences is None:
            differences = [get_seasonality(self.freq)]

        longest_ts_length = train_data.num_timesteps_per_item().max()
        if longest_ts_length <= sum(differences):
            logger.warning(
                f"Chosen differences {differences} require that time series have length "
                f">= sum(differences) (at least {sum(differences)}), "
                f"but longest time series length = {longest_ts_length}. Disabling differencing."
            )
            target_transforms = None
        else:
            target_transforms = [Differences(differences)]

        return {
            "lags": lags,
            "date_features": date_features,
            "target_transforms": target_transforms,
        }

    def _to_mlforecast_df(
        self,
        data: TimeSeriesDataFrame,
        static_features: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """Convert TimeSeriesDataFrame to a format expected by MLForecast methods `predict` and `preprocess`.

        Each row contains unique_id, ds, y, and (optionally) known covariates & static features.
        """
        # past_covariates & lags for known_covariates are not supported
        selected_columns = self.metadata.known_covariates_real.copy()
        column_name_mapping = {ITEMID: "unique_id", TIMESTAMP: "ds"}
        if include_target:
            selected_columns += [self.target]
            column_name_mapping[self.target] = "y"

        df = pd.DataFrame(data)[selected_columns].reset_index()
        if static_features is not None:
            df = pd.merge(df, static_features, how="left", on=ITEMID, suffixes=(None, "_static_feat"))
        return df.rename(columns=column_name_mapping)

    def _get_features_dataframe(
        self,
        data: TimeSeriesDataFrame,
        last_k_values: Optional[int] = None,
        return_X_y: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Construct feature matrix containing lags, covariates, and target time series values.

        Rows where the regression target equals NaN are dropped, but rows where the features are missing are kept.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Time series data that needs to be converted.
        last_k_values : int, optional
            If given, only last `last_k_values` rows will be kept for each time series.
        return_X_y : bool
            If False, the method will return a single dataframe containing both features and target. If True, the
            method will return the feature matrix and the regression target as a tuple (X, y).
        """
        df = self._to_mlforecast_df(data, data.static_features)
        # FIXME: keep_last_n produces a bug if time series too short -> manually select tail of each series
        features = self.mlf.preprocess(
            df,
            dropna=False,
            static_features=None,  # we handle static features in `_to_mlforecast_df`, without relying on MLForecast
        )
        if last_k_values is not None:
            features = features.groupby("unique_id", sort=False).tail(last_k_values)
        features.dropna(subset=self.mlf.ts.target_col, inplace=True)
        if return_X_y:
            return features[self.mlf.ts.features_order_], features[self.mlf.ts.target_col]
        else:
            return features[self.mlf.ts.features_order_ + [self.mlf.ts.target_col]]

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        self._check_fit_params()
        from mlforecast import MLForecast

        # Do not use external val_data as tuning_data to avoid overfitting
        train_data, val_data = train_data.train_test_split(self.prediction_length)
        model_params = self._get_model_params().copy()
        self._setup_transformer(model_params)
        train_data = self.transformer.fit_transform(train_data)
        val_data = self.transformer.fit_transform(val_data)

        # TabularEstimator is passed to MLForecast later to include tuning_data
        mlforecast_init_args = self._get_mlforecast_init_args(train_data, model_params)
        self.mlf = MLForecast(models={}, freq=self.freq, **mlforecast_init_args)

        tuning_data = self._get_features_dataframe(val_data, last_k_values=self.prediction_length)
        tabular_hyperparameters = model_params.get("tabular_hyperparameters", self.default_tabular_hyperparameters)
        estimator = TabularEstimator(
            predictor_init_kwargs={
                "path": self.path,
                "problem_type": ag.constants.REGRESSION,
                "eval_metric": self.TIMESERIES_METRIC_TO_TABULAR_METRIC[self.eval_metric],
                "verbosity": verbosity - 2,
            },
            predictor_fit_kwargs={
                "time_limit": time_limit,
                "hyperparameters": tabular_hyperparameters,
                "tuning_data": tuning_data,
            },
        )
        self.mlf.models = {"mean": estimator}

        X_train, y_train = self._get_features_dataframe(train_data, return_X_y=True)
        with statsmodels_warning_filter():
            self.mlf.fit_models(X_train, y_train)

        # Use residuals to compute quantiles
        val_data_future = val_data.slice_by_timestep(-self.prediction_length, None)
        val_forecast = self._predict_without_quantiles(train_data, val_data_future.drop(self.target, axis=1))
        residuals = val_forecast["mean"] - val_data_future[self.target]
        for q in self.quantile_levels:
            self.quantile_adjustments[q] = np.quantile(residuals, q)

    def _predict_without_quantiles(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame = None,
    ) -> pd.DataFrame:
        """Generate a point forecast of the future values using MLForecast.

        Returns
        -------
        predictions : pd.DataFrame
            Predictions with a single column "mean" containing the point forecast.
        """
        new_data = self._to_mlforecast_df(data, data.static_features)
        if known_covariates is not None:
            dynamic_dfs = [self._to_mlforecast_df(known_covariates, data.static_features, include_target=False)]
        else:
            dynamic_dfs = None
        with statsmodels_warning_filter():
            raw_predictions = self.mlf.predict(
                horizon=self.prediction_length,
                new_data=new_data,
                dynamic_dfs=dynamic_dfs,
            )
        predictions = raw_predictions.rename(columns={"unique_id": ITEMID, "ds": TIMESTAMP})
        forecast_index = get_forecast_horizon_index_ts_dataframe(data, self.prediction_length)
        return predictions.set_index([ITEMID, TIMESTAMP]).reindex(forecast_index)

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        data = self.transformer.fit_transform(data)
        predictions = self._predict_without_quantiles(data, known_covariates)
        for q in self.quantile_levels:
            predictions[str(q)] = predictions["mean"] + self.quantile_adjustments[q]
        return self.transformer.inverse_transform_predictions(TimeSeriesDataFrame(predictions))

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
