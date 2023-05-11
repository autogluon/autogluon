import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator

from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.seasonality import get_seasonality
from autogluon.timeseries.utils.warning_filters import statsmodels_warning_filter

logger = logging.getLogger(__name__)


class BaseMLForecastModel(AbstractTimeSeriesModel):
    """Base class for models based on the MLForecast library."""

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
        from mlforecast import MLForecast

        from .utils import StandardScaler

        self.mlf: Optional[MLForecast] = None
        self.required_ts_length: int = 1
        self.scaler: Optional[StandardScaler] = None
        self.residuals_std: float = 0.0

    @staticmethod
    def _get_date_features(freq: str) -> List[Callable]:
        # TODO: Use categorical variables for date features
        from gluonts.time_feature import time_features_from_frequency_str

        return time_features_from_frequency_str(freq)

    def _get_mlforecast_init_args(self, train_data: TimeSeriesDataFrame, model_params: dict) -> dict:
        from gluonts.time_feature import get_lags_for_frequency
        from mlforecast.target_transforms import Differences

        from .utils import StandardScaler

        lags = model_params.get("lags")
        if lags is None:
            lags = get_lags_for_frequency(self.freq)

        date_features = model_params.get("date_features")
        if date_features is None:
            date_features = self._get_date_features(self.freq)

        differences = model_params.get("differences")
        if differences is None:
            differences = [get_seasonality(self.freq)]

        ts_lengths = train_data.num_timesteps_per_item()
        required_ts_length = sum(differences) + 1
        all_train_ts_are_long_enough = ts_lengths.min() >= required_ts_length
        some_ts_available_for_validation = ts_lengths.max() >= required_ts_length + self.prediction_length
        if not (all_train_ts_are_long_enough and some_ts_available_for_validation):
            logger.warning(
                f"\tTime series in the dataset are too short for chosen differences {differences}. "
                f"Setting differences to [1]."
            )
            differences = [1]

        target_transforms = []
        if len(differences) > 0:
            target_transforms.append(Differences(differences))
        self.required_ts_length = sum(differences) + 1

        if model_params.get("standardize", True):
            self.scaler = StandardScaler()
            target_transforms.append(self.scaler)

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
        # FIXME: If unique_id column is not sorted, MLForecast will assign incorrect IDs to forecasts
        return df.rename(columns=column_name_mapping).sort_values(by="unique_id", kind="stable")

    def _get_features_and_target(
        self,
        data: TimeSeriesDataFrame,
        last_k_values: Optional[int] = None,
        max_num_samples: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Construct the feature matrix X and the respective target time series values y.

        Rows where the regression target equals NaN are dropped, but rows where the features are missing are kept.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Time series data that needs to be converted.
        last_k_values : int, optional
            If given, only last `last_k_values` rows will be kept for each time series.
        max_num_samples : int, optional
            If given, the output will contain at most this many rows.
        """
        item_ids_to_exclude = data.item_ids[data.num_timesteps_per_item() < self.required_ts_length]
        if len(item_ids_to_exclude) > 0:
            data = data.drop(item_ids_to_exclude, level=0)
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
        if max_num_samples is not None and len(features) > max_num_samples:
            rows_per_item = int(max_num_samples / data.num_items) + 1
            features = features.groupby("unique_id", sort=False).tail(rows_per_item)
        return features[self.mlf.ts.features_order_], features[self.mlf.ts.target_col]

    def _get_estimator(self, predictor_init_kwargs: dict, predictor_fit_kwargs: dict) -> BaseEstimator:
        """Get the estimator object that implements .fit(X, y) and .predict(X) methods."""
        raise NotImplementedError

    def _after_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

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

        # TabularEstimator is passed to MLForecast later to include tuning_data
        model_params = self._get_model_params().copy()
        mlforecast_init_args = self._get_mlforecast_init_args(train_data, model_params)
        self.mlf = MLForecast(models={}, freq=self.freq, **mlforecast_init_args)

        # Do not use external val_data as tuning_data to avoid overfitting
        max_num_samples = model_params.get("max_num_samples", 1_000_000)
        train_subset, val_subset = train_data.train_test_split(self.prediction_length)
        X_train, y_train = self._get_features_and_target(
            train_subset,
            max_num_samples=max_num_samples,
        )
        X_val, y_val = self._get_features_and_target(
            val_subset,
            last_k_values=self.prediction_length,
            max_num_samples=max_num_samples,
        )
        estimator = self._get_estimator(
            predictor_init_kwargs={
                "path": self.path,
                "verbosity": verbosity - 2,
            },
            predictor_fit_kwargs={
                "time_limit": time_limit,
                "hyperparameters": model_params.get("tabular_hyperparameters", {"GBM": {}}),
                "tuning_data": pd.concat([X_val, y_val], axis=1),
                **model_params.get("tabular_fit_kwargs", {}),
            },
        )
        self.mlf.models = {"mean": estimator}

        with statsmodels_warning_filter():
            self.mlf.fit_models(X_train, y_train)

        self._after_fit(X_train=X_train, y_train=y_train)

    def _get_mean_and_scale_per_item(self, item_ids: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """Extract the mean and scale for each time series from the scaler, if available."""
        if self.scaler is not None:
            stats = self.scaler.stats_.copy().reindex(item_ids)
            return stats["_mean"], stats["_std"]
        else:
            return pd.Series(0.0, index=item_ids), pd.Series(1.0, index=item_ids)

    def _predict_point_forecast(
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
        return predictions.set_index([ITEMID, TIMESTAMP]).reindex(data.item_ids, level=ITEMID)

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if (data.num_timesteps_per_item() < self.required_ts_length).any():
            # Raise a RuntimeError to avoid a numba segfault that kills the Python process
            raise RuntimeError(f"{self.name} requires that all time series have length >= {self.required_ts_length}")
        return TimeSeriesDataFrame(self._predict(data=data, known_covariates=known_covariates))

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
