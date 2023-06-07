import logging
import math
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator

import autogluon.core as ag
from autogluon.tabular import TabularPredictor
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
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
        return self.predictor.predict(X).values


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
    standardize : bool, default = True
        If True, time series values will be standardized by subtracting mean & dividing by standard deviation.
    tabular_hyperparameters : Dict[Dict[str, Any]], optional
        Hyperparameters dictionary passed to ``TabularPredictor.fit``. Contains the names of models that should be fit.
        Defaults to ``{"GBM": {}}``.
    tabular_fit_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to ``TabularPredictor.fit``. Defaults to an empty dict.
    max_num_samples : int, default = 1_000_000
        If given, training and validation datasets will contain at most this many rows (starting from the end of each
        series).

    """

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

        from .utils import StandardScaler

        self.mlf: Optional[MLForecast] = None
        self.scaler: Optional[StandardScaler] = None
        self.required_ts_length: int = 1
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

    def _get_features_dataframe(
        self,
        data: TimeSeriesDataFrame,
        last_k_values: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Construct feature matrix containing lags, covariates, and target time series values.

        Rows where the regression target equals NaN are dropped, but rows where the features are missing are kept.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Time series data that needs to be converted.
        last_k_values : int, optional
            If given, only last `last_k_values` rows will be kept for each time series.
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
        del self.mlf.ts.features_
        if last_k_values is not None:
            features = features.groupby("unique_id", sort=False).tail(last_k_values)
        features.dropna(subset=self.mlf.ts.target_col, inplace=True)
        features = features.reset_index(drop=True)
        return features[self.mlf.ts.features_order_], features[self.mlf.ts.target_col]

    @staticmethod
    def _subsample_data_to_avoid_oom(data: TimeSeriesDataFrame, max_num_rows: int = 30_000_000) -> TimeSeriesDataFrame:
        """Subsample time series from the dataset to avoid out of memory errors inside MLForecast.preprocess."""
        if len(data) > max_num_rows:
            item_ids = data.item_ids
            num_items_to_keep = math.ceil(len(item_ids) * max_num_rows / len(data))
            items_to_keep = np.random.choice(item_ids, num_items_to_keep, replace=False)
            logger.debug(
                f"\tRandomly selected {num_items_to_keep} ({num_items_to_keep / len(item_ids):.1%}) time series "
                "to limit peak memory usage"
            )
            data = data.query("item_id in @items_to_keep")
        return data

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

        # TODO: Find a better way to ensure that the model does not run out of memory that check available RAM
        train_data = self._subsample_data_to_avoid_oom(train_data)

        # Do not use external val_data as tuning_data to avoid overfitting
        train_subset, val_subset = train_data.train_test_split(self.prediction_length)

        max_num_samples = model_params.get("max_num_samples", 1_000_000)
        max_rows_per_item = math.ceil(max_num_samples / train_data.num_items)
        X_train, y_train = self._get_features_dataframe(train_subset, last_k_values=max_rows_per_item)
        X_val, y_val = self._get_features_dataframe(
            val_subset, last_k_values=min(self.prediction_length, max_rows_per_item)
        )

        estimator = TabularEstimator(
            predictor_init_kwargs={
                "path": self.path + os.sep + "point_predictor",
                "problem_type": ag.constants.REGRESSION,
                "eval_metric": self.TIMESERIES_METRIC_TO_TABULAR_METRIC[self.eval_metric],
                "verbosity": verbosity - 2,
            },
            predictor_fit_kwargs={
                "tuning_data": pd.concat([X_val, y_val], axis=1),
                "time_limit": time_limit,
                "hyperparameters": model_params.get("tabular_hyperparameters", {"GBM": {}}),
                **model_params.get("tabular_fit_kwargs", {}),
            },
        )
        self.mlf.models = {"mean": estimator}

        with statsmodels_warning_filter():
            self.mlf.fit_models(X_train, y_train)

        self.residuals_std = (self.mlf.models_["mean"].predict(X_train) - y_train).std()

    def _predict_with_mlforecast(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame = None,
    ) -> pd.DataFrame:
        """Generate a point forecast with MLForecast.

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
        return predictions.set_index([ITEMID, TIMESTAMP])

    def _get_scale_per_item(self, item_ids: pd.Index) -> pd.Series:
        """Extract the 'std' values from the scaler object, if available."""
        if self.scaler is not None:
            return self.scaler.stats_["_std"].copy().reindex(item_ids)
        else:
            return pd.Series(1.0, index=item_ids)

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if (data.num_timesteps_per_item() < self.required_ts_length).any():
            # Raise a RuntimeError to avoid a numba segfault that kills the Python process
            raise RuntimeError(f"{self.name} requires that all time series have length >= {self.required_ts_length}")
        predictions = self._predict_with_mlforecast(data=data, known_covariates=known_covariates)
        scale_per_item = self._get_scale_per_item(predictions.index.unique(level=ITEMID))

        num_items = int(len(predictions) / self.prediction_length)
        sqrt_h = np.sqrt(np.arange(1, self.prediction_length + 1))
        # Series where normal_scale_per_timestep.loc[item_id].loc[N] = sqrt(1 + N) for N in range(prediction_length)
        normal_scale_per_timestep = pd.Series(np.tile(sqrt_h, num_items), index=predictions.index)

        std_per_timestep = self.residuals_std * scale_per_item * normal_scale_per_timestep
        for q in self.quantile_levels:
            predictions[str(q)] = predictions["mean"] + norm.ppf(q) * std_per_timestep
        return TimeSeriesDataFrame(predictions.reindex(data.item_ids, level=ITEMID))

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
