import logging
from typing import List, Type, Union

import pandas as pd

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.local.abstract_local_model import AbstractLocalModel
from autogluon.timeseries.utils.hashing import hash_ts_dataframe_items
from autogluon.timeseries.utils.warning_filters import statsmodels_warning_filter

logger = logging.getLogger(__name__)


class AbstractStatsForecastModel(AbstractLocalModel):
    """Wrapper for StatsForecast models.

    Cached predictions are stored inside the model to speed up validation & ensemble training downstream.

    Attributes
    ----------
    allowed_local_model_args : List[str]
        List of allowed arguments that can be passed to the underlying model.
        Arguments not in this list will be filtered out and not passed to the underlying model.
    """

    allowed_local_model_args: List[str] = []
    DEFAULT_N_JOBS: Union[float, int] = -1

    def get_model_type(self) -> Type:
        raise NotImplementedError

    def _fit(self, train_data, time_limit=None, verbosity=2, **kwargs) -> None:
        """Prepare hyperparameters that will be passed to the underlying model.

        As for all local models, actual fitting + predictions are delegated to the ``predict`` method.
        """
        # TODO: Find a way to ensure that SF models respect time_limit, e.g. https://docs.python.org/3/library/concurrent.futures.html
        # Fitting usually takes >= 15 seconds
        if time_limit is not None:
            if time_limit < 10:
                raise TimeLimitExceeded
            elif time_limit < 30:
                logger.warning(
                    f"Warning: {self.__class__.__name__} does not support early stopping "
                    f"and may exceed the remaining time_limit of {time_limit:.1f}s"
                )
        super()._fit(train_data=train_data, time_limit=time_limit, verbosity=verbosity, **kwargs)
        # seasonal_period is called season_length in StatsForecast
        self._local_model_args["season_length"] = self._local_model_args.pop("seasonal_period")
        return self

    def _to_statsforecast_dataframe(self, data: TimeSeriesDataFrame) -> pd.DataFrame:
        target = data[[self.target]]
        return target.reset_index().rename({ITEMID: "unique_id", TIMESTAMP: "ds", self.target: "y"}, axis=1)

    def _fit_and_cache_predictions(self, data: TimeSeriesDataFrame, **kwargs):
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

            with statsmodels_warning_filter():
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

    def hyperparameter_tune(self, **kwargs):
        # FIXME: multiprocessing.pool.ApplyResult.get() hangs inside StatsForecast.forecast if HPO enabled - needs investigation
        if self.n_jobs != 1:
            raise NotImplementedError(f"{self.__class__.__name__} does not support hyperparameter tuning.")


class AutoARIMAModel(AbstractStatsForecastModel):
    """Automatically tuned ARIMA model.

    Automatically selects the best (p,d,q,P,D,Q) model parameters using an information criterion

    Based on `statsforecast.models.AutoARIMA <https://nixtla.github.io/statsforecast/models.html#autoarima>`_.

    Other Parameters
    ----------------
    d : int, optional
        Order of first differencing. If None, will be determined automatically using a statistical test.
    D : int, optional
        Order of seasonal differencing. If None, will be determined automatically using a statistical test.
    max_p : int, default = 5
        Maximum number of autoregressive terms.
    max_q : int, default = 5
        Maximum order of moving average.
    max_P : int, default = 2
        Maximum number of seasonal autoregressive terms.
    max_Q : int, default = 2
        Maximum order of seasonal moving average.
    max_d : int, default = 2
        Maximum order of first differencing.
    max_D : int, default = 1
        Maximum order of seasonal differencing.
    start_p : int, default = 2
        Starting value of p in stepwise procedure.
    start_q : int, default = 2
        Starting value of q in stepwise procedure.
    start_P : int, default = 1
        Starting value of P in stepwise procedure.
    start_Q : int, default = 1
        Starting value of Q in stepwise procedure.
    stationary : bool, default = False
        Restrict search to stationary models.
    seasonal : bool, default = True
        Whether to consider seasonal models.
    approximation : bool, default = True
        Approximate optimization for faster convergence.
    allowdrift : bool, default = False
        If True, drift term is allowed.
    allowmean : bool, default = True
        If True, non-zero mean is allowed.
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, seasonality will be disabled.
    n_jobs : int or float, default = -1
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    """

    allowed_local_model_args = [
        "d",
        "D",
        "max_p",
        "max_q",
        "max_P",
        "max_Q",
        "max_d",
        "max_D",
        "start_p",
        "start_q",
        "start_P",
        "start_Q",
        "stationary",
        "seasonal",
        "approximatio",
        "allowdrift",
        "allowmean",
        "seasonal_period",
    ]

    def _update_local_model_args(self, local_model_args: dict, data: TimeSeriesDataFrame) -> dict:
        local_model_args.setdefault("approximation", True)
        local_model_args.setdefault("allowmean", True)
        return local_model_args

    def get_model_type(self):
        from statsforecast.models import AutoARIMA as AutoARIMA_

        return AutoARIMA_


class AutoETSModel(AbstractStatsForecastModel):
    """Automatically tuned exponential smoothing with trend and seasonality.

    Automatically selects the best ETS (Error, Trend, Seasonality) model using an information criterion

    Based on `statsforecast.models.AutoETS <https://nixtla.github.io/statsforecast/models.html#autoets>`_.

    Other Parameters
    ----------------
    model : str, default = "ZZZ"
        Model string describing the configuration of the E (error), T (trend) and S (seasonal) model components.
        Each component can be one of "M" (multiplicative), "A" (additive), "N" (omitted). For example when model="ANN"
        (additive error, no trend, and no seasonality), ETS will explore only a simple exponential smoothing.
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, seasonality will be disabled.
    n_jobs : int or float, default = -1
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    """

    allowed_local_model_args = [
        "model",
        "seasonal_period",
    ]

    def get_model_type(self):
        from statsforecast.models import AutoETS as AutoETS_

        return AutoETS_


class DynamicOptimizedThetaModel(AbstractStatsForecastModel):
    """Optimized Theta forecasting model from Fiorucci et al. (2016).

    Based on `statsforecast.models.DynamicOptimizedTheta <https://nixtla.github.io/statsforecast/models.html#dynamic-optimized-theta-method>`_.


    References
    ----------
    Fiorucci, Jose et al.
    "Models for optimising the theta method and their relationship to state space models."
    International journal of forecasting 32.4 (2016): 1151-1161.


    Other Parameters
    ----------------
    decomposition_type : {"multiplicative", "additive"}, default = "multiplicative"
        Seasonal decomposition type.
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, seasonality will be disabled.
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    """

    allowed_local_model_args = [
        "decomposition_type",
        "seasonal_period",
    ]

    def get_model_type(self):
        from statsforecast.models import DynamicOptimizedTheta

        return DynamicOptimizedTheta
