import logging
from typing import Any, Dict, Type

import pandas as pd

from .abstract_local_model import AbstractLocalModel

logger = logging.getLogger(__name__)


class AbstractStatsForecastModel(AbstractLocalModel):
    """Wrapper for StatsForecast models."""

    init_time_in_seconds = 15  # numba compilation for the first run

    def _update_local_model_args(self, local_model_args: Dict[str, Any]) -> Dict[str, Any]:
        seasonal_period = local_model_args.pop("seasonal_period")
        local_model_args["season_length"] = seasonal_period
        return local_model_args

    def _get_model_type(self) -> Type:
        raise NotImplementedError

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        model_type = self._get_model_type()
        model = model_type(**local_model_args)

        # Code does conversion between confidence levels and quantiles
        levels = []
        quantile_to_key = {}
        for q in self.quantile_levels:
            level = round(abs(q - 0.5) * 200, 1)
            suffix = "lo" if q < 0.5 else "hi"
            levels.append(level)
            quantile_to_key[str(q)] = f"{suffix}-{level}"
        levels = sorted(list(set(levels)))

        forecast = model.forecast(h=self.prediction_length, y=time_series.values.ravel(), level=levels)
        predictions = {"mean": forecast["mean"]}
        for q, key in quantile_to_key.items():
            predictions[q] = forecast[key]
        return pd.DataFrame(predictions)


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
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
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
        "approximation",
        "allowdrift",
        "allowmean",
        "seasonal_period",
    ]

    def _update_local_model_args(self, local_model_args: dict) -> dict:
        local_model_args = super()._update_local_model_args(local_model_args)
        local_model_args.setdefault("approximation", True)
        local_model_args.setdefault("allowmean", True)
        return local_model_args

    def _get_model_type(self):
        from statsforecast.models import AutoARIMA

        return AutoARIMA


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
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    allowed_local_model_args = [
        "model",
        "seasonal_period",
    ]

    def _get_model_type(self):
        from statsforecast.models import AutoETS

        return AutoETS


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
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    allowed_local_model_args = [
        "decomposition_type",
        "seasonal_period",
    ]

    def _get_model_type(self):
        from statsforecast.models import DynamicOptimizedTheta

        return DynamicOptimizedTheta


class ThetaModel(AbstractStatsForecastModel):
    """Theta forecasting model from Assimakopoulos and Nikolopoulos (2000).

    Based on `statsforecast.models.Theta <https://nixtla.github.io/statsforecast/models.html#theta>`_.


    References
    ----------
    Assimakopoulos, Vassilis, and Konstantinos Nikolopoulos.
    "The theta model: a decomposition approach to forecasting."
    International journal of forecasting 16.4 (2000): 521-530.


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
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    allowed_local_model_args = [
        "decomposition_type",
        "seasonal_period",
    ]

    def _get_model_type(self):
        from statsforecast.models import Theta

        return Theta
