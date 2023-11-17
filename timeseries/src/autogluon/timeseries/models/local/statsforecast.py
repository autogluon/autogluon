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

    Based on `statsforecast.models.AutoARIMA <https://nixtla.mintlify.app/statsforecast/docs/models/autoarima.html>`_.

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


class ARIMAModel(AbstractStatsForecastModel):
    """Autoregressive Integrated Moving Average (ARIMA) model with fixed parameters.

    Based on `statsforecast.models.ARIMA <https://nixtla.github.io/statsforecast/models.html#arima>`_.


    Other Parameters
    ----------------
    order: Tuple[int, int, int], default = (1, 1, 1)
        The (p, d, q) order of the model for the number of AR parameters, differences, and MA parameters to use.
    seasonal_order: Tuple[int, int, int], default = (0, 0, 0)
        The (P, D, Q) parameters of the seasonal ARIMA model. Setting to (0, 0, 0) disables seasonality.
    include_mean : bool, default = True
        Should the ARIMA model include a mean term?
    include_drift : bool, default = False
        Should the ARIMA model include a linear drift term?
    include_constant : bool, optional
        If True, then includ_mean is set to be True for undifferenced series and include_drift is set to be True for
        differenced series.
    blambda : float, optional
        Box-Cox transformation parameter.
    biasadj : bool, default = False
        Use adjusted back-transformed mean Box-Cox.
    method : {"CSS-ML", "CSS", "ML"}, default = "CSS-ML"
        Fitting method: CSS (conditional sum of squares), ML (maximum likelihood), CSS-ML (initialize with CSS, then
        optimize with ML).
    fixed : Dict[str, float], optional
        Dictionary containing fixed coefficients for the ARIMA model.
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
        "order",
        "seasonal_order",
        "include_mean",
        "include_drift",
        "include_constant",
        "blambda",
        "biasadj",
        "method",
        "fixed",
        "seasonal_period",
    ]

    def _update_local_model_args(self, local_model_args: dict) -> dict:
        local_model_args = super()._update_local_model_args(local_model_args)
        local_model_args.setdefault("order", (1, 1, 1))
        return local_model_args

    def _get_model_type(self):
        from statsforecast.models import ARIMA

        return ARIMA


class AutoETSModel(AbstractStatsForecastModel):
    """Automatically tuned exponential smoothing with trend and seasonality.

    Automatically selects the best ETS (Error, Trend, Seasonality) model using an information criterion

    Based on `statsforecast.models.AutoETS <https://nixtla.mintlify.app/statsforecast/docs/models/autoets.html>`_.

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
    damped : bool, default = True
        Whether to dampen the trend.
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
        "damped",
        "model",
        "seasonal_period",
    ]

    def _get_model_type(self):
        from statsforecast.models import AutoETS

        return AutoETS

    def _update_local_model_args(self, local_model_args: dict) -> dict:
        local_model_args = super()._update_local_model_args(local_model_args)
        local_model_args.setdefault("model", "ZZZ")
        return local_model_args

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        # Disable seasonality if time series too short for chosen season_length, season_length is too high, or
        # season_length == 1. Otherwise model will crash
        season_length = local_model_args["season_length"]
        if len(time_series) < 2 * season_length or season_length == 1 or season_length > 24:
            # changing last character to "N" disables seasonality, e.g., model="AAA" -> model="AAN"
            local_model_args["model"] = local_model_args["model"][:-1] + "N"
        return super()._predict_with_local_model(time_series=time_series, local_model_args=local_model_args)


class ETSModel(AutoETSModel):
    """Exponential smoothing with trend and seasonality.

    The E (error), T (trend) and S (seasonal) components are fixed and provided by the user.

    This is an alias for `statsforecast.models.AutoETS <https://nixtla.mintlify.app/statsforecast/docs/models/autoets.html>`_.

    Other Parameters
    ----------------
    model : str, default = "AAA"
        Model string describing the configuration of the E (error), T (trend) and S (seasonal) model components.
        Each component can be one of "M" (multiplicative), "A" (additive), "N" (omitted). For example when model="ANN"
        (additive error, no trend, and no seasonality), ETS will explore only a simple exponential smoothing.
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, seasonality will be disabled.
    damped : bool, default = False
        Whether to dampen the trend.
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    def _update_local_model_args(self, local_model_args: dict) -> dict:
        local_model_args = super()._update_local_model_args(local_model_args)
        local_model_args.setdefault("model", "AAA")
        local_model_args.setdefault("damped", False)
        return local_model_args


class DynamicOptimizedThetaModel(AbstractStatsForecastModel):
    """Optimized Theta forecasting model [Fiorucci2016]_.

    Based on `statsforecast.models.DynamicOptimizedTheta <https://nixtla.github.io/statsforecast/models.html#dynamic-optimized-theta-method>`_.


    References
    ----------
    .. [Fiorucci2016] Fiorucci, Jose et al.
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
    """Theta forecasting model [Assimakopoulos2000]_.

    Based on `statsforecast.models.Theta <https://nixtla.mintlify.app/statsforecast/docs/models/autotheta.html>`_.


    References
    ----------
    .. [Assimakopoulos2000] Assimakopoulos, Vassilis, and Konstantinos Nikolopoulos.
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
