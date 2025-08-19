import logging
from typing import Any, Dict, Optional, Type

import numpy as np
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

    def _get_model_type(self, variant: Optional[str] = None) -> Type:
        raise NotImplementedError

    def _get_local_model(self, local_model_args: Dict):
        local_model_args = local_model_args.copy()
        variant = local_model_args.pop("variant", None)
        model_type = self._get_model_type(variant)
        return model_type(**local_model_args)

    def _get_point_forecast(
        self,
        time_series: pd.Series,
        local_model_args: Dict,
    ) -> np.ndarray:
        return self._get_local_model(local_model_args).forecast(
            h=self.prediction_length, y=time_series.values.ravel()
        )["mean"]

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        raise NotImplementedError


class AbstractProbabilisticStatsForecastModel(AbstractStatsForecastModel):
    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        levels, quantile_to_key = self._get_confidence_levels()

        forecast = self._get_local_model(local_model_args).forecast(
            h=self.prediction_length, y=time_series.values.ravel(), level=levels
        )
        predictions = {"mean": forecast["mean"]}
        for q, key in quantile_to_key.items():
            predictions[q] = forecast[key]
        return pd.DataFrame(predictions)

    def _get_confidence_levels(self) -> tuple[list[float], dict[str, str]]:
        """Get StatsForecast compatible levels from quantiles"""
        levels = []
        quantile_to_key = {}
        for q in self.quantile_levels:
            level = round(abs(q - 0.5) * 200, 1)
            suffix = "lo" if q < 0.5 else "hi"
            levels.append(level)
            quantile_to_key[str(q)] = f"{suffix}-{level}"
        levels = sorted(list(set(levels)))
        return levels, quantile_to_key


class AutoARIMAModel(AbstractProbabilisticStatsForecastModel):
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
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    default_priority = 60
    init_time_in_seconds = 0  # C++ models require no compilation
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

    def _get_model_type(self, variant: Optional[str] = None):
        from statsforecast.models import AutoARIMA

        return AutoARIMA


class ARIMAModel(AbstractProbabilisticStatsForecastModel):
    """Autoregressive Integrated Moving Average (ARIMA) model with fixed parameters.

    Based on `statsforecast.models.ARIMA <https://nixtla.mintlify.app/statsforecast/src/core/models.html#arima>`_.


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
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    default_priority = 10
    init_time_in_seconds = 0  # C++ models require no compilation
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

    def _get_model_type(self, variant: Optional[str] = None):
        from statsforecast.models import ARIMA

        return ARIMA


class AutoETSModel(AbstractProbabilisticStatsForecastModel):
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
    damped : bool, default = False
        Whether to dampen the trend.
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    default_priority = 70
    init_time_in_seconds = 0  # C++ models require no compilation
    allowed_local_model_args = [
        "damped",
        "model",
        "seasonal_period",
    ]

    def _get_model_type(self, variant: Optional[str] = None):
        from statsforecast.models import AutoETS

        return AutoETS

    def _update_local_model_args(self, local_model_args: dict) -> dict:
        local_model_args = super()._update_local_model_args(local_model_args)
        local_model_args.setdefault("model", "ZZZ")
        local_model_args.setdefault("damped", False)
        return local_model_args

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        # Disable seasonality if time series too short for chosen season_length, season_length is too high, or
        # season_length == 1. Otherwise model will crash
        season_length = local_model_args["season_length"]
        if len(time_series) < 2 * season_length or season_length == 1:
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
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    default_priority = 80

    def _update_local_model_args(self, local_model_args: dict) -> dict:
        local_model_args = super()._update_local_model_args(local_model_args)
        local_model_args.setdefault("model", "AAA")
        return local_model_args


class DynamicOptimizedThetaModel(AbstractProbabilisticStatsForecastModel):
    """Optimized Theta forecasting model [Fiorucci2016]_.

    Based on `statsforecast.models.DynamicOptimizedTheta <https://nixtla.mintlify.app/statsforecast/src/core/models.html#dynamic-optimized-theta-method>`_.


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
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    default_priority = 75
    allowed_local_model_args = [
        "decomposition_type",
        "seasonal_period",
    ]

    def _get_model_type(self, variant: Optional[str] = None):
        from statsforecast.models import DynamicOptimizedTheta

        return DynamicOptimizedTheta


class ThetaModel(AbstractProbabilisticStatsForecastModel):
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
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    default_priority = 75
    allowed_local_model_args = [
        "decomposition_type",
        "seasonal_period",
    ]

    def _get_model_type(self, variant: Optional[str] = None):
        from statsforecast.models import Theta

        return Theta


class AbstractConformalizedStatsForecastModel(AbstractStatsForecastModel):
    """Applies "naive pooled" conformalization to the model, where critical nonconformity scores
    are computed on the quantiles of absolute forecast residuals, which are "pooled" across the
    prediction horizon. The forecasts are then generated by adding corresponding offsets with
    critical nonconformity scores to the original mean forecast.

    This implementation uses the ``ceil((1 - alpha) * (n + 1)) / n`` quantile of nonconformity scores
    as the critical value. By definition, it follows that for small alpha and small n, this may result
    in "infinite" forecast intervals. In this case, the absolute residual quantiles are clipped to the
    sample maximum. This is why this method can be expected to undercover when the calibration window
    is short and the required alpha is small (forecasts for very low or high quantiles are required).
    """

    max_num_conformalization_windows = 5

    def _get_nonconformity_scores(
        self,
        time_series: pd.Series,
        local_model_args: Dict,
    ) -> np.ndarray:
        h = self.prediction_length
        y = time_series.values.ravel()

        if len(y) <= h:
            # if there is only prediction_length many time steps in sample, we fall back to
            # the naive-1 forecaster to compute residuals for as many time steps as possible
            nonconf_scores = np.abs(y - y[0])
            if len(y) > 1:
                # discard the first residual (0 by definition)
                nonconf_scores = np.full((h,), y[-1])
                nonconf_scores[: (len(y) - 1)] = y[1:]
            return nonconf_scores.reshape(1, -1)

        test_length = min(len(y) - 1, h * self.max_num_conformalization_windows)
        cutoffs = list(range(-h, -test_length - 1, -h))

        nonconf_scores = np.full((len(cutoffs), h), np.nan)
        for i, cutoff in enumerate(cutoffs, start=0):
            forecast = self._get_point_forecast(pd.Series(y[:cutoff]), local_model_args)
            forecast_horizon = y[cutoff:] if cutoff + h == 0 else y[cutoff : cutoff + h]
            nonconf_scores[i] = np.abs(forecast - forecast_horizon)

        return nonconf_scores

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        nonconf_scores = self._get_nonconformity_scores(time_series, local_model_args).ravel()

        # conformalize with naive pooling of nonconformity scores
        n = len(nonconf_scores)
        levels = np.array(self.quantile_levels)
        alpha = 1 - np.abs(2 * levels - 1)  # failure probabilities corresponding to quantiles
        q_sign = np.sign(2 * levels - 1)
        ehat = np.quantile(
            nonconf_scores,
            q=np.clip(
                np.ceil((1 - alpha) * (n + 1)) / n,
                a_min=0.0,
                a_max=1.0,
            ),
            method="lower",
        )

        point_forecast = self._get_point_forecast(time_series, local_model_args)
        predictions = {
            "mean": point_forecast,
            **{str(q): point_forecast + q_sign[i] * ehat[i] for i, q in enumerate(levels)},
        }
        return pd.DataFrame(predictions)


class AutoCESModel(AbstractProbabilisticStatsForecastModel):
    """Forecasting with an Complex Exponential Smoothing model where the model selection is performed using the
    Akaike Information Criterion [Svetunkov2022]_.

    Based on `statsforecast.models.AutoCES <https://nixtla.mintlify.app/statsforecast/docs/models/autoces.html>`_.


    References
    ----------
    .. [Svetunkov2022] Svetunkov, Ivan, Nikolaos Kourentzes, and John Keith Ord. "Complex exponential
        smoothing." Naval Research Logistics (NRL) 69.8 (2022): 1108-1123.


    Other Parameters
    ----------------
    model : {"Z", "N", "S", "P", "F"}, default = "Z"
        Defines type of CES model, "N" for simple CES, "S" for simple seasonality, "P" for partial seasonality
        (without complex part), "F" for full seasonality. When "Z" is selected, the best model is selected using
        Akaike Information Criterion (AIC).
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, seasonality will be disabled.
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    default_priority = 10
    allowed_local_model_args = [
        "model",
        "seasonal_period",
    ]

    def _get_model_type(self, variant: Optional[str] = None):
        from statsforecast.models import AutoCES

        return AutoCES

    def _update_local_model_args(self, local_model_args: dict) -> dict:
        local_model_args = super()._update_local_model_args(local_model_args)
        local_model_args.setdefault("model", "Z")
        return local_model_args

    def _get_point_forecast(self, time_series: pd.Series, local_model_args: Dict):
        # Disable seasonality if time series too short for chosen season_length or season_length == 1,
        # otherwise model will crash
        if len(time_series) < 5:
            # AutoCES does not handle "tiny" datasets, fall back to naive
            return np.full(self.prediction_length, time_series.values[-1])
        if len(time_series) < 2 * local_model_args["season_length"] + 1 or local_model_args["season_length"] == 1:
            local_model_args["model"] = "N"
        return super()._get_point_forecast(time_series, local_model_args)


class AbstractStatsForecastIntermittentDemandModel(AbstractConformalizedStatsForecastModel):
    def _update_local_model_args(self, local_model_args: Dict[str, Any]) -> Dict[str, Any]:
        _ = local_model_args.pop("seasonal_period")
        return local_model_args

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        # intermittent demand models clip their predictions at 0 or lower if the time series has lower values
        predictions = super()._predict_with_local_model(time_series=time_series, local_model_args=local_model_args)
        return predictions.clip(lower=min(0, time_series.min()))


class ADIDAModel(AbstractStatsForecastIntermittentDemandModel):
    """Intermittent demand forecasting model using the Aggregate-Dissagregate Intermittent
    Demand Approach [Nikolopoulos2011]_.

    Based on `statsforecast.models.ADIDA <https://nixtla.mintlify.app/statsforecast/docs/models/adida.html>`_.


    References
    ----------
    .. [Nikolopoulos2011] Nikolopoulos, K., Syntetos, A., Boylan, J. et al. An aggregate–disaggregate
        intermittent demand approach (ADIDA) to forecasting: an empirical proposition and analysis.
        J Oper Res Soc 62, 544–554 (2011). https://doi.org/10.1057/jors.2010.32


    Other Parameters
    ----------------
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    default_priority = 10

    def _get_model_type(self, variant: Optional[str] = None):
        from statsforecast.models import ADIDA

        return ADIDA


class CrostonModel(AbstractStatsForecastIntermittentDemandModel):
    """Intermittent demand forecasting model using Croston's model from [Croston1972]_ and [SyntetosBoylan2001]_.

    References
    ----------
    .. [Croston1972] Croston, John D. "Forecasting and stock control for intermittent demands." Journal of
        the Operational Research Society 23.3 (1972): 289-303.
    .. [SyntetosBoylan2001] Syntetos, Aris A., and John E. Boylan. "On the bias of intermittent
        demand estimates." International journal of production economics 71.1-3 (2001): 457-466.


    Other Parameters
    ----------------
    variant : {"SBA", "classic", "optimized"}, default = "SBA"
        Variant of the Croston model that is used. Available options:

        - ``"classic"`` - variant of the Croston method where the smoothing parameter is fixed to 0.1 (based on `statsforecast.models.CrostonClassic <https://nixtla.mintlify.app/statsforecast/docs/models/crostonclassic.html>`_)
        - ``"SBA"`` - variant of the Croston method based on Syntetos-Boylan Approximation (based on `statsforecast.models.CrostonSBA <https://nixtla.mintlify.app/statsforecast/docs/models/crostonsba.html>`_)
        - ``"optimized"`` - variant of the Croston method where the smoothing parameter is optimized (based on `statsforecast.models.CrostonOptimized <https://nixtla.mintlify.app/statsforecast/docs/models/crostonoptimized.html>`_)

    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    _aliases = ["CrostonSBA"]
    default_priority = 80
    allowed_local_model_args = [
        "variant",
    ]

    def _get_model_type(self, variant: Optional[str] = None):
        from statsforecast.models import CrostonClassic, CrostonOptimized, CrostonSBA

        model_variants = {
            "classic": CrostonClassic,
            "sba": CrostonSBA,
            "optimized": CrostonOptimized,
        }

        if not isinstance(variant, str) or variant.lower() not in model_variants:
            raise ValueError(
                f"Invalid model variant '{variant}'. Available Croston model variants: {list(model_variants)}"
            )
        else:
            return model_variants[variant.lower()]

    def _update_local_model_args(self, local_model_args: dict) -> dict:
        local_model_args = super()._update_local_model_args(local_model_args)
        local_model_args.setdefault("variant", "SBA")
        return local_model_args


class IMAPAModel(AbstractStatsForecastIntermittentDemandModel):
    """Intermittent demand forecasting model using the Intermittent Multiple Aggregation Prediction Algorithm
    [Petropoulos2015]_.


    Based on `statsforecast.models.IMAPA <https://nixtla.mintlify.app/statsforecast/docs/models/imapa.html>`_.


    References
    ----------
    .. [Petropoulos2015] Petropoulos, Fotios, and Nikolaos Kourentzes. "Forecast combinations for intermittent
        demand." Journal of the Operational Research Society 66.6 (2015): 914-924.


    Other Parameters
    ----------------
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    default_priority = 10

    def _get_model_type(self, variant: Optional[str] = None):
        from statsforecast.models import IMAPA

        return IMAPA


class ZeroModel(AbstractStatsForecastIntermittentDemandModel):
    """A naive forecaster that always returns 0 forecasts across the prediction horizon, where the prediction
    intervals are computed using conformal prediction.

    Other Parameters
    ----------------
    n_jobs : int or float, default = joblib.cpu_count(only_physical_cores=True)
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : int, default = 2500
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    default_priority = 100

    def _get_model_type(self, variant: Optional[str] = None):
        # ZeroModel does not depend on a StatsForecast implementation
        raise NotImplementedError

    def _get_point_forecast(
        self,
        time_series: pd.Series,
        local_model_args: Dict,
    ):
        return np.zeros(self.prediction_length)
