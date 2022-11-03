import logging

from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.theta import ThetaForecaster

from . import AbstractSktimeModel

logger = logging.getLogger(__name__)


class ThetaSktimeModel(AbstractSktimeModel):
    """Theta model for forecasting.

    Based on `sktime.forecasting.theta.ThetaForecaster <https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.forecasting.theta.ThetaForecaster.html>`_


    Other Parameters
    ----------------
    initial_level : float or None, default = None
        The alpha value of the simple exponential smoothing, if the value is set then
        this will be used, otherwise it will be estimated from the data.
    seasonal: bool, default = True
        If True, data is seasonally adjusted.
    seasonal_period: int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For
        example, 4 for quarterly data with an annual cycle, or 7 for daily data with a
        weekly cycle.
        When set to None, seasonal_period will be inferred from the frequency of the
        training data. Can also be specified manually by providing an integer > 1.
    fail_if_misconfigured: bool, default = False
        If True, the model will raise an exception and fail when given an invalid
        configuration (e.g., selected seasonality is incompatible with seasonal_period).
        If False, the model will instead raise a warning and try to adjust the
        configuration (e.g., turn off seasonality).
        Setting this parameter to True is useful during HPO to avoid training multiple
        models that all fall back to the same configuration.
    """

    sktime_forecaster_class = ThetaForecaster
    sktime_allowed_init_args = ["initial_level", "deseasonalize", "sp"]

    def _get_sktime_forecaster_init_args(self, min_length: int, inferred_period: int = 1):
        sktime_init_args = self._get_model_params().copy()
        fail_if_misconfigured = sktime_init_args.pop("fail_if_misconfigured", False)
        # Below code handles both cases when seasonal_period is set to None explicitly & implicitly
        seasonal_period = sktime_init_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = inferred_period
        seasonal = sktime_init_args.pop("seasonal", True)
        sktime_init_args["deseasonalize"] = seasonal
        sktime_init_args["sp"] = seasonal_period

        if seasonal and (min_length < 2 * seasonal_period or seasonal_period <= 1):
            error_message = (
                f"{self.name} with seasonal = {seasonal} requires training series of length "
                f"at least 2 * seasonal_period and seasonal_period > 1 "
                f"(received min_length = {min_length} and seasonal_period = {seasonal_period})."
            )
            if fail_if_misconfigured:
                raise ValueError(error_message)
            else:
                logger.warning(error_message + "\nSetting seasonal to False.")
                sktime_init_args["deseasonalize"] = False
                sktime_init_args["sp"] = 1
        return sktime_init_args


class TBATSSktimeModel(AbstractSktimeModel):
    """TBATS forecaster with multiple seasonalities.

    This model automatically tries all combinations of hyperparameters (e.g.,
    use_box_cox, use_trend, use_arma_errors), and selects the best model

    Based on `sktime.forecasting.tbats.TBATS <https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.forecasting.tbats.TBATS.html>`_

    Caution: The fitting time for this model can be very long, and the saved model can
    take up a lot of disk space when applied to large datasets.


    Other Parameters
    ----------------
    use_box_cox: bool or None, default = None
        Whether to use the Box-Cox transform of the data.
        When None, both options are considered and the best one is chosen based on AIC.
    use_trend: bool or None, default = None
        Whether to use a trend component.
        When None, both options are considered and the best one is chosen based on AIC.
    use_damped_trend: bool or None, default = None
        Whether to damp the trend component.
        When None, both options are considered and the best one is chosen based on AIC.
    use_arma_erros: bool or None, default = None
        Whether to model the residuals with ARMA.
        When None, both options are considered and the best one is chosen based on AIC.
    seasonal_period: int, float, array or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For
        example, 4 for quarterly data with an annual cycle, or 7 for daily data with a
        weekly cycle.
        When set to None, seasonal_period will be inferred from the frequency of the
        training data. Setting to 1 disables seasonality.
        It's possible to capture multiple trend components by setting seasonal_period
        to an array of frequencies.

    Other parameters listed in `sktime_allowed_init_args` can be passed to the underlying
    sktime model. See docstring of `sktime.forecasting.tbats.TBATS` for their description.
    """

    sktime_forecaster_class = TBATS
    sktime_allowed_init_args = [
        "use_box_cox",
        "box_cox_bounds",
        "use_trend",
        "use_damped_trend",
        "sp",
        "use_arma_errors",
        "show_warnings",
        "n_jobs",
        "multiprocessing_start_method",
        "context",
    ]

    def _get_sktime_forecaster_init_args(self, min_length: int, inferred_period: int = 1):
        sktime_init_args = self._get_model_params().copy()
        seasonal_period = sktime_init_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = inferred_period

        if seasonal_period == 1:
            sktime_init_args["sp"] = None
        else:
            sktime_init_args["sp"] = seasonal_period
        return sktime_init_args


class AutoETSSktimeModel(AbstractSktimeModel):
    """AutoETS model from sktime.

    See `AbstractSktimeModel` for common parameters.

    Other Parameters
    ----------------
    error: str, default = "add"
        Error model. Allowed values are "add" and "mul".
    trend: str or None, default = None
        Trend component model. Allowed values are "add", "mul" and None.
    damped_trend: bool, default = False
        Whether or not the included trend component is damped.
    seasonal: str or None, default = "add"
        Seasonality model. Allowed values are "add", "mul" and None.
    seasonal_period: int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For
        example, 4 for quarterly data with an annual cycle, or 7 for daily data with a
        weekly cycle.
        When set to None, seasonal_period will be inferred from the frequency of the
        training data. Can also be specified manually by providing an integer > 1.
    initialization_method: str, default = "estimated"
        Method for initializing the model parameters. Allowed values:

        * "estimated" - learn parameters from the data with maximum likelihood
        * "heuristic" - select parameters with a heuristic. Faster than "estimated" but
        can be less accurate and requires a series with at least 8 elements
    fail_if_misconfigured: bool, default = False
        If True, the model will raise an exception and fail when given an invalid
        configuration (e.g., selected seasonality is incompatible with seasonal_period).
        If False, the model will instead raise a warning and try to adjust the
        configuration (e.g., turn off seasonality).
        Setting this parameter to True is useful during HPO to avoid training multiple
        models that all fall back to the same configuration.

    Other parameters listed in `sktime_allowed_init_args` can be passed to the underlying
    sktime model. See docstring of `sktime.forecasting.ets.AutoETS` for their description.
    """

    sktime_forecaster_class = AutoETS
    sktime_allowed_init_args = [
        "error",
        "trend",
        "damped_trend",
        "seasonal",
        "sp",
        "initialization_method",
        "initial_level",
        "initial_trend",
        "initial_seasonal",
        "bounds",
        "dates",
        "freq",
        "missing",
        "start_params",
        "maxiter",
        "full_output",
        "disp",
        "callback",
        "return_params",
        "auto",
        "information_criterion",
        "allow_multiplicative_trend",
        "restrict",
        "additive_only",
        "ignore_inf_ic",
        "n_jobs",
        "random_state",
    ]

    def _get_sktime_forecaster_init_args(self, min_length: int, inferred_period: int = 1):
        sktime_init_args = self._get_model_params().copy()
        fail_if_misconfigured = sktime_init_args.pop("fail_if_misconfigured", False)

        if min_length < 8 and sktime_init_args.get("initialization_method") == "heuristic":
            error_message = f"{self.name}: Training series too short for initialization_method='heuristic'."
            if fail_if_misconfigured:
                raise ValueError(error_message)
            else:
                logger.warning(error_message + "\nFalling back to initialization_method='estimated'")
                sktime_init_args["initialization_method"] = "estimated"

        seasonal = sktime_init_args.pop("seasonal", "add")
        if seasonal not in ["add", "mul", None]:
            raise ValueError(f"Invalid seasonal {seasonal} for model {self.name} (must be one of 'add', 'mul', None)")

        seasonal_period = sktime_init_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = inferred_period

        sktime_init_args["seasonal"] = seasonal
        sktime_init_args["sp"] = seasonal_period

        # Check if seasonality and seasonal_period are compatible
        if seasonal in ["add", "mul"]:
            if min_length < 2 * seasonal_period or seasonal_period <= 1:
                error_message = (
                    f"{self.name} with seasonal = {seasonal} requires training series of length "
                    f"at least 2 * seasonal_period and seasonal_period > 1 "
                    f"(received min_length = {min_length} and seasonal_period = {seasonal_period})."
                )
                if fail_if_misconfigured:
                    raise ValueError(error_message)
                else:
                    logger.warning(error_message + "\nSetting seasonality to None.")
                    sktime_init_args["seasonal"] = None
                    sktime_init_args["sp"] = 1
        return sktime_init_args


class ARIMASktimeModel(AbstractSktimeModel):
    """ARIMA model from sktime.

    See `AbstractSktimeModel` for common parameters.

    Other Parameters
    ----------------
    order: Tuple[int, int, int], default = (1, 0, 0)
        The (p, d, q) order of the model for the number of AR parameters, differences,
        and MA parameters to use.
    seasonal_order: Tuple[int, int, int], default = (1, 0, 1)
        The (P, D, Q) parameters of the seasonal ARIMA model. Setting to (0, 0, 0)
        disables seasonality.
    seasonal_period: int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For
        example, 4 for quarterly data with an annual cycle or 7 for daily data with a
        weekly cycle.
        When set to None, seasonal period will be inferred from the frequency of the
        training data. Can also be specified manually by providing an integer > 1.
    fail_if_misconfigured: bool, default = False
        If True, the model will raise an exception and fail when given an invalid
        configuration (e.g., selected seasonal_order is incompatible with seasonal_period).
        If False, the model will instead raise a warning and try to adjust the
        configuration (e.g., turn off seasonality).
        Setting this parameter to True is useful during HPO to avoid training multiple
        models that all fall back to the same configuration.

    Other parameters listed in `sktime_allowed_init_args` can be passed to the underlying
    sktime model. See docstring of `sktime.forecasting.arima.ARIMA` for their description.
    """

    sktime_forecaster_class = ARIMA
    sktime_allowed_init_args = [
        "order",
        "seasonal_order",
        "start_params",
        "method",
        "maxiter",
        "suppress_warnings",
        "out_of_sample_size",
        "scoring",
        "scoring_args",
        "trend",
        "with_intercept",
        "time_varying_regression",
        "enforce_stationarity",
        "enforce_invertibility",
        "simple_differencing",
        "measurement_error",
        "mle_regression",
        "hamilton_representation",
        "concentrate_scale",
    ]

    def _get_sktime_forecaster_init_args(self, min_length: int, inferred_period: int = 1):
        sktime_init_args = self._get_model_params().copy()
        fail_if_misconfigured = sktime_init_args.pop("fail_if_misconfigured", False)
        seasonal_order = sktime_init_args.pop("seasonal_order", (1, 0, 1))
        seasonal_period = sktime_init_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = inferred_period

        seasonal_order_is_valid = len(seasonal_order) == 3 and all(isinstance(p, int) for p in seasonal_order)
        if not seasonal_order_is_valid:
            raise ValueError(
                f"{self.name} can't interpret received seasonal_order {seasonal_order} as a "
                "tuple with 3 nonnegative integers (P, D, Q)."
            )

        sktime_init_args["seasonal_order"] = tuple(seasonal_order) + (seasonal_period,)

        if seasonal_period <= 1 and any(s > 0 for s in seasonal_order):
            error_message = (
                f"{self.name} with seasonal_order {seasonal_order} expects "
                f"seasonal_period > 1 (received seasonal_period = {seasonal_period})."
            )
            if fail_if_misconfigured:
                raise ValueError(error_message)
            else:
                logger.warning(error_message + "\nSetting seasonal_order to (0, 0, 0).")
                sktime_init_args["seasonal_order"] = (0, 0, 0, 0)

        return sktime_init_args


class AutoARIMASktimeModel(AbstractSktimeModel):
    """AutoARIMA model from sktime.

    This model automatically selects the (p, d, q) and (P, D, Q) parameters of ARIMA by
    fitting multiple models with different configurations and choosing the best one
    based on the AIC criterion.

    Other Parameters
    ----------------
    seasonal_period: int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For
        example, 4 for quarterly data with an annual cycle or 7 for daily data with a
        weekly cycle.
        When set to None, seasonal period will be inferred from the frequency of the
        training data. Can also be specified manually by providing an integer > 1.

    Other parameters listed in `sktime_allowed_init_args` can be passed to the underlying
    sktime model. See docstring of `sktime.forecasting.arima.AutoARIMA` for their description.
    """

    sktime_forecaster_class = AutoARIMA
    sktime_allowed_init_args = [
        "start_p",
        "d",
        "start_q",
        "max_p",
        "max_d",
        "max_q",
        "start_P",
        "D",
        "start_Q",
        "max_P",
        "max_D",
        "max_Q",
        "max_order",
        "sp",
        "seasonal",
        "stationary",
        "information_criterion",
        "alpha",
        "test",
        "seasonal_test",
        "stepwise",
        "n_jobs",
        "start_params",
        "trend",
        "method",
        "maxiter",
        "offset_test_args",
        "seasonal_test_args",
        "suppress_warnings",
        "error_action",
        "trace",
        "random",
        "random_state",
        "n_fits",
        "out_of_sample_size",
        "scoring",
        "scoring_args",
        "with_intercept",
        "time_varying_regression",
        "enforce_stationarity",
        "enforce_invertibility",
        "simple_differencing",
        "measurement_error",
        "mle_regression",
        "hamilton_representation",
        "concentrate_scale",
    ]

    def _get_sktime_forecaster_init_args(self, min_length: int, inferred_period: int = 1):
        sktime_init_args = self._get_model_params().copy()
        seasonal_period = sktime_init_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = inferred_period

        sktime_init_args["sp"] = seasonal_period
        return sktime_init_args
