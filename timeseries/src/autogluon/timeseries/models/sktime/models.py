from sktime.forecasting.arima import AutoARIMA, ARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.theta import ThetaForecaster

from . import AbstractSktimeModel


class ThetaModel(AbstractSktimeModel):
    sktime_forecaster_class = ThetaForecaster
    sktime_allowed_init_args = ["initial_level", "deseasonalize", "sp"]


class TBATSModel(AbstractSktimeModel):
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


class AutoETSModel(AbstractSktimeModel):
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


class ARIMAModel(AbstractSktimeModel):
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


class AutoARIMAModel(AbstractSktimeModel):
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
