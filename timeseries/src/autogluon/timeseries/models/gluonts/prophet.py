from autogluon.core.utils import warning_filter

with warning_filter():
    from gluonts.model.estimator import Estimator as GluonTSEstimator, DummyEstimator
    from gluonts.model.prophet import ProphetPredictor

from .abstract_gluonts import AbstractGluonTSModel


class _ProphetDummyEstimator(DummyEstimator):
    def train(self, train_data, validation_data=None, **kwargs):
        return self.predictor


class ProphetModel(AbstractGluonTSModel):
    """Wrapper around `Prophet <https://github.com/facebook/prophet>`_, which wraps the
    library through GluonTS's wrapper.

    In order to use it you need to install the package::

        pip install prophet

    Other Parameters
    ----------------
    hyperparameters : Dict[str, Any]
        Model hyperparameters that will be passed directly to the `Prophet`
        class. See Prophet documentation for available parameters.
    """

    gluonts_estimator_class = _ProphetDummyEstimator
    allowed_prophet_parameters = [
        "growth",
        "changepoints",
        "n_changepoints",
        "changepoint_range",
        "yearly_seasonality",
        "weekly_seasonality",
        "daily_seasonality",
        "holidays",
        "seasonality_mode",
        "seasonality_prior_scale",
        "holidays_prior_scale",
        "changepoint_prior_scale",
        "mcmc_samples",
        "interval_width",
        "uncertainty_samples",
    ]

    def _get_estimator(self) -> GluonTSEstimator:
        model_init_params = self._get_estimator_init_args()

        return _ProphetDummyEstimator(
            predictor_cls=ProphetPredictor,
            freq=model_init_params["freq"],
            prediction_length=model_init_params["prediction_length"],
            prophet_params={k: v for k, v in model_init_params.items() if k in self.allowed_prophet_parameters},
        )
