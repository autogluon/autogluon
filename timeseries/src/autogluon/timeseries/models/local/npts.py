import pandas as pd
from autogluon.timeseries.models.local.abstract_local_model import AbstractLocalModel


class NPTSModel(AbstractLocalModel):
    """Non-Parametric Time Series Forecaster.

    This models is especially well suited for forecasting sparse or intermittent time series with many zero values.

    Based on `gluonts.model.npts.NPTSPredictor <https://ts.gluon.ai/stable/api/gluonts/gluonts.model.npts.html>`_.
    See GluonTS documentation for more information about the model.

    Other Parameters
    ----------------
    kernel_type : {"exponential", "uniform"}, default = "exponential"
        Kernel used by the model.
    exp_kernel_weights : float, default = 1.0
        Scaling factor used in the exponential kernel.
    use_seasonal_variant : bool, default = True
        Whether to use the seasonal variant of the model.
    """

    allowed_local_model_args = [
        "kernel_type",
        "exp_kernel_weights",
        "use_seasonal_model",
    ]

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        from gluonts.model.npts import NPTSPredictor

        predictor = NPTSPredictor(
            freq=self.freq,
            prediction_length=self.prediction_length,
            **local_model_args,
        )
        ts = time_series.copy(deep=False)
        ts.index = ts.index.to_period()
        forecast = predictor.predict_time_series(ts, num_samples=100)
        forecast_dict = {"mean": forecast.mean}
        for q in self.quantile_levels:
            forecast_dict[str(q)] = forecast.quantile(q)
        return pd.DataFrame(forecast_dict)
