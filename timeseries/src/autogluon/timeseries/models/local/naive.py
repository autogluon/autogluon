import numpy as np
import pandas as pd

from autogluon.timeseries.models.local.abstract_local_model import (
    AbstractLocalModel,
    get_quantile_function,
    seasonal_naive_forecast,
)


class NaiveModel(AbstractLocalModel):
    """Baseline model that sets the forecast equal to the last observed value.

    Quantiles are obtained by assuming that the residuals follow zero-mean normal distribution, scale of which is
    estimated from the empirical distribution of the residuals.
    As described in https://otexts.com/fpp3/prediction-intervals.html

    Other Parameters
    ----------------
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    """

    allowed_local_model_args = ["seasonal_period"]

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        return seasonal_naive_forecast(
            target=time_series.values.ravel(),
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            seasonal_period=1,
        )

    def _more_tags(self) -> dict:
        return {"allow_nan": True}


class SeasonalNaiveModel(AbstractLocalModel):
    """Baseline model that sets the forecast equal to the last observed value from the same season.

    Quantiles are obtained by assuming that the residuals follow zero-mean normal distribution, scale of which is
    estimated from the empirical distribution of the residuals.
    As described in https://otexts.com/fpp3/prediction-intervals.html


    Other Parameters
    ----------------
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, will fall back to Naive forecast.
        Seasonality will also be disabled, if the length of the time series is < seasonal_period.
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    """

    allowed_local_model_args = ["seasonal_period"]

    def _predict_with_local_model(
        self,
        time_series: np.ndarray,
        local_model_args: dict,
    ) -> pd.DataFrame:
        return seasonal_naive_forecast(
            target=time_series.values.ravel(),
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            seasonal_period=local_model_args["seasonal_period"],
        )

    def _more_tags(self) -> dict:
        return {"allow_nan": True}


class AverageModel(AbstractLocalModel):
    """Baseline model that sets the forecast equal to the historical average or quantile.

    Other Parameters
    ----------------
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : Optional[int], default = None
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    allowed_local_model_args = ["seasonal_period"]
    default_max_ts_length = None

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        agg_functions = ["mean"] + [get_quantile_function(q) for q in self.quantile_levels]
        stats_marginal = time_series.agg(agg_functions)
        stats_repeated = np.tile(stats_marginal.values, [self.prediction_length, 1])
        return pd.DataFrame(stats_repeated, columns=stats_marginal.index)

    def _more_tags(self) -> dict:
        return {"allow_nan": True}


class SeasonalAverageModel(AbstractLocalModel):
    """Baseline model that sets the forecast equal to the historical average or quantile in the same season.

    Other Parameters
    ----------------
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, will fall back to Naive forecast.
        Seasonality will also be disabled, if the length of the time series is < seasonal_period.
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    max_ts_length : Optional[int], default = None
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    """

    allowed_local_model_args = ["seasonal_period"]
    default_max_ts_length = None

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        seasonal_period = local_model_args["seasonal_period"]
        agg_functions = ["mean"] + [get_quantile_function(q) for q in self.quantile_levels]

        # Compute mean & quantiles for each season
        ts_df = time_series.reset_index(drop=True).to_frame()
        ts_df["season"] = ts_df.index % seasonal_period
        stats_per_season = ts_df.groupby("season")[self.target].agg(agg_functions)

        next_season = ts_df["season"].iloc[-1] + 1
        season_in_forecast_horizon = np.arange(next_season, next_season + self.prediction_length) % seasonal_period
        result = stats_per_season.reindex(season_in_forecast_horizon)

        if np.any(result.isna().values):
            # Use statistics over all timesteps to fill values for seasons that are missing from training data
            stats_marginal = time_series.agg(agg_functions)
            result = result.fillna(stats_marginal)
        return result

    def _more_tags(self) -> dict:
        return {"allow_nan": True}
