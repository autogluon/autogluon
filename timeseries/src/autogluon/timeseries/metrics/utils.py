import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import ITEMID


def _get_seasonal_diffs(*, y_past: pd.Series, seasonal_period: int = 1) -> pd.Series:
    return y_past.groupby(level=ITEMID, sort=False).diff(seasonal_period).abs()


def _in_sample_abs_seasonal_error(*, y_past: pd.Series, seasonal_period: int = 1) -> pd.Series:
    """Compute seasonal naive forecast error (predict value from seasonal_period steps ago) for each time series."""
    seasonal_diffs = _get_seasonal_diffs(y_past=y_past, seasonal_period=seasonal_period)
    return seasonal_diffs.groupby(level=ITEMID, sort=False).mean().fillna(1.0)


def _in_sample_squared_seasonal_error(*, y_past: pd.Series, seasonal_period: int = 1) -> pd.Series:
    seasonal_diffs = _get_seasonal_diffs(y_past=y_past, seasonal_period=seasonal_period)
    return seasonal_diffs.pow(2.0).groupby(level=ITEMID, sort=False).mean().fillna(1.0)
