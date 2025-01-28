import warnings
from typing import Optional

import numpy as np
import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame


def get_forecast_horizon_index_single_time_series(
    past_timestamps: pd.DatetimeIndex, freq: str, prediction_length: int
) -> pd.DatetimeIndex:
    """Get timestamps for the next prediction_length many time steps of the time series with given frequency."""
    offset = pd.tseries.frequencies.to_offset(freq)
    if offset is None:
        raise ValueError(f"Invalid frequency: {freq}")
    start_ts = past_timestamps.max() + 1 * offset
    return pd.date_range(start=start_ts, periods=prediction_length, freq=freq, name=TIMESTAMP)


# TODO: Deprecate this method, add this functionality to `TimeSeriesPredictor`
def get_forecast_horizon_index_ts_dataframe(
    ts_dataframe: TimeSeriesDataFrame,
    prediction_length: int,
    freq: Optional[str] = None,
) -> pd.MultiIndex:
    """For each item in the dataframe, get timestamps for the next `prediction_length` time steps into the future.

    Returns a pandas.MultiIndex, where
    - level 0 ("item_id") contains the same item_ids as the input ts_dataframe.
    - level 1 ("timestamp") contains the next prediction_length time steps starting from the end of each time series.
    """
    last = ts_dataframe.reset_index()[[ITEMID, TIMESTAMP]].groupby(by=ITEMID, sort=False, as_index=False).last()
    item_ids = np.repeat(last[ITEMID], prediction_length)

    if freq is None:
        freq = ts_dataframe.freq
    offset = pd.tseries.frequencies.to_offset(freq)
    last_ts = pd.DatetimeIndex(last[TIMESTAMP])
    # Non-vectorized offsets like BusinessDay may produce a PerformanceWarning - we filter them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        timestamps = np.dstack([last_ts + step * offset for step in range(1, prediction_length + 1)]).ravel()  # type: ignore[operator]
    return pd.MultiIndex.from_arrays([item_ids, timestamps], names=[ITEMID, TIMESTAMP])
