import warnings
from typing import Optional

import numpy as np
import pandas as pd

from autogluon.common.utils.deprecated_utils import Deprecated
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame


def get_forecast_horizon_index_single_time_series(
    past_timestamps: pd.DatetimeIndex, freq: str, prediction_length: int
) -> pd.DatetimeIndex:
    """Get timestamps for the next prediction_length many time steps of the time series with given frequency."""
    offset = pd.tseries.frequencies.to_offset(freq)
    if offset is None:
        raise ValueError(f"Invalid frequency: {freq}")
    start_ts = past_timestamps.max() + 1 * offset
    return pd.date_range(start=start_ts, periods=prediction_length, freq=freq, name=TimeSeriesDataFrame.TIMESTAMP)


@Deprecated(
    min_version_to_warn="1.3", min_version_to_error="2.0", new="TimeSeriesPredictor.forecast_horizon_data_frame"
)
def get_forecast_horizon_index_ts_dataframe(*args, **kwargs) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(make_future_data_frame(*args, **kwargs))


def make_future_data_frame(
    ts_dataframe: TimeSeriesDataFrame,
    prediction_length: int,
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """For each item in the dataframe, get timestamps for the next `prediction_length` time steps into the future.

    Returns a pandas.DataFrame, with columns "item_id" and "timestamp" corresponding to the forecast horizon.
    """
    indptr = ts_dataframe.get_indptr()
    last = ts_dataframe.index[indptr[1:] - 1].to_frame(index=False)
    item_ids = np.repeat(last[TimeSeriesDataFrame.ITEMID].to_numpy(), prediction_length)

    if freq is None:
        freq = ts_dataframe.freq
    offset = pd.tseries.frequencies.to_offset(freq)
    last_ts = pd.DatetimeIndex(last[TimeSeriesDataFrame.TIMESTAMP])
    # Non-vectorized offsets like BusinessDay may produce a PerformanceWarning - we filter them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        timestamps = np.dstack([last_ts + step * offset for step in range(1, prediction_length + 1)]).ravel()  # type: ignore[operator]
    return pd.DataFrame({TimeSeriesDataFrame.ITEMID: item_ids, TimeSeriesDataFrame.TIMESTAMP: timestamps})
