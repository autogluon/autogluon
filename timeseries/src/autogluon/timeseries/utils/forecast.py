import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame


def get_forecast_horizon_index_single_time_series(
    past_timestamps: pd.DatetimeIndex, freq: str, prediction_length: int
) -> pd.DatetimeIndex:
    """Get timestamps for the next prediction_length many time steps of the time series with given frequency."""
    start_ts = past_timestamps.max() + 1 * pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=start_ts, periods=prediction_length, freq=freq, name=TIMESTAMP)


def get_forecast_horizon_index_ts_dataframe(
    ts_dataframe: TimeSeriesDataFrame, prediction_length: int
) -> pd.MultiIndex:
    """For each item in the dataframe, get timestamps for the next prediction_length many time steps into the future.

    Returns a pandas.MultiIndex, where
    - level 0 ("item_id") contains the same item_ids as the input ts_dataframe.
    - level 1 ("timestamp") contains the next prediction_length time steps starting from the end of each time series.
    """

    timestamps = ts_dataframe.reset_index(level=TIMESTAMP)[TIMESTAMP]
    last_ts = timestamps.groupby(level=ITEMID, sort=False).tail(1)
    offset = pd.tseries.frequencies.to_offset(ts_dataframe.freq)

    def get_index_single_item(item_id, cutoff):
        return pd.DataFrame(
            {
                ITEMID: [item_id] * prediction_length,
                TIMESTAMP: pd.date_range(start=cutoff + offset, freq=offset, periods=prediction_length),
            }
        )

    index_per_item = []
    for item_id, cutoff in last_ts.items():
        index_per_item.append(get_index_single_item(item_id, cutoff))
    return pd.MultiIndex.from_frame(pd.concat(index_per_item))
