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

    def get_series_with_timestamps_per_item(group: pd.DataFrame) -> pd.DataFrame:
        timestamps = group.index.get_level_values(TIMESTAMP)
        return get_forecast_horizon_index_single_time_series(
            past_timestamps=timestamps, freq=ts_dataframe.freq, prediction_length=prediction_length
        ).to_frame()

    return ts_dataframe.groupby(ITEMID, sort=False).apply(get_series_with_timestamps_per_item).index
