import pandas as pd


def get_forecast_horizon_timestamps(
    past_timestamps: pd.DatetimeIndex, freq: str, prediction_length: int
) -> pd.DatetimeIndex:
    start_ts = past_timestamps.max() + 1 * pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=start_ts, periods=prediction_length, freq=freq)
