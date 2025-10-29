from typing import Callable

import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.utils.datetime import get_lags_for_frequency, get_time_features_for_frequency, norm_freq_str
from autogluon.timeseries.utils.datetime.seasonality import DEFAULT_SEASONALITIES
from autogluon.timeseries.utils.forecast import make_future_data_frame

from ..common import ALL_PANDAS_FREQUENCIES, to_supported_pandas_freq


@pytest.mark.parametrize("freq", ["D", "W", "ME", "QE", "YE", "h", "min", "s", "30min", "2h", "17s"])
@pytest.mark.parametrize("prediction_length", [1, 7])
def test_when_start_times_dont_match_freq_then_forecast_timestamps_are_correct(freq, prediction_length):
    freq = to_supported_pandas_freq(freq)
    item_ids_to_length = {"B": 14, "A": 12, "1": 7}
    start_timestamps = {
        "B": pd.Timestamp("2020-01-05 12:05:01"),
        "A": pd.Timestamp("2017-04-20 07:14:55"),
        "1": pd.Timestamp("1901-12-31 20:14:07"),
    }
    dfs = []
    for item_id, length in item_ids_to_length.items():
        timestamps = pd.date_range(start=start_timestamps[item_id], periods=length, freq=freq)
        index = pd.MultiIndex.from_product(
            [(item_id,), timestamps], names=[TimeSeriesDataFrame.ITEMID, TimeSeriesDataFrame.TIMESTAMP]
        )
        dfs.append(pd.DataFrame({"CustomTarget": np.random.rand(length)}, index=index))
    ts_dataframe = TimeSeriesDataFrame(pd.concat(dfs))

    prediction_index = pd.MultiIndex.from_frame(make_future_data_frame(ts_dataframe, prediction_length))
    preds = TimeSeriesDataFrame(pd.DataFrame({"mean": np.random.rand(len(prediction_index))}, index=prediction_index))
    offset = pd.tseries.frequencies.to_offset(freq)
    for item_id in ts_dataframe.item_ids:
        for i, timestamp in enumerate(preds.loc[item_id].index):
            assert timestamp == ts_dataframe.loc[item_id].index[-1] + (i + 1) * offset


@pytest.mark.parametrize("freq", ALL_PANDAS_FREQUENCIES)
@pytest.mark.parametrize("multiplier", ["", 1, 3])
def test_when_computing_seasonality_then_all_pandas_frequencies_are_supported(freq, multiplier):
    freq_str = f"{multiplier}{freq}"
    offset = pd.tseries.frequencies.to_offset(freq_str)
    offset_name = norm_freq_str(offset)
    assert offset_name in DEFAULT_SEASONALITIES


@pytest.mark.parametrize("freq", ALL_PANDAS_FREQUENCIES)
@pytest.mark.parametrize("multiplier", ["", 1, 3])
def test_when_computing_lags_then_all_pandas_frequencies_are_supported(freq, multiplier):
    freq_str = f"{multiplier}{freq}"
    lags = get_lags_for_frequency(freq_str)
    assert all(isinstance(lag, int) for lag in lags)


@pytest.mark.parametrize("freq", ALL_PANDAS_FREQUENCIES)
@pytest.mark.parametrize("multiplier", ["", 1, 3])
def test_when_computing_time_features_then_all_pandas_frequencies_are_supported(freq, multiplier):
    freq_str = f"{multiplier}{freq}"
    time_features = get_time_features_for_frequency(freq_str)
    assert all(isinstance(f, Callable) for f in time_features)
