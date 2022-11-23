import numpy as np
import pandas as pd
import pytest

from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe


@pytest.mark.parametrize("freq", ["H", "min", "S", "D", "W", "M", "Q", "Y", "3H", "17S"])
@pytest.mark.parametrize("prediction_length", [1, 7])
def test_when_start_times_dont_match_freq_then_forecast_timestamps_are_correct(freq, prediction_length):
    item_ids_to_length = {"B": 14, "A": 12, "1": 7}
    start_timestamps = {
        "B": pd.Timestamp("2020-01-05 12:05:01"),
        "A": pd.Timestamp("2017-04-20 07:14:55"),
        "1": pd.Timestamp("1901-12-31 20:14:07"),
    }
    dfs = []
    for item_id, length in item_ids_to_length.items():
        timestamps = pd.date_range(start=start_timestamps[item_id], periods=length, freq=freq)
        index = pd.MultiIndex.from_product([(item_id,), timestamps], names=[ITEMID, TIMESTAMP])
        dfs.append(pd.DataFrame({"CustomTarget": np.random.rand(length)}, index=index))
    ts_dataframe = TimeSeriesDataFrame(pd.concat(dfs))

    prediction_index = get_forecast_horizon_index_ts_dataframe(ts_dataframe, prediction_length)
    preds = TimeSeriesDataFrame(pd.DataFrame({"mean": np.random.rand(len(prediction_index))}, index=prediction_index))
    offset = pd.tseries.frequencies.to_offset(freq)
    for item_id in ts_dataframe.item_ids:
        for i, timestamp in enumerate(preds.loc[item_id].index):
            assert timestamp == ts_dataframe.loc[item_id].index[-1] + (i + 1) * offset
