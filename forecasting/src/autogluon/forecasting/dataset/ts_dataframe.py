from __future__ import annotations

from typing import Any, Tuple

import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset


class TimeSeriesDataFrame(pd.DataFrame):

    def __init__(self, data: Any, *args, **kwargs):
        if isinstance(data, ListDataset):
            data = self.from_gluonts(data)
        self.validate(data)
        super().__init__(data=data, *args, **kwargs)

    @classmethod
    def sample_dataframe(cls):
        target = list(range(9))
        datetime_index = tuple(pd.date_range(pd.Timestamp("01-01-2019"), periods=3, freq='D'))
        item_ids = (0, 1, 2)
        multi_index = pd.MultiIndex.from_product([item_ids, datetime_index], names=['item_id', 'timestamp'])
        ts_df = pd.Series(target, name='target', index=multi_index).to_frame()
        return TimeSeriesDataFrame(ts_df)

    @classmethod
    def validate(cls, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, get {type(data)}")
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"data must have pd.MultiIndex, get {type(data.index)}")
        if 'target' not in data.columns:
            raise ValueError(f"data must have a column called target")
        if not data.index.dtypes.array[0] == np.dtype(np.int64):
            raise ValueError(f"for item_id, the only NumPy dtype allowed is ‘np.int64’.")
        if not data.index.dtypes.array[1] == np.dtype('datetime64[ns]'):
            raise ValueError(f"for timestamp, the only NumPy dtype allowed is ‘datetime64[ns]’.")

    @classmethod
    def from_gluonts(
            cls, list_dataset: ListDataset
    ) -> TimeSeriesDataFrame:
        all_ts = []
        for i, ts in enumerate(list_dataset):
            start_timestamp = ts['start']
            target = ts['target']
            datetime_index = tuple(pd.date_range(start_timestamp, periods=len(target)))
            idx = pd.MultiIndex.from_product([(i, ), datetime_index], names=['item_id', 'timestamp'])
            ts_df = pd.Series(target, name='target', index=idx).to_frame()
            all_ts.append(ts_df)
        return TimeSeriesDataFrame(pd.concat(all_ts))

    def split_by_time(
            self, cutoff_time: pd.Timestamp
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        nanosecond_before_cutoff = cutoff_time - pd.Timedelta(nanoseconds=1)
        data_before = self.loc[(slice(None), slice(None, nanosecond_before_cutoff)), :]
        data_after = self.loc[(slice(None), slice(cutoff_time, None)), :]
        return TimeSeriesDataFrame(data_before), TimeSeriesDataFrame(data_after)

    def split_by_item(
            self, cutoff_item: int
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        data_before = self.loc[(slice(None, cutoff_item-1), slice(None)), :]
        data_after = self.loc[(slice(cutoff_item, None), slice(None)), :]
        return TimeSeriesDataFrame(data_before), TimeSeriesDataFrame(data_after)

    def subsequence(
            self, start: pd.Timestamp, end: pd.Timestamp
    ) -> TimeSeriesDataFrame:
        nanosecond_before_end = end - pd.Timedelta(nanoseconds=1)
        return TimeSeriesDataFrame(self.loc[(slice(None), slice(start, nanosecond_before_end)), :])

