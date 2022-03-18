from __future__ import annotations

from typing import Any, Tuple

import pandas as pd
from gluonts.dataset.common import ListDataset


class TimeSeriesDataFrame(pd.DataFrame):

    def __init__(self, data: Any, *args, **kwargs):
        # infer if data is a known type and attempt to cast to dataframe
        # for example:
        if isinstance(data, ListDataset):
            # todo: validate data in ListDataset
            # validate time series structure
            # assign indexes (index by item id and timestamp)
            data = self.from_gluonts(data)

        # todo: validate data DataFrame format
        super().__init__(data=data, *args, **kwargs)

        # todo: infer freq
        self.freq = None  # infer frequency string

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
