from __future__ import annotations

from typing import Any, Tuple, Dict, List
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class TimeSeriesListDataset:
    data_iter: List[Dict[str, Any]]
    freq: str


class TimeSeriesDataFrame(pd.DataFrame):
    """TimeSeriesDataFrame to represent time-series dataset.

    Parameters
    ----------
    data : Any
        Time-series data to construct a TimeSeriesDataFrame.
        It currently supports two formats:
            1. Time-series data in TimeSeriesListDataset format.
            2. Time-series data pd.DataFrame format with multi-index on item and timestamp.
            An example can be found using example_dataframe() function.
    """

    def __init__(self, data: Any, *args, **kwargs):
        if isinstance(data, TimeSeriesListDataset):
            data = self.from_list_dataset(data)
        self._validate_data_frame(data)
        super().__init__(data=data, *args, **kwargs)

    @classmethod
    def example_dataframe(cls):
        """An example TimeSeriesDataFrame.

        Returns
        -------
        ts_df : TimeSeriesDataFrame
            It returns an example TimeSeriesDataFrame as:
                                    target
            item_id timestamp
            0       2019-01-01       0
                    2019-01-02       1
                    2019-01-03       2
            1       2019-01-01       3
                    2019-01-02       4
                    2019-01-03       5
            2       2019-01-01       6
                    2019-01-02       7
                    2019-01-03       8
        """

        target = list(range(9))
        datetime_index = tuple(
            pd.date_range(pd.Timestamp("01-01-2019"), periods=3, freq="D")
        )
        item_ids = (0, 1, 2)
        multi_index = pd.MultiIndex.from_product(
            [item_ids, datetime_index], names=["item_id", "timestamp"]
        )
        ts_df = TimeSeriesDataFrame(
            pd.Series(target, name="target", index=multi_index).to_frame()
        )
        return ts_df

    @classmethod
    def _validate_data_frame(cls, data: pd.DataFrame):
        """Validate a pd.DataFrame can be converted to TimeSeriesDataFrame

        Parameters:
        -----------
        data : pd.DataFrame
            a data frame in pd.DataFrame format.
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, get {type(data)}")
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"data must have pd.MultiIndex, get {type(data.index)}")
        if "target" not in data.columns:
            raise ValueError(f"data must have a column called target")
        if not data.index.dtypes.array[0] == np.dtype(np.int64):
            raise ValueError(
                f"for item_id, the only NumPy dtype allowed is ‘np.int64’."
            )
        if not data.index.dtypes.array[1] == np.dtype("datetime64[ns]"):
            raise ValueError(
                f"for timestamp, the only NumPy dtype allowed is ‘datetime64[ns]’."
            )

    @classmethod
    def _validate_list_dataset(cls, list_dataset):
        if not isinstance(list_dataset, TimeSeriesListDataset):
            raise ValueError(
                f"list_dataset must be a TimeSeriesListDataset, get {type(list_dataset)}"
            )
        if len(list_dataset.data_iter) == 0:
            raise ValueError(f"list_dataset has no time-series.")
        for i, ts in enumerate(list_dataset.data_iter):
            if not isinstance(ts, dict):
                raise ValueError(
                    f"{i}'th time-series in list_dataset must be a dict, get{type(ts)}"
                )
            if not ("target" in ts and "start" in ts):
                raise ValueError(
                    f"{i}'th time-series in list_dataset must have 'target' and 'start', get{ts.keys()}"
                )

    @classmethod
    def from_list_dataset(
        cls, list_dataset: TimeSeriesListDataset
    ) -> TimeSeriesDataFrame:
        """Convert TimeSeriesListDataset to TimeSeriesDataFrame.

        Parameters:
        -----------
        list_dataset : TimeSeriesListDataset
            Data in TimeSeriesListDataset format. For example:
            example_ts_list_dataset = TimeSeriesListDataset(
                data_iter=[
                    {"target": [0, 1, 2], "start": pd.Timestamp("01-01-2019")},
                    {"target": [3, 4, 5], "start": pd.Timestamp("01-01-2019")},
                    {"target": [6, 7, 8], "start": pd.Timestamp("01-01-2019")}
                ],
                freq='D'
            )

        Returns:
        --------
        ts_df : TimeSeriesDataFrame
            A data frame in TimeSeriesDataFrame format.
        """

        cls._validate_list_dataset(list_dataset)
        all_ts = []
        for i, ts in enumerate(list_dataset.data_iter):
            start_timestamp = ts["start"]
            target = ts["target"]
            datetime_index = tuple(
                pd.date_range(
                    start_timestamp, periods=len(target), freq=list_dataset.freq
                )
            )
            idx = pd.MultiIndex.from_product(
                [(i,), datetime_index], names=["item_id", "timestamp"]
            )
            ts_df = pd.Series(target, name="target", index=idx).to_frame()
            all_ts.append(ts_df)
        ts_df = TimeSeriesDataFrame(pd.concat(all_ts))
        return ts_df

    def split_by_time(
        self, cutoff_time: pd.Timestamp
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Split dataframe by a cutoff_time.

        Parameters
        ----------
        cutoff_time : pd.Timestamp
            The time to Split the current data frame into two data frames, all in TimeSeriesDataFrame format.

        Returns
        -------
        data_before : TimeSeriesDataFrame
            The first one after split contains time-series before the cutoff_time (exclude cutoff_time).
        data_after : TimeSeriesDataFrame
            The second one after split contains time-series after the cutoff_time (include cutoff_time).
        """

        nanosecond_before_cutoff = cutoff_time - pd.Timedelta(nanoseconds=1)
        data_before = self.loc[(slice(None), slice(None, nanosecond_before_cutoff)), :]
        data_after = self.loc[(slice(None), slice(cutoff_time, None)), :]
        return TimeSeriesDataFrame(data_before), TimeSeriesDataFrame(data_after)

    def split_by_item(
        self, cutoff_item: int
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Split dataframe by an item_id cutoff_item.

        Parameters
        ----------
        cutoff_item : int
            The item_id to Split the current data frame into two data frames, all in TimeSeriesDataFrame format.

        Returns
        -------
        data_before : TimeSeriesDataFrame
            The first one after split contains time-series before the cutoff_item (exclude cutoff_item).
        data_after : TimeSeriesDataFrame
            The second one after split contains time-series after the cutoff_item (include cutoff_item).
        """

        data_before = self.loc[(slice(None, cutoff_item - 1), slice(None)), :]
        data_after = self.loc[(slice(cutoff_item, None), slice(None)), :]
        return TimeSeriesDataFrame(data_before), TimeSeriesDataFrame(data_after)

    def subsequence(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> TimeSeriesDataFrame:
        """Extract time-series between start (inclusive) and end (exclusive) time.

        Parameters
        ----------
        start : pd.Timestamp
            The start time (inclusive) of a time range that will be used for subsequence.
        end : pd.Timestamp
            The end time (exclusive) of a time range that will be used for subsequence.

        Returns
        -------
        ts_df : TimeSeriesDataFrame
            A new data frame in TimeSeriesDataFrame format contains time-series in a time range
            defined between start and end time.
        """

        if end < start:
            raise ValueError(f"end time {end} is earlier than stat time {start}")

        nanosecond_before_end = end - pd.Timedelta(nanoseconds=1)
        ts_df = TimeSeriesDataFrame(
            self.loc[(slice(None), slice(start, nanosecond_before_end)), :]
        )
        return ts_df
