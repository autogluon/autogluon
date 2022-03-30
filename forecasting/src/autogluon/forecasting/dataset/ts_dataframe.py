from __future__ import annotations

from typing import Any, Tuple, Dict, List
from dataclasses import dataclass
from collections import UserList

import pandas as pd
import numpy as np


@dataclass
class TimeSeriesListDataset(UserList):
    """TimeSeriesListDataset to represent time-series dataset.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        List of time series represented by dict. The time series dict should have a
        'target' key with list of observation as values and a 'start' key with a timestamp as value.
    freq : str
        Frequency string as in pandas offset aliases.

    An example is as following:

    ts_list = TimeSeriesListDataset(
        data=[
            {"target": [0, 1, 2], "start": START_TIMESTAMP},
            {"target": [3, 4, 5], "start": START_TIMESTAMP},
            {"target": [6, 7, 8], "start": START_TIMESTAMP},
        ],
        freq="D",
    )
    """

    data: List[Dict[str, Any]]
    freq: str


class TimeSeriesDataFrame(pd.DataFrame):
    """TimeSeriesDataFrame to represent time-series dataset.

    Parameters
    ----------
    data : Any
        Time-series data to construct a TimeSeriesDataFrame.
        It currently supports three input formats:

            1. Time-series data in TimeSeriesListDataset format. For example:

                TimeSeriesListDataset(
                        data=[
                            {"target": [0, 1, 2], "start": START_TIMESTAMP},
                            {"target": [3, 4, 5], "start": START_TIMESTAMP},
                            {"target": [6, 7, 8], "start": START_TIMESTAMP},
                        ],
                        freq="D",
                    )
            2. Time-series data in pd.DataFrame format without multi-index. For example:

                   item_id  timestamp  target
                0        0 2019-01-01       0
                1        0 2019-01-02       1
                2        0 2019-01-03       2
                3        1 2019-01-01       3
                4        1 2019-01-02       4
                5        1 2019-01-03       5
                6        2 2019-01-01       6
                7        2 2019-01-02       7
                8        2 2019-01-03       8

            3. Time-series data in pd.DataFrame format with multi-index on item_id and timestamp. For example:

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
                This example can be found using example() function.
    """

    def __init__(self, data: Any, *args, **kwargs):
        if isinstance(data, TimeSeriesListDataset):
            data = self.from_list_dataset(data)
        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.MultiIndex):
                data = self.from_data_frame(data)
        self._validate_multi_index_data_frame(data)
        super().__init__(data=data, *args, **kwargs)

    @classmethod
    def example(cls):
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

        target = np.arange(9)
        datetime_index = tuple(
            pd.date_range(pd.Timestamp("01-01-2019"), periods=3, freq="D")
        )
        item_ids = (0, 1, 2)
        multi_index = pd.MultiIndex.from_product(
            [item_ids, datetime_index], names=["item_id", "timestamp"]
        )
        return TimeSeriesDataFrame(
            pd.Series(target, name="target", index=multi_index).to_frame()
        )

    @classmethod
    def _validate_data_frame(cls, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, got {type(df)}")
        if "item_id" not in df.columns:
            raise ValueError(f"data must have a `item_id` column")
        if "timestamp" not in df.columns:
            raise ValueError(f"data must have a `timestamp` column")
        if df["item_id"].isnull().any():
            raise ValueError(f"`item_id` columns can not have nan")
        if df["timestamp"].isnull().any():
            raise ValueError(f"`timestamp` columns can not have nan")
        if not df["item_id"].dtype == "int64":
            raise ValueError(f"for item_id, the only pandas dtype allowed is ‘int64’.")
        if not df["timestamp"].dtype == "datetime64[ns]":
            raise ValueError(
                f"for timestamp, the only pandas dtype allowed is ‘datetime64[ns]’."
            )

    @classmethod
    def _validate_list_dataset(cls, list_dataset):
        if not isinstance(list_dataset, TimeSeriesListDataset):
            raise ValueError(
                f"list_dataset must be a TimeSeriesListDataset, got {type(list_dataset)}"
            )
        if len(list_dataset.data) == 0:
            raise ValueError(f"list_dataset has no time-series.")
        for i, ts in enumerate(list_dataset.data):
            if not isinstance(ts, dict):
                raise ValueError(
                    f"{i}'th time-series in list_dataset must be a dict, got{type(ts)}"
                )
            if not ("target" in ts and "start" in ts):
                raise ValueError(
                    f"{i}'th time-series in list_dataset must have 'target' and 'start', got{ts.keys()}"
                )

    @classmethod
    def _validate_multi_index_data_frame(cls, data: pd.DataFrame):
        """Validate a multi-index pd.DataFrame can be converted to TimeSeriesDataFrame

        Parameters:
        -----------
        data : pd.DataFrame
            a data frame in pd.DataFrame format.
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, got {type(data)}")
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"data must have pd.MultiIndex, got {type(data.index)}")
        if not data.index.dtypes.array[0] == "int64":
            raise ValueError(f"for item_id, the only pandas dtype allowed is ‘int64’.")
        if not data.index.dtypes.array[1] == "datetime64[ns]":
            raise ValueError(
                f"for timestamp, the only pandas dtype allowed is ‘datetime64[ns]’."
            )
        if not data.index.names == ("item_id", "timestamp"):
            raise ValueError(
                f"data must have index names as ('item_id', 'timestamp'), got {type(data.index.names)}"
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
        for i, ts in enumerate(list_dataset.data):
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
        return TimeSeriesDataFrame(pd.concat(all_ts))

    @classmethod
    def from_data_frame(cls, df: pd.DataFrame) -> TimeSeriesDataFrame:
        """Convert a normal pd.DataFrame to a TimeSeriesDataFrame

        Parameters:
        -----------
        df: pd.DataFrame
            A pd.DataFrame with 'item_id' and 'timestamp' as columns. For example:

               item_id  timestamp  target
            0        0 2019-01-01       0
            1        0 2019-01-02       1
            2        0 2019-01-03       2
            3        1 2019-01-01       3
            4        1 2019-01-02       4
            5        1 2019-01-03       5
            6        2 2019-01-01       6
            7        2 2019-01-02       7
            8        2 2019-01-03       8

        Returns:
        --------
        ts_df : TimeSeriesDataFrame
            A data frame in TimeSeriesDataFrame format.
        """
        cls._validate_data_frame(df)
        return TimeSeriesDataFrame(df.set_index(["item_id", "timestamp"]))

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
        return TimeSeriesDataFrame(
            self.loc[(slice(None), slice(start, nanosecond_before_end)), :]
        )
