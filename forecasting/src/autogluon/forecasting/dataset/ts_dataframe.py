from __future__ import annotations

from typing import Any, Tuple
from collections.abc import Iterable

import pandas as pd
import numpy as np


ITEMID = "item_id"
TIMESTAMP = "timestamp"


class TimeSeriesDataFrame(pd.DataFrame):
    """TimeSeriesDataFrame to represent time-series dataset.

    Parameters
    ----------
    data : Any
        Time-series data to construct a TimeSeriesDataFrame.
        It currently supports three input formats:

            1. Time-series data in Iterable format. For example:

                iterable_dataset = [
                    {"target": [0, 1, 2], "start": pd.Timestamp("01-01-2019", freq='D')},
                    {"target": [3, 4, 5], "start": pd.Timestamp("01-01-2019", freq='D')},
                    {"target": [6, 7, 8], "start": pd.Timestamp("01-01-2019", freq='D')}
                ]

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
        if isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                self._validate_multi_index_data_frame(data)
            else:
                data = self.from_data_frame(data)
        else:
            data = self.from_iterable_dataset(data)
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
            [item_ids, datetime_index], names=[ITEMID, TIMESTAMP]
        )
        return TimeSeriesDataFrame(
            pd.Series(target, name="target", index=multi_index).to_frame()
        )

    @classmethod
    def _validate_iterable(cls, data: Iterable):
        if not isinstance(data, Iterable):
            raise ValueError(f"data must be of type Iterable.")

        if len(data) == 0:
            raise ValueError(f"data has no time-series.")

        for i, ts in enumerate(data):
            if not isinstance(ts, dict):
                raise ValueError(
                    f"{i}'th time-series in data must be a dict, got{type(ts)}"
                )
            if not ("target" in ts and "start" in ts):
                raise ValueError(
                    f"{i}'th time-series in data must have 'target' and 'start', got{ts.keys()}"
                )
            if not isinstance(ts["start"], pd.Timestamp) or ts["start"].freq is None:
                raise ValueError(
                    f"{i}'th time-series must have timestamp as 'start' with freq specified, got {ts['start']}"
                )

    @classmethod
    def _validate_data_frame(cls, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, got {type(df)}")
        if ITEMID not in df.columns:
            raise ValueError(f"data must have a `{ITEMID}` column")
        if TIMESTAMP not in df.columns:
            raise ValueError(f"data must have a `{TIMESTAMP}` column")
        if df[ITEMID].isnull().any():
            raise ValueError(f"`{ITEMID}` column can not have nan")
        if df[TIMESTAMP].isnull().any():
            raise ValueError(f"`{TIMESTAMP}` column can not have nan")
        if not df[ITEMID].dtype == "int64":
            raise ValueError(f"for {ITEMID}, the only pandas dtype allowed is ‘int64’.")
        if not df[TIMESTAMP].dtype == "datetime64[ns]":
            raise ValueError(
                f"for {TIMESTAMP}, the only pandas dtype allowed is ‘datetime64[ns]’."
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
            raise ValueError(f"for {ITEMID}, the only pandas dtype allowed is ‘int64’.")
        if not data.index.dtypes.array[1] == "datetime64[ns]":
            raise ValueError(
                f"for {TIMESTAMP}, the only pandas dtype allowed is ‘datetime64[ns]’."
            )
        if not data.index.names == (f"{ITEMID}", f"{TIMESTAMP}"):
            raise ValueError(
                f"data must have index names as ('{ITEMID}', '{TIMESTAMP}'), got {data.index.names}"
            )

    @classmethod
    def from_iterable_dataset(
        cls, iterable_dataset: Iterable
    ) -> TimeSeriesDataFrame:
        """Convenient function to Iterable dataset to TimeSeriesDataFrame.

        Parameters:
        -----------
        iterable_dataset : Iterable
            The iterable_dataset must have the following format:

            iterable_dataset = [
                {"target": [0, 1, 2], "start": pd.Timestamp("01-01-2019", freq='D')},
                {"target": [3, 4, 5], "start": pd.Timestamp("01-01-2019", freq='D')},
                {"target": [6, 7, 8], "start": pd.Timestamp("01-01-2019", freq='D')}
            ]

        Returns:
        --------
        ts_df : TimeSeriesDataFrame
            A data frame in TimeSeriesDataFrame format.
        """

        cls._validate_iterable(iterable_dataset)

        all_ts = []
        for i, ts in enumerate(iterable_dataset):
            start_timestamp = ts["start"]
            target = ts["target"]
            datetime_index = tuple(
                pd.date_range(
                    start_timestamp, periods=len(target), freq=start_timestamp.freq
                )
            )
            idx = pd.MultiIndex.from_product(
                [(i,), datetime_index], names=[ITEMID, TIMESTAMP]
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
        return TimeSeriesDataFrame(df.set_index([ITEMID, TIMESTAMP]))

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
