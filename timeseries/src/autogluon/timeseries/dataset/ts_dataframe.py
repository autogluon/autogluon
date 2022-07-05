from __future__ import annotations

import itertools
from typing import Any, Tuple, Type, Optional
from collections.abc import Iterable

import pandas as pd
from pandas.core.internals import ArrayManager, BlockManager

ITEMID = "item_id"
TIMESTAMP = "timestamp"


class TimeSeriesDataFrame(pd.DataFrame):
    """``TimeSeriesDataFrame`` s represent a collection of time series, where each row
    identifies the values of an (``item_id``, ``timestamp``) pair.

    For example, a time series data frame could represent the daily sales of a collection
    of products, where each ``item_id`` identifies a product and ``timestamp`` s correspond to
    the days.

    Parameters
    ----------
    data: Any
        Time-series data to construct a ``TimeSeriesDataFrame``. The class currently supports three
        input formats.

        1. Time-series data in Iterable format. For example::

                iterable_dataset = [
                    {"target": [0, 1, 2], "start": pd.Timestamp("01-01-2019", freq='D')},
                    {"target": [3, 4, 5], "start": pd.Timestamp("01-01-2019", freq='D')},
                    {"target": [6, 7, 8], "start": pd.Timestamp("01-01-2019", freq='D')}
                ]

        2. Time-series data in a pandas DataFrame format without multi-index. For example::

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

        3. Time-series data in pandas DataFrame format with multi-index on item_id and timestamp. For example::

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

    Attributes
    ----------
    freq: str
        A pandas and gluon-ts compatible string describing the frequency of the time series. For example
        "D" is daily data, etc. Also see,
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    num_items: int
        Number of items (time series) in the data set.
    """

    index: pd.MultiIndex

    def __init__(self, data: Any, *args, **kwargs):
        if isinstance(data, (BlockManager, ArrayManager)):
            # necessary for copy constructor to work. see _constructor
            # and pandas.DataFrame
            pass
        elif isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                self._validate_multi_index_data_frame(data)
            else:
                data = self._construct_pandas_frame_from_data_frame(data)
        elif isinstance(data, Iterable):
            data = self._construct_pandas_frame_from_iterable_dataset(data)
        else:
            raise ValueError(
                "Data input type not recognized, must be DataFrame or iterable."
            )
        super().__init__(data=data, *args, **kwargs)

    @property
    def _constructor(self) -> Type[TimeSeriesDataFrame]:
        return TimeSeriesDataFrame

    @property
    def freq(self):
        ts_index = self.index.levels[1]  # noqa
        freq = (
            ts_index.freq
            or ts_index.inferred_freq
            or self.loc[0].index.freq  # fall back to freq of first item
            or self.loc[0].index.inferred_freq
        )
        if freq is None:
            raise ValueError("Frequency not provided and cannot be inferred")
        if isinstance(freq, str):
            return freq
        elif isinstance(freq, pd._libs.tslibs.BaseOffset):
            return freq.freqstr
        return freq

    def iter_items(self) -> Iterable[Any]:
        return iter(self.index.levels[0])

    @property
    def num_items(self):
        return len(self.index.levels[0])

    @classmethod
    def _validate_iterable(cls, data: Iterable):
        if not isinstance(data, Iterable):
            raise ValueError("data must be of type Iterable.")

        first = next(iter(data), None)
        if first is None:
            raise ValueError("data has no time-series.")

        for i, ts in enumerate(itertools.chain([first], data)):
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
        if not df[TIMESTAMP].dtype == "datetime64[ns]":
            raise ValueError(
                f"for {TIMESTAMP}, the only pandas dtype allowed is ‘datetime64[ns]’."
            )

        # TODO: check if time series are irregularly sampled. this check was removed as
        # TODO: pandas is inconsistent in identifying freq when period-end timestamps
        # TODO: are provided.

    @classmethod
    def _validate_multi_index_data_frame(cls, data: pd.DataFrame):
        """Validate a multi-index pd.DataFrame can be converted to TimeSeriesDataFrame

        Parameters
        ----------
        data: pd.DataFrame
            a data frame in pd.DataFrame format.
        """

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, got {type(data)}")
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"data must have pd.MultiIndex, got {type(data.index)}")
        if not data.index.dtypes.array[1] == "datetime64[ns]":
            raise ValueError(
                f"for {TIMESTAMP}, the only pandas dtype allowed is ‘datetime64[ns]’."
            )
        if not data.index.names == (f"{ITEMID}", f"{TIMESTAMP}"):
            raise ValueError(
                f"data must have index names as ('{ITEMID}', '{TIMESTAMP}'), got {data.index.names}"
            )

    @classmethod
    def _construct_pandas_frame_from_iterable_dataset(
        cls, iterable_dataset: Iterable
    ) -> pd.DataFrame:
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
        return pd.concat(all_ts)

    @classmethod
    def from_iterable_dataset(cls, iterable_dataset: Iterable) -> pd.DataFrame:
        """Construct a ``TimeSeriesDataFrame`` from an Iterable of dictionaries each of which
        represent a single time series.

        This function also offers compatibility with GluonTS data sets, see
        https://ts.gluon.ai/_modules/gluonts/dataset/common.html#ListDataset.

        Parameters
        ----------
        iterable_dataset: Iterable
            An iterator over dictionaries, each with a ``target`` field specifying the value of the
            (univariate) time series, and a ``start`` field that features a pandas Timestamp with features.
            Example::

                iterable_dataset = [
                    {"target": [0, 1, 2], "start": pd.Timestamp("01-01-2019", freq='D')},
                    {"target": [3, 4, 5], "start": pd.Timestamp("01-01-2019", freq='D')},
                    {"target": [6, 7, 8], "start": pd.Timestamp("01-01-2019", freq='D')}
                ]

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A data frame in TimeSeriesDataFrame format.
        """
        return cls(cls._construct_pandas_frame_from_iterable_dataset(iterable_dataset))

    @classmethod
    def _construct_pandas_frame_from_data_frame(
        cls,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
    ) -> pd.DataFrame:

        df = df.copy()
        if id_column is not None:
            assert id_column in df.columns, f"Column {id_column} not found!"
            df.rename(columns={id_column: ITEMID}, inplace=True)

        if timestamp_column is not None:
            assert (
                timestamp_column in df.columns
            ), f"Column {timestamp_column} not found!"
            df.rename(columns={timestamp_column: TIMESTAMP}, inplace=True)

        cls._validate_data_frame(df)
        return df.set_index([ITEMID, TIMESTAMP])

    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
    ) -> TimeSeriesDataFrame:
        """Construct a ``TimeSeriesDataFrame`` from a pandas DataFrame.

        Parameters
        ----------
        df: pd.DataFrame
            A pd.DataFrame with 'item_id' and 'timestamp' as columns. For example:

            .. code-block::

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
        id_column: str
            Name of the 'item_id' column if column name is different
        timestamp_column: str
            Name of the 'timestamp' column if column name is different

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A data frame in TimeSeriesDataFrame format.
        """
        return cls(
            cls._construct_pandas_frame_from_data_frame(
                df, id_column=id_column, timestamp_column=timestamp_column
            )
        )

    def split_by_time(
        self, cutoff_time: pd.Timestamp
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Split dataframe to two different ``TimeSeriesDataFrame`` s before and after a certain
        ``cutoff_time``.

        Parameters
        ----------
        cutoff_time: pd.Timestamp
            The time to split the current data frame into two data frames.

        Returns
        -------
        data_before: TimeSeriesDataFrame
            Data frame containing time series before the ``cutoff_time`` (exclude ``cutoff_time``).
        data_after: TimeSeriesDataFrame
            Data frame containing time series after the ``cutoff_time`` (include ``cutoff_time``).
        """

        nanosecond_before_cutoff = cutoff_time - pd.Timedelta(nanoseconds=1)
        data_before = self.loc[(slice(None), slice(None, nanosecond_before_cutoff)), :]
        data_after = self.loc[(slice(None), slice(cutoff_time, None)), :]
        return TimeSeriesDataFrame(data_before), TimeSeriesDataFrame(data_after)

    def split_by_item(
        self, cutoff_item: int
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Split dataframe to two data frames containing items before and after a ``cutoff_item``.

        Parameters
        ----------
        cutoff_item: int
            The item_id to split the current data frame into two data frames.

        Returns
        -------
        data_before: TimeSeriesDataFrame
            Data frame containing time-series before the ``cutoff_item`` (exclude ``cutoff_item``).
        data_after: TimeSeriesDataFrame
            Data frame containing time-series after the ``cutoff_item`` (include ``cutoff_item``).
        """

        data_before = self.loc[(slice(None, cutoff_item - 1), slice(None)), :]
        data_after = self.loc[(slice(cutoff_item, None), slice(None)), :]
        return TimeSeriesDataFrame(data_before), TimeSeriesDataFrame(data_after)

    def slice_by_timestep(self, time_step_slice: slice) -> TimeSeriesDataFrame:
        """Return a slice of time steps (with no regards to the actual timestamp) from within
        each item in a time series data frame. For example, if a data frame is constructed as::

            item_id  timestamp  target
                  0 2019-01-01       0
                  0 2019-01-02       1
                  0 2019-01-03       2
                  1 2019-01-02       3
                  1 2019-01-03       4
                  1 2019-01-04       5
                  2 2019-01-03       6
                  2 2019-01-04       7
                  2 2019-01-05       8

        then :code:`df.slice_by_timestep(time_step_slice=slice(-2, None))` would return the last two
        time steps from each item::

            item_id  timestamp  target
                  0 2019-01-02       1
                  0 2019-01-03       2
                  1 2019-01-03       4
                  1 2019-01-04       5
                  2 2019-01-04       7
                  2 2019-01-05       8

        Note that this function returns a copy of the original data. This function is useful for
        constructing holdout sets for validation.

        Parameters
        ----------
        time_step_slice: slice
            A python slice object representing the slices to return from each item

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            Data frame containing only the time steps of each ``item_id`` sliced according to the
            input ``time_step_slice``.
        """
        slice_gen = (
            (i, self.loc[i].iloc[time_step_slice]) for i in self.index.levels[0]
        )
        slices = []
        for ix, data_slice in slice_gen:
            idx = pd.MultiIndex.from_product(
                [(ix,), data_slice.index], names=[ITEMID, TIMESTAMP]
            )
            data_slice.set_index(idx, inplace=True)
            slices.append(data_slice)
        return self.__class__(pd.concat(slices))

    def subsequence(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> TimeSeriesDataFrame:
        """Extract time-series between start (inclusive) and end (exclusive) time.

        Parameters
        ----------
        start: pd.Timestamp
            The start time (inclusive) of a time range that will be used for subsequence.
        end: pd.Timestamp
            The end time (exclusive) of a time range that will be used for subsequence.

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A new data frame in ``TimeSeriesDataFrame`` format contains time-series in a time range
            defined between start and end time.
        """

        if end < start:
            raise ValueError(f"end time {end} is earlier than stat time {start}")

        nanosecond_before_end = end - pd.Timedelta(nanoseconds=1)
        return TimeSeriesDataFrame(
            self.loc[(slice(None), slice(start, nanosecond_before_end)), :]
        )

    @classmethod
    def from_pickle(cls, filepath_or_buffer: Any) -> "TimeSeriesDataFrame":
        """Convenience method to read pickled time series data frames. If the read pickle
        file refers to a plain pandas DataFrame, it will be cast to a TimeSeriesDataFrame.

        Parameters
        ----------
        filepath_or_buffer: Any
            Filename provided as a string or an ``IOBuffer`` containing the pickled object.

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            The pickled time series data frame.
        """
        try:
            data = pd.read_pickle(filepath_or_buffer)
            return data if isinstance(data, cls) else cls(data)
        except Exception as err:  # noqa
            raise IOError(f"Could not load pickled data set due to error: {str(err)}")
