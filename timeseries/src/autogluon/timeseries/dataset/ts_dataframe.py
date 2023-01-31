from __future__ import annotations

import copy
import itertools
import warnings
from collections.abc import Iterable
from typing import Any, Optional, Tuple, Type

import numpy as np
import pandas as pd
from pandas.core.internals import ArrayManager, BlockManager

from autogluon.common.loaders import load_pd

ITEMID = "item_id"
TIMESTAMP = "timestamp"

IRREGULAR_TIME_INDEX_FREQSTR = "IRREG"


class TimeSeriesDataFrame(pd.DataFrame):
    """``TimeSeriesDataFrame`` s represent a collection of time series, where each row
    identifies the values of an (``item_id``, ``timestamp``) pair.

    For example, a time series data frame could represent the daily sales of a collection
    of products, where each ``item_id`` identifies a product and ``timestamp`` s correspond to
    the days.

    Parameters
    ----------
    data : Any
        Time-series data to construct a ``TimeSeriesDataFrame``. The class currently supports four input formats.

        1. Time-series data in a pandas DataFrame format without multi-index. For example::

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

        2. Time-series data in pandas DataFrame format with multi-index on item_id and timestamp. For example::

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

        3. Path to a data file in CSV or Parquet format. The file must contain columns ``item_id`` and ``timestamp``, as well as columns with time series values. This is similar to Option 1 above (pandas DataFrame format without multi-index). Both remote (e.g., S3) and local paths are accepted.

        4. Time-series data in Iterable format. For example::

                iterable_dataset = [
                    {"target": [0, 1, 2], "start": pd.Timestamp("01-01-2019", freq='D')},
                    {"target": [3, 4, 5], "start": pd.Timestamp("01-01-2019", freq='D')},
                    {"target": [6, 7, 8], "start": pd.Timestamp("01-01-2019", freq='D')}
                ]

    static_features : Optional[pd.DataFrame]
        An optional data frame describing the metadata attributes of individual items in the item index. These
        may be categorical or real valued attributes for each item. For example, if the item index refers to
        time series data of individual households, static features may refer to time-independent demographic
        features. When provided during ``fit``, the ``TimeSeriesPredictor`` expects the same metadata to be available
        during prediction time. When provided, the index of the ``static_features`` index must match the item index
        of the ``TimeSeriesDataFrame``.

        ``TimeSeriesDataFrame`` will ensure consistency of static features during serialization/deserialization,
        copy and slice operations although these features should be considered experimental.

    Attributes
    ----------
    freq : str
        A pandas and gluon-ts compatible string describing the frequency of the time series. For example
        "D" is daily data, etc. Also see,
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    num_items : int
        Number of items (time series) in the data set.
    item_ids : pd.Index
        List of unique time series IDs contained in the data set.
    """

    DUMMY_INDEX_START_TIME = pd.Timestamp("1900-01-01 00:00:00")
    index: pd.MultiIndex
    _metadata = ["_static_features", "_cached_freq"]

    def __init__(self, data: Any, static_features: Optional[pd.DataFrame] = None, *args, **kwargs):
        if isinstance(data, (BlockManager, ArrayManager)):
            # necessary for copy constructor to work. see _constructor
            # and pandas.DataFrame
            pass
        elif isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                self._validate_multi_index_data_frame(data)
            else:
                data = self._construct_pandas_frame_from_data_frame(data)
        elif isinstance(data, str):
            data = self._load_data_frame_from_file(data)
        elif isinstance(data, Iterable):
            data = self._construct_pandas_frame_from_iterable_dataset(data)
        else:
            raise ValueError("Data input type not recognized, must be DataFrame, iterable or string.")
        super().__init__(data=data, *args, **kwargs)
        self._static_features: Optional[pd.DataFrame] = None
        if static_features is not None:
            self.static_features = static_features

        # internal value for cached frequency values that are inferred. corresponds to either a
        # pandas-compatible frequency string, the value IRREGULAR_TIME_INDEX_FREQSTR that signals
        # the time series have irregular timestamps (in which case tsdf.freq returns None), or None
        # if inference was not yet performed.
        self._cached_freq: Optional[str] = None

    @property
    def _constructor(self) -> Type[TimeSeriesDataFrame]:
        return TimeSeriesDataFrame

    @property
    def item_ids(self) -> pd.Index:
        return self.index.unique(level=ITEMID)

    @property
    def static_features(self):
        return self._static_features

    @static_features.setter
    def static_features(self, value: Optional[pd.DataFrame]):
        # if the current item index is not a multiindex, then we are dealing with a single
        # item slice. this should only happen when the user explicitly requests only a
        # single item or during `slice_by_timestep`. In this case we do not set static features
        if not isinstance(self.index, pd.MultiIndex):
            return

        if value is not None:
            if isinstance(value, pd.Series):
                value = value.to_frame()
            if not isinstance(value, pd.DataFrame):
                raise ValueError(f"static_features must be a pandas DataFrame (received object of type {type(value)})")
            missing_item_ids = self.item_ids.difference(value.index)
            if len(missing_item_ids) > 0:
                raise ValueError(
                    f"Following item_ids are missing from the index of static_features: {missing_item_ids.to_list()}"
                )
            # if provided static features are a strict superset of the item index, we take a subset to ensure consistency
            if len(value.index.difference(self.item_ids)) > 0:
                value = value.loc[self.item_ids]
            # Avoid modifying static features inplace
            value = value.copy()
            if value.index.name != ITEMID:
                value.index.rename(ITEMID, inplace=True)

        self._static_features = value

    @property
    def freq(self):
        if self._cached_freq is not None and self._cached_freq == IRREGULAR_TIME_INDEX_FREQSTR:
            return None  # irregularly sampled time series
        elif self._cached_freq:
            return self._cached_freq

        def get_freq(series):
            return series.index.freq or series.index.inferred_freq

        # check the frequencies of the first 100 items to see if frequencies are consistent and
        # can be inferred
        freq_for_each_series = [get_freq(self.loc[idx]) for idx in self.item_ids[:100]]
        freq = freq_for_each_series[0]
        if len(set(freq_for_each_series)) > 1 or freq is None:
            self._cached_freq = IRREGULAR_TIME_INDEX_FREQSTR
            return None

        freq = freq.freqstr if isinstance(freq, pd._libs.tslibs.BaseOffset) else freq
        self._cached_freq = freq
        return freq

    @property
    def num_items(self):
        return len(self.item_ids)

    def num_timesteps_per_item(self) -> pd.Series:
        """Length of each time series in the dataframe."""
        return self.groupby(level=ITEMID, sort=False).size()

    @classmethod
    def _validate_iterable(cls, data: Iterable):
        if not isinstance(data, Iterable):
            raise ValueError("data must be of type Iterable.")

        first = next(iter(data), None)
        if first is None:
            raise ValueError("data has no time-series.")

        for i, ts in enumerate(itertools.chain([first], data)):
            if not isinstance(ts, dict):
                raise ValueError(f"{i}'th time-series in data must be a dict, got{type(ts)}")
            if not ("target" in ts and "start" in ts):
                raise ValueError(f"{i}'th time-series in data must have 'target' and 'start', got{ts.keys()}")
            if not isinstance(ts["start"], (pd.Timestamp, pd.Period)) or ts["start"].freq is None:
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
            raise ValueError(f"for {TIMESTAMP}, the only pandas dtype allowed is `datetime64[ns]`.")
        item_id_column = df[ITEMID]
        # workaround for pd.api.types.is_string_dtype issue https://github.com/pandas-dev/pandas/issues/15585
        item_id_is_string = (item_id_column == item_id_column.astype(str)).all()
        item_id_is_int = pd.api.types.is_integer_dtype(item_id_column)
        if not (item_id_is_string or item_id_is_int):
            raise ValueError(f"all entries in column `{ITEMID}` must be of integer or string dtype")

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
            raise ValueError(f"for {TIMESTAMP}, the only pandas dtype allowed is `datetime64[ns]`.")
        if not data.index.names == (f"{ITEMID}", f"{TIMESTAMP}"):
            raise ValueError(f"data must have index names as ('{ITEMID}', '{TIMESTAMP}'), got {data.index.names}")
        item_id_index = data.index.get_level_values(level=ITEMID)
        # workaround for pd.api.types.is_string_dtype issue https://github.com/pandas-dev/pandas/issues/15585
        item_id_is_string = (item_id_index == item_id_index.astype(str)).all()
        item_id_is_int = pd.api.types.is_integer_dtype(item_id_index)
        if not (item_id_is_string or item_id_is_int):
            raise ValueError(f"all entries in index `{ITEMID}` must be of integer or string dtype")

    @classmethod
    def _construct_pandas_frame_from_iterable_dataset(cls, iterable_dataset: Iterable) -> pd.DataFrame:
        cls._validate_iterable(iterable_dataset)

        all_ts = []
        for i, ts in enumerate(iterable_dataset):
            start_timestamp = ts["start"]
            freq = start_timestamp.freq
            if isinstance(start_timestamp, pd.Period):
                start_timestamp = start_timestamp.to_timestamp(how="S")
            target = ts["target"]
            datetime_index = tuple(pd.date_range(start_timestamp, periods=len(target), freq=freq))
            idx = pd.MultiIndex.from_product([(i,), datetime_index], names=[ITEMID, TIMESTAMP])
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
    def _load_data_frame_from_file(cls, path: str) -> pd.DataFrame:
        df = load_pd.load(path)
        return cls._construct_pandas_frame_from_data_frame(df)

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
            assert timestamp_column in df.columns, f"Column {timestamp_column} not found!"
            df.rename(columns={timestamp_column: TIMESTAMP}, inplace=True)

        if TIMESTAMP in df.columns:
            df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])

        cls._validate_data_frame(df)
        return df.set_index([ITEMID, TIMESTAMP])

    @classmethod
    def from_path(
        cls,
        path: str,
        id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
    ) -> TimeSeriesDataFrame:
        """Construct a ``TimeSeriesDataFrame`` from a CSV or Parquet file.

        Parameters
        ----------
        path : str
            Path to a local or remote (e.g., S3) file containing the time series data in CSV or Parquet format.
            Example file contents::

                item_id,timestamp,target
                0,2019-01-01,0
                0,2019-01-02,1
                0,2019-01-03,2
                1,2019-01-01,3
                1,2019-01-02,4
                1,2019-01-03,5
                2,2019-01-01,6
                2,2019-01-02,7
                2,2019-01-03,8

        id_column: str
            Name of the 'item_id' column if column name is different
        timestamp_column: str
            Name of the 'timestamp' column if column name is different

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A data frame in TimeSeriesDataFrame format.
        """
        df = load_pd.load(path)
        return cls.from_data_frame(df, id_column=id_column, timestamp_column=timestamp_column)

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
            cls._construct_pandas_frame_from_data_frame(df, id_column=id_column, timestamp_column=timestamp_column)
        )

    def copy(self: TimeSeriesDataFrame, deep: bool = True) -> pd.DataFrame:  # noqa
        obj = super().copy(deep=deep)

        # also perform a deep copy for static features
        if deep:
            for k in obj._metadata:
                setattr(obj, k, copy.deepcopy(getattr(obj, k)))
        return obj

    def __finalize__(  # noqa
        self: TimeSeriesDataFrame, other, method: Optional[str] = None, **kwargs
    ) -> TimeSeriesDataFrame:
        super().__finalize__(other=other, method=method, **kwargs)
        # when finalizing the copy/slice operation, we use the property setter to stay consistent
        # with the item index
        if hasattr(other, "_static_features"):
            self.static_features = other._static_features
        if hasattr(other, "_cached_freq"):
            self._cached_freq = other._cached_freq
        return self

    def split_by_time(self, cutoff_time: pd.Timestamp) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Split dataframe to two different ``TimeSeriesDataFrame`` s before and after a certain ``cutoff_time``.

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
        before = TimeSeriesDataFrame(data_before, static_features=self.static_features)
        after = TimeSeriesDataFrame(data_after, static_features=self.static_features)
        before._cached_freq = self._cached_freq
        after._cached_freq = self._cached_freq
        return before, after

    def slice_by_timestep(
        self, start_index: Optional[int] = None, end_index: Optional[int] = None
    ) -> TimeSeriesDataFrame:
        """Select a subsequence from each time series between start (inclusive) and end (exclusive) indices.

        This operation is equivalent to selecting a slice ``[start_index : end_index]`` from each time series, and then
        combining these slices into a new ``TimeSeriesDataFrame``. See examples below.

        Returns a copy of the original data. This is useful for constructing holdout sets for validation.

        Parameters
        ----------
        start_index : int or None
            Start index (inclusive) of the slice for each time series.
            Negative values are counted from the end of each time series.
            When set to None, the slice starts from the beginning of each time series.
        end_index : int or None
            End index (exclusive) of the slice for each time series.
            Negative values are counted from the end of each time series.
            When set to None, the slice includes the end of each time series.

        Returns
        -------
        ts_df : TimeSeriesDataFrame
            A new time series dataframe containing entries of the original time series between start and end indices.

        Examples
        --------
        >>> print(ts_dataframe)
                            target
        item_id timestamp
        0       2019-01-01       0
                2019-01-02       1
                2019-01-03       2
        1       2019-01-02       3
                2019-01-03       4
                2019-01-04       5
        2       2019-01-03       6
                2019-01-04       7
                2019-01-05       8

        Select the first entry of each time series

        >>> df.slice_by_timestep(0, 1)
                            target
        item_id timestamp
        0       2019-01-01       0
        1       2019-01-02       3
        2       2019-01-03       6

        Select the last 2 entries of each time series

        >>> df.slice_by_timestep(-2, None)
                            target
        item_id timestamp
        0       2019-01-02       1
                2019-01-03       2
        1       2019-01-03       4
                2019-01-04       5
        2       2019-01-04       7
                2019-01-05       8

        Select all except the last entry of each time series

        >>> df.slice_by_timestep(None, -1)
                            target
        item_id timestamp
        0       2019-01-01       0
                2019-01-02       1
        1       2019-01-02       3
                2019-01-03       4
        2       2019-01-03       6
                2019-01-04       7

        Copy the entire dataframe

        >>> df.slice_by_timestep(None, None)
                            target
        item_id timestamp
        0       2019-01-01       0
                2019-01-02       1
                2019-01-03       2
        1       2019-01-02       3
                2019-01-03       4
                2019-01-04       5
        2       2019-01-03       6
                2019-01-04       7
                2019-01-05       8

        """
        if start_index is not None and not isinstance(start_index, int):
            raise ValueError(f"start_index must be of type int or None (got {type(start_index)})")
        if end_index is not None and not isinstance(end_index, int):
            raise ValueError(f"end_index must be of type int or None (got {type(end_index)})")

        time_step_slice = slice(start_index, end_index)
        result = self.groupby(level=ITEMID, sort=False, as_index=False).nth(time_step_slice)
        result.static_features = self.static_features
        result._cached_freq = self._cached_freq
        return result

    def slice_by_time(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> TimeSeriesDataFrame:
        """Select a subsequence from each time series between start (inclusive) and end (exclusive) timestamps.

        Parameters
        ----------
        start_time: pd.Timestamp
            Start time (inclusive) of the slice for each time series.
        end_time: pd.Timestamp
            End time (exclusive) of the slice for each time series.

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A new time series dataframe containing entries of the original time series between start and end timestamps.
        """

        if end_time < start_time:
            raise ValueError(f"end_time {end_time} is earlier than start_time {start_time}")

        nanosecond_before_end_time = end_time - pd.Timedelta(nanoseconds=1)
        return TimeSeriesDataFrame(
            self.loc[(slice(None), slice(start_time, nanosecond_before_end_time)), :],
            static_features=self.static_features,
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

    def get_reindexed_view(self, freq: str = "S") -> TimeSeriesDataFrame:
        """Returns a new TimeSeriesDataFrame object with the same underlying data and
        static features as the current data frame, except the time index is replaced by
        a new "dummy" time series index with the given frequency. This is useful when
        suggesting AutoGluon-TimeSeries to "ignore" the time information, for example when
        dealing with irregularly sampled time series or sequences (e.g., financial time
        series).

        Parameters
        ----------
        freq: str
            Frequency string of the new time series data index.

        Returns
            TimeSeriesDataFrame: the new view object with replaced index, but the same underlying
            data. Note that the underlying data is not copied.
        """
        df_view = self.iloc[:]  # return a view without copying data

        # build the surrogate index
        indexes = []
        for i in self.item_ids:
            idx = pd.MultiIndex.from_product(
                [(i,), pd.date_range(self.DUMMY_INDEX_START_TIME, periods=len(self.loc[i]), freq=freq)]
            )
            indexes.append(idx)

        new_index = pd.MultiIndex.from_tuples(np.concatenate(indexes), names=[ITEMID, TIMESTAMP])
        df_view.set_index(new_index, inplace=True)
        df_view._cached_freq = freq

        return df_view

    def to_regular_index(self, freq: str) -> "TimeSeriesDataFrame":
        """Fill the gaps in an irregularly-sampled time series with NaNs.

        Parameters
        ----------
        freq: str
            Frequency string of the new time series data index.

        Examples
        --------
        >>> print(ts_dataframe)
                            target
        item_id timestamp
        0       2019-01-01     NaN
                2019-01-03     1.0
                2019-01-06     2.0
                2019-01-07     NaN
        1       2019-02-04     3.0
                2019-02-07     4.0

        >>> print(ts_dataframe.to_regular_index(freq="D"))
                            target
        item_id timestamp
        0       2019-01-01     NaN
                2019-01-02     NaN
                2019-01-03     1.0
                2019-01-04     NaN
                2019-01-05     NaN
                2019-01-06     2.0
                2019-01-07     NaN
        1       2019-02-04     3.0
                2019-02-05     NaN
                2019-02-06     NaN
                2019-02-07     4.0

        """
        if self.freq is not None:
            if self.freq != freq:
                raise ValueError(
                    f"TimeSeriesDataFrame already has a regular index with freq '{self.freq}' "
                    f"that cannot be converted to the given freq '{freq}'"
                )
            else:
                return self

        filled_series = []
        for item_id, time_series in self.groupby(level=ITEMID, sort=False):
            time_series = time_series.droplevel(ITEMID)
            timestamps = time_series.index
            resampled_ts = time_series.resample(freq).asfreq()
            if not timestamps.isin(resampled_ts.index).all():
                raise ValueError(
                    f"Irregularly-sampled timestamps in this TimeSeriesDataFrame are not compatible "
                    f"with the given frequency '{freq}'"
                )
            filled_series.append(pd.concat({item_id: resampled_ts}, names=[ITEMID]))

        return TimeSeriesDataFrame(pd.concat(filled_series), static_features=self.static_features)

    def fill_missing_values(self, method: str = "auto", value: float = 0.0) -> "TimeSeriesDataFrame":
        """Fill missing values represented by NaN.

        Parameters
        ----------
        method : str, default = "auto"
            Method used to impute missing values.

            - "auto" - first forward fill (to fill the in-between and trailing NaNs), then backward fill (to fill the leading NaNs)
            - "ffill" or "pad" - propagate last valid observation forward. Note: missing values at the start of the time series are not filled.
            - "bfill" or "backfill" - use next valid observation to fill gap. Note: this may result in information leakage; missing values at the end of the time series are not filled.
            - "constant" - replace NaNs with the given constant ``value``.
            - "interpolate" - fill NaN values using linear interpolation. Note: this may result in information leakage.
        value : float, default = 0.0
            Value used by the "constant" imputation method.

        Examples
        --------
        >>> print(ts_dataframe)
                            target
        item_id timestamp
        0       2019-01-01     NaN
                2019-01-02     NaN
                2019-01-03     1.0
                2019-01-04     NaN
                2019-01-05     NaN
                2019-01-06     2.0
                2019-01-07     NaN
        1       2019-02-04     NaN
                2019-02-05     3.0
                2019-02-06     NaN
                2019-02-07     4.0

        >>> print(ts_dataframe.fill_missing_values(method="auto"))
                            target
        item_id timestamp
        0       2019-01-01     1.0
                2019-01-02     1.0
                2019-01-03     1.0
                2019-01-04     1.0
                2019-01-05     1.0
                2019-01-06     2.0
                2019-01-07     2.0
        1       2019-02-04     3.0
                2019-02-05     3.0
                2019-02-06     3.0
                2019-02-07     4.0

        """
        if self.freq is None:
            raise ValueError(
                "Please make sure that all time series have a regular index before calling `fill_missing_values`"
                "(for example, using the `to_regular_index` method)."
            )

        grouped_df = pd.DataFrame(self).groupby(level=ITEMID, sort=False, group_keys=False)
        if method == "auto":
            filled_df = grouped_df.fillna(method="ffill").fillna(method="bfill")
        elif method in ["ffill", "pad"]:
            filled_df = grouped_df.fillna(method="ffill")
        elif method in ["bfill", "backfill"]:
            filled_df = grouped_df.fillna(method="bfill")
        elif method == "constant":
            filled_df = self.fillna(value=value)
        elif method == "interpolate":
            filled_df = grouped_df.apply(lambda ts: ts.interpolate())
        else:
            raise ValueError(
                "Invalid fill method. Expecting one of "
                "{'auto', 'ffill', 'pad', 'bfill', 'backfill', 'constant', 'interpolate'}. "
                f"Got {method}"
            )
        return TimeSeriesDataFrame(filled_df, static_features=self.static_features)

    def dropna(self, how: str = "any") -> "TimeSeriesDataFrame":
        """Drop rows containing NaNs.

        Parameters
        ----------
        how : {"any", "all"}, default = "any"
            Determine if row or column is removed from TimeSeriesDataFrame, when we have at least one NaN or all NaN.
            - "any" : If any NaN values are present, drop that row or column.
            - "all" : If all values are NaN, drop that row or column.
        """
        # We need to cast to a DataFrame first. Calling self.dropna() results in an exception because self.T
        # (used inside dropna) is not supported for TimeSeriesDataFrame
        dropped_df = pd.DataFrame(self).dropna(how=how)
        return TimeSeriesDataFrame(dropped_df, static_features=self.static_features)
