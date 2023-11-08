from __future__ import annotations

import copy
import itertools
import logging
import reprlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, Union

import pandas as pd
from joblib.parallel import Parallel, delayed
from pandas.core.internals import ArrayManager, BlockManager

from autogluon.common.loaders import load_pd

logger = logging.getLogger(__name__)

ITEMID = "item_id"
TIMESTAMP = "timestamp"

IRREGULAR_TIME_INDEX_FREQSTR = "IRREG"


class TimeSeriesDataFrame(pd.DataFrame):
    """A collection of univariate time series, where each row is identified by an (``item_id``, ``timestamp``) pair.

    For example, a time series data frame could represent the daily sales of a collection of products, where each
    ``item_id`` corresponds to a product and ``timestamp`` corresponds to the day of the record.

    Parameters
    ----------
    data : pd.DataFrame, str, pathlib.Path or Iterable
        Time series data to construct a ``TimeSeriesDataFrame``. The class currently supports four input formats.

        1. Time series data in a pandas DataFrame format without multi-index. For example::

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

            You can also use :meth:`~autogluon.timeseries.TimeSeriesDataFrame.from_data_frame` for loading data in such format.

        2. Path to a data file in CSV or Parquet format. The file must contain columns ``item_id`` and ``timestamp``, as well as columns with time series values. This is similar to Option 1 above (pandas DataFrame format without multi-index). Both remote (e.g., S3) and local paths are accepted. You can also use :meth:`~autogluon.timeseries.TimeSeriesDataFrame.from_path` for loading data in such format.

        3. Time series data in pandas DataFrame format with multi-index on ``item_id`` and ``timestamp``. For example::

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

        4. Time series data in Iterable format. For example::

                iterable_dataset = [
                    {"target": [0, 1, 2], "start": pd.Period("01-01-2019", freq='D')},
                    {"target": [3, 4, 5], "start": pd.Period("01-01-2019", freq='D')},
                    {"target": [6, 7, 8], "start": pd.Period("01-01-2019", freq='D')}
                ]

            You can also use :meth:`~autogluon.timeseries.TimeSeriesDataFrame.from_iterable_dataset` for loading data in such format.

    static_features : pd.DataFrame, str or pathlib.Path, optional
        An optional data frame describing the metadata of each individual time series that does not change with time.
        Can take real-valued or categorical values. For example, if ``TimeSeriesDataFrame`` contains sales of various
        products, static features may refer to time-independent features like color or brand.

        The index of the ``static_features`` index must contain a single entry for each item present in the respective
        ``TimeSeriesDataFrame``. For example, the following ``TimeSeriesDataFrame``::

                                target
            item_id timestamp
            A       2019-01-01       0
                    2019-01-02       1
                    2019-01-03       2
            B       2019-01-01       3
                    2019-01-02       4
                    2019-01-03       5

        is compatible with the following ``static_features``::

                     feat_1 feat_2
            item_id
            A           2.0    bar
            B           5.0    foo

        ``TimeSeriesDataFrame`` will ensure consistency of static features during serialization/deserialization, copy
        and slice operations.

        If ``static_features`` are provided during ``fit``, the ``TimeSeriesPredictor`` expects the same metadata to be
        available during prediction time.
    id_column : str, optional
        Name of the ``item_id`` column, if it's different from the default. This argument is only used when
        constructing a TimeSeriesDataFrame using format 1 (DataFrame without multi-index) or 2 (path to a file).
    timestamp_column : str, optional
        Name of the ``timestamp`` column, if it's different from the default. This argument is only used when
        constructing a TimeSeriesDataFrame using format 1 (DataFrame without multi-index) or 2 (path to a file).
    num_cpus : int, default = -1
        Number of CPU cores used to process the iterable dataset in parallel. Set to -1 to use all cores. This argument
        is only used when constructing a TimeSeriesDataFrame using format 4 (iterable dataset).

    Attributes
    ----------
    freq : str
        A pandas-compatible string describing the frequency of the time series. For example ``"D"`` for daily data,
        ``"H"`` for hourly data, etc. This attribute is determined automatically based on the timestamps. For the full
        list of possible values, see
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    num_items : int
        Number of items (time series) in the data set.
    item_ids : pd.Index
        List of unique time series IDs contained in the data set.
    """

    index: pd.MultiIndex
    _metadata = ["_static_features", "_cached_freq"]

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path, Iterable],
        static_features: Optional[Union[pd.DataFrame, str, Path]] = None,
        id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        num_cpus: int = -1,
        *args,
        **kwargs,
    ):
        if isinstance(data, (BlockManager, ArrayManager)):
            # necessary for copy constructor to work in pandas <= 2.0.x. In >= 2.1.x this is replaced by _constructor_from_mgr
            pass
        elif isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                self._validate_multi_index_data_frame(data)
            else:
                data = self._construct_tsdf_from_data_frame(
                    data, id_column=id_column, timestamp_column=timestamp_column
                )
        elif isinstance(data, (str, Path)):
            data = self._construct_tsdf_from_data_frame(
                load_pd.load(str(data)), id_column=id_column, timestamp_column=timestamp_column
            )
        elif isinstance(data, Iterable):
            data = self._construct_tsdf_from_iterable_dataset(data, num_cpus=num_cpus)
        else:
            raise ValueError(f"data must be a pd.DataFrame, Iterable, string or Path (received {type(data)}).")
        super().__init__(data=data, *args, **kwargs)
        self._static_features: Optional[pd.DataFrame] = None
        if static_features is not None:
            self.static_features = self._construct_static_features(static_features, id_column=id_column)

        # internal value for cached frequency values that are inferred. corresponds to either a
        # pandas-compatible frequency string, the value IRREGULAR_TIME_INDEX_FREQSTR that signals
        # the time series have irregular timestamps (in which case tsdf.freq returns None), or None
        # if inference was not yet performed.
        self._cached_freq: Optional[str] = None

    @property
    def _constructor(self) -> Type[TimeSeriesDataFrame]:
        return TimeSeriesDataFrame

    def _constructor_from_mgr(self, mgr, axes):
        # Use the default constructor when constructing from _mgr. Otherwise pandas enters an infinite recursion by
        # repeatedly calling TimeSeriesDataFrame constructor
        df = self._from_mgr(mgr, axes=axes)
        df._static_features = self._static_features
        df._cached_freq = self._cached_freq
        return df

    @classmethod
    def _construct_tsdf_from_data_frame(
        cls,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        if id_column is not None:
            assert id_column in df.columns, f"Column '{id_column}' not found!"
            if id_column != ITEMID and ITEMID in df.columns:
                logger.warning(f"Renaming existing column '{ITEMID}' -> '__{ITEMID}' to avoid name collisions.")
                df.rename(columns={ITEMID: "__" + ITEMID}, inplace=True)
            df.rename(columns={id_column: ITEMID}, inplace=True)

        if timestamp_column is not None:
            assert timestamp_column in df.columns, f"Column '{timestamp_column}' not found!"
            if timestamp_column != TIMESTAMP and TIMESTAMP in df.columns:
                logger.warning(f"Renaming existing column '{TIMESTAMP}' -> '__{TIMESTAMP}' to avoid name collisions.")
                df.rename(columns={TIMESTAMP: "__" + TIMESTAMP}, inplace=True)
            df.rename(columns={timestamp_column: TIMESTAMP}, inplace=True)

        if TIMESTAMP in df.columns:
            df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])

        cls._validate_data_frame(df)
        return df.set_index([ITEMID, TIMESTAMP])

    @classmethod
    def _construct_tsdf_from_iterable_dataset(cls, iterable_dataset: Iterable, num_cpus: int = -1) -> pd.DataFrame:
        def load_single_item(item_id: int, ts: dict) -> pd.DataFrame:
            start_timestamp = ts["start"]
            freq = start_timestamp.freq
            if isinstance(start_timestamp, pd.Period):
                start_timestamp = start_timestamp.to_timestamp(how="S")
            target = ts["target"]
            datetime_index = tuple(pd.date_range(start_timestamp, periods=len(target), freq=freq))
            idx = pd.MultiIndex.from_product([(item_id,), datetime_index], names=[ITEMID, TIMESTAMP])
            return pd.Series(target, name="target", index=idx).to_frame()

        cls._validate_iterable(iterable_dataset)
        all_ts = Parallel(n_jobs=num_cpus)(
            delayed(load_single_item)(item_id, ts) for item_id, ts in enumerate(iterable_dataset)
        )
        return pd.concat(all_ts)

    @classmethod
    def _validate_multi_index_data_frame(cls, data: pd.DataFrame):
        """Validate a multi-index pd.DataFrame can be converted to TimeSeriesDataFrame"""

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, got {type(data)}")
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"data must have pd.MultiIndex, got {type(data.index)}")
        if not pd.api.types.is_datetime64_dtype(data.index.dtypes[TIMESTAMP]):
            raise ValueError(f"for {TIMESTAMP}, the only pandas dtype allowed is `datetime64`.")
        if not data.index.names == (f"{ITEMID}", f"{TIMESTAMP}"):
            raise ValueError(f"data must have index names as ('{ITEMID}', '{TIMESTAMP}'), got {data.index.names}")
        item_id_index = data.index.get_level_values(level=ITEMID)
        if not (pd.api.types.is_integer_dtype(item_id_index) or pd.api.types.is_string_dtype(item_id_index)):
            raise ValueError(f"all entries in index `{ITEMID}` must be of integer or string dtype")

    @classmethod
    def _validate_data_frame(cls, df: pd.DataFrame):
        """Validate that a pd.DataFrame with ITEMID and TIMESTAMP columns can be converted to TimeSeriesDataFrame"""
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
        if not pd.api.types.is_datetime64_dtype(df[TIMESTAMP]):
            raise ValueError(f"for {TIMESTAMP}, the only pandas dtype allowed is `datetime64`.")
        item_id_column = df[ITEMID]
        if not (pd.api.types.is_integer_dtype(item_id_column) or pd.api.types.is_string_dtype(item_id_column)):
            raise ValueError(f"all entries in column `{ITEMID}` must be of integer or string dtype")

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
            if not isinstance(ts["start"], pd.Period):
                raise ValueError(f"{i}'th time-series must have a pandas Period as 'start', got {ts['start']}")

    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        static_features_df: Optional[pd.DataFrame] = None,
    ) -> TimeSeriesDataFrame:
        """Construct a ``TimeSeriesDataFrame`` from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            A pd.DataFrame with 'item_id' and 'timestamp' as columns. For example::

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
        id_column : str, optional
            Name of the 'item_id' column if column name is different
        timestamp_column : str, optional
            Name of the 'timestamp' column if column name is different
        static_features_df : pd.DataFrame, optional
            A pd.DataFrame with 'item_id' column that contains the static features for each time series. For example::

                   item_id feat_1   feat_2
                0        0 foo         0.5
                1        1 foo         2.2
                2        2 bar         0.1

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A data frame in TimeSeriesDataFrame format.
        """
        return cls(df, static_features=static_features_df, id_column=id_column, timestamp_column=timestamp_column)

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        static_features_path: Optional[Union[str, Path]] = None,
    ) -> TimeSeriesDataFrame:
        """Construct a ``TimeSeriesDataFrame`` from a CSV or Parquet file.

        Parameters
        ----------
        path : str or pathlib.Path
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

        id_column : str, optional
            Name of the 'item_id' column if column name is different
        timestamp_column : str, optional
            Name of the 'timestamp' column if column name is different
        static_features_path : str or pathlib.Path, optional
            Path to a local or remote (e.g., S3) file containing static features in CSV or Parquet format.
            Example file contents::

                item_id,feat_1,feat_2
                0,foo,0.5
                1,foo,2.2
                2,bar,0.1

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A data frame in TimeSeriesDataFrame format.
        """
        return cls(path, static_features=static_features_path, id_column=id_column, timestamp_column=timestamp_column)

    @classmethod
    def from_iterable_dataset(cls, iterable_dataset: Iterable, num_cpus: int = -1) -> TimeSeriesDataFrame:
        """Construct a ``TimeSeriesDataFrame`` from an Iterable of dictionaries each of which
        represent a single time series.

        This function also offers compatibility with GluonTS data sets, see
        https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.common.html#gluonts.dataset.common.ListDataset.

        Parameters
        ----------
        iterable_dataset: Iterable
            An iterator over dictionaries, each with a ``target`` field specifying the value of the
            (univariate) time series, and a ``start`` field with the starting time as a pandas Period .
            Example::

                iterable_dataset = [
                    {"target": [0, 1, 2], "start": pd.Period("01-01-2019", freq='D')},
                    {"target": [3, 4, 5], "start": pd.Period("01-01-2019", freq='D')},
                    {"target": [6, 7, 8], "start": pd.Period("01-01-2019", freq='D')}
                ]
        num_cpus : int, default = -1
            Number of CPU cores used to process the iterable dataset in parallel. Set to -1 to use all cores.

        Returns
        -------
        ts_df: TimeSeriesDataFrame
            A data frame in TimeSeriesDataFrame format.
        """
        return cls(iterable_dataset, num_cpus=num_cpus)

    @property
    def item_ids(self) -> pd.Index:
        return self.index.unique(level=ITEMID)

    @property
    def static_features(self):
        return self._static_features

    def _construct_static_features(
        cls,
        static_features: Union[pd.DataFrame, str, Path],
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        if isinstance(static_features, (str, Path)):
            static_features = load_pd.load(str(static_features))
        if not isinstance(static_features, pd.DataFrame):
            raise ValueError(
                f"static_features must be a pd.DataFrame, string or Path (received {type(static_features)})"
            )

        if id_column is not None:
            assert id_column in static_features.columns, f"Column '{id_column}' not found in static_features!"
            if id_column != ITEMID and ITEMID in static_features.columns:
                logger.warning(f"Renaming existing column '{ITEMID}' -> '__{ITEMID}' to avoid name collisions.")
                static_features.rename(columns={ITEMID: "__" + ITEMID}, inplace=True)
            static_features.rename(columns={id_column: ITEMID}, inplace=True)
        return static_features

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
            if isinstance(value.index, pd.MultiIndex):
                raise ValueError("static_features cannot have a MultiIndex")

            # Avoid modifying static features inplace
            value = value.copy()
            if ITEMID in value.columns and value.index.name != ITEMID:
                value = value.set_index(ITEMID)
            if value.index.name != ITEMID:
                value.index.rename(ITEMID, inplace=True)
            missing_item_ids = self.item_ids.difference(value.index)
            if len(missing_item_ids) > 0:
                raise ValueError(
                    "Following item_ids are missing from the index of static_features: "
                    f"{reprlib.repr(missing_item_ids.to_list())}"
                )
            # if provided static features are a strict superset of the item index, we take a subset to ensure consistency
            if len(value.index.difference(self.item_ids)) > 0:
                value = value.reindex(self.item_ids)

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
        >>> ts_df
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
        ts_df : TimeSeriesDataFrame
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
    def from_pickle(cls, filepath_or_buffer: Any) -> TimeSeriesDataFrame:
        """Convenience method to read pickled time series data frames. If the read pickle
        file refers to a plain pandas DataFrame, it will be cast to a TimeSeriesDataFrame.

        Parameters
        ----------
        filepath_or_buffer: Any
            Filename provided as a string or an ``IOBuffer`` containing the pickled object.

        Returns
        -------
        ts_df : TimeSeriesDataFrame
            The pickled time series data frame.
        """
        try:
            data = pd.read_pickle(filepath_or_buffer)
            return data if isinstance(data, cls) else cls(data)
        except Exception as err:  # noqa
            raise IOError(f"Could not load pickled data set due to error: {str(err)}")

    def fill_missing_values(self, method: str = "auto", value: float = 0.0) -> TimeSeriesDataFrame:
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
        >>> ts_df
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

        >>> ts_df.fill_missing_values(method="auto")
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
                "(for example, using the `convert_frequency` method)."
            )

        grouped_df = pd.DataFrame(self).groupby(level=ITEMID, sort=False, group_keys=False)
        if method == "auto":
            filled_df = grouped_df.ffill()
            # Fill missing values at the start of each time series with bfill
            filled_df = filled_df.groupby(level=ITEMID, sort=False, group_keys=False).bfill()
        elif method in ["ffill", "pad"]:
            filled_df = grouped_df.ffill()
        elif method in ["bfill", "backfill"]:
            filled_df = grouped_df.bfill()
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

    def dropna(self, how: str = "any") -> TimeSeriesDataFrame:
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

    def get_model_inputs_for_scoring(
        self, prediction_length: int, known_covariates_names: Optional[List[str]] = None
    ) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        """Prepare model inputs necessary to predict the last ``prediction_length`` time steps of each time series in the dataset.

        Parameters
        ----------
        prediction_length : int
            The forecast horizon, i.e., How many time steps into the future must be predicted.
        known_covariates_names : List[str], optional
            Names of the dataframe columns that contain covariates known in the future.
            See :attr:`known_covariates_names` of :class:`~autogluon.timeseries.TimeSeriesPredictor` for more details.

        Returns
        -------
        past_data : TimeSeriesDataFrame
            Data, where the last ``prediction_length`` time steps have been removed from the end of each time series.
        known_covariates : TimeSeriesDataFrame or None
            If ``known_covariates_names`` was provided, dataframe with the values of the known covariates during the
            forecast horizon. Otherwise, ``None``.
        """
        past_data = self.slice_by_timestep(None, -prediction_length)
        if known_covariates_names is not None and len(known_covariates_names) > 0:
            future_data = self.slice_by_timestep(-prediction_length, None)
            known_covariates = future_data[known_covariates_names]
        else:
            known_covariates = None
        return past_data, known_covariates

    def train_test_split(
        self,
        prediction_length: int,
        end_index: Optional[int] = None,
        suffix: Optional[str] = None,
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Generate a train/test split from the given dataset.
        This method can be used to generate splits for multi-window backtesting.

        Parameters
        ----------
        prediction_length : int
            Number of time steps in a single evaluation window.
        end_index : int, optional
            If given, all time series will be shortened up to ``end_idx`` before the train/test splitting. In other
            words, test data will include the slice ``[:end_index]`` of each time series, and train data will include
            the slice ``[:end_index - prediction_length]``.
        suffix : str, optional
            Suffix appended to all entries in the ``item_id`` index level.

        Returns
        -------
        train_data : TimeSeriesDataFrame
            Train portion of the data. Contains the slice ``[:-prediction_length]`` of each time series in ``test_data``.
        test_data : TimeSeriesDataFrame
            Test portion of the data. Contains the slice ``[:end_idx]`` of each time series in the original dataset.
        """
        test_data = self.slice_by_timestep(None, end_index)
        train_data = test_data.slice_by_timestep(None, -prediction_length)

        if suffix is not None:
            for data in [train_data, test_data]:
                new_item_id = data.index.levels[0].astype(str) + suffix
                data.index = data.index.set_levels(levels=new_item_id, level=0)
                if data.static_features is not None:
                    data.static_features.index = data.static_features.index.astype(str)
                    data.static_features.index += suffix
        return train_data, test_data

    def convert_frequency(
        self,
        freq: Union[str, pd.DateOffset],
        agg_numeric: str = "mean",
        agg_categorical: str = "first",
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Convert each time series in the data frame to the given frequency.

        This method is useful for two purposes:

        1. Converting an irregularly-sampled time series to a regular time index.
        2. Aggregating time series data by downsampling (e.g., convert daily sales into weekly sales)

        Parameters
        ----------
        freq : Union[str, pd.DateOffset]
            Frequency to which the data should be converted. See [pandas frequency aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
            for supported values.
        agg_numeric : {"max", "min", "sum", "mean", "median", "first", "last"}, default = "mean"
            Aggregation method applied to numeric columns.
        agg_categorical : {"first", "last"}, default = "first"
            Aggregation method applied to categorical columns.
        **kwargs
            Additional keywords arguments that will be passed to ``pandas.DataFrameGroupBy.resample``.

        Returns
        -------
        ts_df : TimeSeriesDataFrame
            A new time series dataframe with time series resampled at the new frequency. Output may contain missing
            values represented by ``NaN`` if original data does not have information for the given period.

        Examples
        --------
        Convert irregularly-sampled time series data to a regular index

        >>> ts_df
                            target
        item_id timestamp
        0       2019-01-01     NaN
                2019-01-03     1.0
                2019-01-06     2.0
                2019-01-07     NaN
        1       2019-02-04     3.0
                2019-02-07     4.0
        >>> ts_df.convert_frequency(freq="D")
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

        Downsample quarterly data to yearly frequency

        >>> ts_df
                            target
        item_id timestamp
        0       2020-03-31     1.0
                2020-06-30     2.0
                2020-09-30     3.0
                2020-12-31     4.0
                2021-03-31     5.0
                2021-06-30     6.0
                2021-09-30     7.0
                2021-12-31     8.0
        >>> ts_df.convert_frequency("Y")
                            target
        item_id timestamp
        0       2020-12-31     2.5
                2021-12-31     6.5
        >>> ts_df.convert_frequency("Y", agg_numeric="sum")
                            target
        item_id timestamp
        0       2020-12-31    10.0
                2021-12-31    26.0
        """
        if self.freq == pd.tseries.frequencies.to_offset(freq).freqstr:
            return self

        # We need to aggregate categorical columns separately because .agg("mean") deletes all non-numeric columns
        aggregation = {}
        for col in self.columns:
            if pd.api.types.is_numeric_dtype(self.dtypes[col]):
                aggregation[col] = agg_numeric
            else:
                aggregation[col] = agg_categorical

        resampled_df = TimeSeriesDataFrame(
            self.groupby(level=ITEMID, sort=False).resample(freq, level=TIMESTAMP, **kwargs).agg(aggregation)
        )
        resampled_df.static_features = self.static_features
        return resampled_df

    def __dir__(self) -> List[str]:
        # This hides method from IPython autocomplete, but not VSCode autocomplete
        deprecated = ["get_reindexed_view", "to_regular_index"]
        return [d for d in super().__dir__() if d not in deprecated]

    def get_reindexed_view(self, *args, **kwargs) -> TimeSeriesDataFrame:
        raise ValueError(
            "`TimeSeriesDataFrame.get_reindexed_view` has been deprecated. If your data has irregular timestamps, "
            "please convert it to a regular frequency with `convert_frequency`."
        )

    def to_regular_index(self, *args, **kwargs) -> TimeSeriesDataFrame:
        raise ValueError(
            "`TimeSeriesDataFrame.to_regular_index` has been deprecated. "
            "Please use `TimeSeriesDataFrame.convert_frequency` instead."
        )
