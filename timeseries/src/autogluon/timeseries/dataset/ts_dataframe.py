from __future__ import annotations

import copy
import itertools
import logging
import reprlib
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Type, overload

import numpy as np
import pandas as pd
from joblib.parallel import Parallel, delayed
from pandas.core.internals import ArrayManager, BlockManager  # type: ignore
from typing_extensions import Self

from autogluon.common.loaders import load_pd

logger = logging.getLogger(__name__)


class TimeSeriesDataFrame(pd.DataFrame):
    """A collection of univariate time series, where each row is identified by an (``item_id``, ``timestamp``) pair.

    For example, a time series dataframe could represent the daily sales of a collection of products, where each
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
        An optional dataframe describing the metadata of each individual time series that does not change with time.
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

    """

    index: pd.MultiIndex  # type: ignore
    _metadata = ["_static_features"]

    IRREGULAR_TIME_INDEX_FREQSTR: Final[str] = "IRREG"
    ITEMID: Final[str] = "item_id"
    TIMESTAMP: Final[str] = "timestamp"

    def __init__(
        self,
        data: pd.DataFrame | str | Path | Iterable,
        static_features: pd.DataFrame | str | Path | None = None,
        id_column: str | None = None,
        timestamp_column: str | None = None,
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
        super().__init__(data=data, *args, **kwargs)  # type: ignore
        self._static_features: pd.DataFrame | None = None
        if static_features is not None:
            self.static_features = self._construct_static_features(static_features, id_column=id_column)

    @property
    def _constructor(self) -> Type[TimeSeriesDataFrame]:
        return TimeSeriesDataFrame

    def _constructor_from_mgr(self, mgr, axes):
        # Use the default constructor when constructing from _mgr. Otherwise pandas enters an infinite recursion by
        # repeatedly calling TimeSeriesDataFrame constructor
        df = self._from_mgr(mgr, axes=axes)
        df._static_features = self._static_features
        return df

    @classmethod
    def _construct_tsdf_from_data_frame(
        cls,
        df: pd.DataFrame,
        id_column: str | None = None,
        timestamp_column: str | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        if id_column is not None:
            assert id_column in df.columns, f"Column '{id_column}' not found!"
            if id_column != cls.ITEMID and cls.ITEMID in df.columns:
                logger.warning(
                    f"Renaming existing column '{cls.ITEMID}' -> '__{cls.ITEMID}' to avoid name collisions."
                )
                df.rename(columns={cls.ITEMID: "__" + cls.ITEMID}, inplace=True)
            df.rename(columns={id_column: cls.ITEMID}, inplace=True)

        if timestamp_column is not None:
            assert timestamp_column in df.columns, f"Column '{timestamp_column}' not found!"
            if timestamp_column != cls.TIMESTAMP and cls.TIMESTAMP in df.columns:
                logger.warning(
                    f"Renaming existing column '{cls.TIMESTAMP}' -> '__{cls.TIMESTAMP}' to avoid name collisions."
                )
                df.rename(columns={cls.TIMESTAMP: "__" + cls.TIMESTAMP}, inplace=True)
            df.rename(columns={timestamp_column: cls.TIMESTAMP}, inplace=True)

        if cls.TIMESTAMP in df.columns:
            df[cls.TIMESTAMP] = pd.to_datetime(df[cls.TIMESTAMP])

        cls._validate_data_frame(df)
        return df.set_index([cls.ITEMID, cls.TIMESTAMP])

    @classmethod
    def _construct_tsdf_from_iterable_dataset(cls, iterable_dataset: Iterable, num_cpus: int = -1) -> pd.DataFrame:
        def load_single_item(item_id: int, ts: dict) -> pd.DataFrame:
            start_timestamp = ts["start"]
            freq = start_timestamp.freq
            if isinstance(start_timestamp, pd.Period):
                start_timestamp = start_timestamp.to_timestamp(how="S")
            target = ts["target"]
            datetime_index = tuple(pd.date_range(start_timestamp, periods=len(target), freq=freq))
            idx = pd.MultiIndex.from_product([(item_id,), datetime_index], names=[cls.ITEMID, cls.TIMESTAMP])
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
        if not pd.api.types.is_datetime64_dtype(data.index.dtypes[cls.TIMESTAMP]):
            raise ValueError(f"for {cls.TIMESTAMP}, the only pandas dtype allowed is `datetime64`.")
        if not data.index.names == (f"{cls.ITEMID}", f"{cls.TIMESTAMP}"):
            raise ValueError(
                f"data must have index names as ('{cls.ITEMID}', '{cls.TIMESTAMP}'), got {data.index.names}"
            )
        item_id_index = data.index.levels[0]
        if not (pd.api.types.is_integer_dtype(item_id_index) or pd.api.types.is_string_dtype(item_id_index)):
            raise ValueError(f"all entries in index `{cls.ITEMID}` must be of integer or string dtype")

    @classmethod
    def _validate_data_frame(cls, df: pd.DataFrame):
        """Validate that a pd.DataFrame with ITEMID and TIMESTAMP columns can be converted to TimeSeriesDataFrame"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"data must be a pd.DataFrame, got {type(df)}")
        if cls.ITEMID not in df.columns:
            raise ValueError(f"data must have a `{cls.ITEMID}` column")
        if cls.TIMESTAMP not in df.columns:
            raise ValueError(f"data must have a `{cls.TIMESTAMP}` column")
        if df[cls.ITEMID].isnull().any():
            raise ValueError(f"`{cls.ITEMID}` column can not have nan")
        if df[cls.TIMESTAMP].isnull().any():
            raise ValueError(f"`{cls.TIMESTAMP}` column can not have nan")
        if not pd.api.types.is_datetime64_dtype(df[cls.TIMESTAMP]):
            raise ValueError(f"for {cls.TIMESTAMP}, the only pandas dtype allowed is `datetime64`.")
        item_id_column = df[cls.ITEMID]
        if not (pd.api.types.is_integer_dtype(item_id_column) or pd.api.types.is_string_dtype(item_id_column)):
            raise ValueError(f"all entries in column `{cls.ITEMID}` must be of integer or string dtype")

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
        id_column: str | None = None,
        timestamp_column: str | None = None,
        static_features_df: pd.DataFrame | None = None,
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
            A dataframe in TimeSeriesDataFrame format.
        """
        return cls(df, static_features=static_features_df, id_column=id_column, timestamp_column=timestamp_column)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        id_column: str | None = None,
        timestamp_column: str | None = None,
        static_features_path: str | Path | None = None,
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
            A dataframe in TimeSeriesDataFrame format.
        """
        return cls(path, static_features=static_features_path, id_column=id_column, timestamp_column=timestamp_column)

    @classmethod
    def from_iterable_dataset(cls, iterable_dataset: Iterable, num_cpus: int = -1) -> TimeSeriesDataFrame:
        """Construct a ``TimeSeriesDataFrame`` from an Iterable of dictionaries each of which
        represent a single time series.

        This function also offers compatibility with GluonTS `ListDataset format <https://ts.gluon.ai/stable/api/gluonts/gluonts.dataset.common.html#gluonts.dataset.common.ListDataset>`_.

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
            A dataframe in TimeSeriesDataFrame format.
        """
        return cls(iterable_dataset, num_cpus=num_cpus)

    @property
    def item_ids(self) -> pd.Index:
        """List of unique time series IDs contained in the data set."""
        return self.index.unique(level=self.ITEMID)

    @classmethod
    def _construct_static_features(
        cls,
        static_features: pd.DataFrame | str | Path,
        id_column: str | None = None,
    ) -> pd.DataFrame:
        if isinstance(static_features, (str, Path)):
            static_features = load_pd.load(str(static_features))
        if not isinstance(static_features, pd.DataFrame):
            raise ValueError(
                f"static_features must be a pd.DataFrame, string or Path (received {type(static_features)})"
            )

        if id_column is not None:
            assert id_column in static_features.columns, f"Column '{id_column}' not found in static_features!"
            if id_column != cls.ITEMID and cls.ITEMID in static_features.columns:
                logger.warning(
                    f"Renaming existing column '{cls.ITEMID}' -> '__{cls.ITEMID}' to avoid name collisions."
                )
                static_features.rename(columns={cls.ITEMID: "__" + cls.ITEMID}, inplace=True)
            static_features.rename(columns={id_column: cls.ITEMID}, inplace=True)
        return static_features

    @property
    def static_features(self):
        return self._static_features

    @static_features.setter
    def static_features(self, value: pd.DataFrame | None):
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
            if self.ITEMID in value.columns and value.index.name != self.ITEMID:
                value = value.set_index(self.ITEMID)
            if value.index.name != self.ITEMID:
                value.index.rename(self.ITEMID, inplace=True)
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

    def infer_frequency(self, num_items: int | None = None, raise_if_irregular: bool = False) -> str:
        """Infer the time series frequency based on the timestamps of the observations.

        Parameters
        ----------
        num_items : int or None, default = None
            Number of items (individual time series) randomly selected to infer the frequency. Lower values speed up
            the method, but increase the chance that some items with invalid frequency are missed by subsampling.

            If set to ``None``, all items will be used for inferring the frequency.
        raise_if_irregular : bool, default = False
            If True, an exception will be raised if some items have an irregular frequency, or if different items have
            different frequencies.

        Returns
        -------
        freq : str
            If all time series have a regular frequency, returns a pandas-compatible `frequency alias <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.

            If some items have an irregular frequency or if different items have different frequencies, returns string
            ``IRREG``.
        """
        ts_df = self
        if num_items is not None and ts_df.num_items > num_items:
            items_subset = ts_df.item_ids.to_series().sample(n=num_items, random_state=123)
            ts_df = ts_df.loc[items_subset]

        if not ts_df.index.is_monotonic_increasing:
            ts_df = ts_df.sort_index()

        indptr = ts_df.get_indptr()
        item_ids = ts_df.item_ids
        timestamps = ts_df.index.get_level_values(level=1)
        candidate_freq = ts_df.index.levels[1].freq

        frequencies = []
        irregular_items = []
        for i in range(len(indptr) - 1):
            start, end = indptr[i], indptr[i + 1]
            item_timestamps = timestamps[start:end]
            inferred_freq = item_timestamps.inferred_freq

            # Fallback option: maybe original index has a `freq` attribute that pandas fails to infer (e.g., 'SME')
            if inferred_freq is None and candidate_freq is not None:
                try:
                    # If this line does not raise an exception, then candidate_freq is a compatible frequency
                    item_timestamps.freq = candidate_freq
                except ValueError:
                    inferred_freq = None
                else:
                    inferred_freq = candidate_freq.freqstr

            if inferred_freq is None:
                irregular_items.append(item_ids[i])
            else:
                frequencies.append(inferred_freq)

        unique_freqs = list(set(frequencies))
        if len(unique_freqs) != 1 or len(irregular_items) > 0:
            if raise_if_irregular:
                if irregular_items:
                    raise ValueError(
                        f"Cannot infer frequency. Items with irregular frequency: {reprlib.repr(irregular_items)}"
                    )
                else:
                    raise ValueError(f"Cannot infer frequency. Multiple frequencies detected: {unique_freqs}")
            else:
                return self.IRREGULAR_TIME_INDEX_FREQSTR
        else:
            return pd.tseries.frequencies.to_offset(unique_freqs[0]).freqstr

    @property
    def freq(self):
        """Inferred pandas-compatible frequency of the timestamps in the dataframe.

        Computed using a random subset of the time series for speed. This may sometimes result in incorrectly inferred
        values. For reliable results, use :meth:`~autogluon.timeseries.TimeSeriesDataFrame.infer_frequency`.
        """
        inferred_freq = self.infer_frequency(num_items=50)
        return None if inferred_freq == self.IRREGULAR_TIME_INDEX_FREQSTR else inferred_freq

    @property
    def num_items(self):
        """Number of items (time series) in the data set."""
        return len(self.item_ids)

    def num_timesteps_per_item(self) -> pd.Series:
        """Number of observations in each time series in the dataframe.

        Returns a ``pandas.Series`` with ``item_id`` as index and number of observations per item as values.
        """
        counts = pd.Series(self.index.codes[0]).value_counts(sort=False)
        counts.index = self.index.levels[0][counts.index]
        return counts

    def copy(self: TimeSeriesDataFrame, deep: bool = True) -> TimeSeriesDataFrame:
        """Make a copy of the TimeSeriesDataFrame.

        When ``deep=True`` (default), a new object will be created with a copy of the calling object's data and
        indices. Modifications to the data or indices of the copy will not be reflected in the original object.

        When ``deep=False``, a new object will be created without copying the calling object's data or index (only
        references to the data and index are copied). Any changes to the data of the original will be reflected in the
        shallow copy (and vice versa).

        For more details, see `pandas documentation <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.copy.html>`_.
        """
        obj = super().copy(deep=deep)

        # also perform a deep copy for static features
        if deep:
            for k in obj._metadata:
                setattr(obj, k, copy.deepcopy(getattr(obj, k)))
        return obj

    def __finalize__(  # noqa
        self: TimeSeriesDataFrame, other, method: str | None = None, **kwargs
    ) -> TimeSeriesDataFrame:
        super().__finalize__(other=other, method=method, **kwargs)
        # when finalizing the copy/slice operation, we use the property setter to stay consistent
        # with the item index
        if hasattr(other, "_static_features"):
            self.static_features = other._static_features
        return self

    def split_by_time(self, cutoff_time: pd.Timestamp) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Split dataframe to two different ``TimeSeriesDataFrame`` s before and after a certain ``cutoff_time``.

        Parameters
        ----------
        cutoff_time: pd.Timestamp
            The time to split the current dataframe into two dataframes.

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
        return before, after

    def slice_by_timestep(self, start_index: int | None = None, end_index: int | None = None) -> TimeSeriesDataFrame:
        """Select a subsequence from each time series between start (inclusive) and end (exclusive) indices.

        This operation is equivalent to selecting a slice ``[start_index : end_index]`` from each time series, and then
        combining these slices into a new ``TimeSeriesDataFrame``. See examples below.

        It is recommended to sort the index with ``ts_df.sort_index()`` before calling this method to take advantage of
        a fast optimized algorithm.

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

        if start_index is None and end_index is None:
            # Return a copy to avoid in-place modification.
            # self.copy() is much faster than self.loc[ones(len(self), dtype=bool)]
            return self.copy()

        if self.index.is_monotonic_increasing:
            # Use a fast optimized algorithm if the index is sorted
            indptr = self.get_indptr()
            lengths = np.diff(indptr)
            starts = indptr[:-1]

            slice_start = (
                np.zeros_like(lengths)
                if start_index is None
                else np.clip(np.where(start_index >= 0, start_index, lengths + start_index), 0, lengths)
            )
            slice_end = (
                lengths.copy()
                if end_index is None
                else np.clip(np.where(end_index >= 0, end_index, lengths + end_index), 0, lengths)
            )

            # Filter out invalid slices where start >= end
            valid_slices = slice_start < slice_end
            if not np.any(valid_slices):
                # Return empty dataframe with same structure
                return self.loc[np.zeros(len(self), dtype=bool)]

            starts = starts[valid_slices]
            slice_start = slice_start[valid_slices]
            slice_end = slice_end[valid_slices]

            # We put 1 at the slice_start index for each item and -1 at the slice_end index for each item.
            # After we apply cumsum we get the indicator mask selecting values between slice_start and slice_end
            # cumsum([0, 0, 1, 0, 0, -1, 0]) -> [0, 0, 1, 1, 1, 0, 0]
            # We need array of size len(self) + 1 in case events[starts + slice_end] tries to access position len(self)
            events = np.zeros(len(self) + 1, dtype=np.int8)
            events[starts + slice_start] += 1
            events[starts + slice_end] -= 1
            mask = np.cumsum(events)[:-1].astype(bool)
            # loc[mask] returns a view of the original data - modifying it will produce a SettingWithCopyWarning
            return self.loc[mask]
        else:
            # Fall back to a slow groupby operation
            result = self.groupby(level=self.ITEMID, sort=False, as_index=False).nth(slice(start_index, end_index))
            result.static_features = self.static_features
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
        """Convenience method to read pickled time series dataframes. If the read pickle
        file refers to a plain pandas DataFrame, it will be cast to a TimeSeriesDataFrame.

        Parameters
        ----------
        filepath_or_buffer: Any
            Filename provided as a string or an ``IOBuffer`` containing the pickled object.

        Returns
        -------
        ts_df : TimeSeriesDataFrame
            The pickled time series dataframe.
        """
        try:
            data = pd.read_pickle(filepath_or_buffer)
            return data if isinstance(data, cls) else cls(data)
        except Exception as err:  # noqa
            raise IOError(f"Could not load pickled data set due to error: {str(err)}")

    def fill_missing_values(self, method: str = "auto", value: float = 0.0) -> TimeSeriesDataFrame:
        """Fill missing values represented by NaN.

        .. note::
            This method assumes that the index of the TimeSeriesDataFrame is sorted by [item_id, timestamp].

            If the index is not sorted, this method will log a warning and may produce an incorrect result.

        Parameters
        ----------
        method : str, default = "auto"
            Method used to impute missing values.

            - ``"auto"`` - first forward fill (to fill the in-between and trailing NaNs), then backward fill (to fill the leading NaNs)
            - ``"ffill"`` or ``"pad"`` - propagate last valid observation forward. Note: missing values at the start of the time series are not filled.
            - ``"bfill"`` or ``"backfill"`` - use next valid observation to fill gap. Note: this may result in information leakage; missing values at the end of the time series are not filled.
            - ``"constant"`` - replace NaNs with the given constant ``value``.
            - ``"interpolate"`` - fill NaN values using linear interpolation. Note: this may result in information leakage.
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
        # Convert to pd.DataFrame for faster processing
        df = pd.DataFrame(self)

        # Skip filling if there are no NaNs
        if not df.isna().any(axis=None):
            return self

        if not self.index.is_monotonic_increasing:
            logger.warning(
                "Trying to fill missing values in an unsorted dataframe. "
                "It is highly recommended to call `ts_df.sort_index()` before calling `ts_df.fill_missing_values()`"
            )

        grouped_df = df.groupby(level=self.ITEMID, sort=False, group_keys=False)
        if method == "auto":
            filled_df = grouped_df.ffill()
            # If necessary, fill missing values at the start of each time series with bfill
            if filled_df.isna().any(axis=None):
                filled_df = filled_df.groupby(level=self.ITEMID, sort=False, group_keys=False).bfill()
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

    def dropna(self, how: str = "any") -> TimeSeriesDataFrame:  # type: ignore[override]
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

    # added for static type checker compatibility
    def assign(self, **kwargs) -> TimeSeriesDataFrame:
        """Assign new columns to the time series dataframe. See :meth:`pandas.DataFrame.assign` for details."""
        return super().assign(**kwargs)  # type: ignore

    # added for static type checker compatibility
    def sort_index(self, *args, **kwargs) -> TimeSeriesDataFrame:
        return super().sort_index(*args, **kwargs)  # type: ignore

    def get_model_inputs_for_scoring(
        self, prediction_length: int, known_covariates_names: list[str] | None = None
    ) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame | None]:
        """Prepare model inputs necessary to predict the last ``prediction_length`` time steps of each time series in the dataset.

        Parameters
        ----------
        prediction_length : int
            The forecast horizon, i.e., How many time steps into the future must be predicted.
        known_covariates_names : list[str], optional
            Names of the dataframe columns that contain covariates known in the future.
            See ``known_covariates_names`` of :class:`~autogluon.timeseries.TimeSeriesPredictor` for more details.

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
        end_index: int | None = None,
        suffix: str | None = None,
    ) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Generate a train/test split from the given dataset.

        This method can be used to generate splits for multi-window backtesting.

        .. note::
            This method automatically sorts the TimeSeriesDataFrame by [item_id, timestamp].

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
        df = self
        if not df.index.is_monotonic_increasing:
            logger.warning("Sorting the dataframe index before generating the train/test split.")
            df = df.sort_index()
        test_data = df.slice_by_timestep(None, end_index)
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
        freq: str | pd.DateOffset,
        agg_numeric: str = "mean",
        agg_categorical: str = "first",
        num_cpus: int = -1,
        chunk_size: int = 100,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Convert each time series in the dataframe to the given frequency.

        This method is useful for two purposes:

        1. Converting an irregularly-sampled time series to a regular time index.
        2. Aggregating time series data by downsampling (e.g., convert daily sales into weekly sales)

        Standard ``df.groupby(...).resample(...)`` can be extremely slow for large datasets, so we parallelize this
        operation across multiple CPU cores.

        Parameters
        ----------
        freq : str | pd.DateOffset
            Frequency to which the data should be converted. See `pandas frequency aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            for supported values.
        agg_numeric : {"max", "min", "sum", "mean", "median", "first", "last"}, default = "mean"
            Aggregation method applied to numeric columns.
        agg_categorical : {"first", "last"}, default = "first"
            Aggregation method applied to categorical columns.
        num_cpus : int, default = -1
            Number of CPU cores used when resampling in parallel. Set to -1 to use all cores.
        chunk_size : int, default = 100
            Number of time series in a chunk assigned to each parallel worker.
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
        >>> ts_df.convert_frequency("YE")
                            target
        item_id timestamp
        0       2020-12-31     2.5
                2021-12-31     6.5
        >>> ts_df.convert_frequency("YE", agg_numeric="sum")
                            target
        item_id timestamp
        0       2020-12-31    10.0
                2021-12-31    26.0
        """
        offset = pd.tseries.frequencies.to_offset(freq)

        # We need to aggregate categorical columns separately because .agg("mean") deletes all non-numeric columns
        aggregation = {}
        for col in self.columns:
            if pd.api.types.is_numeric_dtype(self.dtypes[col]):
                aggregation[col] = agg_numeric
            else:
                aggregation[col] = agg_categorical

        def split_into_chunks(iterable: Iterable, size: int) -> Iterable[Iterable]:
            # Based on https://stackoverflow.com/a/22045226/5497447
            iterable = iter(iterable)
            return iter(lambda: tuple(islice(iterable, size)), ())

        def resample_chunk(chunk: Iterable[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
            resampled_dfs = []
            for item_id, df in chunk:
                resampled_df = df.resample(offset, level=self.TIMESTAMP, **kwargs).agg(aggregation)
                resampled_dfs.append(pd.concat({item_id: resampled_df}, names=[self.ITEMID]))
            return pd.concat(resampled_dfs)

        # Resampling time for 1 item < overhead time for a single parallel job. Therefore, we group items into chunks
        # so that the speedup from parallelization isn't dominated by the communication costs.
        df = pd.DataFrame(self)
        # Make sure that timestamp index has dtype 'datetime64[ns]', otherwise index may contain NaT values.
        # See https://github.com/autogluon/autogluon/issues/4917
        df.index = df.index.set_levels(df.index.levels[1].astype("datetime64[ns]"), level=self.TIMESTAMP)
        chunks = split_into_chunks(df.groupby(level=self.ITEMID, sort=False), chunk_size)
        resampled_chunks = Parallel(n_jobs=num_cpus)(delayed(resample_chunk)(chunk) for chunk in chunks)
        resampled_df = TimeSeriesDataFrame(pd.concat(resampled_chunks))
        resampled_df.static_features = self.static_features
        return resampled_df

    def to_data_frame(self) -> pd.DataFrame:
        """Convert ``TimeSeriesDataFrame`` to a ``pandas.DataFrame``"""
        return pd.DataFrame(self)

    def get_indptr(self) -> np.ndarray:
        """[Advanced] Get a numpy array of shape [num_items + 1] that points to the start and end of each time series.

        This method assumes that the TimeSeriesDataFrame is sorted by [item_id, timestamp].
        """
        return np.concatenate([[0], np.cumsum(self.num_timesteps_per_item().to_numpy())]).astype(np.int32)

    # inline typing stubs for various overridden methods
    if TYPE_CHECKING:

        def query(  # type: ignore
            self, expr: str, *, inplace: bool = False, **kwargs
        ) -> Self: ...

        def reindex(*args, **kwargs) -> Self: ...  # type: ignore

        @overload
        def __new__(cls, data: pd.DataFrame, static_features: pd.DataFrame | None = None) -> Self: ...  # type: ignore
        @overload
        def __new__(
            cls,
            data: pd.DataFrame | str | Path | Iterable,
            static_features: pd.DataFrame | str | Path | None = None,
            id_column: str | None = None,
            timestamp_column: str | None = None,
            num_cpus: int = -1,
            *args,
            **kwargs,
        ) -> Self:
            """This overload is needed since in pandas, during type checking, the default constructor resolves to __new__"""
            ...

        @overload
        def __getitem__(self, items: list[str]) -> Self: ...  # type: ignore
        @overload
        def __getitem__(self, item: str) -> pd.Series: ...  # type: ignore


# TODO: remove with v2.0
# module-level constants kept for backward compatibility.
ITEMID = TimeSeriesDataFrame.ITEMID
TIMESTAMP = TimeSeriesDataFrame.TIMESTAMP
