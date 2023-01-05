import logging
from typing import Tuple, Union

import pandas as pd

from .dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame

logger = logging.getLogger(__name__)


class AbstractTimeSeriesSplitter:
    """A class that handles train / validation splitting of timeseries datasets."""

    def split(
        self, ts_dataframe: TimeSeriesDataFrame, prediction_length: int
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Split each series in the input dataset into one train and potentially multiple validation series.

        The ``item_id`` of each validation series is a string of format ``f"{idx}_[{start}:{end}]"``. Here ``idx`` is
        the ``item_id`` of the input series, and ``start`` and ``end`` correspond to the slice parameters used to generate
        the validation series.

        For example, the series ``val_data.loc["0_[None:-10]"]`` in the validation dataset can equivalently be
        obtained as ``ts_dataframe.loc[0][None:-10]``.

        Parameters
        ----------
        ts_dataframe: TimeSeriesDataFrame
            Dataset containing of series that should be split.
        prediction_length: int
            The forecast horizon, i.e., how many time points into the future forecasters should be trained to predict.

        Returns
        -------
        train_data: TimeSeriesDataFrame
            Time series used for training. Has the same number of items as in the input dataset.
        val_data: TimeSeriesDataFrame
            Time series used for tuning / validation.
        """
        return self._split(ts_dataframe=ts_dataframe, prediction_length=prediction_length)

    def _split(
        self, ts_dataframe: TimeSeriesDataFrame, prediction_length: int
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        raise NotImplementedError

    def describe_validation_strategy(self, prediction_length: int) -> None:
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.name}()"

    @staticmethod
    def _append_suffix_to_item_id(
        ts_dataframe: Union[TimeSeriesDataFrame, pd.DataFrame], suffix: str
    ) -> TimeSeriesDataFrame:
        """Append a suffix to each item_id in a TimeSeriesDataFrame."""
        result = ts_dataframe.copy(deep=False)
        if result.index.nlevels == 1:
            result.index = result.index.astype(str) + suffix
        elif result.index.nlevels == 2:
            new_item_id = result.index.levels[0].astype(str) + suffix
            result.index = result.index.set_levels(levels=new_item_id, level=0)
        return result


class MultiWindowSplitter(AbstractTimeSeriesSplitter):
    """Reserve multiple windows at the end of each time series as the validation set.

    The first validation series contains the entire series (i.e. the last ``prediction_length`` elements are used for
    computing the validation score). For each following validation series we cut off the last ``prediction_length``
    time steps. This process is repeated until ``num_windows`` validation series are generated, or until all training
    series have length less than ``prediction_length + 1``.

    MultiWindowSplitter guarantees that each training series has length of at least ``prediction_length + 1``.

    Example:
        input_series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], prediction_length = 3, num_windows = 2

    Validation:
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # val score computed on [8, 9, 10]
        [1, 2, 3, 4, 5, 6, 7]  # val score computed on [5, 6, 7]

    Train:
        [1, 2, 3, 4]  # train loss computed on [1, 2, 3, 4]


    Parameters
    ----------
    num_windows: int, default = 3
        Number of windows to generate from each time series in the dataset.

    Examples
    --------
    >>> print(ts_dataframe)
                        target
    item_id timestamp
    0       1970-01-01       1
            1970-01-02       2
            1970-01-03       3
            1970-01-04       4
            1970-01-05       5
            1970-01-06       6
            1970-01-07       7
            1970-01-08       8
            1970-01-09       9
            1970-01-10      10

    >>> splitter = SlidingWindowSplitter(num_windows=2)
    >>> train_data, val_data = splitter.split(ts_dataframe, prediction_length=3)
    >>> print(train_data)
                            target
    item_id timestamp
    0       1970-01-01       1
            1970-01-02       2
            1970-01-03       3
            1970-01-04       4

    >>> print(val_data)
                                target
    item_id         timestamp
    0_[None:None]   1970-01-01       1
                    1970-01-02       2
                    1970-01-03       3
                    1970-01-04       4
                    1970-01-05       5
                    1970-01-06       6
                    1970-01-07       7
                    1970-01-08       8
                    1970-01-09       9
                    1970-01-10      10
    0_[None:-3]     1970-01-01       1
                    1970-01-02       2
                    1970-01-03       3
                    1970-01-04       4
                    1970-01-05       5
                    1970-01-06       6
                    1970-01-07       7
    """

    def __init__(self, num_windows: int = 3):
        self.num_windows = num_windows

    def describe_validation_strategy(self, prediction_length):
        return (
            f"Will use the last {self.num_windows} windows (each with prediction_length = {prediction_length} time "
            f"steps) as a hold-out validation set."
        )

    def _split(
        self, ts_dataframe: TimeSeriesDataFrame, prediction_length: int
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        original_freq = ts_dataframe.freq

        static_features_available = ts_dataframe.static_features is not None

        train_dataframes = []
        validation_dataframes = []
        train_static_features = []
        validation_static_features = []
        for window_idx in range(self.num_windows):
            num_timesteps_per_item = ts_dataframe.num_timesteps_per_item()

            item_index = num_timesteps_per_item.index
            long_enough = num_timesteps_per_item > 2 * prediction_length
            # Convert boolean indicator into item_id index
            can_be_split = item_index[long_enough]
            cannot_be_split = item_index[~long_enough]

            train_dataframes.append(ts_dataframe.query("item_id in @cannot_be_split"))
            if static_features_available:
                train_static_features.append(ts_dataframe.static_features.query("item_id in @cannot_be_split"))
            # Keep timeseries that are long enough for the next round of splitting
            ts_dataframe = ts_dataframe.query("item_id in @can_be_split")

            if window_idx == 0:
                suffix = "_[None:None]"
                # TODO: Should we also warn users if there are too few items in the validation set?
                if len(can_be_split) == 0:
                    raise ValueError(
                        f"Cannot create a validation set because all training time series are too short. "
                        f"At least some time series in train_data must have length >= 2 * prediction_length + 1 "
                        f"(at least {2 * prediction_length + 1}) but the longest training series has length "
                        f"{num_timesteps_per_item.max()}. Please decrease prediction_length, provide longer "
                        f"time series in train_data, or provide tuning_data."
                    )
            else:
                suffix = f"_[None:-{window_idx * prediction_length}]"
                if len(can_be_split) == 0:
                    break

            validation_dataframes.append(self._append_suffix_to_item_id(ts_dataframe, suffix))
            if static_features_available:
                validation_static_features.append(self._append_suffix_to_item_id(ts_dataframe.static_features, suffix))

            ts_dataframe = ts_dataframe.slice_by_timestep(None, -prediction_length)

        train_dataframes.append(ts_dataframe)

        train_data = pd.concat(train_dataframes)
        train_data._cached_freq = original_freq
        val_data = pd.concat(validation_dataframes)

        val_data = pd.concat(validation_dataframes)
        val_data._cached_freq = original_freq

        if static_features_available:
            train_static_features.append(ts_dataframe.static_features)
            train_data.static_features = pd.concat(train_static_features)
            val_data.static_features = pd.concat(validation_static_features)

        return train_data, val_data


class LastWindowSplitter(MultiWindowSplitter):
    """Reserves the last prediction_length steps of each time series for validation."""

    def __init__(self):
        super().__init__(num_windows=1)

    def describe_validation_strategy(self, prediction_length: int):
        return (
            f"Will use the last prediction_length = {prediction_length} time steps of each time series as a hold-out "
            "validation set."
        )
