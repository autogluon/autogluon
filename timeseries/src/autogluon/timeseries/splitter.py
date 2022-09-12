import logging
from typing import Tuple

import pandas as pd

from .dataset import TimeSeriesDataFrame

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


def append_suffix_to_item_id(ts_dataframe: TimeSeriesDataFrame, suffix: str) -> TimeSeriesDataFrame:
    """Append a suffix to each item_id in a TimeSeriesDataFrame."""

    def add_suffix(multiindex_element):
        item_id, timestamp = multiindex_element
        return (f"{item_id}{suffix}", timestamp)

    result = ts_dataframe.copy(deep=False)
    result.index = result.index.map(add_suffix)
    return result


class MultiWindowSplitter(AbstractTimeSeriesSplitter):
    """Slide window from the end of each series to generate validation series.

    The first valdation series contains the entire series (i.e. the last ``prediction_length`` elements are used for
    computing the validation score). The end of each following validation series is moved
    ``prediction_length - overlap`` steps to the left.

    The validation set has up to ``self.num_windows`` as many items as the input dataset (can have fewer items if some
    training series are too short to split into ``self.num_windows`` many windows).

    Example: input_series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], prediction_length = 3, overlap = 1, num_windows = 2

    Validation:
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # val score computed on [8, 9, 10]
        [1, 2, 3, 4, 5, 6, 7, 8]  # val score computed on [6, 7, 8]

    Train:
        [1, 2, 3, 4, 5]  # train loss computed on [1, 2, 3, 4, 5]


    Parameters
    ----------
    num_windows: int, default = 3
        Number of windows to generate from each time series in the dataset.
    overlap: int, default = 0
        Number of steps shared between two consecutive validation windows. Can be used to increase the # of validation
        windows while keeping more data for training.

    Example
    -------
    .. code-block:: python

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
        >>> splitter = SlidingWindowSplitter(num_windows=2, overlap=1)
        >>> train_data, val_data = splitter.split(ts_dataframe, prediction_length=3)
        >>> print(train_data)
                                target
        item_id timestamp
        0       1970-01-01       1
                1970-01-02       2
                1970-01-03       3
                1970-01-04       4
                1970-01-05       5
        >>> print(val_data)
                                    target
        item_id       timestamp
        0_[None:None] 1970-01-01       1
                      1970-01-02       2
                      1970-01-03       3
                      1970-01-04       4
                      1970-01-05       5
                      1970-01-06       6
                      1970-01-07       7
                      1970-01-08       8
                      1970-01-09       9
                      1970-01-10      10
        0_[None:-2]   1970-01-01       1
                      1970-01-02       2
                      1970-01-03       3
                      1970-01-04       4
                      1970-01-05       5
                      1970-01-06       6
                      1970-01-07       7
                      1970-01-08       8
    """

    def __init__(self, num_windows: int = 3, overlap: int = 0):
        self.num_windows = num_windows
        self.overlap = overlap

    def describe_validation_strategy(self, prediction_length):
        return (
            f"Will use the last {self.num_windows} windows (each with prediction_length {prediction_length} "
            f"time steps and overlap {self.overlap}) as a hold-out validation set."
        )

    def _split(
        self, ts_dataframe: TimeSeriesDataFrame, prediction_length: int
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        if self.overlap >= prediction_length:
            raise ValueError(
                f"SlidingWindowSplitter.overlap {self.overlap} must be < prediction_length {prediction_length}"
            )
        step_size = prediction_length - self.overlap

        length_per_series = ts_dataframe.index.get_level_values(0).value_counts(sort=False)
        num_total_validation_steps = prediction_length + step_size * (self.num_windows - 1)
        num_too_short_series = (length_per_series <= num_total_validation_steps).sum()
        if num_too_short_series > 0:
            logger.warning(f"{num_too_short_series} are too short and won't appear in the training set")
        if num_too_short_series == ts_dataframe.num_items:
            raise ValueError(f"{self.name} produced an empty training set since all sequences are too short")

        validation_dataframes = [append_suffix_to_item_id(ts_dataframe, "_[None:None]")]

        for window_idx in range(1, self.num_windows):
            ts_dataframe = ts_dataframe.slice_by_timestep(None, -step_size)
            total_offset = step_size * window_idx
            next_val_dataframe = append_suffix_to_item_id(ts_dataframe, f"_[None:{-total_offset}]")
            validation_dataframes.append(next_val_dataframe)

        train_data = ts_dataframe.slice_by_timestep(None, -prediction_length)
        val_data = pd.concat(validation_dataframes)
        val_data._cached_freq = train_data._cached_freq

        return train_data, val_data


class LastWindowSplitter(MultiWindowSplitter):
    """Reserves the last prediction_length steps of each time series for validation."""

    def __init__(self):
        super().__init__(num_windows=1)

    def describe_validation_strategy(self, prediction_length: int):
        return f"Will use the last prediction_length {prediction_length} time steps as a hold-out validation set."
