from typing import Optional, Tuple
import pandas as pd

from .dataset import TimeSeriesDataFrame


def append_suffix_to_item_id(ts_dataframe: TimeSeriesDataFrame, suffix: str) -> TimeSeriesDataFrame:
    """Append a suffix to each item_id in a TimeSeriesDataFrame.

    Returns a copy of the data, the original TimeSeriesDataFrame is not modified.
    """

    def add_suffix(multiindex_element):
        item_id, timestamp = multiindex_element
        return (f"{item_id}{suffix}", timestamp)

    result = ts_dataframe.copy()
    result.index = result.index.map(add_suffix)
    return result


class AbstractTimeSeriesSplitter:
    """A class that handles train / validation splitting of timeseries datasets."""

    def split(
        self, ts_dataframe: TimeSeriesDataFrame, prediction_length: int
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Split each series in the input dataset into one train and potentially multiple validation series.

        The `item_id` of each validation series is a string of format `f"{idx}_slice({start}, {end})". Here `idx` is
        the `item_id` of the input series, and `start` and `end` correspond to the slice parameters used to generate
        the validation series.

        For example, the series `val_data.loc["0_slice(None, -10)"]` in the validation dataset can equivalently be
        obtained as `ts_dataframe.loc[0].slice_by_timestep(slice(None, -10))`.

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


class SlidingWindowSplitter(AbstractTimeSeriesSplitter):
    """Slide window from the end of each series to generate validation series.

    Each val sequence has length `prediction_length`. We start by selecting the last `prediction_length` elements,
    an then end of each following val seqience gets shifted by `prediction_length - overlap` steps to the left.

    The validation set has up to `self.num_windows` as many items as the input dataset (can have fewer items if some
    training series are too short to split into `self.num_windows` many windows).

    Parameters
    ----------
    num_windows: int
        Number of windows to generate from each time series in the dataset.
    overlap: Optional[int], default = None
        Number of steps shared between two consecutive validation windows. When set to None, overlap is set
        automatically to `prediction_length // 2`.

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
        >>> train_data, tuning_data = splitter.split(ts_dataframe, prediction_length=3)
        >>> print(train_data)
                                target
        item_id timestamp
        0       1970-01-01       1
                1970-01-02       2
                1970-01-03       3
                1970-01-04       4
                1970-01-05       5
        >>> print(tuning_data)
                                            target
        item_id             timestamp
        0_slice(None, None) 1970-01-01       1
                            1970-01-02       2
                            1970-01-03       3
                            1970-01-04       4
                            1970-01-05       5
                            1970-01-06       6
                            1970-01-07       7
                            1970-01-08       8
                            1970-01-09       9
                            1970-01-10      10
        0_slice(None, -2)   1970-01-01       1
                            1970-01-02       2
                            1970-01-03       3
                            1970-01-04       4
                            1970-01-05       5
                            1970-01-06       6
                            1970-01-07       7
                            1970-01-08       8
    """

    def __init__(self, num_windows: int, overlap: Optional[int] = None):
        self.num_windows = num_windows
        self.overlap = overlap

    def _split(
        self, ts_dataframe: TimeSeriesDataFrame, prediction_length: int
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        if self.overlap is None:
            overlap = prediction_length // 2
        else:
            overlap = self.overlap
        step_size = prediction_length - overlap
        validation_dataframes = [append_suffix_to_item_id(ts_dataframe, "_slice(None, None)")]

        for window_idx in range(1, self.num_windows):
            ts_dataframe = ts_dataframe.slice_by_timestep(slice(None, -step_size))
            total_offset = step_size * window_idx
            next_val_dataframe = append_suffix_to_item_id(ts_dataframe, f"_slice(None, {-total_offset})")
            validation_dataframes.append(next_val_dataframe)

        train_data = ts_dataframe.slice_by_timestep(slice(None, -prediction_length))

        return train_data, pd.concat(validation_dataframes)


class LastWindowSplitter(SlidingWindowSplitter):
    """Reserves the last prediction interval of each time series for validation."""

    def __init__(self):
        super().__init__(num_windows=1)
