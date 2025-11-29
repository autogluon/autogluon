from typing import Iterator

from autogluon.timeseries.dataset import TimeSeriesDataFrame

__all__ = [
    "AbstractWindowSplitter",
    "ExpandingWindowSplitter",
]


class AbstractWindowSplitter:
    def __init__(self, prediction_length: int, num_val_windows: int = 1):
        self.prediction_length = prediction_length
        self.num_val_windows = num_val_windows

    def split(self, data: TimeSeriesDataFrame) -> Iterator[tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]]:
        raise NotImplementedError


class ExpandingWindowSplitter(AbstractWindowSplitter):
    """For each train / validation split, training data includes all available past data.

    For example, for ``prediction_length=2``, ``num_val_windows=3`` and ``val_step_size=1`` the folds are::

        |-------------------|
        | x x x x x y y - - |
        | x x x x x x y y - |
        | x x x x x x x y y |

    where ``x`` are the train time steps and ``y`` are the validation time steps.

    Train data includes only time steps denoted by ``x``, and validation data includes both ``x`` and ``y`` time steps.

    Parameters
    ----------
    prediction_length
        Length of the forecast horizon.
    num_val_windows
        Number of windows to generate from each time series in the dataset.
    val_step_size
        The end of each subsequent window is moved this many time steps forward.
    """

    def __init__(self, prediction_length: int, num_val_windows: int = 1, val_step_size: int | None = None):
        super().__init__(prediction_length=prediction_length, num_val_windows=num_val_windows)
        if val_step_size is None:
            val_step_size = prediction_length
        self.val_step_size = val_step_size

    def split(self, data: TimeSeriesDataFrame) -> Iterator[tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]]:
        """Generate train and validation folds for a time series dataset."""
        for window_idx in range(1, self.num_val_windows + 1):
            val_end = -(self.num_val_windows - window_idx) * self.val_step_size
            train_end = val_end - self.prediction_length
            if val_end == 0:
                val_end = None
            train_data = data.slice_by_timestep(None, train_end)
            val_data = data.slice_by_timestep(None, val_end)
            yield train_data, val_data
