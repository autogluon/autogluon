import time
from typing import Callable, Optional

import numpy as np
import torch

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame


class ChronosInferenceDataset:
    """A container for time series datasets that implements the ``torch.utils.data.Dataset`` interface"""

    def __init__(
        self,
        target_df: TimeSeriesDataFrame,
        context_length: int,
        target_column: str = "target",
    ):
        assert context_length > 0
        self.context_length = context_length
        self.target_array = target_df[target_column].to_numpy(dtype=np.float32)
        self.freq = target_df.freq

        # store pointer to start:end of each time series
        cum_sizes = target_df.num_timesteps_per_item().values.cumsum()
        self.indptr = np.append(0, cum_sizes).astype(np.int32)

    def __len__(self):
        return len(self.indptr) - 1  # noqa

    def _get_context(self, a: np.ndarray, pad_value=np.nan):
        a = a[-self.context_length :]
        pad_size = self.context_length - len(a)
        if pad_size > 0:
            pad = np.full(shape=(pad_size,), fill_value=pad_value)
            a = np.concatenate((pad, a))
        return a

    def __getitem__(self, idx) -> np.ndarray:
        start_idx = self.indptr[idx]
        end_idx = self.indptr[idx + 1]

        return self._get_context(self.target_array[start_idx:end_idx])


class ChronosInferenceDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.callback: Callable = kwargs.pop("on_batch", lambda: None)
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for item in super().__iter__():
            yield item
            self.callback()


def timeout_callback(seconds: Optional[float]) -> Callable:
    """Return a callback object that raises an exception if time limit is exceeded."""
    start_time = time.time()

    def callback() -> None:
        if seconds is not None and time.time() - start_time > seconds:
            raise TimeLimitExceeded

    return callback
