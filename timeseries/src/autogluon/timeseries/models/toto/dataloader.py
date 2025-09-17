import time
from typing import Any, Callable, Iterator, Literal, Optional, Union

import numpy as np
import torch

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries import TimeSeriesDataFrame

from ._internal.dataset import MaskedTimeseries, freq_to_seconds


class TotoInferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        target_df: TimeSeriesDataFrame,
        context_length: int,
        target_column: str = "target",
    ):
        assert context_length > 0
        self.context_length = context_length
        self.target_array = target_df[target_column].to_numpy(dtype=np.float32)

        target_df = target_df.sort_index()

        # store pointer to start:end of each time series
        self.indptr = target_df.get_indptr()

        self.freq = target_df.freq

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


class TotoDataLoader:
    def __init__(
        self,
        dataset: TotoInferenceDataset,
        freq: Optional[str] = None,
        batch_size: int = 1,
        time_limit: Optional[Union[int, float]] = None,
        device: Any = None,
        missing_value_handling: Literal["ffill", None] = "ffill",
    ):
        self.batch_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
        )
        self.on_batch = self._get_timeout_callback(time_limit) if time_limit is not None else (lambda *a, **k: None)
        self.device = torch.device(device)

        self.freq: str = freq or dataset.freq or "h"
        self.missing_value_handling: Optional[str] = missing_value_handling

    @staticmethod
    def _get_timeout_callback(seconds: Optional[float]) -> Callable:
        start_time = time.monotonic()

        def callback() -> None:
            if seconds is not None and time.monotonic() - start_time > seconds:
                raise TimeLimitExceeded

        return callback

    @staticmethod
    def _ffill(tensor):
        assert tensor.ndim > 1
        nan_mask = torch.isnan(tensor)
        indices = torch.where(nan_mask, 0, torch.arange(tensor.shape[-1], device=tensor.device).expand_as(tensor))
        last_valid = torch.cummax(indices, dim=-1).values
        return torch.gather(tensor, dim=-1, index=last_valid)

    def __iter__(self) -> Iterator[MaskedTimeseries]:
        fill_fn = self._ffill if self.missing_value_handling == "ffill" else lambda x: x

        for batch in self.batch_loader:
            time_series = fill_fn(batch.unsqueeze(1).to(self.device))
            current_batch_size = batch.shape[0]

            mts = MaskedTimeseries(
                time_series,
                padding_mask=~torch.isnan(time_series),
                id_mask=torch.arange(current_batch_size, dtype=torch.int, device=self.device)[:, None, None],
                timestamp_seconds=torch.zeros_like(time_series, dtype=torch.int),
                time_interval_seconds=torch.full(
                    (current_batch_size, 1),
                    fill_value=freq_to_seconds(self.freq),
                    device=self.device,
                    dtype=torch.int,
                ),
            )

            yield mts

            self.on_batch()
