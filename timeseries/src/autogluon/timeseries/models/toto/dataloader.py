import functools
import time
from typing import Any, Callable, Iterator, Optional, Union

import numpy as np
import torch

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries import TimeSeriesDataFrame

from ._internal.dataset import MaskedTimeseries, freq_to_seconds


class TotoInferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        target_df: TimeSeriesDataFrame,
        max_context_length: int,
        target_column: str = "target",
    ):
        assert max_context_length > 0
        self.max_context_length = max_context_length
        self.target_array = target_df[target_column].to_numpy(dtype=np.float32)

        # store pointer to start:end of each time series
        self.indptr = target_df.get_indptr()

        self.freq = target_df.freq

    def __len__(self):
        return len(self.indptr) - 1  # noqa

    def __getitem__(self, idx) -> np.ndarray:
        start_idx = self.indptr[idx]
        end_idx = self.indptr[idx + 1]

        if end_idx - start_idx > self.max_context_length:
            start_idx = end_idx - self.max_context_length

        return self.target_array[start_idx:end_idx]


class TotoDataLoader:
    def __init__(
        self,
        dataset: TotoInferenceDataset,
        freq: Optional[str] = None,
        batch_size: int = 1,
        time_limit: Optional[Union[int, float]] = None,
        device: Any = None,
    ):
        self.device = torch.device(device)
        self.batch_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=functools.partial(self._collate, device=self.device),
        )
        self.on_batch = self._get_timeout_callback(time_limit) if time_limit is not None else (lambda *a, **k: None)

        self.freq: str = freq or dataset.freq or "h"

    @staticmethod
    def _get_timeout_callback(seconds: Optional[float]) -> Callable:
        start_time = time.monotonic()

        def callback() -> None:
            if seconds is not None and time.monotonic() - start_time > seconds:
                raise TimeLimitExceeded

        return callback

    @staticmethod
    def _collate(time_series: list[np.ndarray], device: Any) -> torch.Tensor:
        max_len = max(len(c) for c in time_series)
        padded = []
        for c in time_series:
            padding = torch.full(
                size=(max_len - len(c),),
                fill_value=torch.nan,
                device=device,
                dtype=torch.float32,
            )
            data = torch.tensor(c, device=device, dtype=torch.float32)
            padded.append(torch.concat((padding, data)))
        return torch.stack(padded)

    def __iter__(self) -> Iterator[MaskedTimeseries]:
        for batch in self.batch_loader:
            time_series = batch.unsqueeze(1).to(self.device).to(torch.float32)
            nan_mask = torch.isnan(time_series)
            time_series[nan_mask] = 0.0  # pad with zeros instead of nan

            current_batch_size, _, context_length = time_series.shape

            id_mask = torch.arange(current_batch_size, dtype=torch.int, device=self.device)[:, None, None].repeat(
                1, 1, context_length
            )

            time_interval_seconds = torch.full(
                (current_batch_size, 1),
                fill_value=freq_to_seconds(self.freq),
                device=self.device,
                dtype=torch.int,
            )

            yield MaskedTimeseries(
                time_series,
                padding_mask=~nan_mask,
                id_mask=id_mask,
                timestamp_seconds=torch.zeros_like(time_series, dtype=torch.int),
                time_interval_seconds=time_interval_seconds,
            )

            self.on_batch()
