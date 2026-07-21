import functools
import time
from typing import Any, Callable, Iterator

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
        freq: str | None = None,
        batch_size: int = 1,
        time_limit: int | float | None = None,
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
    def _get_timeout_callback(seconds: float | None) -> Callable:
        start_time = time.monotonic()

        def callback() -> None:
            if seconds is not None and time.monotonic() - start_time > seconds:
                raise TimeLimitExceeded

        return callback

    @staticmethod
    def _collate(time_series: list[np.ndarray], device: Any) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            sequences=[torch.tensor(c, device=device, dtype=torch.float32) for c in time_series],
            batch_first=True,
            padding_value=torch.nan,
            padding_side="left",
        )

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


class Toto2DataLoader:
    """Data loader for Toto 2.0. Yields batches as the input dict accepted by ``toto2.Toto2Model.forecast``.

    All tensors have a singleton variate dimension: ``(batch, 1, time)`` for ``target``/``target_mask`` and
    ``(batch, 1)`` for ``series_ids``.
    """

    def __init__(
        self,
        dataset: TotoInferenceDataset,
        batch_size: int = 1,
        pad_to_multiple: int = 1,
        time_limit: int | float | None = None,
        device: Any = None,
    ):
        assert pad_to_multiple >= 1
        self.device = torch.device(device)
        self.batch_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=functools.partial(self._collate, device=self.device, pad_to_multiple=pad_to_multiple),
        )
        self.on_batch = self._get_timeout_callback(time_limit) if time_limit is not None else (lambda *a, **k: None)

    @staticmethod
    def _get_timeout_callback(seconds: float | None) -> Callable:
        start_time = time.monotonic()

        def callback() -> None:
            if seconds is not None and time.monotonic() - start_time > seconds:
                raise TimeLimitExceeded

        return callback

    @staticmethod
    def _collate(time_series: list[np.ndarray], device: Any, pad_to_multiple: int = 1) -> torch.Tensor:
        # Left-pad shorter series with NaN so that all series in the batch share the same length.
        # NaNs are converted into the mask (and replaced with zeros) in __iter__.
        batch = torch.nn.utils.rnn.pad_sequence(
            sequences=[torch.tensor(c, device=device, dtype=torch.float32) for c in time_series],
            batch_first=True,
            padding_value=torch.nan,
            padding_side="left",
        )
        # Toto-2 requires the context length to be divisible by the model's patch size. We trim the oldest
        # observations down to the nearest lower multiple rather than left-pad, since a mostly-masked leading
        # patch destabilizes the causal scaler. Only pad up if the batch is shorter than a single patch.
        length = batch.shape[-1]
        remainder = length % pad_to_multiple
        if remainder != 0:
            if length > remainder:
                batch = batch[..., remainder:]
            else:
                batch = torch.nn.functional.pad(batch, (pad_to_multiple - remainder, 0), value=torch.nan)
        return batch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for batch in self.batch_loader:
            # Add a singleton variate dimension -> (batch, 1, time)
            target = batch.unsqueeze(1).to(self.device).to(torch.float32)

            # Toto-2 handles missing values natively via the mask, so we do not impute.
            # The mask is False for NaN entries (both padding and genuine missing values).
            target_mask = ~torch.isnan(target)
            target = torch.where(target_mask, target, torch.zeros_like(target))

            current_batch_size = target.shape[0]
            # Each item is an independent univariate series -> series_ids of shape (batch, 1)
            series_ids = torch.zeros(current_batch_size, 1, dtype=torch.long, device=self.device)

            yield {"target": target, "target_mask": target_mask, "series_ids": series_ids}

            self.on_batch()
