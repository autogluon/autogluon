import functools
import time
from typing import Any, Callable, Iterator, NamedTuple

import numpy as np
import torch

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries import TimeSeriesDataFrame


class Toto2Batch(NamedTuple):
    """A batch of univariate time series prepared for ``Toto2Model.forecast``.

    All tensors have a singleton variate dimension, i.e., shape ``(batch, 1, time)`` for
    ``target``/``target_mask`` and ``(batch, 1)`` for ``series_ids``.
    """

    target: torch.Tensor
    target_mask: torch.Tensor
    series_ids: torch.Tensor


class Toto2InferenceDataset(torch.utils.data.Dataset):
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

    def __len__(self):
        return len(self.indptr) - 1  # noqa

    def __getitem__(self, idx) -> np.ndarray:
        start_idx = self.indptr[idx]
        end_idx = self.indptr[idx + 1]

        if end_idx - start_idx > self.max_context_length:
            start_idx = end_idx - self.max_context_length

        return self.target_array[start_idx:end_idx]


class Toto2DataLoader:
    def __init__(
        self,
        dataset: Toto2InferenceDataset,
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
        # Toto-2 requires the context length to be divisible by the model's patch size.
        # Left-pad with additional NaN entries (masked out) so the length is a multiple of pad_to_multiple.
        remainder = batch.shape[-1] % pad_to_multiple
        if remainder != 0:
            batch = torch.nn.functional.pad(batch, (pad_to_multiple - remainder, 0), value=torch.nan)
        return batch

    def __iter__(self) -> Iterator[Toto2Batch]:
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

            yield Toto2Batch(target=target, target_mask=target_mask, series_ids=series_ids)

            self.on_batch()
