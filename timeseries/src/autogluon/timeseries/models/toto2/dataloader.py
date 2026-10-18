import functools
import time
from typing import Any, Callable, Iterator

import numpy as np
import torch

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries import TimeSeriesDataFrame


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
    """Data loader for Toto 2.0. Yields batches as the input dict accepted by ``Toto2Model.forecast``.

    All tensors have a singleton variate dimension: ``(batch, 1, time)`` for ``target``/``target_mask`` and
    ``(batch, 1)`` for ``series_ids``.
    """

    def __init__(
        self,
        dataset: Toto2InferenceDataset,
        batch_size: int = 1,
        pad_to_multiple: int = 1,
        time_limit: float | None = None,
        device: Any = None,
    ):
        assert pad_to_multiple >= 1
        self.device = torch.device(device)
        self.batch_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=functools.partial(self._collate, device=self.device, pad_to_multiple=pad_to_multiple),
        )
        self.on_batch = self._get_timeout_callback(time_limit) if time_limit is not None else (lambda: None)

    @staticmethod
    def _get_timeout_callback(seconds: float | None) -> Callable:
        start_time = time.monotonic()

        def callback() -> None:
            if seconds is not None and time.monotonic() - start_time > seconds:
                raise TimeLimitExceeded

        return callback

    @staticmethod
    def _collate(time_series: list[np.ndarray], device: Any, pad_to_multiple: int = 1) -> torch.Tensor:
        # Trim each series so that its first observation falls on a patch boundary. Toto-2 requires the
        # context length to be divisible by the patch size, and a partially observed leading patch
        # destabilizes the causal scaler (https://github.com/DataDog/toto/issues/78).
        if pad_to_multiple > 1:
            time_series = [c[len(c) % pad_to_multiple :] if len(c) >= pad_to_multiple else c for c in time_series]

        # Left-pad shorter series with NaN so that all series in the batch share the same length.
        batch = torch.nn.utils.rnn.pad_sequence(
            sequences=[torch.tensor(c, device=device, dtype=torch.float32) for c in time_series],
            batch_first=True,
            padding_value=torch.nan,
            padding_side="left",
        )
        # Series shorter than a single patch cannot be trimmed, so pad the batch up to a full patch instead.
        remainder = batch.shape[-1] % pad_to_multiple
        if remainder != 0:
            batch = torch.nn.functional.pad(batch, (pad_to_multiple - remainder, 0), value=torch.nan)
        return batch

    @staticmethod
    def _fill_missing(target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """Forward-fill missing entries, backfilling leading entries with the first observed value.

        Series with no observed values are filled with zeros.
        """
        positions = torch.arange(target.shape[-1], device=target.device)
        last_observed = torch.where(target_mask, positions, torch.zeros_like(positions)).cummax(dim=-1).values
        filled = torch.gather(target.nan_to_num(), -1, last_observed)

        first_observed = target_mask.to(torch.uint8).argmax(dim=-1, keepdim=True)
        first_value = torch.gather(target, -1, first_observed)
        return torch.where(positions < first_observed, first_value, filled).nan_to_num()

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for batch in self.batch_loader:
            # Add a singleton variate dimension -> (batch, 1, time)
            target = batch.unsqueeze(1).to(self.device).to(torch.float32)

            # `Toto2Model.forecast` marks the final context patch as observed regardless of the mask, so
            # missing entries are filled with neighboring observations rather than zeros.
            target_mask = ~torch.isnan(target)
            target = self._fill_missing(target, target_mask)

            current_batch_size = target.shape[0]
            # Each item is an independent univariate series -> series_ids of shape (batch, 1)
            series_ids = torch.zeros(current_batch_size, 1, dtype=torch.long, device=self.device)

            yield {"target": target, "target_mask": target_mask, "series_ids": series_ids}

            self.on_batch()
