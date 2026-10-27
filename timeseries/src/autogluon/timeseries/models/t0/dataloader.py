import dataclasses
import time
from typing import Any, Callable

import numpy as np
import torch
from t0 import MaskType, TimeSeries, VariateType

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries import TimeSeriesDataFrame


class T0InferenceDataset(torch.utils.data.Dataset):
    """Convert each AutoGluon item into a T0 ``TimeSeries``.

    Parameters
    ----------
    data
        AutoGluon's historical data. It contains the target, past-only covariates, and the historical portion of
        known covariates.
    max_context_length
        Maximum number of recent historical timesteps retained for each item.
    target_column
        Name of the target column in ``data``.
    past_covariate_columns
        Numeric covariates observed only over the historical context, supplied as T0 ``HISTORICAL`` rows.
    known_covariate_columns
        Names of the numeric covariates known over both history and the forecast horizon.
    known_covariates
        AutoGluon's horizon-only known-covariate data passed to ``predict``. It must contain ``prediction_length``
        rows per item. The historical portion remains separate in ``data``.
    prediction_length
        Number of future timesteps in ``known_covariates``.
    """

    def __init__(
        self,
        data: TimeSeriesDataFrame,
        max_context_length: int,
        target_column: str = "target",
        past_covariate_columns: list[str] | None = None,
        known_covariate_columns: list[str] | None = None,
        known_covariates: TimeSeriesDataFrame | None = None,
        prediction_length: int = 1,
    ):
        assert max_context_length > 0
        self.max_context_length = max_context_length
        self.prediction_length = prediction_length
        past_covariate_columns = past_covariate_columns or []
        known_covariate_columns = known_covariate_columns or []
        self.target_array = data[target_column].to_numpy(dtype=np.float32)
        self.past_covariate_array = data[past_covariate_columns].to_numpy(dtype=np.float32)
        self.past_known_covariate_array = data[known_covariate_columns].to_numpy(dtype=np.float32)
        self.indptr = data.get_indptr()
        self.future_known_covariate_array: np.ndarray | None = None
        self.future_known_indptr: np.ndarray | None = None

        if known_covariate_columns:
            if known_covariates is None:
                raise ValueError("known_covariates must be provided when known real-valued covariates are enabled")
            if not data.item_ids.equals(known_covariates.item_ids):
                raise ValueError("known_covariates must contain the same items in the same order as data")
            self.future_known_covariate_array = known_covariates[known_covariate_columns].to_numpy(dtype=np.float32)
            self.future_known_indptr = known_covariates.get_indptr()
            if not np.all(np.diff(self.future_known_indptr) == prediction_length):
                raise ValueError(
                    f"known_covariates must contain exactly prediction_length={prediction_length} rows per item"
                )

    def __len__(self):
        return len(self.indptr) - 1  # noqa

    def __getitem__(self, idx) -> tuple[TimeSeries, int]:
        return self.make_batch(idx, idx + 1)

    def make_batch(self, start: int, stop: int) -> tuple[TimeSeries, int]:
        """Convert items directly into one right-aligned T0 batch, without intermediate per-item tensors."""
        context_lengths = np.minimum(np.diff(self.indptr[start : stop + 1]), self.max_context_length)
        context_length = int(context_lengths.max())
        num_known = self.past_known_covariate_array.shape[1]
        num_past = self.past_covariate_array.shape[1]
        horizon = self.prediction_length if num_known else 0
        width = context_length + horizon
        shape = (stop - start, 1 + num_known + num_past, width)
        values = np.zeros(shape, dtype=np.float32)
        mask = np.full(shape, MaskType.PAD, dtype=np.int8)
        group_ids = np.full(shape, -1, dtype=np.int64)
        variate_type = np.full(shape, -1, dtype=np.int64)
        row_types = np.array(
            [VariateType.TARGET] + [VariateType.FUTURE] * num_known + [VariateType.HISTORICAL] * num_past
        )

        for batch_idx, item_idx in enumerate(range(start, stop)):
            end_idx = self.indptr[item_idx + 1]
            start_idx = end_idx - context_lengths[batch_idx]
            left = context_length - context_lengths[batch_idx]
            history = np.concatenate(
                [
                    self.target_array[None, start_idx:end_idx],
                    self.past_known_covariate_array[start_idx:end_idx].T,
                    self.past_covariate_array[start_idx:end_idx].T,
                ],
                axis=0,
            )
            values[batch_idx, :, left:context_length] = np.where(np.isnan(history), 0.0, history)
            mask[batch_idx, :, left:context_length] = np.where(np.isnan(history), MaskType.MISSING, MaskType.VALID)
            if num_past:
                values[batch_idx, 1 + num_known :, left:context_length] = np.nan_to_num(history[1 + num_known :])
            group_ids[batch_idx, :, left:] = batch_idx
            variate_type[batch_idx, :, left:] = row_types[:, None]
            if num_known:
                assert self.future_known_covariate_array is not None and self.future_known_indptr is not None
                future_start, future_end = self.future_known_indptr[item_idx : item_idx + 2]
                future = self.future_known_covariate_array[future_start:future_end].T
                values[batch_idx, 1 : 1 + num_known, context_length:] = np.where(np.isnan(future), 0.0, future)
                mask[batch_idx, 0, context_length:] = MaskType.WITHHELD
                mask[batch_idx, 1 : 1 + num_known, context_length:] = np.where(
                    np.isnan(future), MaskType.MISSING, MaskType.VALID
                )
                mask[batch_idx, 1 + num_known :, context_length:] = MaskType.MISSING

        return TimeSeries(
            *(torch.from_numpy(array.reshape(-1, width)) for array in (values, mask, group_ids, variate_type))
        ), context_length


class T0DataLoader:
    """Build batches of converted T0 inputs.

    The dataset fills one batch of arrays using T0's padding and metadata conventions.
    This loader preserves AutoGluon item indices, groups items for cross learning, and
    enforces the prediction time limit. The caller chooses each batch's items, so it
    can retry a batch with fewer of them.

    With ``cross_learning``, each batch is joined into one T0 series, so every row of
    every item in it (targets and covariates) attends to the others. Otherwise each
    item is forecast on its own.
    """

    def __init__(
        self,
        dataset: T0InferenceDataset,
        time_limit: float | None = None,
        device: Any = None,
        cross_learning: bool = False,
    ):
        self.dataset = dataset
        self.cross_learning = cross_learning
        self.device = torch.device(device)
        self.on_batch = self._get_timeout_callback(time_limit) if time_limit is not None else (lambda: None)

    @staticmethod
    def _get_timeout_callback(seconds: float | None) -> Callable:
        start_time = time.monotonic()

        def callback() -> None:
            if seconds is not None and time.monotonic() - start_time > seconds:
                raise TimeLimitExceeded

        return callback

    def make_batch(self, start: int, stop: int) -> tuple[list[int], TimeSeries, int]:
        """Items ``[start, stop)`` as one T0 input on the device, with their indices and maximum context length."""
        indices = list(range(start, stop))
        batch, context_length = self.dataset.make_batch(start, stop)
        if self.cross_learning:
            batch = dataclasses.replace(batch, group_ids=torch.where(batch.group_ids >= 0, 0, -1))
        return indices, batch.to(self.device), context_length
