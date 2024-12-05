import logging
import os
import re
import time
from itertools import chain, cycle
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Literal, Optional

import numpy as np
import torch
from gluonts.dataset.field_names import FieldName
from gluonts.transform import ExpectedNumInstanceSampler, InstanceSplitter, ValidationSplitSampler
from torch.utils.data import IterableDataset
from transformers import TrainerCallback

from autogluon.common.loaders.load_s3 import download, list_bucket_prefix_suffix_contains_s3
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.gluonts.abstract_gluonts import SimpleGluonTSDataset

if TYPE_CHECKING:
    # TODO: fix the underlying reason for this circular import, the pipeline should handle tokenization
    from autogluon.timeseries.models.chronos.pipeline.chronos import ChronosTokenizer


logger = logging.getLogger("autogluon.timeseries.models.chronos")


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_size
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_size: int = 100) -> None:
        super().__init__()
        assert shuffle_buffer_size > 0
        self.base_dataset = base_dataset
        self.shuffle_buffer_size = shuffle_buffer_size
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_size:
                idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ChronosFineTuningDataset(IterableDataset):
    """
    Dataset wrapper to convert a ``TimeSeriesDataFrame`` into an iterable dataset
    compatible with Chronos models.

    When a ``tokenizer`` is provided, data is converted into HuggingFace-compatible set of
    ``input_ids``, ``attention_mask`` and ``labels``, used by the original Chronos models.

    When the ``tokenizer`` is omitted, data is converted into the format compatible with
    ChronosBolt models, i.e., ``context`` and ``target``.

    Parameters
    ----------
    target_df : TimeSeriesDataFrame
        The ``TimeSeriesDataFrame`` to be converted
    target_column : str, default = "target"
        The name of the column which contains the target time series, by default "target"
    context_length : int, default = 512
        The length of the historical context
    prediction_length : int, default = 64
        The prediction_length, i.e., length of label or target
    tokenizer : ``ChronosTokenizer``, default = None
        When a ``ChronosTokenizer`` object is provided, data will be converted into the
        HuggingFace format accepted by the original Chronos models using this ``ChronosTokenizer``.
        If None, data will be converted into the format accepted by ChronosBolt models.
    mode : Literal["training", "validation"], default = "training"
        When ``training``, random slices from the time series will be returned for training purposes.
        If ``validation``, the last slice of each time series returned in the original order.
    """

    def __init__(
        self,
        target_df: TimeSeriesDataFrame,
        target_column: str = "target",
        context_length: int = 512,
        prediction_length: int = 64,
        tokenizer: Optional["ChronosTokenizer"] = None,
        mode: Literal["training", "validation"] = "training",
    ) -> None:
        super().__init__()

        assert mode in ("training", "validation")

        # A dummy hourly freq is used because the model doesn't actually need the freq
        self.gluonts_dataset = SimpleGluonTSDataset(target_df=target_df, freq="h", target_column=target_column)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.mode = mode

    def _create_instance_splitter(self, mode: str):
        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=self.prediction_length, min_instances=1
            ),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def _create_training_data(self, data: Iterable[dict]):
        data = chain.from_iterable(cycle([data]))
        split_transform = self._create_instance_splitter("training")
        data = split_transform.apply(data, is_train=True)
        return data

    def _create_validation_data(self, data: Iterable[dict]):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_chronos_format(self, entry: dict) -> dict:
        """Converts an entry from GluonTS data format with past and future targets
        to the HuggingFace format accepted by the original Chronos models using the ChronosTokenizer.

        Parameters
        ----------
        entry : dict
            time series data entry in GluonTS format with ``past_target`` and ``future_target`` keys

        Returns
        -------
        dict
            time series data entry in HuggingFace format with ``input_ids``, ``attention_mask``, and ``labels``
        """
        assert self.tokenizer is not None, "A ChronosTokenizer is required to convert data into the Chronos format"
        past_target = torch.tensor(entry[f"past_{FieldName.TARGET}"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(past_target)
        future_target = torch.tensor(entry[f"future_{FieldName.TARGET}"]).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        labels[labels_mask == 0] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def to_chronos_bolt_format(self, entry: dict) -> dict:
        """Converts an entry from GluonTS data format with past and future targets
        to the format accepted by the ChronosBolt models.

        Parameters
        ----------
        entry : dict
            time series data entry in GluonTS format with ``past_target`` and ``future_target`` keys

        Returns
        -------
        dict
            time series data entry in ChronosBolt format with ``context`` and ``target``
        """
        past_target = torch.tensor(entry[f"past_{FieldName.TARGET}"])
        future_target = torch.tensor(entry[f"future_{FieldName.TARGET}"])

        return {"context": past_target, "target": future_target}

    def __iter__(self) -> Iterator:
        if self.mode == "training":
            iterable = self._create_training_data(self.gluonts_dataset)
        elif self.mode == "validation":
            iterable = self._create_validation_data(self.gluonts_dataset)

        format_transform_fn = self.to_chronos_format if self.tokenizer is not None else self.to_chronos_bolt_format
        for entry in iterable:
            yield format_transform_fn(entry)

    def shuffle(self, shuffle_buffer_size: Optional[int] = None):
        """Returns a (pseudo) shuffled version of this iterable dataset.

        Parameters
        ----------
        shuffle_buffer_size : int, optional, default = None
            The shuffle buffer size used for pseudo shuffling
        """
        assert shuffle_buffer_size is None or shuffle_buffer_size >= 0
        if not shuffle_buffer_size:
            return self
        return PseudoShuffledIterableDataset(self, shuffle_buffer_size)


def left_pad_and_stack_1D(tensors: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(size=(max_len - len(c),), fill_value=torch.nan, device=c.device)
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


def cache_model_from_s3(s3_uri: str, force=False):
    if re.match("^s3://([^/]+)/(.*?([^/]+)/?)$", s3_uri) is None:
        raise ValueError(f"Not a valid S3 URI: {s3_uri}")

    # we expect the prefix to point to a "directory" on S3
    if not s3_uri.endswith("/"):
        s3_uri += "/"

    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)
    bucket_cache_path = cache_home / "autogluon" / "timeseries" / bucket

    for obj_path in list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=prefix):
        destination_path = bucket_cache_path / obj_path
        if not force and destination_path.exists():
            continue
        download(bucket, obj_path, local_path=str(destination_path))

    return str(bucket_cache_path / prefix)


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


class EvaluateAndSaveFinalStepCallback(TrainerCallback):
    """Callback to evaluate and save the model at last training step."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_log = True
            control.should_evaluate = True
            control.should_save = True


class TimeLimitCallback(TrainerCallback):
    def __init__(self, time_limit: int):
        """
        Callback to stop training once a specified time has elapsed.

        Parameters
        ----------
        time_limit: int
            maximum time allowed for training in seconds.
        """
        self.time_limit = time_limit
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.monotonic()  # type: ignore

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.monotonic() - self.start_time  # type: ignore
        if elapsed_time > self.time_limit:
            logger.log(15, "Stopping fine-tuning since time_limit is reached")
            control.should_training_stop = True


class LoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)


def timeout_callback(seconds: Optional[float]) -> Callable:
    """Return a callback object that raises an exception if time limit is exceeded."""
    start_time = time.monotonic()

    def callback() -> None:
        if seconds is not None and time.monotonic() - start_time > seconds:
            raise TimeLimitExceeded

    return callback
