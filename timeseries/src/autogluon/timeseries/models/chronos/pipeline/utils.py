import logging
import os
import re
import time
from itertools import chain, cycle
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.field_names import FieldName
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler, TestSplitSampler
from gluonts.transform.split import TFTInstanceSplitter
from torch.utils.data import IterableDataset
from transformers import TrainerCallback

from autogluon.common.loaders.load_s3 import download, list_bucket_prefix_suffix_contains_s3
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.gluonts.dataset import SimpleGluonTSDataset
from autogluon.timeseries.utils.features import CovariateMetadata

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
        covariate_metadata: CovariateMetadata,
        target_column: str = "target",
        context_length: int = 512,
        prediction_length: int = 64,
        tokenizer: Optional["ChronosTokenizer"] = None,
        mode: Literal["training", "validation"] = "training",
        dynamic_dims: int = 0,
        past_dynamic_dims: int = 0,
        static_dims: int = 0,
        static_cardinalities: Optional[List[int]] = None,
        dynamic_cardinalities: Optional[List[int]] = None,
        past_dynamic_cardinalities: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        assert mode in ("training", "validation")

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.mode = mode

        self.dynamic_dims = dynamic_dims
        self.past_dynamic_dims = past_dynamic_dims
        self.static_dims = static_dims
        self.static_cardinalities = static_cardinalities or []
        self.dynamic_cardinalities = dynamic_cardinalities or []
        self.past_dynamic_cardinalities = past_dynamic_cardinalities or []

        self.gluonts_dataset = ChronosFineTuningDataset.construct_gluonts_dataset(
            time_series_df=target_df,
            known_covariates=None,
            covariate_metadata=covariate_metadata,
            target_column=target_column,
            dynamic_dims=self.dynamic_dims,
            past_dynamic_dims=self.past_dynamic_dims,
            static_dims=self.static_dims,
            static_cardinalities=self.static_cardinalities,
            dynamic_cardinalities=self.dynamic_cardinalities,
            past_dynamic_cardinalities=self.past_dynamic_cardinalities,
            prediction_length=prediction_length,
        )

    @staticmethod
    def construct_gluonts_dataset(
        time_series_df: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame],
        covariate_metadata: CovariateMetadata,
        target_column: str,
        dynamic_dims: int,
        past_dynamic_dims: int,
        static_dims: int,
        static_cardinalities: List[int],
        dynamic_cardinalities: List[int],
        past_dynamic_cardinalities: List[int],
        prediction_length: int,
    ):
        if len(static_cardinalities) > 0:
            assert time_series_df.static_features is not None, (
                "Static features must be provided if len(static_cardinalities) > 0"
            )
            feat_static_cat = time_series_df.static_features[covariate_metadata.static_features_cat].to_numpy()
        else:
            feat_static_cat = None

        if static_dims > 0:
            assert time_series_df.static_features is not None, "Static features must be provided if static_dims > 0"
            feat_static_real = time_series_df.static_features[covariate_metadata.static_features_real].to_numpy()
        else:
            feat_static_real = None

        expected_known_covariates_len = len(time_series_df) + prediction_length * time_series_df.num_items
        # Convert TSDF -> DF to avoid overhead / input validation
        df = pd.DataFrame(time_series_df)
        if known_covariates is not None:
            known_covariates = pd.DataFrame(known_covariates)  # type: ignore
        if len(dynamic_cardinalities) > 0:
            feat_dynamic_cat = df[covariate_metadata.known_covariates_cat].to_numpy()
            if known_covariates is not None:
                feat_dynamic_cat = np.concatenate(
                    [feat_dynamic_cat, known_covariates[covariate_metadata.known_covariates_cat].to_numpy()]
                )
                assert len(feat_dynamic_cat) == expected_known_covariates_len
        else:
            feat_dynamic_cat = None

        if dynamic_dims > 0:
            feat_dynamic_real = df[covariate_metadata.known_covariates_real].to_numpy()
            if known_covariates is not None:
                feat_dynamic_real = np.concatenate(
                    [feat_dynamic_real, known_covariates[covariate_metadata.known_covariates_real].to_numpy()]
                )
                assert len(feat_dynamic_real) == expected_known_covariates_len
        else:
            feat_dynamic_real = None

        if len(past_dynamic_cardinalities) > 0:
            past_feat_dynamic_cat = df[covariate_metadata.past_covariates_cat].to_numpy()
        else:
            past_feat_dynamic_cat = None

        if past_dynamic_dims > 0:
            past_feat_dynamic_real = df[covariate_metadata.past_covariates_real].to_numpy()
        else:
            past_feat_dynamic_real = None

        return SimpleGluonTSDataset(
            target_df=time_series_df[[target_column]],
            freq="h",  # Dummy freq, unused by the model
            target_column=target_column,
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            feat_dynamic_cat=feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
            past_feat_dynamic_cat=past_feat_dynamic_cat,
            past_feat_dynamic_real=past_feat_dynamic_real,
            includes_future=known_covariates is not None,
            prediction_length=prediction_length,
        )

    def _create_instance_splitter(self, mode: str):
        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=self.prediction_length, min_instances=1
            ),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        ts_fields = []
        if self.dynamic_dims > 0:
            ts_fields.append(FieldName.FEAT_DYNAMIC_REAL)
        if len(self.dynamic_cardinalities) > 0:
            ts_fields.append(FieldName.FEAT_DYNAMIC_CAT)
        past_ts_fields = []
        if len(self.past_dynamic_cardinalities) > 0:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_CAT)
        if self.past_dynamic_dims:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_REAL)

        return TFTInstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            observed_value_field=None,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
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
        model_entry = {
            "context": torch.tensor(entry[f"past_{FieldName.TARGET}"]),
            "target": torch.tensor(entry[f"future_{FieldName.TARGET}"]),
        }

        if self.dynamic_dims > 0:
            model_entry["feat_dynamic_real"] = entry["feat_dynamic_real"]
        if self.past_dynamic_dims > 0:
            model_entry["past_feat_dynamic_real"] = entry["past_feat_dynamic_real"]
        if self.static_dims > 0:
            model_entry["feat_static_real"] = entry["feat_static_real"]
        if len(self.static_cardinalities) > 0:
            model_entry["feat_static_cat"] = entry["feat_static_cat"]
        if len(self.dynamic_cardinalities) > 0:
            model_entry["feat_dynamic_cat"] = entry["feat_dynamic_cat"]
        if len(self.past_dynamic_cardinalities) > 0:
            model_entry["past_feat_dynamic_cat"] = entry["past_feat_dynamic_cat"]

        return model_entry

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


class ChronosInferenceDataset(IterableDataset):
    """A container for time series datasets that implements the ``torch.utils.data.Dataset`` interface"""

    def __init__(
        self,
        target_df: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame],
        covariate_metadata: CovariateMetadata,
        context_length: int = 512,
        prediction_length: int = 64,
        target_column: str = "target",
        dynamic_dims: int = 0,
        past_dynamic_dims: int = 0,
        static_dims: int = 0,
        static_cardinalities: Optional[List[int]] = None,
        dynamic_cardinalities: Optional[List[int]] = None,
        past_dynamic_cardinalities: Optional[List[int]] = None,
    ):
        assert context_length > 0
        assert prediction_length > 0
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.dynamic_dims = dynamic_dims
        self.past_dynamic_dims = past_dynamic_dims
        self.static_dims = static_dims
        self.static_cardinalities = static_cardinalities or []
        self.dynamic_cardinalities = dynamic_cardinalities or []
        self.past_dynamic_cardinalities = past_dynamic_cardinalities or []

        self.gluonts_dataset = ChronosFineTuningDataset.construct_gluonts_dataset(
            time_series_df=target_df,
            known_covariates=known_covariates,
            covariate_metadata=covariate_metadata,
            target_column=target_column,
            dynamic_dims=self.dynamic_dims,
            past_dynamic_dims=self.past_dynamic_dims,
            static_dims=self.static_dims,
            static_cardinalities=self.static_cardinalities,
            dynamic_cardinalities=self.dynamic_cardinalities,
            past_dynamic_cardinalities=self.past_dynamic_cardinalities,
            prediction_length=prediction_length,
        )

    def __len__(self):
        return len(self.gluonts_dataset)

    def __iter__(self) -> Iterator:
        instance_sampler = TestSplitSampler()

        ts_fields = []
        if self.dynamic_dims > 0:
            ts_fields.append(FieldName.FEAT_DYNAMIC_REAL)
        if len(self.dynamic_cardinalities) > 0:
            ts_fields.append(FieldName.FEAT_DYNAMIC_CAT)
        past_ts_fields = []
        if len(self.past_dynamic_cardinalities) > 0:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_CAT)
        if self.past_dynamic_dims:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_REAL)

        splitter = TFTInstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            observed_value_field=None,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )

        data = splitter.apply(self.gluonts_dataset, is_train=False)

        for entry in data:
            model_entry = {"context": entry[f"past_{FieldName.TARGET}"]}
            if self.dynamic_dims > 0:
                model_entry["feat_dynamic_real"] = entry["feat_dynamic_real"]
            if self.past_dynamic_dims > 0:
                model_entry["past_feat_dynamic_real"] = entry["past_feat_dynamic_real"]
            if self.static_dims > 0:
                model_entry["feat_static_real"] = entry["feat_static_real"]
            if len(self.static_cardinalities) > 0:
                model_entry["feat_static_cat"] = entry["feat_static_cat"]
            if len(self.dynamic_cardinalities) > 0:
                model_entry["feat_dynamic_cat"] = entry["feat_dynamic_cat"]
            if len(self.past_dynamic_cardinalities) > 0:
                model_entry["past_feat_dynamic_cat"] = entry["past_feat_dynamic_cat"]

            yield model_entry


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
