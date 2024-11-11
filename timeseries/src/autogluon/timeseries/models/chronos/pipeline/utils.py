import os
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from autogluon.common.loaders.load_s3 import download, list_bucket_prefix_suffix_contains_s3
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame


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


def timeout_callback(seconds: Optional[float]) -> Callable:
    """Return a callback object that raises an exception if time limit is exceeded."""
    start_time = time.time()

    def callback() -> None:
        if seconds is not None and time.time() - start_time > seconds:
            raise TimeLimitExceeded

    return callback
