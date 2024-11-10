import os
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

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


def download_prefix(bucket, prefix, local_path, force: bool = False, boto3_session=None) -> None:
    import boto3

    boto3_session = boto3_session or boto3.Session()
    s3_resource = boto3_session.resource("s3")
    bucket = s3_resource.Bucket(bucket)

    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):
            continue
        dest = local_path / bucket.name / obj.key
        if not force and dest.exists():
            continue
        if not dest.parent.exists():
            dest.parent.mkdir(exist_ok=True, parents=True)
        bucket.download_file(obj.key, str(dest))


def cache_model_from_s3(s3_uri: str, force=False):
    assert re.match("^s3://([^/]+)/(.*?([^/]+)/?)$", s3_uri) is not None, f"Not a valid S3 URI: {s3_uri}"
    cache_home = Path(os.environ.get("XGD_CACHE_HOME", os.path.expanduser("~/.cache")))
    cache_dir = cache_home / "autogluon-timeseries"
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)
    download_prefix(bucket=bucket, prefix=prefix, local_path=cache_dir, force=force)
    return cache_dir / bucket / prefix


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
