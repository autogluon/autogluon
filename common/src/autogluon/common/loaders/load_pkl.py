from __future__ import annotations

import io
import logging
import pickle
from typing import Any
from urllib.parse import urlparse

import requests

from ..utils import compression_utils, s3_utils

logger = logging.getLogger(__name__)


def load(path: str, format: str | None = None, verbose: bool = True, **kwargs) -> Any:
    """

    Parameters
    ----------
    path: str
        The path to the pickle file.
        Can be either local path, a web url, or a private s3 path.
        Local Path Example:
            "local/path/to/file.pkl"
        Web Url Example:
            "https://path/to/file.pkl"
        S3 Path Example:
            "s3://bucket/prefix/file.pkl"

    format: str, optional
        Legacy argument, unused.
    verbose: bool, default True
    kwargs

    Returns
    -------
    object
        The contents of the pickle file

    """
    compression_fn = kwargs.get("compression_fn", None)
    compression_fn_kwargs = kwargs.get("compression_fn_kwargs", None)

    if s3_utils.is_s3_url(path):
        format = "s3"
    elif _is_web_url(path=path):
        format = "url"

    if verbose:
        logger.log(15, f"Loading: {path}")

    if format == "s3":
        import boto3

        s3_bucket, s3_prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
        s3 = boto3.resource("s3")
        return pickle.loads(s3.Bucket(s3_bucket).Object(s3_prefix).get()["Body"].read())
    elif format == "url":
        return _load_pickle_from_url(url=path)

    compression_fn_map = compression_utils.get_compression_map()
    validated_path = compression_utils.get_validated_path(path, compression_fn=compression_fn)

    if compression_fn_kwargs is None:
        compression_fn_kwargs = {}

    if compression_fn in compression_fn_map:
        with compression_fn_map[compression_fn]["open"](validated_path, "rb", **compression_fn_kwargs) as fin:
            object = pickle.load(fin)
    else:
        raise ValueError(
            f"compression_fn={compression_fn} or compression_fn_kwargs={compression_fn_kwargs} are not valid."
            f" Valid function values: {compression_fn_map.keys()}"
        )

    return object


def _is_web_url(path: str) -> bool:
    try:
        result = urlparse(path)
        return result.scheme in ("http", "https") and bool(result.netloc)
    except ValueError:
        return False


def _load_pickle_from_url(url: str):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    return pickle.loads(response.content)


def load_with_fn(path, pickle_fn, format=None, verbose=True):
    if s3_utils.is_s3_url(path):
        format = "s3"
    if format == "s3":
        import boto3

        if verbose:
            logger.log(15, "Loading: %s" % path)
        s3_bucket, s3_prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
        s3 = boto3.resource("s3")
        # Has to be wrapped in IO buffer since s3 stream does not implement seek()
        buff = io.BytesIO(s3.Bucket(s3_bucket).Object(s3_prefix).get()["Body"].read())
        return pickle_fn(buff)

    if verbose:
        logger.log(15, "Loading: %s" % path)
    with open(path, "rb") as fin:
        object = pickle_fn(fin)
    return object
