from __future__ import annotations

import io
import logging
import os
import pickle
from typing import Any
from urllib.parse import urlparse

from ..utils import compression_utils, s3_utils

logger = logging.getLogger(__name__)

# Loading a pickle file executes arbitrary code during deserialization (see CWE-502).
# Fetching a pickle from a remote HTTP(S) URL is therefore unsafe: anyone able to supply
# or intercept the URL achieves remote code execution. Loading from a URL is blocked by
# default and must be explicitly opted into by users who fully trust the source.
_ALLOW_REMOTE_PICKLE_ENV_VAR = "AG_ALLOW_REMOTE_PICKLE"


def _is_remote_pickle_allowed() -> bool:
    return os.environ.get(_ALLOW_REMOTE_PICKLE_ENV_VAR, "False").lower() in ("true", "1")


# S3 loading is a core, widely-used feature (e.g. loading a predictor from an ``s3://`` path), so it
# cannot be blocked by default without breaking legitimate workflows. It remains unsafe, however:
# ``pickle.loads`` executes arbitrary code during deserialization, so a compromised or misconfigured
# bucket results in code execution on load (CWE-502). We surface this risk with a one-time warning.
_warned_s3_pickle = False


def _warn_s3_pickle_load(path: str) -> None:
    global _warned_s3_pickle
    if not _warned_s3_pickle:
        logger.warning(
            f"Loading a pickle file from S3 ({path}). Deserializing a pickle file executes arbitrary "
            "code, so a compromised or misconfigured bucket can lead to code execution. Only load "
            "objects from S3 locations you control and trust. This warning is shown once per process."
        )
        _warned_s3_pickle = True


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

        _warn_s3_pickle_load(path)
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
    if not _is_remote_pickle_allowed():
        raise ValueError(
            f"Refusing to load a pickle file from a remote URL: {url}\n"
            "Loading a pickle file executes arbitrary code during deserialization, so fetching one "
            "from an HTTP(S) URL allows anyone who can supply or intercept that URL to run arbitrary "
            "code on this machine (remote code execution).\n"
            "Download the file to local storage and verify its integrity before loading it, or, only if "
            f"you fully trust the source and connection, set the environment variable "
            f"{_ALLOW_REMOTE_PICKLE_ENV_VAR}=True to allow loading pickle files from URLs."
        )

    import requests

    logger.warning(
        f"Loading a pickle file from a remote URL ({url}). This executes arbitrary code during "
        "deserialization; only do this for sources you fully trust."
    )
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
