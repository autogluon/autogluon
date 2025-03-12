import logging
import sys
from pathlib import Path
from typing import Any, Union

from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem

logger = logging.getLogger(__name__)


def is_interactive_env():
    """
    Return whether the current process is running under the interactive mode.
    Check also https://stackoverflow.com/a/64523765
    """
    return hasattr(sys, "ps1")


def get_filesystem(path: Union[str, Path], **kwargs: Any) -> AbstractFileSystem:
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs
