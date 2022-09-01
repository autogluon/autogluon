import logging
import warnings
import functools
from typing import Optional


logger = logging.getLogger(__name__)


def deprecated(message: str = "", version_removed: Optional[str] = None, *args, **kwargs):
    def deprecated_decorator(func):
        warning_message = (
            f"Calling deprecated function {func.__name__}. This function is deprecated "
            + "and will be removed in "
            + (f"v{version_removed}. " if version_removed else "a future release. ")
            + message
        )

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            warnings.warn(message=warning_message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped_func

    return deprecated_decorator
