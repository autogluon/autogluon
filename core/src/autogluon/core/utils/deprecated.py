import logging
import warnings
import functools


logger = logging.getLogger(__name__)


def deprecated(message="", *args, **kwargs):
    def deprecated_decorator(func):
        warning_message = f"Calling deprecated function {func.__name__}. This function is deprecated " \
                           "and will be removed in a future release. " + message
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            warnings.warn(warning_message)
            logger.warn(warning_message)
            return func(*args, **kwargs)
        return wrapped_func
    return deprecated_decorator
