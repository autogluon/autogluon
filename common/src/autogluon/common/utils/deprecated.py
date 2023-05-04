import inspect
import logging

from typing import Optional, Union
from packaging import version

from .log_utils import DuplicateFilter
from ..version import __version__


ag_logger = logging.getLogger("autogluon")
logger = logging.getLogger(__name__)


def deprecation_warning(
    old: str,
    new: Optional[str] = None,
    custom_warning_msg: Optional[str] = None,
    error: Optional[Union[bool, Exception]] = None,
):
    msg = f"`{old}` has been deprecated. "
    msg += (
        f"Please use `{new}` instead"
        if new is not None
        else custom_warning_msg
        if custom_warning_msg is not None
        else ""
    )

    if error is not None:
        if issubclass(error, Exception):
            raise error(msg)
        else:
            raise ValueError(msg)
    else:
        if not any(isinstance(f, DuplicateFilter) for f in ag_logger.filters):
            dup_filter = DuplicateFilter([msg])
            ag_logger.addFilter(dup_filter)
        else:
            dup_filter: DuplicateFilter = next(
                f for f in ag_logger.filters if isinstance(f, DuplicateFilter)
            )
            dup_filter.attach_filter_targets(msg)
        logger.warning(
            f"Deprecation Warning: {msg}. This will raise an error in the future!"
        )


def Deprecated(
    old: Optional[str] = None,
    new: Optional[str] = None,
    custom_warning_msg: Optional[str] = None,
    error: Optional[Union[bool, Exception]] = None,
    min_version_to_error: Optional[str] = None
):
    def _decorator(obj):
        if version.parse(__version__) >= version.parse(min_version_to_error) and error is None:
            error = True
        if inspect.isclass(obj):
            obj_init = obj.__init__

            def patched_init_with_warning_msg(*args, **kwargs):
                deprecation_warning(
                    old=old or obj.__name__,
                    new=new,
                    custom_warning_msg=custom_warning_msg,
                    error=error,
                )
                return obj_init(*args, **kwargs)

            obj.__init__ = patched_init_with_warning_msg
            return obj

        def patched_func_with_warning_msg(*args, **kwargs):
            deprecation_warning(
                old=old or obj.__name__,
                new=new,
                custom_warning_msg=custom_warning_msg,
                error=error,
            )
            return obj(*args, **kwargs)

        return patched_func_with_warning_msg

    return _decorator
