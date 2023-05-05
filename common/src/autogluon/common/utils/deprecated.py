import inspect
import warnings

from typing import Optional
from packaging import version

from ..version import __version__


def _deprecation_warning(
    old: str,
    new: Optional[str] = None,
    custom_warning_msg: Optional[str] = None,
    error: Optional[bool] = None,
    version_to_remove: Optional[str] = None
):
    msg = f"`{old}` has been deprecated. "
    msg += f"Will be removed in version {version_to_remove} " if version_to_remove is not None else ""
    msg += (
        f"Please use `{new}` instead"
        if new is not None
        else custom_warning_msg
        if custom_warning_msg is not None
        else ""
    )

    if error:
        raise ValueError(msg)
    else:
        warnings.warn(f"Deprecation Warning: {msg}. This will raise an error in the future!", category=DeprecationWarning, stacklevel=2)


def Deprecated(
    min_version_to_warn: str,
    min_version_to_error: str,
    old: Optional[str] = None,
    new: Optional[str] = None,
    custom_warning_msg: Optional[str] = None,
    version_to_remove: Optional[str] = None
):
    """
    
    """
    def _decorator(obj):
        error = False
        if version.parse(__version__) < version.parse(min_version_to_warn):
            return obj
        if version.parse(__version__) >= version.parse(min_version_to_error):
            print("wtf")
            error = True
        if inspect.isclass(obj):
            obj_init = obj.__init__

            def patched_init_with_warning_msg(*args, **kwargs):
                _deprecation_warning(
                    old=old or obj.__name__,
                    new=new,
                    custom_warning_msg=custom_warning_msg,
                    error=error,
                    version_to_remove=version_to_remove
                )
                return obj_init(*args, **kwargs)

            obj.__init__ = patched_init_with_warning_msg
            return obj

        def patched_func_with_warning_msg(*args, **kwargs):
            _deprecation_warning(
                old=old or obj.__name__,
                new=new,
                custom_warning_msg=custom_warning_msg,
                error=error,
                version_to_remove=version_to_remove
            )
            return obj(*args, **kwargs)

        return patched_func_with_warning_msg

    return _decorator
