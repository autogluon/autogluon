import functools
import inspect
import warnings
from typing import Any, Callable, Dict, Optional

from packaging import version

from ..version import __version__


def _deprecation_warning(
    old: str,
    new: Optional[str] = None,
    custom_warning_msg: Optional[str] = None,
    version_to_remove: Optional[str] = None,
    error: bool = False,
):
    msg = custom_warning_msg
    if msg is None:
        msg = f"`{old}` has been deprecated"
        msg += f" and will be removed in version {version_to_remove}. " if version_to_remove is not None else ". "
        msg += f"Please use `{new}` instead" if new is not None else ""

    if error:
        raise ValueError(msg)
    else:
        warnings.warn(
            f"{msg}. This will raise an error in the future!",
            category=DeprecationWarning,
            stacklevel=3,
        )


def Deprecated(
    min_version_to_warn: str,
    min_version_to_error: str,
    old: Optional[str] = None,
    new: Optional[str] = None,
    custom_warning_msg: Optional[str] = None,
    version_to_remove: Optional[str] = None,
    _ag_version: Optional[str] = None,
):
    """
    Decorator to add deprecation warnings or raise an error to the decorated object.
    Can be applied to both functions and classes.

    Parameters
    ----------
    min_version_to_warn: str,
        Minimum ag version to show deprecation warning.
        If the installed ag version is >= min_version_to_warn, a deprecation warning will be generated.
    min_version_to_error: str,
        Minimum ag version to raise deprecation error.
        If the installed ag version is >= min_version_to_error, a deprecation error will be raised.
    old: Optional[str], default = None
        A description of the "thing" that is to be deprecated.
        If not specified, will use the decorated object.__name__
    new: Optional[str], default = None,
        A description of the new "thing" that replaces it.
        If not specified, will not prompt the user what to use as replacement.
    custom_warning_msg: Optional[str], default = None,
        Custom warning message to display.
        If not specified, will prompt in the following format:
            ```python
            msg = f"`{old}` has been deprecated. "
            msg += f"And will be removed in version {version_to_remove} " if version_to_remove is not None else ""
            msg += (
                f"Please use `{new}` instead"
                if new is not None
                else ""
            )
            ```
    version_to_remove: Optional[str], default = None,
        AG version when the object will be removed. Will be added to the warning message if specified

    Examples
    --------
    >>> # Deprecate a class
    >>> @Deprecated(min_version_to_warn="1.0", min_version_to_error="1.1", new="NewClass")
    >>> class OldClass:
    >>>     ...
    >>>
    >>> # Deprecate a class method
    >>> class MyClass:
    >>>     @Deprecated(min_version_to_warn="1.0", min_version_to_error="1.1", new="new_class_method")
    >>>     def old_class_method(self):
    >>>         ...
    >>>
    >>> # Deprecate a function
    >>> @Deprecated(min_version_to_warn="1.0", min_version_to_error="1.1", new="new_func")
    >>> def old_func():
    >>>     ...
    """

    def _decorator(obj):
        error = False
        ag_version = __version__
        if _ag_version is not None:
            ag_version = _ag_version
        if version.parse(ag_version) < version.parse(min_version_to_warn):
            return obj
        if version.parse(ag_version) >= version.parse(min_version_to_error):
            error = True
        if inspect.isclass(obj):
            obj_init = obj.__init__

            @functools.wraps(obj)
            def patched_init_with_warning_msg(*args, **kwargs):
                _deprecation_warning(
                    old=old or obj.__name__,
                    new=new,
                    custom_warning_msg=custom_warning_msg,
                    error=error,
                    version_to_remove=version_to_remove,
                )
                return obj_init(*args, **kwargs)

            obj.__init__ = patched_init_with_warning_msg
            return obj

        @functools.wraps(obj)
        def patched_func_with_warning_msg(*args, **kwargs):
            _deprecation_warning(
                old=old or obj.__name__,
                new=new,
                custom_warning_msg=custom_warning_msg,
                error=error,
                version_to_remove=version_to_remove,
            )
            return obj(*args, **kwargs)

        return patched_func_with_warning_msg

    return _decorator


def _rename_kwargs(
    func_name: str,
    kwargs: Dict[str, Any],
    kwargs_mapping: Dict[str, str],
    custom_warning_msg: Optional[str] = None,
    version_to_remove: Optional[str] = None,
    error: bool = False,
):
    for old_name, new_name in kwargs_mapping.items():
        if old_name in kwargs:
            if new_name is not None and new_name in kwargs:
                raise ValueError(
                    f"{func_name} received both {old_name} and {new_name} as arguments."
                    f"{old_name} is deprecated, please use {new_name} instead."
                )
            print(error)
            _deprecation_warning(
                old=old_name,
                new=new_name,
                custom_warning_msg=custom_warning_msg,
                version_to_remove=version_to_remove,
                error=error,
            )
            if new_name is not None:
                kwargs[new_name] = kwargs.pop(old_name)


# Inspired by https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias
def Deprecated_args(
    min_version_to_warn: str,
    min_version_to_error: str,
    custom_warning_msg: Optional[str] = None,
    version_to_remove: Optional[str] = None,
    _ag_version: Optional[str] = None,
    **kwargs_mapping,
):
    """
    Decorator to add deprecation warnings or raise an error to deprecated arguments.
    If not raising error, will replace the deprecated argument with the new one.

    Parameters
    ----------
    min_version_to_warn: str,
        Minimum ag version to show deprecation warning.
        If the installed ag version is >= min_version_to_warn, a deprecation warning will be generated.
    min_version_to_error: str,
        Minimum ag version to raise deprecation error.
        If the installed ag version is >= min_version_to_error, a deprecation error will be raised.
    custom_warning_msg: Optional[str], default = None,
        Custom warning message to display.
        If not specified, will prompt in the following format:
            ```python
            msg = f"`{old}` has been deprecated. "
            msg += f"And will be removed in version {version_to_remove} " if version_to_remove is not None else ""
            msg += (
                f"Please use `{new}` instead"
                if new is not None
                else ""
            )
            ```
    version_to_remove: Optional[str], default = None,
        AG version when the object will be removed. Will be added to the warning message if specified
    kwargs_mapping
        Mapping between deprecated_arg and new_arg, i.e. kwargs_mapping={"deprecated_arg": "new_args"}
        If the argument is being deprecated and won't be replaced by other new arg, set it to None, i.e. kwargs_mapping={"deprecated_arg": None}

    Examples
    --------
    >>> # Deprecate a renamed argument
    >>> @Deprecated_args(min_version_to_warn="1.0", min_version_to_error="1.1", old_arg="new_arg")
    >>> def myfunc(new_arg):  # Old argument can be removed from function signature
    >>>     ...
    >>>
    >>> @Deprecated_args(min_version_to_warn="1.0", min_version_to_error="1.1", deprecated_arg_with_no_replacement=None)
    >>> def myfunc(deprecated_arg_with_no_replacement):  # This argument need to still exists for it to work
    >>>     ...
    """

    def _decorator(obj):
        error = False
        ag_version = __version__
        if _ag_version is not None:
            ag_version = _ag_version
        if version.parse(ag_version) < version.parse(min_version_to_warn):
            return obj
        if version.parse(ag_version) >= version.parse(min_version_to_error):
            error = True

        @functools.wraps(obj)
        def patched_obj_with_warning_msg(*args, **kwargs):
            _rename_kwargs(
                func_name=obj.__name__,
                kwargs=kwargs,
                kwargs_mapping=kwargs_mapping,
                custom_warning_msg=custom_warning_msg,
                version_to_remove=version_to_remove,
                error=error,
            )
            return obj(*args, **kwargs)

        return patched_obj_with_warning_msg

    return _decorator


def construct_deprecated_wrapper(ag_version) -> Callable:
    """Return Deprecated decorator with ag_version as the local AG version for checking"""

    def _decorator(*args, **kwargs):
        return Deprecated(*args, _ag_version=ag_version, **kwargs)

    return _decorator


def construct_deprecated_args_wrapper(ag_version) -> Callable:
    """Return Deprecated_args decorator with ag_version as the local AG version for checking"""

    def _decorator(*args, **kwargs):
        return Deprecated_args(*args, _ag_version=ag_version, **kwargs)

    return _decorator
