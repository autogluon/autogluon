import pytest

from autogluon.common.utils.deprecated import Deprecated, Deprecated_args


@Deprecated(min_version_to_warn="0.0", min_version_to_error="9999.0")
class ShouldDeprecateButNoErrorClass:
    pass


@Deprecated(min_version_to_warn="0.0", min_version_to_error="0.0")
class ShouldDeprecateErrorClass:
    pass


@Deprecated(min_version_to_warn="9999.0", min_version_to_error="9999.0")
class NotDeprecatedClass:
    @Deprecated(min_version_to_warn="0.0", min_version_to_error="9999.0")
    def should_deprecate_but_no_error_func(self):
        pass

    @Deprecated(min_version_to_warn="0.0", min_version_to_error="0.0")
    def should_deprecate_error_func(self):
        pass

    @Deprecated(min_version_to_warn="9999.0", min_version_to_error="9999.0")
    def not_deprecated_func(self):
        pass


@Deprecated(min_version_to_warn="0.0", min_version_to_error="9999.0")
def should_deprecate_but_no_error_func():
    pass


@Deprecated(min_version_to_warn="0.0", min_version_to_error="0.0")
def should_deprecate_error_func():
    pass


@Deprecated(min_version_to_warn="9999.0", min_version_to_error="9999.0")
def not_deprecated_func():
    pass


@Deprecated_args(min_version_to_warn="0.0", min_version_to_error="9999.0", deprecated_arg="new_arg")
def func_with_deprecated_args_no_error(new_arg=None):
    return new_arg


@Deprecated_args(min_version_to_warn="0.0", min_version_to_error="0.0", deprecated_arg="new_arg")
def func_with_deprecated_args_error(new_arg=None):
    pass


@Deprecated_args(min_version_to_warn="9999.0", min_version_to_error="9999.0", deprecated_arg="new_arg")
def func_with_no_deprecated_args(new_arg=None):
    pass


@Deprecated_args(min_version_to_warn="0.0", min_version_to_error="9999.0", deprecated_arg=None)
def func_with_deprecated_args_no_replacement(deprecated_arg=None):
    return deprecated_arg


def test_should_deprecate_warning():
    with pytest.deprecated_call():
        ShouldDeprecateButNoErrorClass()
        NotDeprecatedClass().should_deprecate_but_no_error_func()
        should_deprecate_but_no_error_func()
        new_arg = func_with_deprecated_args_no_error(deprecated_arg=1)
        assert new_arg == 1
        deprecated_arg = func_with_deprecated_args_no_replacement(deprecated_arg=1)
        assert deprecated_arg == 1


def test_should_raise_deprecate_error():
    with pytest.raises(Exception):
        ShouldDeprecateErrorClass()
        NotDeprecatedClass().should_deprecate_error_func()
        should_deprecate_error_func()
        func_with_deprecated_args_no_error(deprecated_arg=1)


def test_should_not_deprecate():
    NotDeprecatedClass().not_deprecated_func()
    not_deprecated_func()
    func_with_no_deprecated_args(new_arg=1)


def test_should_raise_if_both_new_and_deprecated_args_passed():
    with pytest.raises(Exception):
        func_with_deprecated_args_no_error(new_arg=1, deprecated_arg=1)
