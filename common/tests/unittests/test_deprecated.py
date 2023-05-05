from autogluon.common.utils.deprecated import Deprecated

import pytest


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


def test_should_deprecate_warning():
    with pytest.deprecated_call():
        ShouldDeprecateButNoErrorClass()
        NotDeprecatedClass().should_deprecate_but_no_error_func()
        should_deprecate_but_no_error_func()


def test_should_raise_deprecate_error():
    with pytest.raises(Exception):
        ShouldDeprecateErrorClass()
        NotDeprecatedClass().should_deprecate_error_func()
        should_deprecate_error_func()


def test_should_not_deprecate():
    NotDeprecatedClass().not_deprecated_func()
    not_deprecated_func()
