import pytest

from autogluon.common.utils.deprecated_utils import (
    Deprecated_args,
    construct_deprecated_args_wrapper,
    construct_deprecated_wrapper,
)


@pytest.mark.parametrize(
    "mock_version", [("1.0"), ("0.1"), ("0.0.1"), ("0.0.1b20230508"), ("9999b20230508")]
)  # random dev version
def test_should_deprecate_warning(mock_version):
    with pytest.deprecated_call():
        Deprecated = construct_deprecated_wrapper(mock_version)
        Deprecated_args = construct_deprecated_args_wrapper(mock_version)

        # class and function definition needs to go inside the context manager because the decorator code runs at function definition time
        @Deprecated(min_version_to_warn="0.0", min_version_to_error="9999.0")
        class ShouldDeprecateButNoErrorClass:
            pass

        @Deprecated(min_version_to_warn="9999.0", min_version_to_error="9999.0")
        class NotDeprecatedClass:
            @Deprecated(min_version_to_warn="0.0", min_version_to_error="9999.0")
            def should_deprecate_but_no_error_func(self):
                pass

        @Deprecated(min_version_to_warn="0.0", min_version_to_error="9999.0")
        def should_deprecate_but_no_error_func():
            pass

        @Deprecated_args(min_version_to_warn="0.0", min_version_to_error="9999.0", deprecated_arg="new_arg")
        def func_with_deprecated_args_no_error(new_arg=None):
            return new_arg

        @Deprecated_args(min_version_to_warn="0.0", min_version_to_error="9999.0", deprecated_arg=None)
        def func_with_deprecated_args_no_replacement(deprecated_arg=None):
            return deprecated_arg

        ShouldDeprecateButNoErrorClass()
        NotDeprecatedClass().should_deprecate_but_no_error_func()
        should_deprecate_but_no_error_func()
        new_arg = func_with_deprecated_args_no_error(deprecated_arg=1)
        assert new_arg == 1
        deprecated_arg = func_with_deprecated_args_no_replacement(deprecated_arg=1)
        assert deprecated_arg == 1


@pytest.mark.parametrize("mock_version", [("1.0"), ("0.1"), ("0.0.1"), ("0.0.1b20230508")])  # random dev version
def test_should_raise_deprecate_error(mock_version):
    with pytest.raises(ValueError):
        Deprecated = construct_deprecated_wrapper(mock_version)
        Deprecated_args = construct_deprecated_args_wrapper(mock_version)

        @Deprecated(min_version_to_warn="0.0", min_version_to_error="0.0")
        class ShouldDeprecateErrorClass:
            pass

        @Deprecated(min_version_to_warn="9999.0", min_version_to_error="9999.0")
        class NotDeprecatedClass:
            @Deprecated(min_version_to_warn="0.0", min_version_to_error="0.0")
            def should_deprecate_error_func(self):
                pass

        @Deprecated(min_version_to_warn="0.0", min_version_to_error="0.0")
        def should_deprecate_error_func():
            pass

        @Deprecated_args(min_version_to_warn="0.0", min_version_to_error="0.0", deprecated_arg="new_arg")
        def func_with_deprecated_args_error(new_arg=None):
            pass

        ShouldDeprecateErrorClass()
        NotDeprecatedClass().should_deprecate_error_func()
        should_deprecate_error_func()
        func_with_deprecated_args_error(deprecated_arg=1)


@pytest.mark.parametrize(
    "mock_version", [("1.0"), ("0.1"), ("0.0.1"), ("0.0.1b20230508"), ("9999b20230508")]
)  # random dev version
def test_should_not_deprecate(mock_version):
    Deprecated = construct_deprecated_wrapper(mock_version)
    Deprecated_args = construct_deprecated_args_wrapper(mock_version)

    @Deprecated(min_version_to_warn="9999.0", min_version_to_error="9999.0")
    class NotDeprecatedClass:
        @Deprecated(min_version_to_warn="9999.0", min_version_to_error="9999.0")
        def not_deprecated_func(self):
            pass

    @Deprecated(min_version_to_warn="9999.0", min_version_to_error="9999.0")
    def not_deprecated_func():
        pass

    @Deprecated_args(min_version_to_warn="9999.0", min_version_to_error="9999.0", deprecated_arg="new_arg")
    def func_with_no_deprecated_args(new_arg=None):
        pass

    NotDeprecatedClass().not_deprecated_func()
    not_deprecated_func()
    func_with_no_deprecated_args(new_arg=1)


def test_should_raise_if_both_new_and_deprecated_args_passed():
    with pytest.raises(ValueError):

        @Deprecated_args(min_version_to_warn="0.0", min_version_to_error="9999.0", deprecated_arg="new_arg")
        def func_with_deprecated_args_no_error(new_arg=None):
            return new_arg

        func_with_deprecated_args_no_error(new_arg=1, deprecated_arg=1)
