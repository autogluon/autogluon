import copy
from typing import Any, Dict, Optional

from autogluon.core.models import AbstractModel


def _assert_init_user_params(
    params_og: Optional[Dict[str, Any]], expected_params: Dict[str, Any], expected_params_aux: Dict[str, Any], **kwargs
):
    """
    Assert that `AbstractModel._init_user_params` works as intended
    and that `AbstractModel` calls `AbstractModel._init_user_params` in the expected way during init.
    """
    expected_params_og = copy.deepcopy(params_og) if params_og is not None else params_og
    params, params_aux = AbstractModel._init_user_params(params=params_og, **kwargs)
    assert params_og == expected_params_og  # Ensure no outer context update
    assert params == expected_params
    assert params_aux == expected_params_aux

    if kwargs is None or len(kwargs.keys()) == 0:
        abstract_model = AbstractModel(name="", path="", hyperparameters=params_og)
        assert params_og == expected_params_og
        assert abstract_model._user_params == expected_params
        assert abstract_model._user_params_aux == expected_params_aux


def test_init_user_params_none():
    params_og = None
    expected_params = {}
    expected_params_aux = {}
    _assert_init_user_params(
        params_og=params_og, expected_params=expected_params, expected_params_aux=expected_params_aux
    )


def test_init_user_params_simple():
    params_og = {
        "foo": 1,
        "bar": 2,
    }
    expected_params = {
        "foo": 1,
        "bar": 2,
    }
    expected_params_aux = {}
    _assert_init_user_params(
        params_og=params_og, expected_params=expected_params, expected_params_aux=expected_params_aux
    )


def test_init_user_params_ag_args_fit_none():
    params_og = {
        "foo": 1,
        "bar": 2,
        "ag_args_fit": None,
    }
    expected_params = {
        "foo": 1,
        "bar": 2,
    }
    expected_params_aux = {}
    _assert_init_user_params(
        params_og=params_og, expected_params=expected_params, expected_params_aux=expected_params_aux
    )


def test_init_user_params_with_prefix():
    params_og = {
        "foo": 1,
        "bar": 2,
        "ag.foo": 3,
    }
    expected_params = {
        "foo": 1,
        "bar": 2,
    }
    expected_params_aux = {"foo": 3}
    _assert_init_user_params(
        params_og=params_og, expected_params=expected_params, expected_params_aux=expected_params_aux
    )


def test_init_user_params_with_ag_args_fit():
    params_og = {
        "foo": 1,
        "bar": 2,
        "ag_args_fit": {"foo": 3},
    }
    expected_params = {
        "foo": 1,
        "bar": 2,
    }
    expected_params_aux = {"foo": 3}
    _assert_init_user_params(
        params_og=params_og, expected_params=expected_params, expected_params_aux=expected_params_aux
    )


def test_init_user_params_with_ag_args_fit_and_prefix():
    params_og = {
        "foo": 1,
        "bar": 2,
        "ag_args_fit": {"foo": 3, "ag.foo": 4, "ag.bar": 5, "ag.ag.bar": 7},
    }
    expected_params = {
        "foo": 1,
        "bar": 2,
    }
    expected_params_aux = {
        "foo": 4,
        "bar": 5,
        "ag.bar": 7,
    }
    _assert_init_user_params(
        params_og=params_og, expected_params=expected_params, expected_params_aux=expected_params_aux
    )


def test_init_user_params_with_all():
    params_og = {
        "foo": 1,
        "bar": 2,
        "ag.foo": 12,
        "ag_args_fit": {"foo": 3, "ag.foo": 4, "ag.bar": 5, "ag.ag.bar": 7},
    }
    expected_params = {
        "foo": 1,
        "bar": 2,
    }
    expected_params_aux = {
        "foo": 12,
        "bar": 5,
        "ag.bar": 7,
    }
    _assert_init_user_params(
        params_og=params_og, expected_params=expected_params, expected_params_aux=expected_params_aux
    )


def test_init_user_params_with_all_and_custom():
    params_og = {
        "foo": 1,
        "bar": 2,
        "custom.": "hello",
        "ag.foo": 12,
        "ag_args_fit": {"foo": 3, "ag.foo": 4, "ag.bar": 5, "ag.ag.bar": 7},
        "hello": {"custom.5": 22, "ag.custom.5": 33},
    }
    kwargs = {"ag_args_fit": "hello", "ag_arg_prefix": "custom."}
    expected_params = {
        "foo": 1,
        "bar": 2,
        "ag.foo": 12,
        "ag_args_fit": {"foo": 3, "ag.foo": 4, "ag.bar": 5, "ag.ag.bar": 7},
    }
    expected_params_aux = {
        "": "hello",
        "5": 22,
        "ag.custom.5": 33,
    }
    _assert_init_user_params(
        params_og=params_og, expected_params=expected_params, expected_params_aux=expected_params_aux, **kwargs
    )
