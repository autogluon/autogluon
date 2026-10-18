import copy
from unittest import mock

from autogluon.core.models import AbstractModel
from autogluon.core.models.abstract import abstract_model as abstract_model_module


def _assert_merge(hyperparameters, params_trained, expected):
    params_trained_og = copy.deepcopy(params_trained)
    AbstractModel._update_hyperparameters_with_params_trained(hyperparameters, params_trained)
    assert hyperparameters == expected
    assert params_trained == params_trained_og  # Ensure no outer context update


def test_prefixed_key_routes_into_ag_args_fit():
    """An `ag.`-prefixed trained param must land in the `ag_args_fit` sub-dict rather than at the
    top level, so the merged dict carries one encoding per parameter and constructing a template
    from it does not log the "present in both" collision warning."""
    _assert_merge(
        hyperparameters={"ag_args_fit": {"max_rows": 100}},
        params_trained={"ag.max_rows": None},
        expected={"ag_args_fit": {"max_rows": None}},
    )


def test_prefixed_key_creates_ag_args_fit():
    _assert_merge(
        hyperparameters={},
        params_trained={"ag.max_rows": 3},
        expected={"ag_args_fit": {"max_rows": 3}},
    )


def test_unprefixed_keys_update_top_level():
    _assert_merge(
        hyperparameters={"n_estimators": 1, "ag_args_fit": {"max_rows": 100}},
        params_trained={"n_estimators": 7, "learning_rate": 0.1},
        expected={"n_estimators": 7, "learning_rate": 0.1, "ag_args_fit": {"max_rows": 100}},
    )


def test_merged_dict_initializes_without_collision_warning():
    """End-to-end over `_init_user_params`: the merged dict must produce the same params_aux the
    old dual-encoding produced, without the collision warning."""
    hyperparameters = {"ag_args_fit": {"max_rows": 100}}
    AbstractModel._update_hyperparameters_with_params_trained(hyperparameters, {"ag.max_rows": None})
    with mock.patch.object(abstract_model_module.logger, "warning") as warning:
        params, params_aux = AbstractModel._init_user_params(params=hyperparameters)
    assert params == {}
    assert params_aux == {"max_rows": None}
    assert not any("present in both" in str(call) for call in warning.call_args_list)


def test_genuine_user_collision_still_warns():
    """The warning stays for a user who really specifies a parameter both ways."""
    with mock.patch.object(abstract_model_module.logger, "warning") as warning:
        params, params_aux = AbstractModel._init_user_params(
            params={"ag_args_fit": {"max_rows": 100}, "ag.max_rows": 5}
        )
    assert params == {}
    assert params_aux == {"max_rows": 5}
    assert any("present in both" in str(call) for call in warning.call_args_list)
