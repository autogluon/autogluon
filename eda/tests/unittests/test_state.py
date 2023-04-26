import pytest

from autogluon.eda.state import AnalysisState, StateCheckMixin
from autogluon.eda.utils.common import expand_nested_args_into_nested_maps


def test_analysis_state():
    state: AnalysisState = AnalysisState(dict(a=1, b=2, c=3), {"b": 3}, c=4, d=5)
    assert state.a == 1
    assert state.b == 3
    assert state.c == 4
    assert state.d == 5


def test_analysis_state_ignore_non_dict_args():
    state: AnalysisState = AnalysisState(1, 2, 3, q=5)
    assert state.__dict__ == {"q": 5}


def test_analysis_state_nested():
    state: AnalysisState = AnalysisState(a={"b": 4})
    assert state.a.b == 4


def test_analysis_state_nested_missing():
    state: AnalysisState = AnalysisState(a={"b": 4})
    assert state.missing is None


def test_analysis_state_mutation():
    state: AnalysisState = AnalysisState()
    assert state.a is None
    state.a = 42
    assert state.a == 42


def test_analysis_state_nested_mutation():
    state: AnalysisState = AnalysisState()
    assert state.a is None
    state.a = {}
    state.a.b = 42
    assert state.a.b == 42


def test_statecheckmixin_at_least_one_key_must_be_present():
    assert StateCheckMixin().at_least_one_key_must_be_present(AnalysisState(q=42, w=43), "missing") is False
    assert StateCheckMixin().at_least_one_key_must_be_present(AnalysisState(q=42, w=43), "q") is True
    assert StateCheckMixin().at_least_one_key_must_be_present(AnalysisState(q=42, w=43), "w") is True
    assert StateCheckMixin().at_least_one_key_must_be_present(AnalysisState(q=42, w=43), "q", "w") is True
    assert StateCheckMixin().at_least_one_key_must_be_present(AnalysisState(q=42, w=43), "q", "missing") is True
    assert StateCheckMixin().at_least_one_key_must_be_present(AnalysisState(q=None, w=43), "q", "missing") is False


def test_statecheckmixin_all_keys_must_be_present():
    assert StateCheckMixin().all_keys_must_be_present(AnalysisState(q=42, w=43), "missing") is False
    assert StateCheckMixin().all_keys_must_be_present(AnalysisState(q=42, w=43), "q") is True
    assert StateCheckMixin().all_keys_must_be_present(AnalysisState(q=None), "q") is False
    assert StateCheckMixin().all_keys_must_be_present(AnalysisState(q=42, w=43), "w") is True
    assert StateCheckMixin().all_keys_must_be_present(AnalysisState(q=42, w=43), "q", "w") is True
    assert StateCheckMixin().all_keys_must_be_present(AnalysisState(q=42, w=43), "q", "missing") is False


def test_expand_nested_args_into_nested_maps():
    assert expand_nested_args_into_nested_maps({"a": 1, "b.a": 3, "c.a.b": 4}) == {
        "a": 1,
        "b": {"a": 3},
        "c": {"a": {"b": 4}},
    }


def test_expand_nested_args_into_nested_maps__namespaces_overlap():
    with pytest.raises(ValueError):
        expand_nested_args_into_nested_maps({"a": 1, "a.b": 2})
    with pytest.raises(ValueError):
        expand_nested_args_into_nested_maps({"a.b": 1, "a.b.c": 2})
