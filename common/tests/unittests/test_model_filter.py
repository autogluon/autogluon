import pytest

from autogluon.common.model_filter import ModelFilter


@pytest.mark.parametrize(
    "models,included_model_types,excluded_model_types,expected_answer",
    [
        ({"dummy": {}}, ["dummy"], None, {"dummy": {}}),
        ({"dummy": {"dummy": 1}}, ["dummy"], None, {"dummy": {"dummy": 1}}),
        ({"dummy": {}}, ["foo"], None, {}),
        ({"dummy": {}, "foo": {}}, ["foo"], None, {"foo": {}}),
        ({}, ["foo"], None, {}),
        ({"dummy": {}}, None, ["dummy"], {}),
        ({"dummy": {"dummy": 1}}, None, ["dummy"], {}),
        ({"dummy": {}}, None, ["foo"], {"dummy": {}}),
        ({"dummy": {}, "foo": {}}, None, ["foo"], {"dummy": {}}),
        ({}, None, ["foo"], {}),
        ({"dummy": {}}, ["dummy"], ["dummy"], {"dummy": {}}),
        ({"dummy": {"dummy": 1}}, ["dummy"], ["dummy"], {"dummy": {"dummy": 1}}),
        ({"dummy": {}}, ["foo"], ["dummy"], {}),
        ({"dummy": {}, "foo": {}}, ["foo"], ["dummy"], {"foo": {}}),
        ({}, ["foo"], ["foo"], {}),
        ({"dummy": {}}, None, None, {"dummy": {}}),
        (["dummy"], ["dummy"], None, ["dummy"]),
        (["dummy"], ["foo"], None, []),
        (["dummy", "foo"], ["foo"], None, ["foo"]),
        ([], ["foo"], None, []),
        (["dummy"], None, ["foo"], ["dummy"]),
        ([], None, ["foo"], []),
        (["dummy"], None, ["dummy"], []),
        (["dummy", "foo"], None, ["dummy"], ["foo"]),
        (["dummy"], None, None, ["dummy"]),
    ],
)
def test_filter_model(models, included_model_types, excluded_model_types, expected_answer):
    assert (
        ModelFilter.filter_models(
            models=models, included_model_types=included_model_types, excluded_model_types=excluded_model_types
        )
        == expected_answer
    )
