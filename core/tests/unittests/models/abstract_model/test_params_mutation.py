"""Post-construction `params` mutation is deprecated (raises from AutoGluon 1.7)."""

import pickle
import warnings

import pytest

from autogluon.core.models import AbstractModel
from autogluon.core.models.abstract import _mutation_deprecated_dict
from autogluon.core.models.abstract._mutation_deprecated_dict import ParamsDict


def _initialized_model() -> AbstractModel:
    model = AbstractModel(
        name="",
        path="",
        problem_type="binary",
        eval_metric="log_loss",
        hyperparameters={"some_param": 1},
    )
    model.initialize()
    return model


def test_params_is_deprecation_dict():
    model = _initialized_model()
    assert isinstance(model.params, ParamsDict)
    assert model.params["some_param"] == 1


def test_params_mutation_warns():
    model = _initialized_model()
    with pytest.warns(DeprecationWarning, match="`params` after construction is deprecated"):
        model.params["some_param"] = 2
    with pytest.warns(DeprecationWarning, match="starting in AutoGluon 1.7"):
        model.params.pop("some_param")


def test_params_reads_and_copies_do_not_warn():
    model = _initialized_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert model.params["some_param"] == 1
        assert model.params.get("missing") is None
        copied = model.params.copy()
        assert type(copied) is dict
        copied["some_param"] = 2  # copies stay plain and mutable
        # `_get_model_params` (the documented pattern) works on a mutable copy
        model_params = model._get_model_params()
        model_params["some_param"] = 3


def test_params_survives_pickle():
    model = _initialized_model()
    loaded = pickle.loads(pickle.dumps(model))
    assert isinstance(loaded.params, ParamsDict)
    assert loaded.params == model.params


def test_params_mutation_raises_from_1_7(monkeypatch):
    model = _initialized_model()
    monkeypatch.setattr(_mutation_deprecated_dict, "_MUTATION_SHOULD_RAISE", True)
    with pytest.raises(TypeError, match="starting in AutoGluon 1.7"):
        model.params["some_param"] = 2
