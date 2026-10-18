"""Post-construction `params_aux` mutation is deprecated (raises from AutoGluon 1.7)."""

import pickle
import warnings

import pytest

from autogluon.core.models import AbstractModel
from autogluon.core.models.abstract import _mutation_deprecated_dict
from autogluon.core.models.abstract._auxiliary_params import ParamsAuxDict


def _initialized_model() -> AbstractModel:
    model = AbstractModel(name="", path="", problem_type="binary", eval_metric="log_loss")
    model.initialize()
    return model


def test_params_aux_is_deprecation_dict():
    model = _initialized_model()
    assert isinstance(model.params_aux, ParamsAuxDict)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("max_rows", 1),
        lambda d: d.__delitem__("max_memory_usage_ratio"),
        lambda d: d.update({"max_rows": 1}),
        lambda d: d.pop("max_memory_usage_ratio"),
        lambda d: d.popitem(),
        lambda d: d.clear(),
        lambda d: d.setdefault("max_rows", 1),  # missing key: an insertion
        lambda d: d.__ior__({"max_rows": 1}),
    ],
)
def test_params_aux_mutation_warns(mutate):
    model = _initialized_model()
    with pytest.warns(DeprecationWarning, match="starting in AutoGluon 1.7"):
        mutate(model.params_aux)


def test_params_aux_reads_do_not_warn():
    model = _initialized_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert model.params_aux["max_memory_usage_ratio"] == 1.0
        assert model.params_aux.get("max_rows") is None
        # setdefault on a present key is a read, not a mutation
        assert model.params_aux.setdefault("max_memory_usage_ratio", 2.0) == 1.0
        assert "max_rows" not in model.params_aux
        dict(model.params_aux)
        len(model.params_aux)


def test_params_aux_copy_is_plain_mutable_dict():
    # copy-and-edit code (e.g. `get_params_aux_info`) must stay warning-free
    model = _initialized_model()
    copied = model.params_aux.copy()
    assert type(copied) is dict
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        copied["max_rows"] = 1


def test_params_aux_survives_pickle():
    model = _initialized_model()
    loaded = pickle.loads(pickle.dumps(model))
    assert isinstance(loaded.params_aux, ParamsAuxDict)
    assert loaded.params_aux == model.params_aux
    with pytest.warns(DeprecationWarning):
        loaded.params_aux["max_rows"] = 1


def test_params_aux_mutation_raises_from_1_7(monkeypatch):
    model = _initialized_model()
    monkeypatch.setattr(_mutation_deprecated_dict, "_MUTATION_SHOULD_RAISE", True)
    with pytest.raises(TypeError, match="starting in AutoGluon 1.7"):
        model.params_aux["max_rows"] = 1
