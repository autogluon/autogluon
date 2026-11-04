from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models.abstract import abstract_model
from autogluon.core.models.dummy.dummy_model import DummyModel


@pytest.fixture
def fitted_model(tmp_path):
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 2.0, 4.0, 6.0])
    model = DummyModel(path=str(tmp_path), name="Dummy", problem_type="regression")
    model.fit(X=X, y=y)
    return model, X


def _fail_save(**kwargs):
    raise OSError("disk full")


def test_failed_compiled_model_save_preserves_predictions(fitted_model, monkeypatch):
    model, X = fitted_model
    expected = model.predict(X)
    estimator = model.model
    # A compiled estimator is stored separately rather than inside model.pkl.
    model._compiler = SimpleNamespace(save_in_pkl=False)
    monkeypatch.setattr(abstract_model.save_pkl, "save", _fail_save)

    with pytest.raises(OSError, match="disk full"):
        model.save()

    assert model.model is estimator
    np.testing.assert_array_equal(model.predict(X), expected)


@pytest.mark.parametrize("fail_save", [False, True])
def test_copy_refit_preserves_original_model(fitted_model, monkeypatch, fail_save):
    model, X = fitted_model
    expected = model.predict(X)
    original_name, original_path = model.name, model.path
    if fail_save:
        monkeypatch.setattr(abstract_model.save_pkl, "save", _fail_save)
        with pytest.raises(OSError, match="disk full"):
            model.convert_to_refit_full_via_copy()
    else:
        refit = model.convert_to_refit_full_via_copy()
        assert refit.name == f"{original_name}_FULL"
        assert refit.path != original_path
        np.testing.assert_array_equal(refit.predict(X), expected)

    assert model.name == original_name
    assert model.path == original_path
    np.testing.assert_array_equal(model.predict(X), expected)
