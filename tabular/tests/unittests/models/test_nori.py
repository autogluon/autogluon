"""Backend-agnostic tests for NoriModel.

`synthefy-nori` is an optional dependency that is intentionally excluded from the
test bundle (it requires ``huggingface_hub>=1.0``, which conflicts with the
``<1.0`` cap used by the mitra/tabpfnmix extras). These tests therefore inject a
fake ``synthefy_nori`` module so the wrapper's plumbing (including model-size
variant selection) can be exercised without downloading a checkpoint.
"""

import sys
import types

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.nori.nori_model import NoriModel

# Populated by the fake NoriRegressor below with the last constructor kwargs.
_CAPTURED: dict = {}


class _FakeNoriRegressor:
    """Module-level (picklable) stand-in for ``synthefy_nori.NoriRegressor``.

    Mirrors just enough of the real interface for the AutoGluon wrapper: a
    ``device`` attribute (read by ``NoriModel.get_device``), a lazily-cleared
    ``_predictor``, and sklearn-style ``fit``/``predict``.
    """

    def __init__(self, **kwargs):
        _CAPTURED["init_kwargs"] = kwargs
        self.device = kwargs.get("device")
        self._predictor = None

    def fit(self, X, y):
        self._mean = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X, *, output_type="mean", **kwargs):
        return np.full(len(X), self._mean, dtype=float)


def _install_fake_nori(monkeypatch):
    _CAPTURED.clear()
    fake = types.ModuleType("synthefy_nori")
    fake.NoriRegressor = _FakeNoriRegressor
    monkeypatch.setitem(sys.modules, "synthefy_nori", fake)


def _toy_regression_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),  # exercises the label-encode path
            "target": rng.normal(size=n),
        }
    )


def _fit_nori(hyperparameters):
    df = _toy_regression_df()
    predictor = TabularPredictor(label="target", problem_type="regression").fit(
        df,
        hyperparameters={NoriModel: hyperparameters},
        fit_weighted_ensemble=False,
        verbosity=0,
    )
    return predictor.predict(df.drop(columns=["target"]).head(5))


def test_nori_default_variant(monkeypatch):
    """With no `model` hyperparameter, the variant is left to NoriRegressor's default."""
    _install_fake_nori(monkeypatch)
    preds = _fit_nori({})
    assert len(preds) == 5
    assert "model" not in _CAPTURED["init_kwargs"]


def test_nori_30m_variant_forwarded(monkeypatch):
    """`model='nori-30m'` is forwarded verbatim to NoriRegressor (Nori-30M support)."""
    _install_fake_nori(monkeypatch)
    preds = _fit_nori({"model": "nori-30m"})
    assert len(preds) == 5
    assert _CAPTURED["init_kwargs"].get("model") == "nori-30m"
