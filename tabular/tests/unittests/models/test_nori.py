"""Backend-agnostic tests for NoriModel.

Most tests here inject a fake ``synthefy_nori`` module so the wrapper's plumbing
(including model-size variant selection) is exercised without downloading a
checkpoint. ``test_nori`` is the end-to-end fit against the real backend, which
needs the ``nori`` extra installed.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.nori.nori_model import NoriModel
from autogluon.tabular.testing import FitHelper

# Populated by the fake NoriRegressor below with the last constructor kwargs.
_CAPTURED: dict = {}


class _FakeNoriRegressor:
    """Module-level (picklable) stand-in for ``synthefy_nori.NoriRegressor``.

    Mirrors just enough of the real interface for the AutoGluon wrapper: a
    ``device`` attribute (read by ``NoriModel.get_device``), a lazily-cleared
    ``_predictor``, and sklearn-style ``fit``/``predict``. The constructor rejects
    unknown kwargs like the real one (synthefy-nori 0.12.0), so the wrapper
    forwarding a hyperparameter the library doesn't accept fails here too.
    """

    _INIT_KWARGS = {
        "model_path",
        "model",
        "device",
        "inference_config",
        "token",
        "augmentations",
        "yj_skew_threshold",
        "quantile_collapse",
        "bar_temperature",
        "bar_point_estimator",
        "discrete_y_snap_max_unique",
        "discretize",
        "categorical_levels",
        "text_columns",
        "svd_dim",
        "embedder",
        "text_max_cardinality",
        "text_normalize",
    }

    def __init__(self, **kwargs):
        unknown = set(kwargs) - self._INIT_KWARGS
        if unknown:
            raise TypeError(f"NoriRegressor.__init__() got an unexpected keyword argument {min(unknown)!r}")
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
    # Disable the memory-safety check: the fake backend uses ~no memory, so it isn't
    # verifying anything real and would only skip the model on low-RAM runners.
    hyperparameters = {**hyperparameters, "ag.max_memory_usage_ratio": None}
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


@pytest.mark.skip(
    reason="Fits the real backend, which downloads the Nori checkpoint from Hugging Face. A "
    "rejected or throttled download would fail CI, so run this one manually."
)
def test_nori():
    """End-to-end fit test with the real synthefy-nori backend (regression only)."""
    FitHelper.verify_model(
        model_cls=NoriModel,
        model_hyperparameters={},
        verify_load_wo_cuda=True,
        # Nori attends queries over the shared context, so predicting on an
        # individual sample is not guaranteed to match batch prediction exactly.
        verify_single_prediction_equivalent_to_multi=False,
    )


def test_nori_device_hyperparameter_dropped(monkeypatch):
    """A user-supplied `device` hyperparameter is dropped in favor of the device
    resolved from allocated resources (would otherwise duplicate the kwarg)."""
    _install_fake_nori(monkeypatch)
    preds = _fit_nori({"device": "cpu"})
    assert len(preds) == 5
    assert _CAPTURED["init_kwargs"].get("device") in ("cpu", "cuda")


def test_no_feature_cap_and_estimate_saturates():
    """Nori sees at most 256 columns, so the cap is lifted and the estimate must not exceed it.

    Its inference config SVD-projects anything wider than 256 features down to 256 components
    before the model, so a wide fit costs no more memory than a 256-feature one. Measured peak VRAM
    at 1000 rows is 8.80 GB at 256 features and 8.89 GB at 5000 — flat — while the unclamped
    estimate reached 61 GB (6.9x measured), enough to have a wide fit skipped for lack of VRAM.
    """
    assert NoriModel()._get_default_auxiliary_params()["max_features"] is None

    def estimate(n_features: int) -> int:
        X = pd.DataFrame(np.zeros((1000, n_features), dtype="float32"))
        return NoriModel._estimate_gpu_memory_usage_static(X=X)

    # Below the internal width the estimate still scales with the input.
    assert estimate(100) < estimate(200) < estimate(NoriModel._INTERNAL_MAX_FEATURES)
    # At and beyond it, the estimate is constant, because the model's view is.
    saturated = estimate(NoriModel._INTERNAL_MAX_FEATURES)
    for n_features in (300, 1_000, 5_000):
        assert estimate(n_features) == saturated
    # Still above the measured 8.89 GB peak at 5000 features: clamping must not underestimate.
    assert saturated > 8.9 * 1024**3
