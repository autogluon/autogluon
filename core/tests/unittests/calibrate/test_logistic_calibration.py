from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy.special import expit, logit, softmax

import autogluon.core.calibrate.logistic_calibration as lc
from autogluon.core.calibrate.logistic_calibration import LogisticCalibrator, cross_val_calibrated_proba


def _log_loss(y: np.ndarray, y_pred_proba: np.ndarray) -> float:
    if y_pred_proba.ndim == 1:
        y_pred_proba = np.stack([1 - y_pred_proba, y_pred_proba], axis=1)
    return float(-np.mean(np.log(np.clip(y_pred_proba[np.arange(len(y)), y], 1e-15, 1))))


def _overconfident_binary(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p_true = expit(rng.normal(0, 2, n))
    y = (rng.random(n) < p_true).astype(int)
    return expit(2.5 * logit(p_true) + 0.7), y


def _overconfident_multiclass(n: int = 2000, k: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, (n, k))
    p_true = softmax(z, axis=1)
    y = np.array([rng.choice(k, p=row) for row in p_true])
    return softmax(2.0 * z + rng.normal(0, 0.5, k), axis=1), y


def test_binary_platt_scaling_fixes_overconfidence_and_keeps_the_shape():
    p, y = _overconfident_binary()
    calibrator = LogisticCalibrator(problem_type="binary").fit(p, y)
    p_cal = calibrator.predict_proba(p)
    assert p_cal.shape == p.shape
    assert np.all((p_cal > 0) & (p_cal < 1))
    assert _log_loss(y, p_cal) < _log_loss(y, p) - 0.1
    # The slope undoes the overconfidence (logits were multiplied by 2.5).
    assert calibrator.coef_[1] == pytest.approx(1 / 2.5, abs=0.1)


def test_multiclass_matrix_scaling_fixes_overconfidence_and_returns_distributions():
    p, y = _overconfident_multiclass()
    p_cal = LogisticCalibrator(problem_type="multiclass").fit(p, y).predict_proba(p)
    assert p_cal.shape == p.shape
    np.testing.assert_allclose(p_cal.sum(axis=1), 1.0)
    assert _log_loss(y, p_cal) < _log_loss(y, p) - 0.1


def test_calibrated_input_is_left_nearly_unchanged():
    rng = np.random.default_rng(1)
    p = softmax(rng.normal(0, 1.5, (5000, 3)), axis=1)
    y = np.array([rng.choice(3, p=row) for row in p])
    p_cal = LogisticCalibrator(problem_type="multiclass").fit(p, y).predict_proba(p)
    assert np.abs(p_cal - p).max() < 0.05


def test_zero_probabilities_and_absent_classes_stay_finite():
    p, y = _overconfident_multiclass(n=500, k=5)
    p[:50, 0] = 0.0
    p = p / p.sum(axis=1, keepdims=True)
    y[y == 4] = 3  # class 4 never occurs in the calibration data
    p_cal = LogisticCalibrator(problem_type="multiclass").fit(p, y).predict_proba(p)
    assert np.all(np.isfinite(p_cal))
    np.testing.assert_allclose(p_cal.sum(axis=1), 1.0)

    p_bin, y_bin = _overconfident_binary(n=500)
    p_bin[:10] = 0.0
    p_bin[10:20] = 1.0
    assert np.all(np.isfinite(LogisticCalibrator(problem_type="binary").fit(p_bin, y_bin).predict_proba(p_bin)))


def test_fitted_calibrator_pickles():
    p, y = _overconfident_multiclass(n=300)
    calibrator = LogisticCalibrator(problem_type="multiclass").fit(p, y)
    restored = pickle.loads(pickle.dumps(calibrator))
    np.testing.assert_array_equal(restored.predict_proba(p), calibrator.predict_proba(p))


def test_unsupported_problem_type_and_unfitted_use_raise():
    with pytest.raises(ValueError, match="supports"):
        LogisticCalibrator(problem_type="regression")
    with pytest.raises(ValueError, match="backend"):
        LogisticCalibrator(problem_type="multiclass", backend="jax")
    with pytest.raises(AssertionError, match="must be fit"):
        LogisticCalibrator(problem_type="binary").predict_proba(np.array([0.5]))


def test_torch_and_numpy_backends_agree():
    pytest.importorskip("torch")
    p, y = _overconfident_multiclass(n=3000, k=6)
    fits = {b: LogisticCalibrator(problem_type="multiclass", backend=b).fit(p, y) for b in ("numpy", "torch")}
    assert {b: fit.backend_ for b, fit in fits.items()} == {"numpy": "numpy", "torch": "torch"}
    p_numpy, p_torch = fits["numpy"].predict_proba(p), fits["torch"].predict_proba(p)
    assert _log_loss(y, p_torch) == pytest.approx(_log_loss(y, p_numpy), abs=1e-5)
    assert np.abs(p_torch - p_numpy).max() < 1e-3
    assert fits["torch"].inv_temperature_ == pytest.approx(fits["numpy"].inv_temperature_, rel=1e-5)


def test_auto_backend_uses_torch_on_large_inputs_only(monkeypatch):
    torch = pytest.importorskip("torch")
    p, y = _overconfident_multiclass(n=500, k=4)
    monkeypatch.setattr(lc, "_TORCH_MIN_CELLS", 1000)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    assert LogisticCalibrator(problem_type="multiclass").fit(p, y).backend_ == "numpy", (
        "Newton-sized fits stay on numpy"
    )
    monkeypatch.setattr(lc, "_NEWTON_MAX_CLASSES", 2)
    assert LogisticCalibrator(problem_type="multiclass").fit(p, y).backend_ == "torch"
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    assert LogisticCalibrator(problem_type="multiclass").fit(p, y).backend_ == "numpy"
    monkeypatch.setattr(lc, "_TORCH_MIN_CELLS", 10_000)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    assert LogisticCalibrator(problem_type="multiclass").fit(p, y).backend_ == "numpy"


def test_newton_matches_lbfgs_and_falls_back_to_it(monkeypatch):
    p, y = _overconfident_multiclass(n=3000, k=5)
    newton = LogisticCalibrator(problem_type="multiclass", backend="numpy").fit(p, y).predict_proba(p)
    monkeypatch.setattr(lc, "_NEWTON_MAX_CLASSES", 0)
    lbfgs = LogisticCalibrator(problem_type="multiclass", backend="numpy").fit(p, y).predict_proba(p)
    assert _log_loss(y, newton) == pytest.approx(_log_loss(y, lbfgs), abs=1e-6)
    assert np.abs(newton - lbfgs).max() < 1e-3
    monkeypatch.setattr(lc, "_NEWTON_MAX_CLASSES", 8)
    monkeypatch.setattr(lc, "_NEWTON_MAX_EVALS", 2)  # Newton cannot converge: L-BFGS finishes the fit
    fallback = LogisticCalibrator(problem_type="multiclass", backend="numpy").fit(p, y).predict_proba(p)
    assert _log_loss(y, fallback) == pytest.approx(_log_loss(y, lbfgs), abs=1e-6)


def test_separable_binary_input_stays_finite():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.where(y == 1, rng.uniform(0.6, 1.0, 500), rng.uniform(0.0, 0.4, 500))
    calibrator = LogisticCalibrator(problem_type="binary").fit(p, y)
    assert np.all(np.isfinite(calibrator.coef_))
    p_cal = calibrator.predict_proba(p)
    assert np.all(np.isfinite(p_cal)) and _log_loss(y, p_cal) < _log_loss(y, p)


def test_without_torch_auto_falls_back_to_numpy_and_torch_backend_raises(monkeypatch):
    monkeypatch.setattr(lc, "_TORCH_MIN_CELLS", 0)
    monkeypatch.setattr(lc, "_import_torch", lambda: None)
    p, y = _overconfident_multiclass(n=300, k=3)
    assert LogisticCalibrator(problem_type="multiclass").fit(p, y).backend_ == "numpy"
    with pytest.raises(ImportError, match="requires torch"):
        LogisticCalibrator(problem_type="multiclass", backend="torch").fit(p, y)


@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_cross_val_calibrated_proba_is_out_of_fold(problem_type):
    p, y = _overconfident_binary(n=1000) if problem_type == "binary" else _overconfident_multiclass(n=1000)
    oof = cross_val_calibrated_proba(p, y, problem_type=problem_type, n_splits=5)
    in_sample = LogisticCalibrator(problem_type=problem_type).fit(p, y).predict_proba(p)
    assert oof.shape == p.shape
    assert _log_loss(y, oof) < _log_loss(y, p)
    assert _log_loss(y, oof) >= _log_loss(y, in_sample), "held-out rows score no better than in-sample ones"
    assert not np.allclose(oof, in_sample)


def test_cross_val_calibrated_proba_falls_back_to_in_sample_for_a_singleton_class():
    p, y = _overconfident_multiclass(n=200, k=3)
    y[:] = np.where(y == 2, 1, y)
    y[0] = 2  # one row of class 2: no stratified split exists
    oof = cross_val_calibrated_proba(p, y, problem_type="multiclass")
    in_sample = LogisticCalibrator(problem_type="multiclass").fit(p, y).predict_proba(p)
    np.testing.assert_allclose(oof, in_sample)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_matches_probmetrics_logistic_calibrator(backend):
    """The reference implementation this ports; skipped when it is not installed."""
    get_calibrator = pytest.importorskip("probmetrics.calibrators").get_calibrator
    for p, y, binary in [(*_overconfident_binary(), True), (*_overconfident_multiclass(k=6), False)]:
        reference = get_calibrator("logistic").fit(p.astype(np.float64), y).predict_proba(p.astype(np.float64))
        calibrator = LogisticCalibrator(problem_type="binary" if binary else "multiclass", backend=backend)
        ours = calibrator.fit(p, y).predict_proba(p)
        if binary:
            reference = reference[:, 1]
        assert _log_loss(y, ours) == pytest.approx(_log_loss(y, reference), abs=1e-3)
        assert np.abs(ours - reference).max() < 1e-2
