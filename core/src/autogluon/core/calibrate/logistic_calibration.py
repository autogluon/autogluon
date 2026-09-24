"""Logistic post-hoc calibration of predicted class probabilities.

Binary problems use Platt scaling on the logit of the positive-class probability. Multiclass
problems use structured matrix scaling (SMS): temperature scaling, a small mixture with the uniform
distribution, then an affine map of the log-probabilities whose weights are penalized by a ridge
term that shrinks with the number of rows. Both fits are convex and run on numpy and scipy only;
the multiclass fit works in float32, like the reference implementation, and single-threaded BLAS.

References
----------
John Platt. Probabilistic outputs for support vector machines and comparisons to regularized
likelihood methods. Advances in Large Margin Classifiers, 1999.

Eugène Berta, David Holzmüller, Michael I. Jordan, and Francis Bach. Structured matrix scaling for
multi-class calibration. International Conference on Artificial Intelligence and Statistics, 2026.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, softmax

from ..constants import BINARY, MULTICLASS

# Log-probabilities are floored at the log of the smallest normal float32, so a predicted
# probability of 0 gives a large finite logit instead of -inf.
_LOG_FLOOR = float(np.log(np.finfo(np.float32).tiny))


def _log_proba(y_pred_proba: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.maximum(np.log(y_pred_proba), _LOG_FLOOR)


def _single_threaded_blas() -> AbstractContextManager:
    """Limit BLAS to one thread: on the narrow matrices of a calibration fit, many threads cost
    far more than they save (up to 25x slower with 48 OpenBLAS threads)."""
    try:
        from threadpoolctl import threadpool_limits  # installed with scikit-learn
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


class LogisticCalibrator:
    """Fits a logistic map from predicted probabilities to calibrated probabilities.

    Parameters
    ----------
    problem_type : str
        ``"binary"`` or ``"multiclass"``. Binary probabilities are the positive-class column only,
        shape ``(n,)``, as `predict_proba` returns them; multiclass probabilities have shape
        ``(n, n_classes)``. The output has the input's shape.
    reg_lambda : float, default 1.0
        Ridge strength of the multiclass fit. The intercept and the diagonal are penalized by
        ``reg_lambda * K / n`` and the off-diagonal weights by ``reg_lambda * K * (K - 1) / n``, for
        ``K`` classes and ``n`` calibration rows.
    max_iter : int, default 1000
        Iteration limit of the L-BFGS solver.
    tol : float, default 1e-8
        Relative objective decrease at which the multiclass L-BFGS fit stops.
    """

    def __init__(self, problem_type: str, reg_lambda: float = 1.0, max_iter: int = 1000, tol: float = 1e-8):
        if problem_type not in (BINARY, MULTICLASS):
            raise ValueError(f"LogisticCalibrator supports {BINARY} and {MULTICLASS}, got {problem_type!r}.")
        self.problem_type = problem_type
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.inv_temperature_: float | None = None
        self.uniform_weight_: float | None = None

    def fit(self, y_pred_proba: np.ndarray, y: np.ndarray) -> LogisticCalibrator:
        """Fit on predicted probabilities and integer labels ``0..n_classes-1``."""
        y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if self.problem_type == BINARY:
            self._fit_binary(y_pred_proba, y)
        else:
            with _single_threaded_blas():
                self._fit_multiclass(y_pred_proba, y)
        return self

    def predict_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """The calibrated probabilities, in the input's shape."""
        if self.coef_ is None:
            raise AssertionError("LogisticCalibrator must be fit before predict_proba.")
        y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
        if self.problem_type == BINARY:
            a, b = self.coef_
            return expit(a + b * self._binary_logit(y_pred_proba))
        log_q = _log_proba(self._scale_temperature(y_pred_proba))
        k = log_q.shape[1]
        weights, bias = self.coef_[: k * k].reshape(k, k), self.coef_[k * k :]
        return softmax(log_q @ (np.eye(k) + weights).T + bias, axis=1)

    @staticmethod
    def _binary_logit(p: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            log_p = np.clip(np.log(p), _LOG_FLOOR, -_LOG_FLOOR)
            log_1mp = np.clip(np.log1p(-p), _LOG_FLOOR, -_LOG_FLOOR)
        return log_p - log_1mp

    def _fit_binary(self, p: np.ndarray, y: np.ndarray) -> None:
        """Platt scaling: ``sigmoid(a + b * logit(p))`` by unpenalized maximum likelihood."""
        z = self._binary_logit(p)

        def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            s = theta[0] + theta[1] * z
            residual = expit(s) - y
            loss = np.mean(np.logaddexp(0.0, s) - y * s)
            return loss, np.array([residual.mean(), (residual * z).mean()])

        res = minimize(
            loss_and_grad, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter}
        )
        self.coef_ = res.x

    def _fit_temperature(self, log_p: np.ndarray, y: np.ndarray) -> None:
        """Inverse temperature ``exp(u)``, ``u`` in ``[-16, 16]``, at the root of the cross-entropy's derivative.

        The derivative is nondecreasing in ``u``, so a root found by Brent's method is the minimum;
        without a sign change the minimum is at the end of the interval the derivative points to.
        """
        target = log_p[np.arange(len(y)), y].mean(dtype=np.float64)

        def grad(u: float) -> float:
            z = np.exp(u) * log_p
            z -= z.max(axis=1, keepdims=True)
            np.exp(z, out=z)
            return float(((log_p * z).sum(axis=1) / z.sum(axis=1)).mean(dtype=np.float64) - target)

        low, high = -16.0, 16.0
        if grad(low) > 0:
            u = low
        elif grad(high) <= 0:
            u = high
        else:
            u = brentq(grad, low, high, xtol=1e-9)
        self.inv_temperature_ = float(np.exp(u))
        self.uniform_weight_ = 1.0 / (len(y) + 1)

    def _scale_temperature(self, p: np.ndarray) -> np.ndarray:
        q = softmax(self.inv_temperature_ * _log_proba(p), axis=1)
        return (1.0 - self.uniform_weight_) * q + self.uniform_weight_ / q.shape[1]

    def _fit_multiclass(self, p: np.ndarray, y: np.ndarray) -> None:
        """Structured matrix scaling on the temperature-scaled log-probabilities."""
        n, k = p.shape
        self._fit_temperature(_log_proba(p).astype(np.float32), y)
        log_q = _log_proba(self._scale_temperature(p)).astype(np.float32)
        reg_bias = self.reg_lambda * k / n
        reg_matrix = np.full((k, k), self.reg_lambda * k * (k - 1) / n)
        np.fill_diagonal(reg_matrix, self.reg_lambda * k / n)
        rows = np.arange(n)
        identity = np.eye(k)

        def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            weights, bias = theta[: k * k].reshape(k, k), theta[k * k :]
            s = log_q @ (identity + weights).T.astype(np.float32) + bias.astype(np.float32)
            s -= s.max(axis=1, keepdims=True)
            s_true = s[rows, y]
            np.exp(s, out=s)
            norm = s.sum(axis=1)
            loss = np.log(norm).mean(dtype=np.float64) - s_true.mean(dtype=np.float64)
            # `s` becomes the residual: softmax minus the one-hot labels
            s /= norm[:, None]
            s[rows, y] -= 1
            grad_weights = (s.T @ log_q).astype(np.float64) / n + 2 * reg_matrix * weights
            grad_bias = s.sum(axis=0, dtype=np.float64) / n + 2 * reg_bias * bias
            loss += (reg_matrix * weights**2).sum() + reg_bias * (bias**2).sum()
            return float(loss), np.concatenate([grad_weights.ravel(), grad_bias])

        res = minimize(
            loss_and_grad,
            np.zeros(k * (k + 1)),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "maxcor": 30, "ftol": self.tol},
        )
        self.coef_ = res.x


def cross_val_calibrated_proba(
    y_pred_proba: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    n_splits: int = 10,
    random_state: int = 0,
) -> np.ndarray:
    """Out-of-fold calibrated probabilities: each fold is calibrated by a `LogisticCalibrator` fit on the others.

    Uses stratified folds, at most as many as the largest class has rows. Falls back to in-sample
    calibrated probabilities when a stratified split is impossible (fewer than 2 folds, or a class
    with a single row).
    """
    from autogluon.common.utils.cv_splitter import CVSplitter

    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(n_splits, int(counts.max()))
    if n_splits < 2 or counts.min() < 2:
        return LogisticCalibrator(problem_type=problem_type).fit(y_pred_proba, y).predict_proba(y_pred_proba)
    splitter = CVSplitter(n_splits=n_splits, n_repeats=1, stratify=True, random_state=random_state)
    calibrated = np.empty_like(np.asarray(y_pred_proba, dtype=np.float64))
    for train_index, holdout_index in splitter.split(X=None, y=pd.Series(y)):
        calibrator = LogisticCalibrator(problem_type=problem_type).fit(y_pred_proba[train_index], y[train_index])
        calibrated[holdout_index] = calibrator.predict_proba(y_pred_proba[holdout_index])
    return calibrated
