"""Logistic post-hoc calibration of predicted class probabilities.

Binary problems use Platt scaling on the logit of the positive-class probability. Multiclass
problems use structured matrix scaling (SMS): temperature scaling, a small mixture with the uniform
distribution, then an affine map of the log-probabilities whose weights are penalized by a ridge
term that shrinks with the number of rows. Both fits are convex and optimized with scipy's L-BFGS.
The multiclass objective is evaluated in float32, like the reference implementation: with torch
on its CPU threads when torch is installed and the data is large enough to gain from them, else
with numpy. Both backends run the same optimizer from the same start, with numpy's and scipy's
BLAS on a single thread.

References
----------
John Platt. Probabilistic outputs for support vector machines and comparisons to regularized
likelihood methods. Advances in Large Margin Classifiers, 1999.

Eugène Berta, David Holzmüller, Michael I. Jordan, and Francis Bach. Structured matrix scaling for
multi-class calibration. International Conference on Artificial Intelligence and Statistics, 2026.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from types import ModuleType

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


# torch pays a fixed cost per operation and gains from its threads, so "auto" uses it only on
# inputs of at least this many cells (rows times classes) and with more than one torch thread.
_TORCH_MIN_CELLS = 50_000


def _single_threaded_blas() -> AbstractContextManager:
    """Limit numpy's and scipy's BLAS to one thread (torch's own threads are untouched).

    On the narrow matrices of a calibration fit and the short vectors of scipy's L-BFGS, many BLAS
    threads cost far more than they save: up to 25x slower with 48 OpenBLAS threads, and much worse
    while torch's thread pool competes for the same cores.
    """
    try:
        from threadpoolctl import threadpool_limits  # installed with scikit-learn
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def _import_torch() -> ModuleType | None:
    try:
        import torch
    except (ImportError, OSError):
        return None
    return torch


def _temperature_grad_numpy(log_p: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    """The derivative of the cross-entropy of ``softmax(exp(u) * log_p)`` in the inverse temperature."""
    target = log_p[np.arange(len(y)), y].mean(dtype=np.float64)

    def grad(u: float) -> float:
        z = math.exp(u) * log_p
        z -= z.max(axis=1, keepdims=True)
        np.exp(z, out=z)
        return float(((log_p * z).sum(axis=1) / z.sum(axis=1)).mean(dtype=np.float64) - target)

    return grad


def _temperature_grad_torch(torch: ModuleType, log_p: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    """`_temperature_grad_numpy` on torch tensors."""
    log_p = torch.from_numpy(log_p)
    target = float(log_p[torch.arange(len(y)), torch.from_numpy(y)].double().mean())

    def grad(u: float) -> float:
        z = math.exp(u) * log_p
        z -= z.amax(dim=1, keepdim=True)
        z.exp_()
        return float(((log_p * z).sum(dim=1) / z.sum(dim=1)).double().mean()) - target

    return grad


def _sms_cross_entropy_numpy(log_q: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray, np.ndarray], tuple]:
    """Cross-entropy of ``softmax(log_q @ matrix.T + bias)`` and its gradients in ``matrix`` and ``bias``."""
    n = len(y)
    rows = np.arange(n)

    def cross_entropy(matrix: np.ndarray, bias: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        s = log_q @ matrix.T.astype(np.float32) + bias.astype(np.float32)
        s -= s.max(axis=1, keepdims=True)
        s_true = s[rows, y]
        np.exp(s, out=s)
        norm = s.sum(axis=1)
        loss = np.log(norm).mean(dtype=np.float64) - s_true.mean(dtype=np.float64)
        # `s` becomes the residual: softmax minus the one-hot labels
        s /= norm[:, None]
        s[rows, y] -= 1
        return float(loss), (s.T @ log_q).astype(np.float64) / n, s.sum(axis=0, dtype=np.float64) / n

    return cross_entropy


def _sms_cross_entropy_torch(
    torch: ModuleType, log_q: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray, np.ndarray], tuple]:
    """`_sms_cross_entropy_numpy` on torch tensors."""
    n = len(y)
    log_q = torch.from_numpy(log_q)
    labels = torch.from_numpy(y)
    rows = torch.arange(n)

    def cross_entropy(matrix: np.ndarray, bias: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        matrix_t = torch.from_numpy(np.ascontiguousarray(matrix.T, dtype=np.float32))
        s = torch.addmm(torch.from_numpy(bias.astype(np.float32)), log_q, matrix_t)
        s -= s.amax(dim=1, keepdim=True)
        s_true = s[rows, labels]
        s.exp_()
        norm = s.sum(dim=1)
        loss = float(norm.log().double().mean() - s_true.double().mean())
        # `s` becomes the residual: softmax minus the one-hot labels
        s /= norm[:, None]
        s[rows, labels] -= 1
        return loss, (s.T @ log_q).double().numpy() / n, s.sum(dim=0, dtype=torch.float64).numpy() / n

    return cross_entropy


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
    backend : str, default "auto"
        Where the multiclass objective is evaluated: ``"torch"``, ``"numpy"``, or ``"auto"`` for torch
        when it is installed, the input has at least 50,000 cells (rows times classes) and torch has
        more than one thread, else numpy. The backend used is recorded in ``backend_``.
    """

    def __init__(
        self,
        problem_type: str,
        reg_lambda: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        backend: str = "auto",
    ):
        if problem_type not in (BINARY, MULTICLASS):
            raise ValueError(f"LogisticCalibrator supports {BINARY} and {MULTICLASS}, got {problem_type!r}.")
        if backend not in ("auto", "torch", "numpy"):
            raise ValueError(f"backend must be one of ['auto', 'torch', 'numpy'], got {backend!r}.")
        self.problem_type = problem_type
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.tol = tol
        self.backend = backend
        self.backend_: str | None = None
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

    def _fit_temperature(self, grad: Callable[[float], float], n: int) -> None:
        """Inverse temperature ``exp(u)``, ``u`` in ``[-16, 16]``, at the root of the cross-entropy's derivative ``grad``.

        The derivative is nondecreasing in ``u``, so a root found by Brent's method is the minimum;
        without a sign change the minimum is at the end of the interval the derivative points to.
        """
        low, high = -16.0, 16.0
        if grad(low) > 0:
            u = low
        elif grad(high) <= 0:
            u = high
        else:
            u = brentq(grad, low, high, xtol=1e-9)
        self.inv_temperature_ = float(np.exp(u))
        self.uniform_weight_ = 1.0 / (n + 1)

    def _scale_temperature(self, p: np.ndarray) -> np.ndarray:
        q = softmax(self.inv_temperature_ * _log_proba(p), axis=1)
        return (1.0 - self.uniform_weight_) * q + self.uniform_weight_ / q.shape[1]

    def _resolve_backend(self, n_cells: int) -> ModuleType | None:
        """torch for the torch backend, None for numpy."""
        if self.backend == "numpy" or (self.backend == "auto" and n_cells < _TORCH_MIN_CELLS):
            return None
        torch = _import_torch()
        if self.backend == "torch":
            if torch is None:
                raise ImportError("LogisticCalibrator(backend='torch') requires torch.")
            return torch
        return torch if torch is not None and torch.get_num_threads() > 1 else None

    def _fit_multiclass(self, p: np.ndarray, y: np.ndarray) -> None:
        """Structured matrix scaling on the temperature-scaled log-probabilities."""
        n, k = p.shape
        torch = self._resolve_backend(n * k)
        self.backend_ = "numpy" if torch is None else "torch"
        with _single_threaded_blas():
            log_p = _log_proba(p).astype(np.float32)
            if torch is None:
                self._fit_temperature(_temperature_grad_numpy(log_p, y), n=n)
            else:
                self._fit_temperature(_temperature_grad_torch(torch, log_p, y), n=n)
            log_q = _log_proba(self._scale_temperature(p)).astype(np.float32)
            if torch is None:
                cross_entropy = _sms_cross_entropy_numpy(log_q, y)
            else:
                cross_entropy = _sms_cross_entropy_torch(torch, log_q, y)
            reg_bias = self.reg_lambda * k / n
            reg_matrix = np.full((k, k), self.reg_lambda * k * (k - 1) / n)
            np.fill_diagonal(reg_matrix, self.reg_lambda * k / n)
            identity = np.eye(k)

            def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
                weights, bias = theta[: k * k].reshape(k, k), theta[k * k :]
                loss, grad_matrix, grad_bias = cross_entropy(identity + weights, bias)
                loss += (reg_matrix * weights**2).sum() + reg_bias * (bias**2).sum()
                grad_weights = grad_matrix + 2 * reg_matrix * weights
                return float(loss), np.concatenate([grad_weights.ravel(), grad_bias + 2 * reg_bias * bias])

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
