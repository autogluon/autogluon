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
from scipy.optimize import minimize
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
_TORCH_MIN_CELLS = 200_000


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


# The kernels below take class-major arrays, shape ``(n_classes, n_rows)``, so every reduction over
# the classes is an elementwise operation across contiguous rows (much faster for few classes).


def _temperature_derivatives_numpy(log_p: np.ndarray, y: np.ndarray) -> Callable[[float], tuple[float, float]]:
    """First and second derivative in ``t`` of the cross-entropy of ``softmax(t * log_p)``."""
    n = log_p.shape[1]
    log_p_sq = log_p * log_p
    target = log_p[y, np.arange(n)].mean(dtype=np.float64)

    def derivatives(t: float) -> tuple[float, float]:
        z = t * log_p
        z -= z.max(axis=0)
        np.exp(z, out=z)
        z /= z.sum(axis=0)
        mean = (log_p * z).sum(axis=0)
        second = (log_p_sq * z).sum(axis=0)
        return float(mean.mean(dtype=np.float64) - target), float((second - mean * mean).mean(dtype=np.float64))

    return derivatives


def _temperature_derivatives_torch(
    torch: ModuleType, log_p: np.ndarray, y: np.ndarray
) -> Callable[[float], tuple[float, float]]:
    """`_temperature_derivatives_numpy` on torch tensors."""
    n = log_p.shape[1]
    log_p = torch.from_numpy(log_p)
    log_p_sq = log_p * log_p
    target = float(log_p[torch.from_numpy(y), torch.arange(n)].double().mean())

    def derivatives(t: float) -> tuple[float, float]:
        z = t * log_p
        z -= z.amax(dim=0)
        z.exp_()
        z /= z.sum(dim=0)
        mean = (log_p * z).sum(dim=0)
        second = (log_p_sq * z).sum(dim=0)
        return float(mean.double().mean()) - target, float((second - mean * mean).double().mean())

    return derivatives


def _sms_cross_entropy_numpy(log_q: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray, np.ndarray], tuple]:
    """Cross-entropy of ``softmax(matrix @ log_q + bias)`` and its gradients in ``matrix`` and ``bias``."""
    n = log_q.shape[1]
    cols = np.arange(n)
    log_q_rows = np.ascontiguousarray(log_q.T)

    def cross_entropy(matrix: np.ndarray, bias: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        s = matrix.astype(np.float32) @ log_q
        s += bias.astype(np.float32)[:, None]
        s -= s.max(axis=0)
        s_true = s[y, cols]
        np.exp(s, out=s)
        norm = s.sum(axis=0)
        loss = np.log(norm).mean(dtype=np.float64) - s_true.mean(dtype=np.float64)
        # `s` becomes the residual: softmax minus the one-hot labels
        s /= norm
        s[y, cols] -= 1
        return float(loss), (s @ log_q_rows).astype(np.float64) / n, s.sum(axis=1, dtype=np.float64) / n

    return cross_entropy


def _sms_cross_entropy_torch(
    torch: ModuleType, log_q: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray, np.ndarray], tuple]:
    """`_sms_cross_entropy_numpy` on torch tensors."""
    n = log_q.shape[1]
    log_q = torch.from_numpy(log_q)
    log_q_rows = log_q.T
    labels = torch.from_numpy(y)
    cols = torch.arange(n)

    def cross_entropy(matrix: np.ndarray, bias: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        bias = torch.from_numpy(bias.astype(np.float32))[:, None]
        s = torch.addmm(bias, torch.from_numpy(matrix.astype(np.float32)), log_q)
        s -= s.amax(dim=0)
        s_true = s[labels, cols]
        s.exp_()
        norm = s.sum(dim=0)
        loss = float(norm.log().double().mean() - s_true.double().mean())
        # `s` becomes the residual: softmax minus the one-hot labels
        s /= norm
        s[labels, cols] -= 1
        return loss, (s @ log_q_rows).double().numpy() / n, s.sum(dim=1, dtype=torch.float64).numpy() / n

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
        when it is installed, the input has at least 200,000 cells (rows times classes) and torch has
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
            # softplus(s) and sigmoid(s) share one exp(-|s|)
            e = np.exp(-np.abs(s))
            inv = 1.0 / (1.0 + e)
            residual = np.where(s >= 0, inv, e * inv) - y
            loss = np.mean(np.maximum(s, 0.0) + np.log1p(e) - y * s)
            return float(loss), np.array([residual.mean(), (residual * z).mean()])

        res = minimize(
            loss_and_grad, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter}
        )
        self.coef_ = res.x

    def _fit_temperature(self, derivatives: Callable[[float], tuple[float, float]], n: int) -> None:
        """Inverse temperature ``t = exp(u)``, ``u`` in ``[-16, 16]``, minimizing the cross-entropy.

        ``derivatives(t)`` returns the cross-entropy's first and second derivative in ``t``. The
        first is nondecreasing, so Newton steps in ``u`` are safeguarded by bisection of the bracket
        its sign maintains; without a root in the interval the result is the end it points to.
        """
        low, high = -16.0, 16.0
        u = 0.0
        for _ in range(100):
            t = math.exp(u)
            grad, hess = derivatives(t)
            if grad > 0:
                high = u
            else:
                low = u
            u_next = u - grad / (t * hess) if hess > 0 else math.nan
            if not low <= u_next <= high:
                u_next = 0.5 * (low + high)
            # float32 derivatives resolve ``u`` to about 1e-7; smaller steps only chase their noise
            converged = abs(u_next - u) < 1e-7
            u = u_next
            if converged:
                break
        self.inv_temperature_ = math.exp(u)
        self.uniform_weight_ = 1.0 / (n + 1)

    def _scale_temperature(self, p: np.ndarray) -> np.ndarray:
        q = softmax(self.inv_temperature_ * _log_proba(p), axis=1)
        return (1.0 - self.uniform_weight_) * q + self.uniform_weight_ / q.shape[1]

    def _scaled_log_proba(self, log_p: np.ndarray) -> np.ndarray:
        """`_scale_temperature` in log space for class-major float32 log-probabilities, as the fit uses them."""
        z = np.float32(self.inv_temperature_) * log_p
        z -= z.max(axis=0)
        np.exp(z, out=z)
        z *= np.float32(1.0 - self.uniform_weight_) / z.sum(axis=0)
        z += np.float32(self.uniform_weight_ / len(z))
        return _log_proba(z)

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
            log_p = np.ascontiguousarray(_log_proba(p).T, dtype=np.float32)
            if torch is None:
                self._fit_temperature(_temperature_derivatives_numpy(log_p, y), n=n)
            else:
                self._fit_temperature(_temperature_derivatives_torch(torch, log_p, y), n=n)
            log_q = self._scaled_log_proba(log_p)
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
