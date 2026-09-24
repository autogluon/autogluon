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
    global _BLAS_CONTROLLER
    if _BLAS_CONTROLLER is None:
        try:
            from threadpoolctl import ThreadpoolController  # installed with scikit-learn
        except ImportError:
            return nullcontext()
        # Finding the loaded libraries scans every shared object in the process (about 1 ms), so
        # do it once; numpy's and scipy's BLAS are loaded by this module's imports.
        _BLAS_CONTROLLER = ThreadpoolController()
    return _BLAS_CONTROLLER.limit(limits=1, user_api="blas")


_BLAS_CONTROLLER = None


def _import_torch() -> ModuleType | None:
    try:
        import torch
    except (ImportError, OSError):
        return None
    return torch


# Newton's method needs the Hessian, whose cost grows with the number of parameters squared, so the
# multiclass fit uses it up to this many classes (K * (K + 1) = 72 parameters) and L-BFGS above.
# Its float32 gradient also gets too noisy for Newton's stopping test on very large inputs, so it
# stops at this many cells too.
_NEWTON_MAX_CLASSES = 8
_NEWTON_MAX_CELLS = 2_000_000
_NEWTON_MAX_EVALS = 50


def _damped_newton(
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]], x0: np.ndarray, tol: float
) -> tuple[np.ndarray, bool]:
    """Minimize a smooth convex function by Newton steps with Armijo backtracking.

    ``evaluate(x)`` returns the value, gradient and Hessian at ``x``; each accepted step costs one
    evaluation. Converged when the decrease the quadratic model predicts, ``-g @ step / 2``, is at
    most ``tol * max(|f|, 1)``; once it is below ``1e-6 * max(|f|, 1)`` the model is accurate, so
    the full step is taken and the method stops (a line search there would test float32 rounding
    of the value). Returns the minimizer and whether it converged within ``_NEWTON_MAX_EVALS``.
    """
    x = x0
    f, g, h = evaluate(x)
    evals = 1
    while evals < _NEWTON_MAX_EVALS:
        try:
            step = np.linalg.solve(h, -g)
        except np.linalg.LinAlgError:
            step = -g
        slope = float(g @ step)
        if not slope < 0:  # not a descent direction: fall back to the gradient
            step, slope = -g, -float(g @ g)
        scale = max(abs(f), 1.0)
        if -0.5 * slope <= tol * scale:
            return x, True
        if -0.5 * slope <= 1e-6 * scale:
            return x + step, True
        t = 1.0
        while True:
            f_new, g_new, h_new = evaluate(x + t * step)
            evals += 1
            if f_new <= f + 1e-4 * t * slope:
                break
            if evals >= _NEWTON_MAX_EVALS:
                return x, False
            t *= 0.5
        x = x + t * step
        f, g, h = f_new, g_new, h_new
    return x, False


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


def _sms_cross_entropy_numpy(log_q: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray, bool], tuple]:
    """Cross-entropy of ``softmax(weights @ [log_q; 1])``, its gradient in ``weights`` and, on request, its Hessian.

    ``weights`` has shape ``(n_classes, n_classes + 1)``: the matrix, then the bias as the last column,
    so one product computes the logits and one the gradient. The Hessian is over the row-major
    flattened ``weights``.
    """
    k1, n = log_q.shape[0] + 1, log_q.shape[1]
    features = np.vstack([log_q, np.ones((1, n), dtype=np.float32)])
    features_rows = np.ascontiguousarray(features.T)
    label_cells = y * n + np.arange(n)  # flat indices of the labels' cells

    def cross_entropy(weights: np.ndarray, hessian: bool) -> tuple[float, np.ndarray, np.ndarray | None]:
        s = weights.astype(np.float32) @ features
        s -= s.max(axis=0)
        s_flat = s.reshape(-1)
        s_true = s_flat.take(label_cells).mean(dtype=np.float64)
        np.exp(s, out=s)
        norm = s.sum(axis=0)
        loss = np.log(norm).mean(dtype=np.float64) - s_true
        s /= norm  # the softmax
        hess = None
        if hessian:
            # sum over rows of (diag(p) - p p^T) kron (x x^T), with x = [log_q; 1] of the row
            z = (s[:, None, :] * features[None, :, :]).reshape(-1, n)
            hess = -(z @ z.T).astype(np.float64)
            # row (a, j) of z times x^T gives row j of class a's diagonal block
            blocks = (z @ features_rows).astype(np.float64).reshape(-1, k1, k1)
            for a, block in enumerate(blocks):
                hess[a * k1 : (a + 1) * k1, a * k1 : (a + 1) * k1] += block
            hess /= n
        # `s` becomes the residual: softmax minus the one-hot labels
        s_flat[label_cells] -= 1
        return float(loss), (s @ features_rows).astype(np.float64) / n, hess

    return cross_entropy


def _sms_cross_entropy_torch(
    torch: ModuleType, log_q: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray, bool], tuple]:
    """`_sms_cross_entropy_numpy` on torch tensors, without the Hessian."""
    n = log_q.shape[1]
    features = torch.from_numpy(np.vstack([log_q, np.ones((1, n), dtype=np.float32)]))
    features_rows = features.T.contiguous()
    label_cells = torch.from_numpy(y * n + np.arange(n))

    def cross_entropy(weights: np.ndarray, hessian: bool) -> tuple[float, np.ndarray, None]:
        # torch's multi-threaded float32 products are too noisy for Newton's method; L-BFGS only
        assert not hessian, "the torch backend has no Hessian"
        s = torch.from_numpy(weights.astype(np.float32)) @ features
        s -= s.amax(dim=0)
        s_flat = s.view(-1)
        s_true = float(s_flat[label_cells].double().mean())
        s.exp_()
        norm = s.sum(dim=0)
        loss = float(norm.log().double().mean()) - s_true
        # `s` becomes the residual: softmax minus the one-hot labels
        s /= norm
        s_flat[label_cells] -= 1
        return loss, (s @ features_rows).double().numpy() / n, None

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
        Iteration limit of the L-BFGS solver. The binary fit and, on numpy, multiclass fits of up to
        8 classes and 2,000,000 cells use Newton's method instead, with L-BFGS as its fallback.
    tol : float, default 1e-8
        Relative objective decrease at which the multiclass fit stops (Newton's method stops at a
        predicted decrease of ``tol / 100``).
    backend : str, default "auto"
        Where the multiclass objective is evaluated: ``"torch"``, ``"numpy"``, or ``"auto"`` for torch
        when it is installed, the input has at least 200,000 cells (rows times classes) and is too
        large for Newton's method, and torch has more than one thread; else numpy. The backend used
        is recorded in ``backend_``.
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
        coef = self.coef_.reshape(k, k + 1)  # the matrix minus the identity, then the bias column
        return softmax(log_q @ (np.eye(k) + coef[:, :k]).T + coef[:, k], axis=1)

    @staticmethod
    def _binary_logit(p: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            log_p = np.clip(np.log(p), _LOG_FLOOR, -_LOG_FLOOR)
            log_1mp = np.clip(np.log1p(-p), _LOG_FLOOR, -_LOG_FLOOR)
        return log_p - log_1mp

    def _fit_binary(self, p: np.ndarray, y: np.ndarray) -> None:
        """Platt scaling: ``sigmoid(a + b * logit(p))`` by unpenalized maximum likelihood."""
        z = self._binary_logit(p)
        n = len(z)
        y = y.astype(np.float64)
        # With h = s / 2, softplus(s) = h + |h| + log1p(exp(-2|h|)) and sigmoid(s) = (1 + tanh(h)) / 2.
        # The terms linear in the parameters (h, and the labels' y * s) average in closed form from
        # sums taken once, so an evaluation is a handful of vectorized passes (np.logaddexp and
        # scipy's expit are not vectorized, and many times slower).
        y_mean, yz_mean, z_mean = y.mean(), (y @ z) / n, z.mean()

        z_sq = z * z

        def evaluate(theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
            a, b = theta
            h = z * (0.5 * b)
            h += 0.5 * a
            t = np.tanh(h)
            np.abs(h, out=h)
            abs_mean = h.mean()
            h *= -2.0
            np.exp(h, out=h)
            np.log1p(h, out=h)
            loss = 0.5 * (a + b * z_mean) + abs_mean + h.mean() - (a * y_mean + b * yz_mean)
            grad = np.array([0.5 + 0.5 * t.mean() - y_mean, 0.5 * z_mean + 0.5 * (t @ z) / n - yz_mean])
            # sigmoid(s) * (1 - sigmoid(s)) = (1 - tanh(h)^2) / 4
            t *= t
            w_sum, wz_sum, wzz_sum = n - t.sum(), z_mean * n - t @ z, z_sq.sum() - t @ z_sq
            hess = np.array([[w_sum, wz_sum], [wz_sum, wzz_sum]]) / (4 * n)
            return float(loss), grad, hess

        self.coef_, _ = _damped_newton(evaluate, np.array([0.0, 1.0]), tol=1e-12)

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

    def _resolve_backend(self, n_cells: int, newton: bool) -> ModuleType | None:
        """torch for the torch backend, None for numpy; "auto" keeps Newton-sized fits on numpy."""
        if self.backend == "numpy" or (self.backend == "auto" and (newton or n_cells < _TORCH_MIN_CELLS)):
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
        newton = k <= _NEWTON_MAX_CLASSES and n * k < _NEWTON_MAX_CELLS
        torch = self._resolve_backend(n * k, newton=newton)
        newton = newton and torch is None
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
            # The parameters are (K, K + 1), row-major: the matrix minus the identity, then the bias.
            reg = np.full((k, k + 1), self.reg_lambda * k * (k - 1) / n)
            reg[:, k] = self.reg_lambda * k / n
            np.fill_diagonal(reg, self.reg_lambda * k / n)
            identity = np.eye(k, k + 1)
            reg_hess_diagonal = 2 * reg.ravel()

            def evaluate(theta: np.ndarray, hessian: bool = True) -> tuple[float, np.ndarray, np.ndarray | None]:
                delta = theta.reshape(k, k + 1)
                loss, grad, hess = cross_entropy(identity + delta, hessian)
                if hess is not None:
                    hess.flat[:: hess.shape[0] + 1] += reg_hess_diagonal
                return loss + float((reg * delta * delta).sum()), (grad + 2 * reg * delta).ravel(), hess

            theta = np.zeros(k * (k + 1))
            converged = False
            if newton:
                theta, converged = _damped_newton(evaluate, theta, tol=self.tol * 1e-2)
            if not converged:  # more classes or cells, torch, or Newton ran out of evaluations
                res = minimize(
                    lambda theta: evaluate(theta, hessian=False)[:2],
                    theta,
                    jac=True,
                    method="L-BFGS-B",
                    options={"maxiter": self.max_iter, "maxcor": 30, "ftol": self.tol},
                )
                theta = res.x
            self.coef_ = theta


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
