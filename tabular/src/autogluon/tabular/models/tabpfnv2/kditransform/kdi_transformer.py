"""Vendored code for KDI due to outdated original package.

Code Adapted from: https://github.com/calvinmccarter/kditransform
Model: RealMLP
Paper: The Kernel Density Integral Transformation
Authors: Calvin McCarter
Codebase: https://github.com/calvinmccarter/kditransform
License: AGPL-3.0 license.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.interpolate as spip
import scipy.stats as spst
from scipy import integrate
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_is_fitted,
    check_random_state,
)

from .ksum import (
    betas_for_order,
    h_Gauss_to_K,
    ksum_numba,
)

BOUNDS_THRESHOLD = 1e-7


class KDITransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Transform features using kernel density quantiles information.

    This method transforms the features to follow a uniform distribution,
    or transforms them by scaling and translating them into a [0, 1] range,
    or does a (hyperparameter-tunable) interpolation of the two.

    Parameters
    ----------
    alpha: float > 0, 'scott', 'silverman', or None
        Bandwidth factor parameter for kernel density estimator.

    kernel: 'polyexp', 'gaussian' (default='polyexp')
        If 'gaussian', uses scipy's gaussian_kde. If 'polyexp', uses the
        polynomial-exponential kernel approximation from (Hofmeyr, 2019).

    polyexp_order: int, default=4
        Order of the kernel in the polynomial-exponential family.
        Ignored for 'gaussian' kernel.

    polyexp_eval: 'uniform', 'train', 'auto' (default='auto')
        Evaluation locations for numerical integration of polyexp KDE.
        If 'uniform', evaluates KDE at uniform grid-points.
        If 'train', evaluates KDE at train samples and their midpoints.
        If 'auto', combines both above approaches.
        Ignored for 'gaussian' kernel.

    n_quantiles : int or None, default=1000
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator. If None, is set to the number of samples.

    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    subsample : int or None, default=None
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.
        If None, no subsampling is performed.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling and smoothing
        noise.
        Please see ``subsample`` for more details.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    copy : bool, default=True
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array).

    exact: bool, default=False
        If True, does not compute and store integrals at landmarks, and instead
        performs integration during calls to transform.

    Attributes:
    ----------
    n_quantiles_ : int
        The actual number of quantiles used to discretize the cumulative
        distribution function.

    subsamples_: int
        The actual number of subsamples sampled from the data.

    quantiles_ : ndarray of shape (n_quantiles, n_features)
        Quantiles of kernel density estimator, with values corresponding
        to the quantiles of references_.

    references_ : ndarray of shape (n_quantiles, )
        Quantiles of references.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    """

    def __init__(
        self,
        alpha=1.0,
        kernel="polyexp",
        polyexp_order=4,
        polyexp_eval="auto",
        n_quantiles=1000,
        output_distribution="uniform",
        subsample=None,
        random_state=None,
        copy=True,
        exact=False,
    ):
        self.alpha = alpha
        self.kernel = kernel
        self.polyexp_order = polyexp_order
        self.polyexp_eval = polyexp_eval
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
        self.exact = exact

        if kernel not in ("gaussian", "polyexp"):
            raise ValueError(f"unexpected kernel: {kernel}")
        if kernel == "polyexp":
            if subsample is not None:
                warnings.warn(
                    "Subsampling is not needed and not recommended"
                    "for polyexp kernel since it is fast already.",
                    stacklevel=2,
                )
            if polyexp_eval not in ("uniform", "train", "auto"):
                raise ValueError(f"Unexpected polyexp_eval: {polyexp_eval}")
            if int(polyexp_order) != polyexp_order or polyexp_order < 1:
                raise ValueError(f"Invalid polyexp_order {polyexp_order}")
        if exact:
            if output_distribution == "normal":
                raise NotImplementedError("exact only transforms to uniform")
            if kernel == "polyexp":
                raise NotImplementedError("exact only uses gaussian kernel")

        self.n_quantiles_ = None
        self.subsample_ = None
        self.references_ = None
        self.quantiles_ = None
        self.n_features_in_ = None
        self.X_ = None

    def _gaussian_dense_fit(self, X, alphas, random_state):
        """Compute percentiles for dense matrices, using Gaussian kernel.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.

        alphas : list of len n_features
            Bandwidth per each feature.

        Reads
        -----
        subsample_, n_quantiles_, references_

        Modifies
        --------
        quantiles_
        """
        n_samples, n_features = X.shape

        self.quantiles_ = []
        for col, alpha in zip(X.T, alphas):
            if self.subsample_ < n_samples:
                subsample_idx = random_state.choice(
                    n_samples, size=self.subsample_, replace=False
                )
                col = col.take(subsample_idx, mode="clip")
            if np.var(col) == 0:
                # Causes gaussian_kde -> _compute_covariance -> linalg.cholesky error.
                # We instead duplicate QuantileTransformer's behavior here, which is
                # quantiles = np.nanpercentile(col, self.references_ * 100)
                # But krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
                # So instead we hard-code what nanpercentile does in this case:
                quantiles = col[0] * np.ones_like(self.references_)
            else:
                kder = spst.gaussian_kde(col, bw_method=alpha)
                xmin = np.min(col)
                xmax = np.max(col)
                # This reduces the total complexity from O(subsample_ ** 2) to
                # O(n_quantiles_ * subsample_ + subsample_ * log(subsample_)):
                col = np.quantile(col, self.references_)
                N = col.shape[0]
                T = np.zeros(N)
                # TODO: make faster.
                # Each loop, ndtr is called twice with subsample_ points.
                # We could easily eliminate the first call, for xmin.
                # We could also vectorize further, eliminating this loop.
                # Overall though, it is not bad, because ndtr is 100x faster.
                # than norm.cdf, and scipy's gaussian_kde calls ndtr directly.
                # www.cuemacro.com/2021/01/02/python-days-might-not-be-numba-ed
                for n in range(N):
                    T[n] = kder.integrate_box_1d(xmin, col[n])
                intcx1 = kder.integrate_box_1d(xmin, xmin)
                intcxN = kder.integrate_box_1d(xmin, xmax)
                m = 1.0 / (intcxN - intcx1)
                b = -m * intcx1  # intc0 / (intc0 - intcxN)
                # T is the result of nonlinear mapping of X onto [0,1]
                T = m * T + b
                inverse_func = spip.interp1d(
                    T, col, bounds_error=False, fill_value=(xmin, xmax)
                )
                quantiles = inverse_func(self.references_)
            self.quantiles_.append(quantiles)
        self.quantiles_ = np.transpose(self.quantiles_)

        # Make sure that quantiles are monotonically increasing
        self.quantiles_ = np.maximum.accumulate(self.quantiles_, axis=0)

    def _polyexp_dense_fit(self, X, alphas, random_state):
        """Compute percentiles for dense matrices, using polyexp kernel.

        See "Fast exact evaluation of univariate kernel sums" (Hofmeyr, 2019).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.

        alphas : list of len n_features
            Bandwidth per each feature.

        Reads
        -----
        polyexp_order, polyexp_eval, subsample_, n_quantiles_, references_

        Modifies
        --------
        quantiles_
        """
        n_samples, n_features = X.shape

        wgts = np.ones(n_samples).astype(X.dtype)
        betas = betas_for_order(self.polyexp_order)

        # Allocate memory for numba
        if self.polyexp_eval == "uniform":
            n_eval = n_samples if self.n_quantiles is None else self.n_quantiles
        elif self.polyexp_eval == "train":
            n_eval = n_samples + (n_samples - 1) * 1
        elif self.polyexp_eval == "auto":
            n_eval = (
                n_samples
                if self.n_quantiles is None
                else self.n_quantiles + self.n_quantiles_
            )

        density_out = np.zeros(n_eval).astype(X.dtype)
        counts = np.zeros(n_eval).astype(np.int64)
        coefs = np.zeros_like(betas)
        Ly = np.zeros((self.polyexp_order + 1, n_samples), order="C")
        Ry = np.zeros((self.polyexp_order + 1, n_samples), order="C")

        self.quantiles_ = []
        for col, alpha in zip(X.T, alphas):
            if self.subsample_ < n_samples:
                subsample_idx = random_state.choice(
                    n_samples, size=self.subsample_, replace=False
                )
                col = col.take(subsample_idx, mode="clip")
            if np.var(col) == 0:
                quantiles = col[0] * np.ones_like(self.references_)
            else:
                xmin = np.min(col)
                xmax = np.max(col)
                if alpha == "scott":
                    alpha = np.power(n_samples, (-1.0 / (1 + 4)))
                elif alpha == "silverman":
                    alpha = np.power(n_samples * (1 + 2.0) / 4.0, -1.0 / (1 + 4))
                # Bandwidth needs to be shrunk for polyexp kernel:
                h = h_Gauss_to_K(alpha * np.std(col), betas)
                col_mean = np.mean(col)
                col -= col_mean
                col_sort = np.sort(col)
                if self.polyexp_eval == "uniform":
                    col_eval = np.linspace(np.min(col), np.max(col), n_eval)
                elif self.polyexp_eval == "train":
                    midpts = col_sort[:-1] + 0.50 * np.diff(col_sort)
                    col_eval = np.sort(np.concatenate([col_sort, midpts]))
                elif self.polyexp_eval == "auto":
                    col_u = np.linspace(
                        np.min(col),
                        np.max(col),
                        n_samples if self.n_quantiles is None else self.n_quantiles,
                    )
                    col_s = np.quantile(col, self.references_)
                    col_eval = np.sort(np.concatenate([col_u, col_s]))
                    assert len(col_eval) == n_eval

                ksum_numba(
                    col_sort,
                    wgts,
                    col_eval,
                    h,
                    betas,
                    density_out,
                    counts,
                    coefs,
                    Ly,
                    Ry,
                )
                density_out /= n_samples * h
                density_out[np.isnan(density_out)] = 1e-300
                density_out[~np.isfinite(density_out)] = 1e-300
                col += col_mean
                col_sort += col_mean
                col_eval += col_mean
                T = integrate.cumulative_trapezoid(density_out, col_eval, initial=0)
                intcx1 = 0.0
                intcxN = T[-1]
                m = 1.0 / (intcxN - intcx1)
                b = -m * intcx1
                T = m * T + b

                inverse_func = spip.interp1d(
                    T, col_eval, bounds_error=False, fill_value=(xmin, xmax)
                )
                quantiles = inverse_func(self.references_)
            self.quantiles_.append(quantiles)
        self.quantiles_ = np.transpose(self.quantiles_)
        # Make sure that quantiles are monotonically increasing
        self.quantiles_ = np.maximum.accumulate(self.quantiles_, axis=0)

    def fit(self, X, y=None):
        """Compute the kernel-smoothed quantiles used for transforming.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to scale along the features axis.

        y : None
            Ignored.

        Returns:
        -------
        self : object
           Fitted transformer.
        """
        X = self._check_inputs(X, in_fit=True, copy=False)
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)

        if self.exact:
            self.X_ = X.copy()
            return self

        if isinstance(self.alpha, list):
            # We quietly support different alphas for different features.
            assert len(self.alpha) == n_features
            alphas = self.alpha
        else:
            alphas = [self.alpha] * n_features

        if self.n_quantiles is None:
            self.n_quantiles_ = n_samples
        else:
            self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        if self.subsample is None:
            self.subsample_ = n_samples
        else:
            self.subsample_ = min(self.subsample, n_samples)

        if self.n_quantiles_ > self.subsample_:
            raise ValueError(
                "The number of quantiles cannot be greater than"
                f" the number of samples used. Got {self.n_quantiles_} quantiles"
                f" and {self.subsample_} samples."
            )

        # Create the quantiles of reference, with shape (n_quantiles_,)
        self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
        if self.kernel == "gaussian":
            self._gaussian_dense_fit(X, alphas, rng)
        elif self.kernel == "polyexp":
            self._polyexp_dense_fit(X, alphas, rng)

        self.n_features_in_ = n_features

        return self

    def _transform_col(self, X_col, quantiles, inverse):
        """Private function to transform a single feature."""
        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = spst.norm.cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if output_distribution == "normal":
                lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
                upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
            if output_distribution == "uniform":
                lower_bounds_idx = X_col == lower_bound_x
                upper_bounds_idx = X_col == upper_bound_x

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col[isfinite_mask] = 0.5 * (
                np.interp(X_col_finite, quantiles, self.references_)
                - np.interp(-X_col_finite, -quantiles[::-1], -self.references_[::-1])
            )
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite, self.references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = spst.norm.ppf(X_col)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    clip_min = spst.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = spst.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
                    X_col = np.clip(X_col, clip_min, clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged

        return X_col

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform."""
        return self._validate_data(
            X,
            reset=in_fit,
            accept_sparse=False,
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan", # changed for new sklearn version.
        )

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.

        inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns:
        -------
        X : ndarray of shape (n_samples, n_features)
            Projected data.
        """
        for feature_idx in range(X.shape[1]):
            X[:, feature_idx] = self._transform_col(
                X[:, feature_idx], self.quantiles_[:, feature_idx], inverse
            )

        return X

    def _transform_exact(self, X):
        n_test, n_features = X.shape
        self.X_.shape[0]
        if isinstance(self.alpha, list):
            # We quietly support different alphas for different features.
            assert len(self.alpha) == n_features
            alphas = self.alpha
        else:
            alphas = [self.alpha] * n_features

        for feature_idx in range(n_features):
            train_col = self.X_[:, feature_idx]
            test_col = X[:, feature_idx]
            if np.var(train_col) == 0:
                X[:, feature_idx] = test_col > train_col
            elif self.kernel == "gaussian":
                kder = spst.gaussian_kde(train_col, bw_method=alphas[feature_idx])
                xmin = np.min(train_col)
                xmax = np.max(train_col)
                intcx1 = kder.integrate_box_1d(xmin, xmin)
                intcxN = kder.integrate_box_1d(xmin, xmax)
                m = 1.0 / (intcxN - intcx1)
                b = -m * intcx1  # intc0 / (intc0 - intcxN)
                for n in range(n_test):
                    X[n, feature_idx] = m * kder.integrate_box_1d(xmin, test_col[n]) + b
        return X

    def transform(self, X):
        """Feature-wise transformation of the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis.

        Returns:
        -------
        Xt : ndarray of shape (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self)
        X = self._check_inputs(X, in_fit=False, copy=self.copy)

        if self.exact:
            return self._transform_exact(X)

        return self._transform(X, inverse=False)

    def inverse_transform(self, X):
        """Back-projection to the original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to scale along the features axis.

        Returns:
        -------
        Xt : ndarray of (n_samples, n_features)
            The projected data.
        """
        if self.exact:
            raise NotImplementedError("inverse not supported for exact")
        check_is_fitted(self)
        X = self._check_inputs(X, in_fit=False, copy=self.copy)

        return self._transform(X, inverse=True)

    def _more_tags(self):
        return {"allow_nan": False}
