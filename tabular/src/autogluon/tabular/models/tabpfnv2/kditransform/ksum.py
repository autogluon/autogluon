"""Vendored code for KDI due to outdated original package.

Code Adapted from: https://github.com/calvinmccarter/kditransform
Model: RealMLP
Paper: The Kernel Density Integral Transformation
Authors: Calvin McCarter
Codebase: https://github.com/calvinmccarter/kditransform
License: AGPL-3.0 license.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.special import factorial


def norm_const_K(betas):
    factorial_terms = np.array(
        [betas[k - 1] * factorial(k - 1) for k in range(1, len(betas) + 1)]
    )
    return 2 * np.sum(factorial_terms)


def betas_for_order(order):
    unnorm_betas = 1 / factorial(np.arange(order + 1))
    return unnorm_betas / norm_const_K(unnorm_betas)


def roughness_K(betas):
    beta_use = betas / norm_const_K(betas)
    betakj = np.outer(beta_use, beta_use)
    n = len(betas)
    kpj = np.log2(np.outer(2 ** np.arange(n), 2 ** np.arange(n))).astype(np.int64)
    return np.sum(betakj / (2**kpj) * factorial(kpj))


def var_K(betas):
    beta_use = betas / norm_const_K(betas)
    factorial_terms = np.array(
        [beta_use[k - 1] * factorial(k + 1) for k in range(1, len(betas) + 1)]
    )
    return 2 * np.sum(factorial_terms)


def h_Gauss_to_K(h, betas):
    """Converts bandwidth of Gaussian kernel to that of poly-exp kernel."""
    return h * (roughness_K(betas) / (var_K(betas) ** 2) * 2 * np.sqrt(np.pi)) ** 0.2


# @njit(error_model="numpy")
@njit
def ksum_numba(x, y, x_eval, h, betas, output, counts, coefs, Ly, Ry):
    """Implements kernel density estimation with poly-exponential kernel.

    See "Fast exact evaluation of univariate kernel sums" (Hofmeyr, 2019)
    and https://github.com/DavidHofmeyr/FKSUM.
    """
    n = x.shape[0]
    n_eval = x_eval.shape[0]
    order = betas.shape[0] - 1
    output[:] = 0.0
    counts[:] = 0
    coefs[:] = 0.0
    Ly[:, :] = 0.0
    Ry[:, :] = 0.0

    for i in range(order + 1):
        Ly[i, 0] = np.power(-x[0], i) * y[0]
    for i in range(1, n):
        for j in range(order + 1):
            Ly[j, i] = (
                np.power(-x[i], j) * y[i] + np.exp((x[i - 1] - x[i]) / h) * Ly[j, i - 1]
            )
            Ry[j, n - i - 1] = np.exp((x[n - i - 1] - x[n - i]) / h) * (
                np.power(x[n - i], j) * y[n - i] + Ry[j, n - i]
            )

    count = 0
    for i in range(n_eval):
        if x_eval[i] >= x[n - 1]:
            counts[i] = n
        else:
            while count < n and x[count] <= x_eval[i]:
                count += 1
            counts[i] = count

    for orddo in range(order + 1):
        coefs[0] = 1
        coefs[orddo] = 1
        if orddo > 1:
            num = 1.0
            for j in range(2, orddo + 1):
                num *= j
            denom1 = 1.0
            denom2 = num / orddo
            for i in range(2, orddo + 1):
                coefs[i - 1] = num / (denom1 * denom2)
                denom1 *= i
                denom2 /= orddo - i + 1
        denom = np.power(h, orddo)

        ix = 0
        for i in range(n_eval):
            ix = np.round(counts[i])
            if ix == 0:
                exp_mult = np.exp((x_eval[i] - x[0]) / h)
                output[i] += (
                    betas[orddo]
                    * np.power(x[0] - x_eval[i], orddo)
                    / denom
                    * exp_mult
                    * y[0]
                )
                for j in range(orddo + 1):
                    output[i] += (
                        betas[orddo]
                        * coefs[j]
                        * np.power(-x_eval[i], orddo - j)
                        * Ry[j, 0]
                        / denom
                        * exp_mult
                    )
            else:
                exp_mult = np.exp((x[ix - 1] - x_eval[i]) / h)
                for j in range(orddo + 1):
                    output[i] += (
                        betas[orddo]
                        * coefs[j]
                        * (
                            np.power(x_eval[i], orddo - j) * Ly[j, ix - 1] * exp_mult
                            + np.power(-x_eval[i], orddo - j)
                            * Ry[j, ix - 1]
                            / max(exp_mult, 1e-300)
                        )
                        / denom
                    )


def ksum(x, y, x_eval, h=None, betas=None):
    # Assumes x and x_eval are sorted
    n = x.shape[0]
    x_eval.shape[0]

    if betas is None:
        # Smooth first-order kernel
        betas = np.array([0.25, 0.25])
    else:
        betas = betas / norm_const_K(betas)
    print(betas)
    order = betas.shape[0] - 1

    if h is None:
        # Silverman's rule
        h = (
            8 * np.sqrt(np.pi) / 3 * roughness_K(betas) / (var_K(betas) ** 2) / n
        ) ** 0.2 * np.std(x)

    output = np.zeros_like(x_eval)
    counts = np.zeros_like(x_eval).astype(np.int64)
    coefs = np.zeros_like(betas)

    Ly = np.zeros((order + 1, n), order="C")
    Ry = np.zeros((order + 1, n), order="C")

    ksum_numba(x, y, x_eval, h, betas, output, counts, coefs, Ly, Ry)
    return output / n / h


def ksum_density(x, alpha=1.0, order=4):
    col = x
    X = x
    (n_samples,) = x.shape

    wgts = np.ones(n_samples).astype(X.dtype)
    betas = betas_for_order(order)
    n_eval = n_samples + (n_samples - 1) * 3

    np.min(col)
    np.max(col)
    h = alpha * np.std(col)
    col_mean = np.mean(col)
    col -= col_mean
    col_sort = np.sort(col)
    earlypts = col_sort[:-1] + 0.25 * np.diff(col_sort)
    midpts = col_sort[:-1] + 0.50 * np.diff(col_sort)
    latepts = col_sort[:-1] + 0.75 * np.diff(col_sort)
    col_eval = np.sort(np.concatenate([col_sort, earlypts, midpts, latepts]))
    col_eval = np.linspace(np.min(col), np.max(col), 1000)
    n_eval = col_eval.shape[0]

    # Allocate memory for numba
    density_out = np.zeros(n_eval).astype(X.dtype)
    counts = np.zeros(n_eval).astype(np.int64)
    coefs = np.zeros_like(betas)
    Ly = np.zeros((order + 1, n_samples), order="C")
    Ry = np.zeros((order + 1, n_samples), order="C")

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
    density_out[np.isnan(density_out)] = 1e-300
    density_out[~np.isfinite(density_out)] = 1e-300
    col += col_mean
    col_sort += col_mean
    col_eval += col_mean

    return col_eval, density_out


"""
if __name__ == "__main__":
    x = np.sort(np.array([0., 0.1, 0.2, 1., 5.]).astype(np.float64))
    y = np.ones_like(x)
    midpts = x[:-1] + 0.5*np.diff(x)
    x_eval = np.sort(np.concatenate([x, midpts]))
    h = None # 0.1 # np.std(x)
    betas = np.array([1., 1., 1 / 2., 1 / 6., 1 / 24.])
    #betas = None
    #betas = np.array([1., 1.])
    output = ksum(x, y, x_eval, h, betas)
    print(output)
"""
