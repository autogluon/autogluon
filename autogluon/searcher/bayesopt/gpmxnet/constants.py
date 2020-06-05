# This file contains various constants required for the definition of the model
# or to set up the optimization

from typing import NamedTuple
import numpy as np


DEFAULT_ENCODING = "logarithm"  # the other choices is positive

NUMERICAL_JITTER = 1e-9

INITIAL_NOISE_VARIANCE = 1e-3
INITIAL_MEAN_VALUE = 0.0
INITIAL_COVARIANCE_SCALE = 1.0
INITIAL_INVERSE_BANDWIDTHS = 1.0
INITIAL_WARPING = 1.0

INVERSE_BANDWIDTHS_LOWER_BOUND = 1e-4
INVERSE_BANDWIDTHS_UPPER_BOUND = 100

COVARIANCE_SCALE_LOWER_BOUND = 1e-3
COVARIANCE_SCALE_UPPER_BOUND = 1e3

NOISE_VARIANCE_LOWER_BOUND = 1e-9
NOISE_VARIANCE_UPPER_BOUND = 1e6

WARPING_LOWER_BOUND = 0.25
WARPING_UPPER_BOUND = 4.

MIN_POSTERIOR_VARIANCE = 1e-12

MIN_CHOLESKY_DIAGONAL_VALUE = 1e-10

DATA_TYPE = np.float64


class OptimizationConfig(NamedTuple):
    lbfgs_tol: float
    lbfgs_maxiter : int
    verbose : bool
    n_starts : int


class MCMCConfig(NamedTuple):
    n_samples : int
    n_burnin : int
    n_thinning : int


DEFAULT_OPTIMIZATION_CONFIG = OptimizationConfig(
    lbfgs_tol=1e-6,
    lbfgs_maxiter=500,
    verbose=False,
    n_starts=5)

DEFAULT_MCMC_CONFIG = MCMCConfig(
    n_samples=300,
    n_burnin=250,
    n_thinning=5)
