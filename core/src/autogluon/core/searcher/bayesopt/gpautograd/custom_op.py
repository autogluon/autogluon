import autograd.numpy as anp
import autograd.scipy.linalg as aspl
from autograd.extend import primitive, defvjp
import numpy as np
import scipy.linalg as spl
import logging
import math

logger = logging.getLogger(__name__)

__all__ = ['AddJitterOp',
           'flatten_and_concat',
           'cholesky_factorization']


INITIAL_JITTER_FACTOR = 1e-9
JITTER_GROWTH = 10.
JITTER_UPPERBOUND_FACTOR = 1e3


def flatten_and_concat(x: anp.ndarray, sigsq_init: anp.ndarray):
    return anp.append(anp.reshape(x, (-1,)), sigsq_init)


@primitive
def AddJitterOp(inputs: np.ndarray, initial_jitter_factor=INITIAL_JITTER_FACTOR,
                jitter_growth=JITTER_GROWTH, debug_log='false'):      
    """
    Finds smaller jitter to add to diagonal of square matrix to render the
    matrix positive definite (in that linalg.potrf works).

    Given input x (positive semi-definite matrix) and sigsq_init (nonneg
    scalar), find sigsq_final (nonneg scalar), so that:
        sigsq_final = sigsq_init + jitter, jitter >= 0,
        x + sigsq_final * Id positive definite (so that potrf call works)
    We return the matrix x + sigsq_final * Id, for which potrf has not failed.

    For the gradient, the dependence of jitter on the inputs is ignored.

    The values tried for sigsq_final are:
        sigsq_init, sigsq_init + initial_jitter * (jitter_growth ** k),
        k = 0, 1, 2, ...,
        initial_jitter = initial_jitter_factor * max(mean(diag(x)), 1)

    Note: The scaling of initial_jitter with mean(diag(x)) is taken from GPy.
    The rationale is that the largest eigenvalue of x is >= mean(diag(x)), and
    likely of this magnitude.

    There is no guarantee that the Cholesky factor returned is well-conditioned
    enough for subsequent computations to be reliable. A better solution
    would be to estimate the condition number of the Cholesky factor, and to add
    jitter until this is bounded below a threshold we tolerate. See

        Higham, N.
        A Survey of Condition Number Estimation for Triangular Matrices
        MIMS EPrint: 2007.10

    Algorithm 4.1 could work for us.
    """
    assert initial_jitter_factor > 0. and jitter_growth > 1.
    n_square = inputs.shape[0] - 1
    n = int(math.sqrt(n_square))
    assert n_square % n == 0 and n_square // n == n, "x must be square matrix, shape (n, n)"
    x, sigsq_init = np.reshape(inputs[:-1], (n, -1)), inputs[-1]
    
    def _get_constant_identity(x, constant):
        n, _ = x.shape
        return np.diag(np.ones((n,)) * constant)

    def _get_jitter_upperbound(x):
        # To define a safeguard in the while-loop of the forward,
        # we define an upperbound on the jitter we can reasonably add
        # the bound is quite generous, and is dependent on the scale of the input x
        # (the scale is captured via the trace of x)
        # the primary goal is avoid any infinite while-loop.
        return JITTER_UPPERBOUND_FACTOR * max(1., np.mean(np.diag(x)))

    jitter = 0.
    jitter_upperbound = _get_jitter_upperbound(x)
    must_increase_jitter = True
    x_plus_constant = None
    
    while must_increase_jitter and jitter <= jitter_upperbound:
        try:
            x_plus_constant = x + _get_constant_identity(
                x, sigsq_init + jitter)
            # Note: Do not use np.linalg.cholesky here, this can cause
            # locking issues
            L = spl.cholesky(x_plus_constant, lower=True)
            must_increase_jitter = False
        except spl.LinAlgError:
            if debug_log == 'true':
                logger.info("sigsq = {} does not work".format(
                    sigsq_init + jitter))
            if jitter == 0.0:
                jitter = initial_jitter_factor * max(1., np.mean(np.diag(x)))
            else:
                jitter = jitter * jitter_growth

    assert not must_increase_jitter, "The jitter ({}) has reached its upperbound ({}) while the Cholesky of the input matrix still cannot be computed.".format(jitter, jitter_upperbound)
    
    if debug_log == 'true':
        logger.info("sigsq_final = {}".format(sigsq_init + jitter))

    return x_plus_constant


def AddJitterOp_vjp(
        ans: np.ndarray, inputs: np.ndarray,
        initial_jitter_factor=INITIAL_JITTER_FACTOR, jitter_growth=JITTER_GROWTH,
        debug_log='false'):
    return lambda g: anp.append(anp.reshape(g, (-1,)), anp.sum(anp.diag(g)))


defvjp(AddJitterOp, AddJitterOp_vjp)


@primitive
def cholesky_factorization(a):
    """
    Replacement for autograd.numpy.linalg.cholesky. Our backward (vjp) is
    faster and simpler, while somewhat less general (only works if
    a.ndim == 2).

    See https://arxiv.org/abs/1710.08717 for derivation of backward (vjp)
    expression.
    
    :param a: Symmmetric positive definite matrix A
    :return: Lower-triangular Cholesky factor L of A
    """
    # Note: Do not use np.linalg.cholesky here, this can cause locking issues
    return spl.cholesky(a, lower=True)


def copyltu(x):
    return anp.tril(x) + anp.transpose(anp.tril(x, -1))


def cholesky_factorization_backward(l, lbar):
    abar = copyltu(anp.matmul(anp.transpose(l), lbar))
    abar = anp.transpose(aspl.solve_triangular(l, abar, lower=True, trans='T'))
    abar = aspl.solve_triangular(l, abar, lower=True, trans='T')
    return 0.5 * abar


def cholesky_factorization_vjp(l, a):
    return lambda lbar: cholesky_factorization_backward(l, lbar)


defvjp(cholesky_factorization, cholesky_factorization_vjp)
