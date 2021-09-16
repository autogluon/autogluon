import numpy as np
import autograd.numpy as anp
import autograd.scipy.linalg as aspl
from autograd import grad
#from autograd.test_util import check_grads
import time

from autogluon.core.searcher.bayesopt.gpautograd.custom_op import \
    cholesky_factorization


def _testfunc_logdet(a, use_my):
    if use_my:
        l = cholesky_factorization(a)
    else:
        l = anp.linalg.cholesky(a)
    return 2.0 * anp.sum(anp.log(anp.diag(l)))


def _testfunc_mahal(a, b, use_my):
    if use_my:
        l = cholesky_factorization(a)
    else:
        l = anp.linalg.cholesky(a)
    x = aspl.solve_triangular(l, b)
    return anp.sum(anp.square(x))


def _a_from_x(x):
    y = anp.matmul(anp.transpose(x), x)
    onevec = anp.ones_like(x[0])
    return y + 0.01 * anp.diag(onevec)


def _testfunc_logdet_from_x(x, use_my):
    return _testfunc_logdet(_a_from_x(x), use_my)


def _testfunc_mahal_from_xb(xb, use_my):
    a = _a_from_x(xb[:-1])
    b = xb[-1]
    return _testfunc_mahal(a, b, use_my)


def test_cholesky_factorization():
    #num_rep = 10
    #min_n = 100
    #max_n = 2500
    # Not so useful for time comparison, but runs faster:
    num_rep = 8
    min_n = 10
    max_n = 250
    grad_logdet_my = grad(
        lambda x: _testfunc_logdet_from_x(x, use_my=True))
    grad_logdet_cmp = grad(
        lambda x: _testfunc_logdet_from_x(x, use_my=False))
    grad_mahal_my = grad(
        lambda xb: _testfunc_mahal_from_xb(xb, use_my=True))
    grad_mahal_cmp = grad(
        lambda xb: _testfunc_mahal_from_xb(xb, use_my=False))
    for rep in range(num_rep):
        n = np.random.randint(min_n, max_n)
        xmat = np.random.randn(n, n)
        #check_grads(
        #    lambda x: testfunc_logdet_from_x(x, use_my=True),
        #    modes=['rev'], order=1)(xmat)
        # logdet
        print('\nn = {}\nlogdet:'.format(n))
        ts_start = time.time()
        gval_my = grad_logdet_my(xmat)
        time_my = time.time() - ts_start
        ts_start = time.time()
        gval_cmp = grad_logdet_cmp(xmat)
        time_cmp = time.time() - ts_start
        max_diff_grad_logdet = np.max(np.abs(gval_my - gval_cmp))
        print('max_abs_diff_grad = {}, time_my = {}, time_cmp = {}'.format(
            max_diff_grad_logdet, time_my, time_cmp))
        assert max_diff_grad_logdet < 1e-12
        # mahal
        print('mahal:')
        ts_start = time.time()
        gval_my = grad_mahal_my(xmat)
        time_my = time.time() - ts_start
        ts_start = time.time()
        gval_cmp = grad_mahal_cmp(xmat)
        time_cmp = time.time() - ts_start
        max_diff_grad_mahal = np.max(np.abs(gval_my - gval_cmp))
        print('max_abs_diff_grad = {}, time_my = {}, time_cmp = {}'.format(
            max_diff_grad_mahal, time_my, time_cmp))
        assert max_diff_grad_mahal < 1e-11


if __name__ == "__main__":
    test_cholesky_factorization()
