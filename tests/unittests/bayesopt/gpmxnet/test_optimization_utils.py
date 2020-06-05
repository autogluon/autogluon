import mxnet as mx
import numpy as np
from mxnet import autograd

from autogluon.searcher.bayesopt.gpmxnet.optimization_utils import \
    apply_lbfgs_with_multiple_starts, apply_lbfgs


def _create_toy_lbfgs_problem():

    x = -0.5*mx.nd.ones((1,), ctx=mx.cpu(), dtype='float64')
    x.attach_grad()

    ###############
    def executor():

        with autograd.record():
            # Over [-4,4]^3, the function has two minima, one local at around x = -2.
            # and global at around x = +2.91318053.
            # Moreover, it has a maximum around x = 0. (plot it in Google, simply typing y=(x+2)*sin(x+2))
            # If the starting point is about x = -0.5 (note that this is the default value for x),
            # L-BFGS should end-up at the local minima while if the starting is about x = 0.5,
            # L-BFGS should end-up at the global minima
            non_convex_function = (x+2.)*mx.nd.sin(x+2.)

        objective_value = non_convex_function.asscalar()
        non_convex_function.backward()

        return objective_value
    ###############

    arg_dict = {'x':x}
    grad_dict = {'x':x.grad}

    return executor, arg_dict, grad_dict


def test_apply_lbfgs_with_single_restart():

    bounds = {'x' : (-4.,4.)}
    executor, arg_dict_with_single_start, grad_dict = _create_toy_lbfgs_problem()
    apply_lbfgs_with_multiple_starts(executor, arg_dict_with_single_start, grad_dict, bounds, n_starts=1)

    # See explanations above to understand the target value of -2.
    np.testing.assert_almost_equal(arg_dict_with_single_start['x'].asscalar(), -2., decimal=5)

    executor, arg_dict, grad_dict = _create_toy_lbfgs_problem()
    apply_lbfgs(executor, arg_dict, grad_dict, bounds)

    # With a single starting point, apply_lbfgs and apply_lbfgs_with_restarts must coincide
    np.testing.assert_almost_equal(arg_dict['x'].asscalar(), arg_dict_with_single_start['x'].asscalar())


def test_apply_lbfgs_with_multiple_restart():

    bounds = {'x' : (-4.,4.)}
    executor, arg_dict_with_multiple_starts, grad_dict = _create_toy_lbfgs_problem()
    # If this is left at -0.5, then the test may fail even with 10 repetitions
    # In fact, the logic in apply_lbfgs_with_multiple_starts makes little sense.
    # At least, the randomization would have to be with stddev of size
    # related to the norm of the mean.
    arg_dict_with_multiple_starts['x'][:] = 0.1
    apply_lbfgs_with_multiple_starts(executor, arg_dict_with_multiple_starts, grad_dict, bounds, n_starts=10)

    # See explanations above to understand the target value of 2.91318053
    np.testing.assert_almost_equal(arg_dict_with_multiple_starts['x'].asscalar(), 2.91318053, decimal=4)