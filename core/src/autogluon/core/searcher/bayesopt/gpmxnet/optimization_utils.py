"""
optimization_utils
==================

Wrapper of SciPy L-BFGS-B optimizer minimizing objectives given by MXNet
executors. The main issue is to map between parameter dictionaries of
mx.ndarray (executor side) and a single flat np.ndarray (optimizer side).
"""

import mxnet as mx
import numpy as np
from scipy import optimize
import ctypes

__all__ = ['apply_lbfgs',
           'from_executor',
           'apply_lbfgs_with_multiple_starts']


# NOTE: _dtype is the dtype used on the NumPy side (argument, value, and
# gradient of objective.
_dtype = np.float64
default_LBFGS_tol, default_LBFGS_maxiter = 1e-5, 500

N_STARTS = 5
STARTING_POINT_RANDOMIZATION_STD = 1.


# Utility functions for (un-)grouping NDArrays into Numpy arrays.
# Useful e.g. to use optimizers from scipy.optimize to optimize
# functions expressed in MXNet.

def _get_name_to_index(nd_arrays, names):
    name_to_index = {}
    global_position = 0
    for name in names:
        a = nd_arrays[name]
        name_to_index[name] = np.arange(global_position, global_position+a.size)
        global_position += a.size
    return name_to_index


def _zeros_like_nd_list(l, dtype):
    """
    Create a Numpy array with size equal to the
    sum of the sizes of all the NDArrays in the list
    of NDArrays l.
    """
    total_size = np.sum([x.size for x in l])
    return np.zeros(total_size, dtype)


def _copy_into(ndarray, nparray):
    """
    Copy the values from the given ndarray into the given (preallocated) numpy array.
    This can be used to avoid extra memory allocation that ndarray.asnumpy()
    performs.
    """
    assert nparray.size == ndarray.size
    assert nparray.flags.f_contiguous and nparray.flags.behaved
    # NOTE: The copy=False variant of NDArray.astype does not seem to work
    if ndarray.dtype != nparray.dtype:
        ndarray = ndarray.astype(nparray.dtype)
    mx.base.check_call(mx.base._LIB.MXNDArraySyncCopyToCPU(
        ndarray.handle,
        nparray.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(ndarray.size)))


def _copy_to_numpy_array(l, a):
    """
    Copy values from each NDArray in the list l to the numpy array a (in
    order).
    """
    total_size = np.sum([x.size for x in l])
    assert total_size == a.size
    j = 0
    for x in l:
        # a[j:j+x.size] = x.asnumpy().reshape((x.size,))
        _copy_into(x, a[j:j + x.size])
        j += x.size


def _copy_from_numpy_array(a, l):
    """
    Copy values from sub-arrays of the numpy array a to
    the NDArrays in the list l. The sizes of the sub arrays
    correspond to the sizes of the NDArrays, so that this
    performs a copy in the reverse direction of
    copy_to_numpy_array().

    Entries of l can have different dtype than a.
    """
    total_size = np.sum([x.size for x in l])
    assert total_size == a.size
    j = 0
    for x in l:
        x[:] = a[j:j + x.size].reshape(x.shape)
        j += x.size


def _make_objective(
        exec_func, arg_dict, grad_dict, param_names):
    # list of argument and grad NDArrays corresponding to param_names
    param_nd_arrays = [arg_dict[name] for name in param_names]
    grad_nd_arrays = [grad_dict[name] for name in param_names]
    # Numpy arrays for holding parameters and gradients
    # grad output array for direct copy
    grad_numpy_array = _zeros_like_nd_list(grad_nd_arrays, _dtype)

    # Objective and its gradients
    def objective(a):
        # copy parameter values to arg_dict (NDArrays)
        _copy_from_numpy_array(a, param_nd_arrays)
        # compute objective and gradients
        obj_val = exec_func()
        # Copy gradients from grad_dict (NDArray) into numpy array
        _copy_to_numpy_array(grad_nd_arrays, grad_numpy_array)
        return obj_val, grad_numpy_array

    return objective


def _apply_lbfgs_internal(
        exec_func, arg_dict, grad_dict, param_names, param_numpy_array,
        name_to_index, bounds, **kwargs):
    # Define bounds for L-BFGS, None by default
    param_bounds = np.array([(None, None)] * len(param_numpy_array))
    for name, bound in bounds.items():
        if name in param_names:
            param_bounds[name_to_index[name]] = bound
    # Objective from executor
    mx_objective = _make_objective(
        exec_func, arg_dict, grad_dict, param_names)
    # Run L-BFGS-B
    LBFGS_tol = kwargs.get("tol", default_LBFGS_tol)
    LBFGS_maxiter = kwargs.get("maxiter", default_LBFGS_maxiter)
    LBFGS_callback = kwargs.get("callback", None)
    ret_info = None
    try:
        output = optimize.minimize(mx_objective,
                                   param_numpy_array,
                                   jac=True,
                                   method="L-BFGS-B",
                                   bounds=param_bounds,
                                   tol=LBFGS_tol,
                                   options={"maxiter": LBFGS_maxiter},
                                   callback=LBFGS_callback)
        # NOTE: Be aware that the stopping condition based on tol can terminate
        # with a gradient size which is not small.
        # To control L-BFGS convergence conditions for real, have to instead use
        # something like this:
        #                           tol=None,
        #                           options={
        #                               "maxiter": LBFGS_maxiter,
        #                               "ftol": 1e-6,
        #                               "gtol": 1e-1},
        #                           callback=LBFGS_callback)

        # Write result evaluation point back to arg_dict
        optimized_param_numpy_array = output.x
        param_nd_arrays = [arg_dict[name] for name in param_names]
        _copy_from_numpy_array(optimized_param_numpy_array, param_nd_arrays)
    except Exception as inst:
        ret_info = {
            'type': type(inst),
            'args': inst.args,
            'msg': str(inst)}
    return ret_info


# Utility functions for multiple restarts


class ExecutorDecorator:
    """
    This class is a lightweight decorator around the executor passed to L-BFGS
    It adds the functionality of keeping track of the best objective function
    """

    def __init__(self, exec_func):
        self.best_objective = np.inf
        self._exec_func = exec_func

    def exec_func(self):
        objective = self._exec_func()
        self.best_objective = min(self.best_objective, objective)
        return objective


def _deep_copy_arg_dict(input_arg_dict):
    """
    Make a deep copy of the input arg_dict (dict param_name to mx.nd)
    :param input_arg_dict:
    :return: deep copy of input_arg_dict
    """
    output_arg_dict = {}
    for name, param in input_arg_dict.items():
        output_arg_dict[name] = param.copy()
    return output_arg_dict


def _inplace_arg_dict_randomization(arg_dict, mean_arg_dict, bounds, std=STARTING_POINT_RANDOMIZATION_STD):
    """
    In order to initialize L-BFGS from multiple starting points, this function makes it possible to
    randomize, inplace, an arg_dict (as used by executors to communicate parameters to L-BFGS).
    The randomization is centered around mean_arg_dict, with standard deviation std.

    :param arg_dict: dict param_name to mx.nd (as used in executors). This argument is modified inplace
    :param mean_arg_dict: arg_dict around which the random perturbations occur (dict param_name to mx.nd, as used in executors))
    :param bounds: dict param_name to (lower, upper) bounds, as used in L-BFGS
    :param std: standard deviation according to which the (Gaussian) random perturbations happen
    """

    # We check that arg_dict and mean_arg_dict are compatible
    assert arg_dict.keys() == mean_arg_dict.keys()
    for name, param in arg_dict.items():
        assert param.shape == mean_arg_dict[name].shape
        assert param.dtype == mean_arg_dict[name].dtype
        assert param.context == mean_arg_dict[name].context

    # We apply a sort to make the for loop deterministic (especially with the internal calls to mx.random)
    for name, param in sorted(arg_dict.items()):

        arg_dict[name][:] = mean_arg_dict[name] + mx.random.normal(0.0, std, shape=param.shape, dtype=param.dtype, ctx=param.context)

        lower, upper = bounds[name]
        lower = lower if lower is not None else -np.inf
        upper = upper if upper is not None else np.inf

        # We project back arg_dict[name] within its specified lower and upper bounds
        # (in case of we would have perturbed beyond those bounds)
        arg_dict[name][:] = mx.nd.maximum(lower, mx.nd.minimum(upper, arg_dict[name]))


# === Exported functions ===


def apply_lbfgs(exec_func, arg_dict, grad_dict, bounds, **kwargs):
    """Run SciPy L-BFGS-B on criterion given by MXNet code

    Run SciPy L-BFGS-B in order to minimize criterion given by MXNet code.
    Criterion and gradient are computed by:

        crit_val = exec_func()

    Here, arguments are taken from arg_dict, and gradients are written to
    grad_dict (both are dictionaries with NDArray values). crit_val is a
    Python scalar, not an NDArray.
    Think of arg_dict and grad_dict as args and args_grad arguments of an
    MXNet executor.

    The variables which L-BFGS-B is optimizing over are all those in
    grad_dict whose values are not None. Both arg_dict and grad_dict must
    contain values for these keys, and they must have the same shapes.
    Both arg_dict and grad_dict can have additional entries, but these are
    not modified.

    Initial values are taken from arg_dict, and final values are written
    back there.

    L-BFGS-B allows box constraints [a, b] for any coordinate. Here, None
    stands for -infinity (a) or +infinity (b). The default is (None, None),
    so no constraints. In bounds, box constraints can be specified per
    argument (the constraint applies to all coordinates of the argument).
    Pass {} for no constraints.

    If the criterion function is given by an MXNet executor mx_executor,
    you can call

        apply_bfgs(*from_executor(mx_executor), bounds, ...)

    See from_executor comments for details.

    :param exec_func: Function to compute criterion
    :param arg_dict: See above
    :param grad_dict: See above
    :param bounds: See above
    :return: None, or dict with info about exception caught
    """

    param_names = sorted(
        [name for name, value in grad_dict.items() \
         if value is not None])
    name_to_index = _get_name_to_index(arg_dict, param_names)
    # Construct initial evaluation point (NumPy)
    param_nd_arrays = [arg_dict[name] for name in param_names]
    param_numpy_array = _zeros_like_nd_list(param_nd_arrays, _dtype)
    _copy_to_numpy_array(param_nd_arrays, param_numpy_array)

    return _apply_lbfgs_internal(
        exec_func, arg_dict, grad_dict, param_names, param_numpy_array,
        name_to_index, bounds, return_results=False, **kwargs)


def from_executor(executor):
    """Maps MXNet executor to apply_lbfgs arguments

    apply_lbfgs allows to pass exec_func, arg_dict, grad_dict to specify the
    criterion function, its argument and gradient dictionaries. If your
    criterion is given by an MXNet executor mx_executor, you can call

        apply_bfgs(*from_executor(mx_executor), bounds, ...)

    Here, arg_dict = mx_executor.arg_dict, grad_dict = mx_executor.grad_dict.
    This requires that mx_executors represents a loss function (use
    mx.sym.MakeLoss to be safe).

    :param executor: MXNet executor representing a loss function
    :return: exec_func, arg_dict, grad_dict arguments for apply_lbfgs
    """

    def exec_func():
        executor.forward(is_train=True)
        obj_val = executor.outputs[0].asscalar()
        executor.backward()
        return obj_val

    return exec_func, executor.arg_dict, executor.grad_dict


def apply_lbfgs_with_multiple_starts(
        exec_func, arg_dict, grad_dict, bounds, n_starts=N_STARTS, **kwargs):
    """
    When dealing with non-convex problems (e.g., optimization the marginal
    likelihood), we typically need to start from various starting points. This
    function applies this logic around apply_lbfgs, randomizing the starting
    points around the initial values provided in arg_dict (see below
    "copy_of_initial_arg_dict").

    The first optimization happens exactly at arg_dict, so that the case
    n_starts=1 exactly coincides with the previously used apply_lbfgs.
    Importantly, the communication with the L-BFGS solver happens via arg_dict,
    hence all the operations with respect to arg_dict are inplace.

    We catch exceptions and return ret_infos about these. If none of the
    restarts worked, arg_dict is not modified.

    :param exec_func: see above
    :param arg_dict: see above
    :param grad_dict: see above
    :param bounds: see above
    :param n_starts: Number of times we start an optimization with L-BFGS
        (must be >= 1)
    :return: List ret_infos of length n_starts. Entry is None if optimization
        worked, or otherwise has dict with info about exception caught
    """

    assert n_starts >= 1

    copy_of_initial_arg_dict = _deep_copy_arg_dict(arg_dict)
    best_objective_over_restarts = None
    best_arg_dict_over_restarts = copy_of_initial_arg_dict

    # Loop over restarts
    ret_infos = []
    for iter in range(n_starts):
        if iter > 0:
            _inplace_arg_dict_randomization(
                arg_dict, copy_of_initial_arg_dict, bounds)
        decorator = ExecutorDecorator(exec_func)
        ret_info = apply_lbfgs(
            decorator.exec_func, arg_dict, grad_dict, bounds, **kwargs)
        ret_infos.append(ret_info)
        if ret_info is None and (
                best_objective_over_restarts is None or
                decorator.best_objective < best_objective_over_restarts):
            best_objective_over_restarts = decorator.best_objective
            best_arg_dict_over_restarts = _deep_copy_arg_dict(arg_dict)

    # We copy back the values of the best parameters into arg_dict (again,
    # inplace, as required by the executor)
    for name in arg_dict.keys():
        arg_dict[name][:] = best_arg_dict_over_restarts[name]
    return ret_infos
