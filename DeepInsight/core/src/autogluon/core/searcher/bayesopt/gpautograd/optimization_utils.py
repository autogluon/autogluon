import numpy as np
from scipy import optimize
from autograd import value_and_grad

from .gluon import Parameter

__all__ = ['apply_lbfgs',
           'apply_lbfgs_with_multiple_starts',
           'ParamVecDictConverter',
           'make_scipy_objective']


default_LBFGS_tol, default_LBFGS_maxiter = 1e-5, 500
N_STARTS = 5
STARTING_POINT_RANDOMIZATION_STD = 1.


class ParamVecDictConverter(object):
    def __init__(self, param_dict: dict):
        self.param_dict = param_dict
        self.names = sorted(
            [name for name, value in param_dict.items() if value is not None])
        self.shapes = []
        self.name_to_index = dict()
        pos = 0
        for name in self.names:
            shape = param_dict[name].data().shape
            self.shapes.append(shape)
            size = sum(shape)
            self.name_to_index[name] = np.arange(pos, pos + size)
            pos += size

    def from_vec(self, param_vec: np.ndarray):
        pos = 0
        for name, shape in zip(self.names, self.shapes):
            size = sum(shape)
            self.param_dict[name].set_data(
                np.reshape(param_vec[pos:(pos + size)], shape))
            pos += size

    def to_vec(self):
        param_arrays = [self.param_dict[name].data() for name in self.names]
        return np.concatenate([np.reshape(x, (-1,)) for x in param_arrays])


def make_scipy_objective(autograd_func):
    """
    Maps autograd expression into objective (criterion and gradient) for SciPy
    optimizer. The input to autograd_func is a flat param_vec.

    :param autograd_func: Autograd expression
    :return: SciPy optimizer objective
    """
    return value_and_grad(lambda x: autograd_func(x))


def _apply_lbfgs_internal(
        exec_func, param_converter: ParamVecDictConverter, param_numpy_array,
        param_bounds, **kwargs):

    # Run L-BFGS-B
    LBFGS_tol = kwargs.get("tol", default_LBFGS_tol)
    LBFGS_maxiter = kwargs.get("maxiter", default_LBFGS_maxiter)
    LBFGS_callback = kwargs.get("callback", None)
    ret_info = None

    try:
        output = optimize.minimize(exec_func,
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

        # Write result evaluation point back to param_dict
        param_converter.from_vec(output.x)

    except Exception as inst:
        ret_info = {
            'type': type(inst),
            'args': inst.args,
            'msg': str(inst)}
            
    return ret_info


class ExecutorDecorator:
    """
    This class is a lightweight decorator around the executor passed to L-BFGS
    It adds the functionality of keeping track of the best objective function
    """

    def __init__(self, exec_func):
        self.best_objective = np.inf
        self._exec_func = exec_func

    def exec_func(self, param_vec):
        objective, gradient = self._exec_func(param_vec)
        self.best_objective = min(self.best_objective, objective)
        return objective, gradient


def _deep_copy_param_dict(input_param_dict):
    """
    Make a deep copy of the input param_dict
    :param input_param_dict:
    :return: deep copy of input_param_dict
    """
    output_param_dict = {}
    for name, param in input_param_dict.items():
        param_copy = Parameter(name=param.name, shape=param.shape)
        param_copy.initialize()
        param_copy.set_data(param.data())
        output_param_dict[name] = param_copy
    return output_param_dict


def _inplace_param_dict_randomization(param_dict, mean_param_dict, bounds, std=STARTING_POINT_RANDOMIZATION_STD):
    """
    In order to initialize L-BFGS from multiple starting points, this function makes it possible to
    randomize, inplace, an param_dict (as used by executors to communicate parameters to L-BFGS).
    The randomization is centered around mean_param_dict, with standard deviation std.

    :param param_dict: dict param_name to np.ndarray (as used in executors). This argument is modified inplace
    :param mean_param_dict: param_dict around which the random perturbations occur (dict param_name to np.ndarray, as used in executors))
    :param bounds: dict param_name to (lower, upper) bounds, as used in L-BFGS
    :param std: standard deviation according to which the (Gaussian) random perturbations happen
    """
    # We check that param_dict and mean_param_dict are compatible
    assert param_dict.keys() == mean_param_dict.keys()
    for name, param in param_dict.items():
        assert param.shape == mean_param_dict[name].shape
        assert param.dtype == mean_param_dict[name].dtype

    # We apply a sort to make the for loop deterministic (especially with the internal calls to np.random)
    for name, param in sorted(param_dict.items()):

        lower, upper = bounds[name]
        lower = lower if lower is not None else -np.inf
        upper = upper if upper is not None else np.inf
        
        param_value_new = mean_param_dict[name].data() + np.random.normal(0.0, std, size=param.shape)
        # We project back param_dict[name] within its specified lower and upper bounds
        # (in case of we would have perturbed beyond those bounds)
        param_dict[name].set_data(np.maximum(lower,
            np.minimum(upper, param_value_new)))


def apply_lbfgs(exec_func, param_dict, bounds, **kwargs):
    """Run SciPy L-BFGS-B on criterion given by autograd code

    Run SciPy L-BFGS-B in order to minimize criterion given by autograd code.
    Criterion and gradient are computed by:

        crit_val, gradient = exec_func(param_vec)

    Given an autograd expression, use make_scipy_objective to obtain exec_func.
    param_vec must correspond to the parameter dictionary param_dict via
    ParamVecDictConverter. The initial param_vec is taken from param_dict,
    and final values are written back to param_dict (conversions are done
    by ParamVecDictConverter).

    L-BFGS-B allows box constraints [a, b] for any coordinate. Here, None
    stands for -infinity (a) or +infinity (b). The default is (None, None),
    so no constraints. In bounds, box constraints can be specified per
    argument (the constraint applies to all coordinates of the argument).
    Pass {} for no constraints.

    :param exec_func: Function to compute criterion and gradient
    :param param_dict: See above
    :param bounds: See above
    :return: None, or dict with info about exception caught
    """
    param_converter = ParamVecDictConverter(param_dict)
    # Initial evaluation point
    param_numpy_array = param_converter.to_vec()

    # Define bounds for L-BFGS, None by default
    param_bounds = np.array([(None, None)] * len(param_numpy_array))
    name_to_index = param_converter.name_to_index
    param_names = set(param_converter.names)
    for name, bound in bounds.items():
        if name in param_names:
            param_bounds[name_to_index[name]] = bound
    
    ret_info = _apply_lbfgs_internal(
        exec_func, param_converter, param_numpy_array, param_bounds, **kwargs)
    if ret_info is not None:
        # Caught exception: Return parameters for which evaluation failed
        ret_info['params'] = {
            k: v.data() for k, v in param_dict.items()}
        # Restore initial evaluation point
        param_converter.from_vec(param_numpy_array)
    return ret_info

def apply_lbfgs_with_multiple_starts(
        exec_func, param_dict, bounds, n_starts=N_STARTS, **kwargs):
    """
    When dealing with non-convex problems (e.g., optimization the marginal
    likelihood), we typically need to start from various starting points. This
    function applies this logic around apply_lbfgs, randomizing the starting
    points around the initial values provided in param_dict (see below
    "copy_of_initial_param_dict").

    The first optimization happens exactly at param_dict, so that the case
    n_starts=1 exactly coincides with the previously used apply_lbfgs.
    Importantly, the communication with the L-BFGS solver happens via param_dict,
    hence all the operations with respect to param_dict are inplace.

    We catch exceptions and return ret_infos about these. If none of the
    restarts worked, param_dict is not modified.

    :param exec_func: see above
    :param param_dict: see above
    :param bounds: see above
    :param n_starts: Number of times we start an optimization with L-BFGS
        (must be >= 1)
    :return: List ret_infos of length n_starts. Entry is None if optimization
        worked, or otherwise has dict with info about exception caught
    """
    assert n_starts >= 1

    copy_of_initial_param_dict = _deep_copy_param_dict(param_dict)
    best_objective_over_restarts = None
    best_param_dict_over_restarts = copy_of_initial_param_dict
    
    # Loop over restarts
    ret_infos = []
    for iter in range(n_starts):
        if iter > 0:
            _inplace_param_dict_randomization(
                param_dict, copy_of_initial_param_dict, bounds)
            
        decorator = ExecutorDecorator(exec_func)
        ret_info = apply_lbfgs(
            decorator.exec_func, param_dict, bounds, **kwargs)
        
        ret_infos.append(ret_info)
        if ret_info is None and (
                best_objective_over_restarts is None or
                decorator.best_objective < best_objective_over_restarts):
            best_objective_over_restarts = decorator.best_objective
            best_param_dict_over_restarts = _deep_copy_param_dict(param_dict)

    # We copy back the values of the best parameters into param_dict (again,
    # inplace, as required by the executor)
    for name in param_dict.keys():
        param_dict[name].set_data(best_param_dict_over_restarts[name].data())
    return ret_infos
