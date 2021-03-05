import logging
import numpy as np
import autograd.numpy as anp
from typing import Optional, List

from .constants import OptimizationConfig, DEFAULT_OPTIMIZATION_CONFIG
from .gluon_blocks_helpers import encode_unwrap_parameter
from .gp_model import GaussianProcessModel
from .kernel import KernelFunction
from .likelihood import MarginalLikelihood
from .mean import ScalarMeanFunction, MeanFunction
from .optimization_utils import apply_lbfgs_with_multiple_starts, \
    ParamVecDictConverter, make_scipy_objective
from .posterior_state import GaussProcPosteriorState
from .utils import param_to_pretty_string
from autogluon.core.searcher.bayesopt.utils.simple_profiler import SimpleProfiler

logger = logging.getLogger(__name__)


def negative_log_posterior(
        likelihood: MarginalLikelihood, X: np.array, Y: np.array):
    objective_nd = likelihood(X, Y)
    # Add neg log hyperpriors, whenever some are defined
    for param_int, encoding in likelihood.param_encoding_pairs():
        if encoding.regularizer is not None:
            param = encode_unwrap_parameter(
                param_int, encoding, X)
            objective_nd = objective_nd + encoding.regularizer(
                param)
    return objective_nd


class GaussianProcessRegression(GaussianProcessModel):
    """
    Gaussian Process Regression

    Takes as input a mean function (which depends on X only) and a kernel
    function.

    :param kernel: Kernel function (for instance, a Matern52---note we cannot
        provide Matern52() as default argument since we need to provide with
        the dimension of points in X)
    :param mean: Mean function (which depends on X only)
    :param initial_noise_variance: Initial value for noise variance parameter
    :param optimization_config: Configuration that specifies the behavior of
        the optimization of the marginal likelihood.
    :param random_seed: Random seed to be used (optional)
    :param fit_reset_params: Reset parameters to initial values before running
        'fit'? If False, 'fit' starts from the current values

    """
    def __init__(
            self, kernel: KernelFunction, mean: MeanFunction = None,
            initial_noise_variance: float = None,
            optimization_config: OptimizationConfig = None,
            random_seed=None, fit_reset_params: bool = True,
            test_intermediates: Optional[dict] = None):

        if mean is None:
            mean = ScalarMeanFunction()
        if optimization_config is None:
            optimization_config = DEFAULT_OPTIMIZATION_CONFIG
        if random_seed is not None:
            np.random.seed(random_seed)
            anp.random.seed(random_seed)

        self._states = None
        self.fit_reset_params = fit_reset_params
        self.optimization_config = optimization_config
        self._test_intermediates = test_intermediates
        self.likelihood = MarginalLikelihood(
            kernel=kernel, mean=mean,
            initial_noise_variance=initial_noise_variance)
        self.reset_params()

    @property
    def states(self) -> Optional[List[GaussProcPosteriorState]]:
        return self._states

    def _create_lbfgs_arguments(self, X, Y):
        """
        Creates SciPy optimizer objective and param_dict for criterion
        function.

        :param X: data matrix X of size (n, d)
        :param Y: vector of targets of size (n, 1)
        :return: scipy_objective, param_dict
        """
        param_dict = {
            param.name: param
            for param in self.likelihood.collect_params().values()}
        # Wraps param_dict, conversion to/from flat vector:
        param_converter = ParamVecDictConverter(param_dict)

        def executor(param_vec):
            param_converter.from_vec(param_vec)  # Assign param_dict
            objective = negative_log_posterior(self.likelihood, X, Y)
            if self.optimization_config.verbose:
                msg_lst = ["[criterion = {}]".format(objective)]
                for param, encoding in self.likelihood.param_encoding_pairs():
                    msg_lst.append(param_to_pretty_string(param, encoding))
                logger.info('\n'.join(msg_lst))
            return objective

        return make_scipy_objective(executor), param_dict

    def fit(self, X, Y, profiler: SimpleProfiler = None):
        """
        Fit the parameters of the GP by optimizing the marginal likelihood,
        and set posterior states.

        We catch exceptions during the optimization restarts. If any restarts
        fail, log messages are written. If all restarts fail, the current
        parameters are not changed.

        :param X: data matrix X of size (n, d)
        :param Y: matrix of targets of size (n, 1)
        """
        X = self._check_and_format_input(X)
        Y = self._check_and_format_input(Y)
        assert X.shape[0] == Y.shape[0], "X and Y should have the same number of points (received {} and {})".format(
                X.shape[0], Y.shape[0])
        assert Y.shape[1] == 1, "Y cannot be a matrix if parameters are to be fit"
        
        if self.fit_reset_params:
            self.reset_params()
        mean_function = self.likelihood.mean
        if isinstance(mean_function, ScalarMeanFunction):
            mean_function.set_mean_value(np.mean(Y))
        if profiler is not None:
            profiler.start('fithyperpars')
        n_starts = self.optimization_config.n_starts
        ret_infos = apply_lbfgs_with_multiple_starts(
            *self._create_lbfgs_arguments(X, Y),
            bounds=self.likelihood.box_constraints_internal(),
            n_starts=n_starts,
            tol=self.optimization_config.lbfgs_tol,
            maxiter=self.optimization_config.lbfgs_maxiter)
        if profiler is not None:
            profiler.stop('fithyperpars')

        # Logging in response to failures of optimization runs
        n_succeeded = sum(x is None for x in ret_infos)
        if n_succeeded < n_starts:
            log_msg = "[GaussianProcessRegression.fit]\n"
            log_msg += ("{} of the {} restarts failed with the following exceptions:\n".format(
                n_starts - n_succeeded, n_starts))
            copy_params = {
                param.name: param.data()
                for param in self.likelihood.collect_params().values()}
            for i, ret_info in enumerate(ret_infos):
                if ret_info is not None:
                    log_msg += ("- Restart {}: Exception {}\n".format(
                        i, ret_info['type']))
                    log_msg += ("  Message: {}\n".format(ret_info['msg']))
                    log_msg += ("  Args: {}\n".format(ret_info['args']))
                    # Set parameters in order to print them. These are the
                    # parameters for which the evaluation failed
                    self._set_likelihood_params(ret_info['params'])
                    log_msg += ("  Params: " + str(self.get_params()))
                    logger.info(log_msg)
            # Restore parameters
            self._set_likelihood_params(copy_params)
            if n_succeeded == 0:
                logger.info("All restarts failed: Skipping hyperparameter fitting for now")
        # Recompute posterior state for new hyperparameters
        self._recompute_states(X, Y, profiler=profiler)

    def _set_likelihood_params(self, params: dict):
        for param in self.likelihood.collect_params().values():
            vec = params.get(param.name)
            if vec is not None:
                param.set_data(vec)

    def recompute_states(self, X, Y, profiler: SimpleProfiler = None):
        """
        We allow Y to be a matrix with m>1 columns, which is useful to support
        batch decisions by fantasizing.
        """
        X = self._check_and_format_input(X)
        Y = self._check_and_format_input(Y)
        assert X.shape[0] == Y.shape[0], "X and Y should have the same number of points (received {} and {})".format(
                X.shape[0], Y.shape[0])
        assert self._states is not None, self._states
        self._recompute_states(X, Y, profiler=profiler)

    def _recompute_states(self, X, Y, profiler: SimpleProfiler = None):
        if profiler is not None:
            profiler.start('posterstate')
        self._states = [GaussProcPosteriorState(
            X, Y, self.likelihood.mean, self.likelihood.kernel,
            self.likelihood.get_noise_variance(as_ndarray=True),
            debug_log=(self._test_intermediates is not None),
            test_intermediates=self._test_intermediates)]
        if profiler is not None:
            profiler.stop('posterstate')
    
    def get_params(self):
        return self.likelihood.get_params()

    def set_params(self, param_dict):
        self.likelihood.set_params(param_dict)

    def reset_params(self):
        """
        Reset hyperparameters to their initial values (or resample them).
        """
        self.likelihood.initialize(force_reinit=True)
