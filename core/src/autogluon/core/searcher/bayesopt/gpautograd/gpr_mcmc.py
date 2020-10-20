from typing import Callable, Optional, List
import autograd.numpy as anp
from autograd.builtins import isinstance

from .constants import DEFAULT_MCMC_CONFIG, MCMCConfig
from .gluon_blocks_helpers import encode_unwrap_parameter
from .gp_model import GaussianProcessModel
from .gp_regression import negative_log_posterior
from .kernel import KernelFunction
from .likelihood import MarginalLikelihood
from .mean import ScalarMeanFunction
from .posterior_state import GaussProcPosteriorState
from .slice import SliceSampler


class GPRegressionMCMC(GaussianProcessModel):

    def __init__(self,
                 build_kernel: Callable[[], KernelFunction],
                 mcmc_config: MCMCConfig = DEFAULT_MCMC_CONFIG,
                 random_seed: int = None):

        self.mcmc_config = mcmc_config

        if random_seed is not None:
            anp.random.seed(random_seed)
            self.random_seed = random_seed

        self.likelihood = _create_likelihood(build_kernel)
        self._states = None
        self.samples = None
        self.build_kernel = build_kernel

    @property
    def states(self) -> Optional[List[GaussProcPosteriorState]]:
        return self._states

    def fit(self, X, Y, **kwargs):
        X = self._check_and_format_input(X)
        Y = self._check_and_format_input(Y)

        mean_function = self.likelihood.mean
        if isinstance(mean_function, ScalarMeanFunction):
            mean_function.set_mean_value(anp.mean(Y))

        def _log_posterior_density(hp_values: anp.ndarray) -> float:
            # We check box constraints before converting hp_values to
            # internal
            if not self._is_feasible(hp_values):
                return -float('inf')
            # Decode and write into Gluon parameters
            _set_gp_hps(hp_values, self.likelihood)
            neg_log = negative_log_posterior(self.likelihood, X, Y)
            return -neg_log

        slice_sampler = SliceSampler(
            _log_posterior_density, 1.0, self.random_seed)
        init_hp_values = _get_gp_hps(self.likelihood)

        self.samples = slice_sampler.sample(
            init_hp_values, self.mcmc_config.n_samples,
            self.mcmc_config.n_burnin, self.mcmc_config.n_thinning)
        self._states = self._create_posterior_states(self.samples, X, Y)

    def recompute_states(self, X, Y, **kwargs):
        X = self._check_and_format_input(X)
        Y = self._check_and_format_input(Y)
        assert len(self.samples) > 0
        self._states = self._create_posterior_states(self.samples, X, Y)

    def _is_feasible(self, hp_values: anp.ndarray) -> bool:
        pos = 0
        for _, encoding in self.likelihood.param_encoding_pairs():
            lower, upper = encoding.box_constraints()
            dim = encoding.dimension
            if lower is not None or upper is not None:
                values = hp_values[pos:(pos + dim)]
                if (lower is not None) and any(values < lower):
                    return False
                if (upper is not None) and any(values > upper):
                    return False
            pos += dim
        return True

    def _create_posterior_states(self, samples, X, Y):
        states = []
        for sample in samples:
            likelihood = _create_likelihood(self.build_kernel)
            _set_gp_hps(sample, likelihood)
            state = GaussProcPosteriorState(
                X, Y, likelihood.mean, likelihood.kernel,
                likelihood.get_noise_variance(as_ndarray=True))
            states.append(state)
        return states


def _get_gp_hps(likelihood: MarginalLikelihood) -> anp.ndarray:
    """Get GP hyper-parameters as numpy array for a given likelihood object."""
    hp_values = []
    for param_int, encoding in likelihood.param_encoding_pairs():
        hp_values.append(
            encode_unwrap_parameter(param_int, encoding))
    return anp.concatenate(hp_values)


def _set_gp_hps(params_numpy: anp.ndarray, likelihood: MarginalLikelihood):
    """Set GP hyper-parameters from numpy array for a given likelihood object."""
    pos = 0
    for param, encoding in likelihood.param_encoding_pairs():
        dim = encoding.dimension
        values = params_numpy[pos:(pos + dim)]
        if dim == 1:
            internal_values = encoding.decode(values, param.name)
        else:
            internal_values = anp.array(
                [encoding.decode(v, param.name) for v in values])
        param.set_data(internal_values)
        pos += dim


def _create_likelihood(build_kernel) -> MarginalLikelihood:
    """Create a MarginalLikelihood object with default initial GP hyperparameters."""

    likelihood = MarginalLikelihood(
        kernel=build_kernel(),
        mean=ScalarMeanFunction(),
        initial_noise_variance=None
    )
    likelihood.initialize(force_reinit=True)

    return likelihood
