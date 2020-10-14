import mxnet as mx
from mxnet import gluon

from .constants import DATA_TYPE, INITIAL_NOISE_VARIANCE, NOISE_VARIANCE_LOWER_BOUND, NOISE_VARIANCE_UPPER_BOUND, DEFAULT_ENCODING
from .distribution import Gamma
from .gluon_blocks_helpers import encode_unwrap_parameter
from .kernel import KernelFunction
from .mean import ScalarMeanFunction, MeanFunction
from .posterior_state import GaussProcPosteriorState
from .utils import create_encoding, register_parameter


class MarginalLikelihood(gluon.HybridBlock):
    """
    Marginal likelihood of Gaussian process with Gaussian likelihood

    :param kernel: Kernel function (for instance, a Matern52---note we cannot
        provide Matern52() as default argument since we need to provide with
        the dimension of points in X)
    :param mean: Mean function which depends on the input X only (by default,
        a scalar fitted while optimizing the likelihood)
    :param initial_noise_variance: A scalar to initialize the value of the
        residual noise variance
    """

    def __init__(
            self, kernel: KernelFunction, mean: MeanFunction = None,
            initial_noise_variance=None, encoding_type=None, **kwargs):
        super(MarginalLikelihood, self).__init__(**kwargs)
        if mean is None:
            mean = ScalarMeanFunction()
        if initial_noise_variance is None:
            initial_noise_variance = INITIAL_NOISE_VARIANCE
        if encoding_type is None:
            encoding_type=DEFAULT_ENCODING
        self.encoding = create_encoding(
             encoding_type, initial_noise_variance, NOISE_VARIANCE_LOWER_BOUND,
             NOISE_VARIANCE_UPPER_BOUND, 1, Gamma(mean=0.1, alpha=0.1))
        self.mean = mean
        self.kernel = kernel
        with self.name_scope():
            self.noise_variance_internal = register_parameter(
                self.params, 'noise_variance', self.encoding)

    def hybrid_forward(self, F, X, y, noise_variance_internal):
        """
        Actual computation of the marginal likelihood
        See http://www.gaussianprocess.org/gpml/chapters/RW.pdf, equation (2.30)

        :param F: mx.sym or mx.nd
        :param X: input data of size (n, d)
        :param y: targets corresponding to X, of size (n, 1)
        :param noise_variance_internal: self.noise_variance_internal passed as
            sym or nd depending on F (natively handled by gluon.HybridBlock)
        """

        noise_variance = self.encoding.get(F, noise_variance_internal)
        state = GaussProcPosteriorState(
            X, y, self.mean, self.kernel, noise_variance)
        return state.neg_log_likelihood()

    def param_encoding_pairs(self):
        """
        Return a list of tuples with the Gluon parameters of the likelihood and
        their respective encodings
        """
        own_param_encoding_pairs = [
            (self.noise_variance_internal, self.encoding)]
        return own_param_encoding_pairs + \
               self.mean.param_encoding_pairs() + \
               self.kernel.param_encoding_pairs()

    def box_constraints_internal(self):
        """
        Collect the box constraints for all the underlying parameters
        """

        all_box_constraints = {}
        for param, encoding in self.param_encoding_pairs():
            assert encoding is not None, \
                "encoding of param {} should not be None".format(param.name)
            all_box_constraints.update(encoding.box_constraints_internal(param))
        return all_box_constraints

    def get_noise_variance(self, as_ndarray=False):
        noise_variance = encode_unwrap_parameter(
            mx.nd, self.noise_variance_internal, self.encoding)
        return noise_variance if as_ndarray else noise_variance.asscalar()

    def set_noise_variance(self, val):
        self.encoding.set(self.noise_variance_internal, val)

    def get_params(self):
        result = {'noise_variance': self.get_noise_variance()}
        for pref, func in [('kernel_', self.kernel), ('mean_', self.mean)]:
            result.update({
                (pref + k): v for k, v in func.get_params().items()})
        return result

    def set_params(self, param_dict):
        for pref, func in [('kernel_', self.kernel), ('mean_', self.mean)]:
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items()
                if k.startswith(pref)}
            func.set_params(stripped_dict)
        self.set_noise_variance(param_dict['noise_variance'])
