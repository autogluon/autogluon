from abc import ABC, abstractmethod
import mxnet as mx
from mxnet import gluon

from .constants import INITIAL_MEAN_VALUE
from .distribution import Normal
from .gluon_blocks_helpers import IdentityScalarEncoding, encode_unwrap_parameter
from .utils import register_parameter

__all__ = [
    'MeanFunction',
    'ScalarMeanFunction',
    'ZeroMeanFunction'
]


class MeanFunction(gluon.HybridBlock, ABC):
    """
    Mean function, parameterizing a surrogate model together with a kernel function.

    Note: KernelFunction also inherits from this interface.

    """
    def __init__(self, **kwargs):
        gluon.HybridBlock.__init__(self, **kwargs)

    @abstractmethod
    def param_encoding_pairs(self):
        """
        Returns list of tuples
            (param_internal, encoding)
        over all Gluon parameters maintained here.

        :return: List [(param_internal, encoding)]
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        :return: Dictionary with hyperparameter values
        """
        pass

    @abstractmethod
    def set_params(self, param_dict):
        """

        :param param_dict: Dictionary with new hyperparameter values
        :return:
        """
        pass


class ScalarMeanFunction(MeanFunction):
    """
    Mean function defined as a scalar (fitted while optimizing the marginal
    likelihood).

    :param initial_mean_value: A scalar to initialize the value of the mean

    """
    def __init__(self, initial_mean_value = INITIAL_MEAN_VALUE, **kwargs):
        super(ScalarMeanFunction, self).__init__(**kwargs)

        # Even though we do not apply specific transformation to the mean value
        # we use an encoding to handle in a consistent way the box constraints
        # of Gluon parameters (like bandwidths or residual noise variance)
        self.encoding = IdentityScalarEncoding(
            init_val=initial_mean_value, regularizer=Normal(0.0, 1.0))
        with self.name_scope():
            self.mean_value_internal = register_parameter(
                self.params, 'mean_value', self.encoding)

    def hybrid_forward(self, F, X, mean_value_internal):
        """
        Actual computation of the scalar mean function
        We compute mean_value * vector_of_ones, whose dimensions are given by
        the the first column of X

        :param F: mx.sym or mx.nd
        :param X: input data of size (n,d) for which we want to compute the
            mean (here, only useful to extract the right dimension)

        """
        mean_value = self.encoding.get(F, mean_value_internal)
        return F.broadcast_mul(F.ones_like(F.slice_axis(
            F.BlockGrad(X), axis=1, begin=0, end=1)), mean_value)

    def param_encoding_pairs(self):
        return [(self.mean_value_internal, self.encoding)]

    def get_mean_value(self):
        return encode_unwrap_parameter(
            mx.nd, self.mean_value_internal, self.encoding).asscalar()

    def set_mean_value(self, mean_value):
        self.encoding.set(self.mean_value_internal, mean_value)

    def get_params(self):
        return {'mean_value': self.get_mean_value()}

    def set_params(self, param_dict):
        self.set_mean_value(param_dict['mean_value'])


class ZeroMeanFunction(MeanFunction):
    def __init__(self, **kwargs):
        super(ZeroMeanFunction, self).__init__(**kwargs)

    def hybrid_forward(self, F, X):
        return F.zeros_like(F.slice_axis(
            F.BlockGrad(X), axis=1, begin=0, end=1))

    def param_encoding_pairs(self):
        return []

    def get_params(self):
        return dict()

    def set_params(self, param_dict):
        pass
