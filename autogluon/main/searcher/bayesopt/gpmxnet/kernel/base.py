import mxnet as mx
from mxnet import gluon
from abc import abstractmethod

from ..constants import INITIAL_COVARIANCE_SCALE, INITIAL_INVERSE_BANDWIDTHS, DEFAULT_ENCODING, INVERSE_BANDWIDTHS_LOWER_BOUND, INVERSE_BANDWIDTHS_UPPER_BOUND, COVARIANCE_SCALE_LOWER_BOUND, COVARIANCE_SCALE_UPPER_BOUND, NUMERICAL_JITTER
from ..distribution import Uniform, LogNormal
from ..gluon_blocks_helpers import encode_unwrap_parameter
from ..mean import MeanFunction
from ..utils import create_encoding, register_parameter

__all__ = ['KernelFunction', 'Matern52']


class KernelFunction(MeanFunction):
    """
    Base class of kernel (or covariance) functions

    """
    def __init__(self, dimension: int, **kwargs):
        """
        :param dimension: Dimensionality of input points after encoding into
            ndarray
        """
        super(KernelFunction, self).__init__(**kwargs)
        self._dimension = dimension

    @property
    def dimension(self):
        """
        :return: Dimension d of input points
        """
        return self._dimension

    @abstractmethod
    def diagonal(self, F, X):
        """
        :param F: mx.sym or mx.nd
        :param X: Input data, shape (n, d)
        :return: Diagonal of K(X, X), shape (n,)
        """
        pass

    @abstractmethod
    def diagonal_depends_on_X(self):
        """
        For stationary kernels, diagonal does not depend on X

        :return: Does diagonal(F, X) depend on X?
        """
        pass

    def _check_input_shape(self, F, X):
        # This fails if the shape is not (*, dimension)
        return F.reshape(X, shape=(0, self._dimension))


class SquaredDistance(gluon.HybridBlock):
    """
    HybridBlock that is responsible for the computation of matrices of squared
    distances. The distances can possibly be weighted (e.g., ARD
    parametrization). For instance:
        X1 with size (n1,d)
        X2 with size (n2,d)
        inverse_bandwidths with size (1,d)
        results in a matrix of size (n1,n2) with i,j entry equal to
            sum_{k=1}^d (X1[i,k] - X2[j,k])^2 * inverse_bandwidths[k]^2

    if ARD == False, inverse_bandwidths is equal to a scalar broadcast to the
    d components (with d=dimension, i.e., the number of features in X)
    otherwise, inverse_bandwidths is (1,d)
    """

    def __init__(self, dimension, ARD=False, encoding_type=DEFAULT_ENCODING, **kwargs):
        super(SquaredDistance, self).__init__(**kwargs)

        self.ARD = ARD
        inverse_bandwidths_dimension = 1 if not ARD else dimension
        self.encoding = create_encoding(
            encoding_type, INITIAL_INVERSE_BANDWIDTHS,
            INVERSE_BANDWIDTHS_LOWER_BOUND, INVERSE_BANDWIDTHS_UPPER_BOUND,
            inverse_bandwidths_dimension,
            Uniform(INVERSE_BANDWIDTHS_LOWER_BOUND,
                    INVERSE_BANDWIDTHS_UPPER_BOUND))
        with self.name_scope():
            self.inverse_bandwidths_internal = register_parameter(
                self.params, 'inverse_bandwidths', self.encoding,
                shape=(inverse_bandwidths_dimension,))

    def hybrid_forward(self, F, X1, X2, inverse_bandwidths_internal):
        """
        Actual computation of the matrix of squared distances (see details above)

        :param F: mx.sym or mx.nd
        :param X1: input data of size (n1,d)
        :param X2: input data of size (n2,d)
        :param inverse_bandwidths_internal: self.inverse_bandwidths_internal
            passed as sym or nd depending on F (natively handled by
            gluon.HybridBlock)
        """

        inverse_bandwidths = self.encoding.get(F, inverse_bandwidths_internal)
        # in case inverse_bandwidths if of size (1,dimension), dimension>1,
        # ARD is handled by broadcasting
        inverse_bandwidths = F.reshape(inverse_bandwidths, shape=(1, -1))

        if X2 is X1:
            X1_scaled = F.broadcast_mul(X1, inverse_bandwidths)
            D = -2.0 * F.linalg.syrk(X1_scaled)
            X1_squared_norm = F.sum(F.square(X1_scaled), axis=1)
            D = F.broadcast_add(D, F.reshape(X1_squared_norm, shape=(1, -1)))
            D = F.broadcast_add(D, F.reshape(X1_squared_norm, shape=(-1, 1)))
        else:
            X1_scaled = F.broadcast_mul(X1, inverse_bandwidths)
            X2_scaled = F.broadcast_mul(X2, inverse_bandwidths)
            X1_squared_norm = F.sum(F.square(X1_scaled), axis=1)
            X2_squared_norm = F.sum(F.square(X2_scaled), axis=1)
            D = -2.0 * F.linalg.gemm2(
                X1_scaled, X2_scaled, transpose_a=False, transpose_b=True)
            D = F.broadcast_add(D, F.reshape(X1_squared_norm, shape=(-1, 1)))
            D = F.broadcast_add(D, F.reshape(X2_squared_norm, shape=(1, -1)))
        return F.abs(D)

    def get_params(self):
        """
        Parameter keys are inv_bw<k> if dimension > 1, and inv_bw if
        dimension == 1.

        """
        inverse_bandwidths = encode_unwrap_parameter(
            mx.nd, self.inverse_bandwidths_internal,
            self.encoding).asnumpy().reshape((-1,))
        if inverse_bandwidths.size == 1:
            return {'inv_bw': inverse_bandwidths[0]}
        else:
            return {
                'inv_bw{}'.format(k): inverse_bandwidths[k]
                for k in range(inverse_bandwidths.size)}

    def set_params(self, param_dict):
        dimension = self.encoding.dimension
        if dimension == 1:
            inverse_bandwidths = [param_dict['inv_bw']]
        else:
            keys = ['inv_bw{}'.format(k) for k in range(dimension)]
            for k in keys:
                assert k in param_dict, \
                    "'{}' not in param_dict = {}".format(k, param_dict)
            inverse_bandwidths = [param_dict[k] for k in keys]
        self.encoding.set(self.inverse_bandwidths_internal, inverse_bandwidths)


class Matern52(KernelFunction):
    """
    HybridBlock that is responsible for the computation of Matern52 kernel
    matrices. For instance:
        X1 with size (n1,d)
        X2 with size (n2,d)
    results in a matrix of size (n1,n2) with i,j entry equal to the
    Matern52 kernel at (X1[i,:], X2[j,:]).

    If ARD == False, inverse_bandwidths is equal to a scalar broadcast to the
    d components (with d=dimension, i.e., the number of features in X)
    otherwise (ARD == True), inverse_bandwidths is (1,d)

    """
    def __init__(self, dimension, ARD=False, encoding_type=DEFAULT_ENCODING,
                 **kwargs):
        super(Matern52, self).__init__(dimension, **kwargs)
        self.encoding = create_encoding(
            encoding_type, INITIAL_COVARIANCE_SCALE,
            COVARIANCE_SCALE_LOWER_BOUND, COVARIANCE_SCALE_UPPER_BOUND, 1,
            LogNormal(0.0, 1.0))
        self.ARD = ARD
        self.squared_distance = SquaredDistance(
            dimension=dimension, ARD=ARD, encoding_type=encoding_type)

        with self.name_scope():
            self.covariance_scale_internal = register_parameter(
                self.params, 'covariance_scale', self.encoding)

    def hybrid_forward(self, F, X1, X2, covariance_scale_internal):
        """
        Actual computation of the Matern52 kernel matrix (see details above)
        See http://www.gaussianprocess.org/gpml/chapters/RW.pdf,
        equation (4.17)

        :param F: mx.sym or mx.nd
        :param X1: input data of size (n1, d)
        :param X2: input data of size (n2, d)
        """

        covariance_scale = self.encoding.get(F, covariance_scale_internal)
        X1 = self._check_input_shape(F, X1)
        if X2 is not X1:
            X2 = self._check_input_shape(F, X2)
        D = self.squared_distance(X1, X2)
        # Using the plain F.sqrt is numerically unstable for D ~ 0
        # (non-differentiability)
        # that's why we add NUMERICAL_JITTER
        B = F.sqrt(5.0 * D + NUMERICAL_JITTER)
        K = F.broadcast_mul(
            (1.0 + B + 5.0 / 3.0 * D) * F.exp(-B), covariance_scale)

        return K

    def _covariance_scale(self, F, X):
        return encode_unwrap_parameter(
            F, self.covariance_scale_internal, self.encoding, X)

    def diagonal(self, F, X):
        X = self._check_input_shape(F, X)
        covariance_scale = self._covariance_scale(F, X)
        covariance_scale_times_ones = F.broadcast_mul(
            F.ones_like(F.slice_axis(X, axis=1, begin=0, end=1)),
            covariance_scale
        )
        return covariance_scale_times_ones.reshape((-1,))

    def diagonal_depends_on_X(self):
        return False

    def param_encoding_pairs(self):
        return [
            (self.covariance_scale_internal, self.encoding),
            (self.squared_distance.inverse_bandwidths_internal,
             self.squared_distance.encoding)
        ]

    def get_covariance_scale(self):
        return self._covariance_scale(mx.nd, None).asscalar()

    def set_covariance_scale(self, covariance_scale):
        self.encoding.set(self.covariance_scale_internal, covariance_scale)

    def get_params(self):
        result = self.squared_distance.get_params()
        result['covariance_scale'] = self.get_covariance_scale()
        return result

    def set_params(self, param_dict):
        self.squared_distance.set_params(param_dict)
        self.set_covariance_scale(param_dict['covariance_scale'])
