import autograd.numpy as anp

from .base import KernelFunction
from ..constants import COVARIANCE_SCALE_LOWER_BOUND, \
    COVARIANCE_SCALE_UPPER_BOUND, DEFAULT_ENCODING
from ..gluon_blocks_helpers import encode_unwrap_parameter, IdentityScalarEncoding
from ..utils import create_encoding, register_parameter

__all__ = ['FabolasKernelFunction']


class FabolasKernelFunction(KernelFunction):
    """
    The kernel function proposed in:

        Klein, A., Falkner, S., Bartels, S., Hennig, P., & Hutter, np. (2016).
        Fast Bayesian Optimization of Machine Learning Hyperparameters
        on Large Datasets, in AISTATS 2017.
        ArXiv:1605.07079 [Cs, Stat]. Retrieved from http://arxiv.org/abs/1605.07079

    Please note this is only one of the components of the factorized kernel
    proposed in the paper. This is the finite-rank ("degenerate") kernel for
    modelling data subset fraction sizes. Defined as:

        k(x, y) = (U phi(x))^T (U phi(y)),  x, y in [0, 1],
        phi(x) = [1, (1 - x)^2]^T,  U = [[u1, u3], [0, u2]] upper triangular,
        u1, u2 > 0.
    """
    def __init__(self, dimension=1, encoding_type=DEFAULT_ENCODING,
                 u1_init=1.0, u3_init=0.0, **kwargs):
        super(FabolasKernelFunction, self).__init__(dimension=dimension, **kwargs)
        self.encoding_u12 = create_encoding(
            encoding_type, u1_init, COVARIANCE_SCALE_LOWER_BOUND,
            COVARIANCE_SCALE_UPPER_BOUND, 1, None)
        # This is not really needed, but param_encoding_pairs needs an encoding
        # for each parameter
        self.encoding_u3 = IdentityScalarEncoding(init_val=u3_init)
        with self.name_scope():
            self.u1_internal = register_parameter(
                self.params, 'u1', self.encoding_u12)
            self.u2_internal = register_parameter(
                self.params, 'u2', self.encoding_u12)
            self.u3_internal = register_parameter(
                self.params, 'u3', self.encoding_u3)

    @staticmethod
    def _compute_factor(x, u1, u2, u3):
        tvec = (1.0 - x) ** 2
        return anp.concatenate(
           [anp.add(anp.multiply(tvec, u3), u1),
            anp.multiply(tvec, u2)], axis=1)

    def forward(self, X1, X2):
        u1_internal = self.u1_internal.data()
        u2_internal = self.u1_internal.data()
        u3_internal = self.u1_internal.data()
        X1 = self._check_input_shape(X1)
        u1 = self.encoding_u12.get(u1_internal)
        u2 = self.encoding_u12.get(u2_internal)
        u3 = self.encoding_u3.get(u3_internal)
        mat1 = self._compute_factor(X1, u1, u2, u3)
        if X2 is X1:
            return anp.dot(mat1, anp.transpose(mat1))
        else:
            X2 = self._check_input_shape(X2)
            mat2 = self._compute_factor(X2, u1, u2, u3)
            return anp.dot(mat1, anp.transpose(mat2))

    def _get_pars(self, X):
        u1 = encode_unwrap_parameter(self.u1_internal, self.encoding_u12, X)
        u2 = encode_unwrap_parameter(self.u2_internal, self.encoding_u12, X)
        u3 = encode_unwrap_parameter(self.u3_internal, self.encoding_u3, X)
        return (u1, u2, u3)

    def diagonal(self, X):
        X = self._check_input_shape(X)
        u1, u2, u3 = self._get_pars(X)
        mat = self._compute_factor(X, u1, u2, u3)
        return anp.sum(mat ** 2, axis=1)

    def diagonal_depends_on_X(self):
        return True

    def param_encoding_pairs(self):
        return [
            (self.u1_internal, self.encoding_u12),
            (self.u2_internal, self.encoding_u12),
            (self.u3_internal, self.encoding_u3)
        ]

    def get_params(self):
        values = list(self._get_pars(None))
        keys = ['u1', 'u2', 'u3']
        return {k: anp.reshape(v, (1,))[0] for k, v in zip(keys, values)}

    def set_params(self, param_dict):
        self.encoding_u12.set(self.u1_internal, param_dict['u1'])
        self.encoding_u12.set(self.u2_internal, param_dict['u2'])
        self.encoding_u3.set(self.u3_internal, param_dict['u3'])
