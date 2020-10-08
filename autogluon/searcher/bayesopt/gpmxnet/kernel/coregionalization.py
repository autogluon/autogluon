import mxnet as mx

from .base import KernelFunction
from ..constants import INITIAL_NOISE_VARIANCE, NOISE_VARIANCE_LOWER_BOUND, NOISE_VARIANCE_UPPER_BOUND, DATA_TYPE, DEFAULT_ENCODING
from ..gluon_blocks_helpers import unwrap_parameter, IdentityScalarEncoding
from ..utils import create_encoding

__all__ = ['Coregionalization']


class Coregionalization(KernelFunction):
    """
    k(i, j) = K_{ij}, where K = W W^T + diag(rho).
    """
    def __init__(self, num_outputs, num_factors=16,
                 rho_init=INITIAL_NOISE_VARIANCE,
                 encoding_type=DEFAULT_ENCODING, **kwargs):

        super(Coregionalization, self).__init__(dimension=1, **kwargs)

        self.encoding_W_flat = IdentityScalarEncoding(
            dimension=num_outputs * num_factors)
        self.encoding_rho = create_encoding(encoding_type, rho_init,
                                            NOISE_VARIANCE_LOWER_BOUND,
                                            NOISE_VARIANCE_UPPER_BOUND,
                                            dimension=1)

        self.num_outputs = num_outputs
        self.num_factors = num_factors

        with self.name_scope():
            self.W_flat_internal = self.params.get(
                "W_internal", shape=(num_outputs * num_factors,),
                init=mx.init.Normal(),  # TODO: Use Xavier initialization here
                dtype=DATA_TYPE)
            self.rho_internal = self.params.get(
                "rho_internal", shape=(1,),
                init=mx.init.Constant(self.encoding_rho.init_val_int),
                dtype=DATA_TYPE)

    @staticmethod
    def _meshgrid(F, a, b):
        """
        Return coordinate matrices from coordinate vectors.

        Like https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
        (with Cartesian indexing), but only supports two coordinate vectors as input.

        :param a: 1-D array representing the coordinates of a grid (length n) 
        :param b: 1-D array representing the coordinates of a grid (length m) 
        :return: coordinate matrix. 3-D array of shape (2, m, n).
        """
        aa = F.broadcast_mul(F.ones_like(F.expand_dims(a, axis=-1)), b)
        bb = F.broadcast_mul(F.ones_like(F.expand_dims(b, axis=-1)), a)
        return F.stack(bb, F.transpose(aa), axis=0)

    def _compute_gram_matrix(self, F, W_flat, rho):
        W = F.reshape(W_flat, shape=(self.num_outputs, self.num_factors))
        rho_vec = F.broadcast_mul(rho, F.ones(self.num_outputs, dtype=DATA_TYPE))
        return F.linalg.syrk(W) + F.diag(rho_vec)

    def hybrid_forward(self, F, ind1, ind2, W_flat_internal, rho_internal):
        W_flat = self.encoding_W_flat.get(F, W_flat_internal)
        rho = self.encoding_rho.get(F, rho_internal)
        K = self._compute_gram_matrix(F, W_flat, rho)
        ind1 = self._check_input_shape(F, ind1)
        if ind2 is not ind1:
            ind2 = self._check_input_shape(F, ind2)
        ind = self._meshgrid(F, ind1, ind2)
        return F.transpose(F.squeeze(F.gather_nd(K, ind)))

    def diagonal(self, F, ind):
        ind = self._check_input_shape(F, ind)
        W_flat = self.encoding_W_flat.get(F, unwrap_parameter(F, self.W_flat_internal, ind))
        rho = self.encoding_rho.get(F, unwrap_parameter(F, self.rho_internal, ind))
        K = self._compute_gram_matrix(F, W_flat, rho)
        K_diag = F.diag(K)
        return F.take(K_diag, ind)

    def diagonal_depends_on_X(self):
        return True

    def param_encoding_pairs(self):
        return [
            (self.W_flat_internal, self.encoding_W_flat),
            (self.rho_internal, self.encoding_rho),
        ]
