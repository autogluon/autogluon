import numbers
import autograd.numpy as anp
import numpy as np
from autograd.builtins import isinstance
from autograd.tracer import getval

from .constants import DATA_TYPE
from .gluon import Parameter, Block

__all__ = ['ConstantPositiveVector',
           'PositiveScalarEncoding',
           'IdentityScalarEncoding',
           'LogarithmScalarEncoding',
           'unwrap_parameter',
           'encode_unwrap_parameter']

# === Internal ===

def unwrap_parameter(param_internal, some_arg=None):
    assert isinstance(param_internal, Parameter)
    val = param_internal.data()
    return val


def encode_unwrap_parameter(param_internal, encoding, some_arg=None):
    return encoding.get(unwrap_parameter(param_internal, some_arg))


class ScalarEncodingBase(object):
    """
    ScalarEncodingBase
    ==================

    Base class for encoding and box constraints for Gluon parameter,
    represented as gluon.Parameter. The parameter is with shape (dimension,)
    where dimension is 1 by default.

    An encoding is given as

        param = enc(param_internal), param_internal = dec(param)

    The Gluon parameter represents param_internal, while param is what is
    visible to the outside.

    Here, enc and dec are inverses of each other. enc is used in 'get', dec
    is used in 'set'. Use 'IdentityScalarEncoding' for no encoding (identity).
    NOTE: enc (and dec) must be strictly increasing.

    Box constraints are given by 
    constr_lower_int < constr_upper_int. 
    
    Here, None means no constraint. The constraints apply to param_internal. If both
    are None, param_internal is unconstrained (default).
    
    NOTE: Box constraints are just maintained here, they have to be enforced
    by an optimizer!

    If regularizer is given, it specifies a regularization term for the
    (encoded) parameter which can be added to a criterion function. It is
    evaluated as regularizer(param).

    Typical use cases:
    - Unconstrained optimizer, positive scalar > lower:
      Use PositiveScalarEncoding(lower), box constraints = [None, None]
    - Optimizer supports box constaints [constr_lower, constr_upper]:
      Use IdentityScalarEncoding(constr_lower, constr_upper)
    """
    def __init__(
            self, init_val, constr_lower=None, constr_upper=None,
            regularizer=None, dimension=1):
        
        if constr_lower is not None and constr_upper is not None:
            assert constr_lower < constr_upper
        init_val = self._check_or_set_init_val(
            init_val, constr_lower, constr_upper)
        init_val_int = self.decode(init_val, 'init_val')
        
        if constr_lower is not None:
            assert init_val >= constr_lower
            constr_lower_int = self.decode(constr_lower, 'constr_lower')
        else:
            constr_lower_int = None
            
        if constr_upper is not None:
            assert init_val <= constr_upper
            constr_upper_int = self.decode(constr_upper, 'constr_upper')
        else:
            constr_upper_int = None
            
        self.constraints = (constr_lower, constr_upper)
        self.constraints_internal = (constr_lower_int, constr_upper_int)
        self.init_val_int = init_val_int
        self.regularizer = regularizer
        self.dimension = dimension

    def get(self, param_internal):
        raise NotImplementedError("get must be implemented")

    def set(self, param_internal, param_val):
        assert isinstance(param_internal, Parameter)
        assert param_internal.shape == (self.dimension,)

        if isinstance(param_val, (list, np.ndarray)):
            assert len(param_val) == self.dimension
            assert np.array(param_val).ndim == 1
            val_int_list = [self.decode(val, 'param_val') for val in param_val]
        else:
            assert np.isscalar(param_val) is True
            val_int_list = [self.decode(param_val, 'param_val')] * self.dimension
            
        param_internal.set_data(anp.array(val_int_list))

    def decode(self, val, name):
        raise NotImplementedError("decode to be implemented in subclass")

    def box_constraints_internal(self, param_internal):
        assert isinstance(param_internal, Parameter)
        assert param_internal.shape == (self.dimension,)
        return {param_internal.name: self.constraints_internal}

    def box_constraints(self):
        return self.constraints

    @staticmethod
    def _check_or_set_init_val(init_val, constr_lower, constr_upper):
    # Check constraints and init value
        if init_val is not None:
            if constr_upper is not None:
                assert init_val <= constr_upper
            if constr_lower is not None:
                assert constr_lower <= init_val
        else:
            # Just pick some value which is feasible
            if constr_lower is not None:
                if constr_upper is not None:
                    init_val = 0.5 * (constr_upper + constr_lower)
                else:
                    init_val = 1.1 * constr_lower
            else:
                if constr_upper is not None:
                    init_val = 0.9 * constr_upper
                else:
                    init_val = 1.0
        return init_val


class IdentityScalarEncoding(ScalarEncodingBase):
    """
    IdentityScalarEncoding
    ======================

    Identity encoding for scalar and vector:

        param = param_internal

    This does not ensure that param is positive! Use this only if positivity
    is otherwise guaranteed.
    """
    def __init__(
            self, constr_lower=None, constr_upper=None, init_val=None,
            regularizer=None, dimension=1):
        super(IdentityScalarEncoding, self).__init__(
            init_val, constr_lower=constr_lower, constr_upper=constr_upper,
            regularizer=regularizer, dimension=dimension)

    def get(self, param_internal):
        return param_internal

    def decode(self, val, name):
        return val


class LogarithmScalarEncoding(ScalarEncodingBase):
    """
    LogarithmScalarEncoding
    =======================

    Logarithmic encoding for scalar and vector:

        param = exp(param_internal)
        param_internal = param
    """
    def __init__(
            self, constr_lower=None, constr_upper=None, init_val=None,
            regularizer=None, dimension=1):
        super(LogarithmScalarEncoding, self).__init__(
            init_val, constr_lower=constr_lower, constr_upper=constr_upper,
            regularizer=regularizer, dimension=dimension)

    def get(self, param_internal):
        return anp.exp(param_internal)

    def decode(self, val, name):
        assert val > 0.0, '{} = {} must be positive'.format(name, val)
        return anp.log(anp.maximum(val, 1e-15))
        
        
class PositiveScalarEncoding(ScalarEncodingBase):
    """
    PositiveScalarEncoding
    ======================

    Provides encoding for positive scalar and vector: param > lower.
    Here, param is represented as gluon.Parameter. The param
    is with shape (dimension,) where dimension is 1 by default.

    The encoding is given as:

        param = softrelu(param_internal) + lower,
        softrelu(x) = log(1 + exp(x))

    If constr_upper is used, the constraint

        param_internal < dec(constr_upper)

    can be enforced by an optimizer. Since dec is increasing, this translates
    to param < constr_upper.
    NOTE: While lower is enforced by the encoding, the upper bound is not, has
    to be enforced by an optimizer.
    """
    def __init__(
            self, lower, constr_upper=None, init_val=None, regularizer=None, dimension=1):
        assert isinstance(lower, numbers.Real) and lower >= 0.0
        # lower should be a real number
        self.lower = lower
        super(PositiveScalarEncoding, self).__init__(
            init_val, constr_lower=None, constr_upper=constr_upper,
            regularizer=regularizer, dimension=dimension)

    def get(self, param_internal):
        return anp.log1p(anp.exp(param_internal)) + self.lower
        
    def decode(self, val, name):
        assert val > self.lower, '{} = {} must be > self.lower = {}'.format(
                name, val, self.lower)
        # Inverse of encoding: Careful with numerics:
        # val_int = log(exp(arg) - 1) = arg + log(1 - exp(-arg))
        #         = arg + log1p(-exp(-arg))
        arg = val - self.lower
        return arg + anp.log1p(-anp.exp(-arg))


class init_Constant(object):
    def __init__(self, val):
        self.val = val
    def __call__(self, shape = (1,)):
        return anp.ones(shape)*self.val


class ConstantPositiveVector(Block):
    """
    ConstantPositiveVector
    
    # Just constant positive vectors...
    ======================

    Represents constant vector, with positive entry value represented as Gluon
    parameter, to be used in the context of wrapper classes in
    gluon_blocks.py. Shape, dtype, and context are determined from the
    features argument:

    - If features.shape = (n, d):
       shape = (d, 1) if size_cols = True (number cols of features)
       shape = (n, 1) if size_cols = False (number rows of features)
    - dtype = features.dtype, ctx = features.ctx

    Encoding and internal Gluon parameter:
    The positive scalar parameter is encoded via encoding (see
    ScalarEncodingBase). The internal Gluon parameter (before encoding) has the
    name param_name + '_internal'.
    """

    def __init__(self, param_name, encoding, size_cols, **kwargs):
        super(ConstantPositiveVector, self).__init__(**kwargs)
        assert isinstance(encoding, ScalarEncodingBase)
        self.param_name = param_name
        self.encoding = encoding
        self.size_cols = size_cols
        with self.name_scope():
            init_val_int = encoding.init_val_int
            # Note: The initialization values are bogus!
            self.param_internal = self.params.get(
                param_name + '_internal',
                init=init_Constant(init_val_int),
                shape=(1,), dtype=DATA_TYPE)

    def forward(self, features, param_internal):
        """Returns constant positive vector

        If features.shape = (n, d), the shape of the vector returned is
        (d, 1) if size_cols = True, (n, 1) otherwise.

        :param features: Matrix for shape, dtype, ctx
        :param param_internal: Unwrapped parameter
        :return: Constant positive vector
        """
        # Shape, dtype, ctx is determined by extracting column or row from
        # features, then use ones_like
        axis = 0 if self.size_cols else 1
        ones_vec = anp.ones((features.size//getval(features.shape)[axis], 1))
        param = anp.reshape(self.encoding.get(param_internal), (1, 1))
        return anp.multiply(ones_vec, param)
        # returned (para_internal, ..., para_interval)

    def set(self, val):
        self.encoding.set(self.param_internal, val)

    def get(self):
        param_internal = unwrap_parameter(
              self.param_internal, None)
        return self.encoding.get(param_internal)

    def get_box_constraints_internal(self):
        return self.encoding.box_constraints_internal(
            self.param_internal)

    def log_parameters(self):
        return '{} = {}'.format(self.param_name, self.get())

    def get_parameters(self):
        return {self.param_name: self.get()}

    def switch_updating(self, flag):
        """Is the underlying parameter updated during learning?

        By default, the parameter takes part in learning (its grad_req
        attribute is 'write'). For flag == False, the attribute is
        flipped to 'null', and the parameter remains constant during
        learning.

        :param flag: Update parameter during learning?
        """
        grad_req = 'write' if flag else 'null'
        self.param_internal.grad_req = grad_req

    def has_regularizer(self):
        return (self.encoding.regularizer is not None)

    def eval_regularizer(self, features):
        if self.has_regularizer():
            param_internal = unwrap_parameter(
                self.param_internal, features)
            param = self.encoding.get(param_internal)
            return self.encoding.regularizer(param)
        else:
            return 0.0
