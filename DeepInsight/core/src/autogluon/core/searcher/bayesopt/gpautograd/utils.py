from autograd.tracer import getval
from autograd.builtins import isinstance

from .constants import DATA_TYPE
from .gluon import Parameter
from .gluon_blocks_helpers import LogarithmScalarEncoding, PositiveScalarEncoding, init_Constant


def param_to_pretty_string(gluon_param, encoding):
    """
    Take a gluon parameter and transform it to a string amenable to plotting
    If need be, the gluon parameter is appropriately encoded (e.g., log-exp transform).

    :param gluon_param: gluon parameter
    :param encoding: object in charge of encoding/decoding the gluon_param
    """
    assert isinstance(gluon_param, Parameter)
    assert encoding is not None, "encoding of param {} should not be None".format(gluon_param.name)
    param_as_numpy = encoding.get(getval(gluon_param.data()))
    
    return "{}: {}".format(
        gluon_param.name, ";".join(
            "{:.6f}".format(value) for value in param_as_numpy))


def create_encoding(
        encoding_name, init_val, constr_lower, constr_upper, dimension, prior):
    assert encoding_name in ['logarithm', 'positive'], "encoding name can only be 'logarithm' or 'positive'"

    if encoding_name == 'logarithm':
        return LogarithmScalarEncoding(init_val=init_val,
                                       constr_lower=constr_lower,
                                       constr_upper=constr_upper,
                                       dimension=dimension,
                                       regularizer=prior)
    else:
        return PositiveScalarEncoding(lower=constr_lower,
                                      init_val = init_val,
                                      constr_upper=constr_upper,
                                      dimension=dimension,
                                      regularizer=prior)


PARAMETER_POSTFIX = '_internal'

def get_name_internal(name):
    return name + PARAMETER_POSTFIX


def register_parameter(
        params, name, encoding, shape=(1,), dtype=DATA_TYPE):
    return params.get(
        get_name_internal(name), shape=shape,
        init=init_Constant(encoding.init_val_int), dtype=dtype)
