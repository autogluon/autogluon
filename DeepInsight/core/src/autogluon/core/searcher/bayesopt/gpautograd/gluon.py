"""Gluon APIs for autograd"""
import threading
import warnings
import re
from collections import OrderedDict
import autograd.numpy as anp
from autograd.builtins import isinstance

__all__ = ['Block',
           'Parameter',
           'ParameterDict']

def _indent(s_, numSpaces):
    """Indent string
    """
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [first] + [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    return s

def shape_is_known(shape):
    """Check whether a shape is completely known with or without np semantics.
    Please see the doc of is_np_shape for more details.
    """
    if shape is None:
        return False
    unknown_dim_size = -1
    if len(shape) == 0:
        return unknown_dim_size == -1
    for dim_size in shape:
        if dim_size == unknown_dim_size:
            return False
        assert dim_size > unknown_dim_size, "shape dimension size cannot be less than {}, while " \
                                            "received {}".format(unknown_dim_size, dim_size)
    return True


class Parameter(object):
    """A Container holding parameters (weights) of Blocks.
    :py:class:`Parameter` holds a copy of the parameter on each :py:class:`Context` after
    it is initialized with ``Parameter.initialize(...)``. If :py:attr:`grad_req` is
    not ``'null'``, it will also hold a gradient array on each :py:class:`Context`::
        x = np.zeros((16, 100))
        w = Parameter('fc_weight', shape=(16, 100), init=np.random.uniform)
        w.initialize()
        b.initialize()
        z = x + w.data
    Parameters
    ----------
    name : str
        Name of this parameter.
    grad_req : {'write', 'add', 'null'}, default 'write'
        Specifies how to update gradient to grad arrays.
        - ``'write'`` means everytime gradient is written to grad :py:class:`NDArray`.
        - ``'add'`` means everytime gradient is added to the grad :py:class:`NDArray`. You need
          to manually call ``zero_grad()`` to clear the gradient buffer before each
          iteration when using this option.
        - 'null' means gradient is not requested for this parameter. gradient arrays
          will not be allocated.
    shape : int or tuple of int, default None
        Shape of this parameter. By default shape is not specified. Parameter with
        unknown shape can be used for :py:class:`Symbol` API, but ``init`` will throw an error
        when using :py:class:`NDArray` API.
    dtype : numpy.dtype or str, default 'float64'
        Data type of this parameter. For example, ``numpy.float64`` or ``'float64'``.
    lr_mult : float, default 1.0
        Learning rate multiplier. Learning rate will be multiplied by lr_mult
        when updating this parameter with optimizer.
    wd_mult : float, default 1.0
        Weight decay multiplier (L2 regularizer coefficient). Works similar to lr_mult.
    init : Initializer, default None
        Initializer of this parameter. Will use the global initializer by default.
    stype: {'default', 'row_sparse', 'csr'}, defaults to 'default'.
        The storage type of the parameter.
    grad_stype: {'default', 'row_sparse', 'csr'}, defaults to 'default'.
        The storage type of the parameter's gradient.
    Attributes
    ----------
    grad_req : {'write', 'add', 'null'}
        This can be set before or after initialization. Setting ``grad_req`` to ``'null'``
        with ``x.grad_req = 'null'`` saves memory and computation when you don't
        need gradient w.r.t x.
    lr_mult : float
        Local learning rate multiplier for this Parameter. The actual learning rate
        is calculated with ``learning_rate * lr_mult``. You can set it with
        ``param.lr_mult = 2.0``
    wd_mult : float
        Local weight decay multiplier for this Parameter.
    """
    def __init__(self, name, grad_req='write', shape=None, dtype=anp.float64,
                 lr_mult=1.0, wd_mult=1.0, init=None, allow_deferred_init=False,
                 differentiable=True, stype='default', grad_stype='default'):
        self._var = None
        self._data = None
        self._grad = None
        self._ctx_list = None
        self._ctx_map = None
        self._trainer = None
        self._deferred_init = ()
        self._differentiable = differentiable
        if allow_deferred_init:
            raise NotImplementedError('allow_deferred_init is not a valid option in autograd')
        self._allow_deferred_init = allow_deferred_init
        self._grad_req = None
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self.name = name
        self._dtype = dtype
        self.lr_mult = lr_mult
        self.wd_mult = wd_mult
        self.grad_req = grad_req
        self.init = init
        # sparse related storage type information
        valid_stypes = ['default']
        assert grad_stype in valid_stypes, "grad_stype for Parameter '%s' must be " \
            "one of 'default', 'row_sparse', or 'csr', but got '%s'" % (name, grad_stype)
        assert stype in valid_stypes, "stype for Parameter '%s' must be " \
            "one of 'default', 'row_sparse', or 'csr', but got '%s'" % (name, stype)
        self._grad_stype = grad_stype
        self._stype = stype

    def __repr__(self):
        s = 'Parameter {name} (shape={shape}, dtype={dtype})'
        return s.format(name=self.name, shape=self.shape, dtype=self.dtype)

    @property
    def grad_req(self):
        return self._grad_req

    @grad_req.setter
    def grad_req(self, req):
        assert req in ['write', 'add', 'null'], \
            "grad_req must be one of 'write', 'add', or 'null', but got '%s'"%req
        if not self._differentiable:
            req = 'null'
        if self._grad_req == req:
            return
        self._grad_req = req
        if req == 'null' and self._grad is not None:
            self._grad = None
            self._data = [i.detach() for i in self._data]
        elif self._data is not None:
            self._init_grad()

    @property
    def dtype(self):
        """The type of the parameter.
        Setting the dtype value is equivalent to casting the value of the parameter
        """
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self.cast(dtype)

    @property
    def shape(self):
        """The shape of the parameter.
        By default, an unknown dimension size is 0. However, when the NumPy semantic
        is turned on, unknown dimension size is -1.
        """
        if self._shape is None:
            return None
        else:
            # Parameters shouldn't be zero-size. If one of its dimension is 0,
            # it means the parameter isn't initialized. In the NumPy semantics,
            # the unknown dimension should be marked with -1.
            return tuple(i if i != 0 else -1 for i in self._shape)

    @shape.setter
    def shape(self, new_shape):
        if self._shape is None:
            self._shape = new_shape
            return

        assert len(self._shape) == len(new_shape) and \
            all(j in (-1, 0, i) for i, j in zip(new_shape, self._shape)), \
            "Expected shape %s is incompatible with given shape %s."%(
                str(new_shape), str(self._shape))  # -1 means unknown dim size in np_shape mode

        self._shape = new_shape

    def _check_and_get(self, arr_list, ctx):
        if arr_list is not None:
            if ctx is list:
                return arr_list
            if ctx is None:
                if len(arr_list) == 1:
                    return arr_list[0]
                else:
                    ctx = context.current_context()
            ctx_list = self._ctx_map[ctx.device_typeid&1]
            if ctx.device_id < len(ctx_list):
                idx = ctx_list[ctx.device_id]
                if idx is not None:
                    return arr_list[idx]
            raise RuntimeError(
                "Parameter '%s' was not initialized on context %s. "
                "It was only initialized on %s."%(
                    self.name, str(ctx), str(self._ctx_list)))
        if self._deferred_init:
            raise NotImplementedError('Cannot enable deferred init')
        raise RuntimeError(
            "Parameter '%s' has not been initialized. Note that " \
            "you should initialize parameters and create Trainer " \
            "with Block.collect_params() instead of Block.params " \
            "because the later does not include Parameters of " \
            "nested child Blocks"%(self.name))

    def _init_impl(self, data, ctx_list=None):
        """Sets data and grad."""
        self._data = [data]
        self._init_grad()

    def _init_grad(self):
        """Initialize grad buffers."""
        if self.grad_req == 'null':
            self._grad = None
            return

        if self._grad_stype != 'default':
            raise ValueError("numpy.zeros does not support stype = {}"
                             .format(self._grad_stype))
        self._grad = [anp.zeros(shape=i.shape, dtype=i.dtype)
                      for i in self._data]

        # autograd.mark_variables(self._check_and_get(self._data, list),
        #                         self._grad, self.grad_req)

    def initialize(self, init=None, ctx=None, default_init=anp.random.uniform,
                   force_reinit=False):
        """Initializes parameter and gradient arrays. Only used for :py:class:`NDArray` API.
        Parameters
        ----------
        init : Initializer
            The initializer to use. Overrides :py:meth:`Parameter.init` and default_init.
        ctx : Context or list of Context, defaults to :py:meth:`context.current_context()`.
            Initialize Parameter on given context. If ctx is a list of Context, a
            copy will be made for each context.
            .. note::
                Copies are independent arrays. User is responsible for keeping
                their values consistent when updating.
                Normally :py:class:`gluon.Trainer` does this for you.
        default_init : Initializer
            Default initializer is used when both :py:func:`init`
            and :py:meth:`Parameter.init` are ``None``.
        force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.
        Examples
        --------
        >>> weight = mx.gluon.Parameter('weight', shape=(2, 2))
        >>> weight.initialize(ctx=mx.cpu(0))
        >>> weight.data()
        [[-0.01068833  0.01729892]
         [ 0.02042518 -0.01618656]]
        <NDArray 2x2 @cpu(0)>
        >>> weight.grad()
        [[ 0.  0.]
         [ 0.  0.]]
        <NDArray 2x2 @cpu(0)>
        >>> weight.initialize(ctx=[mx.gpu(0), mx.gpu(1)])
        >>> weight.data(mx.gpu(0))
        [[-0.00873779 -0.02834515]
         [ 0.05484822 -0.06206018]]
        <NDArray 2x2 @gpu(0)>
        >>> weight.data(mx.gpu(1))
        [[-0.00873779 -0.02834515]
         [ 0.05484822 -0.06206018]]
        <NDArray 2x2 @gpu(1)>
        """
        if self._data is not None and not force_reinit:
            warnings.warn("Parameter '%s' is already initialized, ignoring. " \
                          "Set force_reinit=True to re-initialize."%self.name,
                          stacklevel=2)
            return
        self._data = self._grad = None

        if init is None:
            init = default_init if self.init is None else self.init
        if not shape_is_known(self.shape):
            if self._allow_deferred_init:
                raise NotImplementedError('deferred_init not implemented for autograd')
                return
            raise ValueError("Cannot initialize Parameter '%s' because it has " \
                             "invalid shape: %s."%(self.name, str(self.shape)))

        try:
            data = init(shape=self.shape)
        except TypeError:
            data = init(size=self.shape)
        self._init_impl(data, ctx_list=ctx)

    def reset_ctx(self, ctx):
        """Re-assign Parameter to other contexts.
        Parameters
        ----------
        ctx : Context or list of Context, default ``context.current_context()``.
            Assign Parameter to given context. If ctx is a list of Context, a
            copy will be made for each context.
        """
        return

    def set_data(self, data):
        """Sets this parameter's value on all contexts."""
        self.shape = data.shape

        if self._data is None:
            assert self._deferred_init, \
                "Parameter '%s' has not been initialized"%self.name
            self._deferred_init = self._deferred_init[:3] + (data,)
            return
        
        # self._check_and_get(self._data, list)
        # added, raise no initialization error
#       for arr in self._check_and_get(self._data, list):
#           arr[:] = data
        for i in range(len(self._data)):
            self._data[i] = anp.array(data, copy=True)

    def data(self, ctx=None):
        """Returns a copy of this parameter on one context. Must have been
        initialized on this context before. For sparse parameters, use
        :py:meth:`Parameter.row_sparse_data` instead.
        Parameters
        ----------
        ctx : Context
            Desired context.
        Returns
        -------
        NDArray on ctx
        """
        if self._stype != 'default':
            raise RuntimeError("Cannot return a copy of Parameter '%s' on ctx %s via data() " \
                               "because its storage type is %s. Please use row_sparse_data() " \
                               "instead." % (self.name, str(ctx), self._stype))
        return self._check_and_get(self._data, ctx)

    def list_data(self):
        """Returns copies of this parameter on all contexts, in the same order
        as creation. For sparse parameters, use :py:meth:`Parameter.list_row_sparse_data`
        instead.
        Returns
        -------
        list of NDArrays
        """
        if self._stype != 'default':
            raise RuntimeError("Cannot return copies of Parameter '%s' on all contexts via " \
                               "list_data() because its storage type is %s. Please use " \
                               "row_sparse_data() instead." % (self.name, self._stype))
        return self._check_and_get(self._data, list)

    def grad(self, ctx=None):
        """Returns a gradient buffer for this parameter on one context.
        Parameters
        ----------
        ctx : Context
            Desired context.
        """
        if self._data is not None and self._grad is None:
            raise RuntimeError(
                "Cannot get gradient array for Parameter '%s' " \
                "because grad_req='null'"%(self.name))
        return self._check_and_get(self._grad, ctx)

    def list_grad(self):
        """Returns gradient buffers on all contexts, in the same order
        as :py:meth:`values`."""
        if self._data is not None and self._grad is None:
            raise RuntimeError(
                "Cannot get gradient array for Parameter '%s' " \
                "because grad_req='null'"%(self.name))
        return self._check_and_get(self._grad, list)

    def list_ctx(self):
        """Returns a list of contexts this parameter is initialized on."""
        if self._data is None:
            if self._deferred_init:
                return self._deferred_init[1]
            raise RuntimeError("Parameter '%s' has not been initialized"%self.name)
        return self._ctx_list

    def zero_grad(self):
        """Sets gradient buffer on all contexts to 0. No action is taken if
        parameter is uninitialized or doesn't require gradient."""
        if self._grad is None:
            return
        for i in self._grad:
            i[:] = 0

    def cast(self, dtype):
        """Cast data and gradient of this Parameter to a new data type.
        Parameters
        ----------
        dtype : str or numpy.dtype
            The new data type.
        """
        self._dtype = dtype
        if self._data is None:
            return

        self._data = [i.astype(dtype) for i in self._data]
        if self._grad is None:
            return
        self._grad = [i.astype(dtype) for i in self._grad]


class ParameterDict(object):
    """A dictionary managing a set of parameters.
    Parameters
    ----------
    prefix : str, default ``''``
        The prefix to be prepended to all Parameters' names created by this dict.
    shared : ParameterDict or None
        If not ``None``, when this dict's :py:meth:`get` method creates a new parameter, will
        first try to retrieve it from "shared" dict. Usually used for sharing
        parameters with another Block.
    """
    def __init__(self, prefix='', shared=None):
        self._prefix = prefix
        self._params = OrderedDict()
        self._shared = shared

    def __repr__(self):
        s = '{name}(\n{content}\n)'
        name = self._prefix+' ' if self._prefix else ''
        return s.format(name=name,
                        content='\n'.join([_indent('  {0}'.format(v), 2)
                                           for v in self.values()]))

    def __getitem__(self, key):
        return self._params[key]

    def __iter__(self):
        return iter(self._params)

    def items(self):
        return self._params.items()

    def keys(self):
        return self._params.keys()

    def values(self):
        return self._params.values()

    @property
    def prefix(self):
        """Prefix of this dict. It will be prepended to :py:class:`Parameter`s' name created
        with :py:func:`get`."""
        return self._prefix

    def _get_impl(self, name):
        if name in self._params:
            return self._params[name]
        if self._shared is not None and name in self._shared._params:
            self._params[name] = self._shared._params[name]
            return self._shared._params[name]
        return None

    def get(self, name, **kwargs):
        """Retrieves a :py:class:`Parameter` with name ``self.prefix+name``. If not found,
        :py:func:`get` will first try to retrieve it from "shared" dict. If still not
        found, :py:func:`get` will create a new :py:class:`Parameter` with key-word arguments and
        insert it to self.
        Parameters
        ----------
        name : str
            Name of the desired Parameter. It will be prepended with this dictionary's
            prefix.
        **kwargs : dict
            The rest of key-word arguments for the created :py:class:`Parameter`.
        Returns
        -------
        Parameter
            The created or retrieved :py:class:`Parameter`.
        """
        name = self.prefix + name
        param = self._get_impl(name)
        if param is None: # pylint: disable=too-many-nested-blocks
            param = Parameter(name, **kwargs)
            self._params[name] = param
        else:
            for k, v in kwargs.items():
                if hasattr(param, k) and getattr(param, k) is not None:
                    existing = getattr(param, k)
                    if k == 'shape' and len(v) == len(existing):
                        inferred_shape = []
                        matched = True
                        for dim1, dim2 in zip(v, existing):
                            if dim1 != dim2 and dim1 > 0 and dim2 > 0:
                                matched = False
                                break
                            elif dim1 == dim2:
                                inferred_shape.append(dim1)
                            elif dim1 in (0, -1):  # -1 means unknown dim size in np_shape mode
                                inferred_shape.append(dim2)
                            else:
                                inferred_shape.append(dim1)

                        if matched:
                            param._shape = tuple(inferred_shape)
                            continue
                    elif k == 'dtype' and anp.dtype(v) == anp.dtype(existing):
                        continue

                    assert v is None or v == existing, \
                        "Cannot retrieve Parameter '%s' because desired attribute " \
                        "does not match with stored for attribute '%s': " \
                        "desired '%s' vs stored '%s'."%(
                            name, k, str(v), str(getattr(param, k)))
                else:
                    setattr(param, k, v)
        return param

    def update(self, other):
        """Copies all Parameters in ``other`` to self."""
        for k, v in other.items():
            if k in self._params:
                assert self._params[k] is v, \
                    "Cannot update self with other because they have different " \
                    "Parameters with the same name '%s'"%k

        for k, v in other.items():
            self._params[k] = v

    def initialize(self, init=anp.random.uniform, ctx=None, verbose=False,
                   force_reinit=False):
        """Initializes all Parameters managed by this dictionary to be used for :py:class:`NDArray`
        API. It has no effect when using :py:class:`Symbol` API.
        Parameters
        ----------
        init : Initializer
            Global default Initializer to be used when :py:meth:`Parameter.init` is ``None``.
            Otherwise, :py:meth:`Parameter.init` takes precedence.
        ctx : Context or list of Context
            Keeps a copy of Parameters on one or many context(s).
        verbose : bool, default False
            Whether to verbosely print out details on initialization.
        force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.
        """
        if verbose:
            init.set_verbosity(verbose=verbose)
        for _, v in self.items():
            v.initialize(None, ctx, init, force_reinit=force_reinit)

    def reset_ctx(self, ctx):
        """Re-assign all Parameters to other contexts.
        Parameters
        ----------
        ctx : Context or list of Context, default :py:meth:`context.current_context()`.
            Assign Parameter to given context. If ctx is a list of Context, a
            copy will be made for each context.
        """
        for i in self.values():
            i.reset_ctx(ctx)

    def list_ctx(self):
        """Returns a list of all the contexts on which the underlying Parameters
        are initialized."""
        s = set()
        for i in self.values():
            s.update(i.list_ctx())
        return list(s)

    def setattr(self, name, value):
        """Set an attribute to a new value for all Parameters.
        For example, set grad_req to null if you don't need gradient w.r.t a
        model's Parameters::
            model.collect_params().setattr('grad_req', 'null')
        or change the learning rate multiplier::
            model.collect_params().setattr('lr_mult', 0.5)
        Parameters
        ----------
        name : str
            Name of the attribute.
        value : valid type for attribute name
            The new value for the attribute.
        """
        for i in self.values():
            setattr(i, name, value)


class NameManager(object):
    """NameManager to do automatic naming.
    Developers can also inherit from this class to change naming behavior.
    """
    _current = threading.local()

    def __init__(self):
        self._counter = {}
        self._old_manager = None

    def get(self, name, hint):
        """Get the canonical name for a symbol.
        This is the default implementation.
        If the user specifies a name,
        the user-specified name will be used.
        When user does not specify a name, we automatically generate a
        name based on the hint string.
        Parameters
        ----------
        name : str or None
            The name specified by the user.
        hint : str
            A hint string, which can be used to generate name.
        Returns
        -------
        full_name : str
            A canonical name for the symbol.
        """
        if name:
            return name
        if hint not in self._counter:
            self._counter[hint] = 0
        name = '%s%d' % (hint, self._counter[hint])
        self._counter[hint] += 1
        return name

    def __enter__(self):
        if not hasattr(NameManager._current, "value"):
            NameManager._current.value = NameManager()
        self._old_manager = NameManager._current.value
        NameManager._current.value = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_manager
        NameManager._current.value = self._old_manager

class Prefix(NameManager):
    """A name manager that attaches a prefix to all names.
    Examples
    --------
    >>> import mxnet as mx
    >>> data = mx.symbol.Variable('data')
    >>> with mx.name.Prefix('mynet_'):
            net = mx.symbol.FullyConnected(data, num_hidden=10, name='fc1')
    >>> net.list_arguments()
    ['data', 'mynet_fc1_weight', 'mynet_fc1_bias']
    """
    def __init__(self, prefix):
        super(Prefix, self).__init__()
        self._prefix = prefix

    def get(self, name, hint):
        name = super(Prefix, self).get(name, hint)
        return self._prefix + name

# initialize the default name manager
NameManager._current.value = NameManager()


class _BlockScope(object):
    """Scope for collecting child `Block` s."""
    _current = threading.local()

    def __init__(self, block):
        self._block = block
        self._counter = {}
        self._old_scope = None
        self._name_scope = None

    @staticmethod
    def create(prefix, params, hint):
        """Creates prefix and params for new `Block`."""
        current = getattr(_BlockScope._current, "value", None)
        if current is None:
            if prefix is None:
                if not hasattr(NameManager._current, "value"):
                    NameManager._current.value = NameManager()
                prefix = NameManager._current.value.get(None, hint) + '_'
            if params is None:
                params = ParameterDict(prefix)
            else:
                params = ParameterDict(params.prefix, params)
            return prefix, params

        if prefix is None:
            count = current._counter.get(hint, 0)
            prefix = '%s%d_'%(hint, count)
            current._counter[hint] = count + 1
        if params is None:
            parent = current._block.params
            params = ParameterDict(parent.prefix+prefix, parent._shared)
        else:
            params = ParameterDict(params.prefix, params)
        return current._block.prefix+prefix, params

    def __enter__(self):
        if self._block._empty_prefix:
            return self
        self._old_scope = getattr(_BlockScope._current, "value", None)
        _BlockScope._current.value = self
        self._name_scope = Prefix(self._block.prefix)
        self._name_scope.__enter__()
        return self

    def __exit__(self, ptype, value, trace):
        if self._block._empty_prefix:
            return
        self._name_scope.__exit__(ptype, value, trace)
        self._name_scope = None
        _BlockScope._current.value = self._old_scope


class Block(object):
    """Base class for all neural network layers and models. Your models should
    subclass this class.
    :py:class:`Block` can be nested recursively in a tree structure. You can create and
    assign child :py:class:`Block` as regular attributes::
        from mxnet.gluon import Block, nn
        from mxnet import ndarray as F
        class Model(Block):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
                # use name_scope to give child Blocks appropriate names.
                with self.name_scope():
                    self.dense0 = nn.Dense(20)
                    self.dense1 = nn.Dense(20)
            def forward(self, x):
                x = F.relu(self.dense0(x))
                return F.relu(self.dense1(x))
        model = Model()
        model.initialize(ctx=mx.cpu(0))
        model(F.zeros((10, 10), ctx=mx.cpu(0)))
    Child :py:class:`Block` assigned this way will be registered and :py:meth:`collect_params`
    will collect their Parameters recursively. You can also manually register
    child blocks with :py:meth:`register_child`.
    Parameters
    ----------
    prefix : str
        Prefix acts like a name space. All children blocks created in parent block's
        :py:meth:`name_scope` will have parent block's prefix in their name.
        Please refer to
        `naming tutorial </api/python/docs/tutorials/packages/gluon/blocks/naming.html>`_
        for more info on prefix and naming.
    params : ParameterDict or None
        :py:class:`ParameterDict` for sharing weights with the new :py:class:`Block`. For example,
        if you want ``dense1`` to share ``dense0``'s weights, you can do::
            dense0 = nn.Dense(20)
            dense1 = nn.Dense(20, params=dense0.collect_params())
    """
    def __init__(self, prefix=None, params=None):
        self._empty_prefix = prefix == ''
        self._prefix, self._params = _BlockScope.create(prefix, params, self._alias())
        self._name = self._prefix[:-1] if self._prefix.endswith('_') else self._prefix
        self._scope = _BlockScope(self)
        self._children = OrderedDict()
        self._reg_params = {}
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block.__repr__(), 2))
                            for key, block in self.__dict__.items() if isinstance(block, Block)])
        return s.format(name=self.__class__.__name__, modstr=modstr)

    def __setattr__(self, name, value):
        """Registers parameters."""

        if hasattr(self, name):
            existing = getattr(self, name)
            if isinstance(existing, (Parameter, Block)) and not isinstance(value, type(existing)):
                raise TypeError('Changing attribute type for {name} from {type1} to {type2}' \
                                'is not allowed.'.format(
                                    name=name, type1=type(existing), type2=type(value)))

        if isinstance(value, Block):
            self.register_child(value, name)
        elif isinstance(value, Parameter):
            assert name not in self._reg_params, \
                "Overriding Parameter attribute %s is not allowed. " \
                "If you want to share parameters between blocks, please set " \
                "'params' at Block construction instead."
            self._reg_params[name] = value

        super(Block, self).__setattr__(name, value)

    def _check_container_with_block(self):
        children = set(self._children.values())
        def _find_unregistered_block_in_container(data):
            # Find whether a nested container structure contains Blocks
            if isinstance(data, (list, tuple)):
                for ele in data:
                    if _find_unregistered_block_in_container(ele):
                        return True
                return False
            elif isinstance(data, dict):
                for _, v in data.items():
                    if _find_unregistered_block_in_container(v):
                        return True
                return False
            elif isinstance(data, Block):
                return not data in children
            else:
                return False
        for k, v in self.__dict__.items():
            if isinstance(v, (list, tuple, dict)) and not (k.startswith('__') or k == '_children'):
                if _find_unregistered_block_in_container(v):
                    warnings.warn('"{name}" is an unregistered container with Blocks. '
                                  'Note that Blocks inside the list, tuple or dict will not be '
                                  'registered automatically. Make sure to register them using '
                                  'register_child() or switching to '
                                  'nn.Sequential/nn.HybridSequential instead. '
                                  .format(name=self.__class__.__name__ + "." + k), stacklevel=3)

    def _alias(self):
        return self.__class__.__name__.lower()

    @property
    def prefix(self):
        """Prefix of this :py:class:`Block`."""
        return self._prefix

    @property
    def name(self):
        """Name of this :py:class:`Block`, without '_' in the end."""
        return self._name

    def name_scope(self):
        """Returns a name space object managing a child :py:class:`Block` and parameter
        names. Should be used within a ``with`` statement::
            with self.name_scope():
                self.dense = nn.Dense(20)
        Please refer to
        `the naming tutorial </api/python/docs/tutorials/packages/gluon/blocks/naming.html>`_
        for more info on prefix and naming.
        """
        return self._scope

    @property
    def params(self):
        """Returns this :py:class:`Block`'s parameter dictionary (does not include its
        children's parameters)."""
        return self._params

    def collect_params(self, select=None):
        """Returns a :py:class:`ParameterDict` containing this :py:class:`Block` and all of its
        children's Parameters(default), also can returns the select :py:class:`ParameterDict`
        which match some given regular expressions.
        For example, collect the specified parameters in ['conv1_weight', 'conv1_bias', 'fc_weight',
        'fc_bias']::
            model.collect_params('conv1_weight|conv1_bias|fc_weight|fc_bias')
        or collect all parameters whose names end with 'weight' or 'bias', this can be done
        using regular expressions::
            model.collect_params('.*weight|.*bias')
        Parameters
        ----------
        select : str
            regular expressions
        Returns
        -------
        The selected :py:class:`ParameterDict`
        """
        # We need to check here because blocks inside containers are not supported.
        self._check_container_with_block()
        ret = ParameterDict(self._params.prefix)
        if not select:
            ret.update(self.params)
        else:
            pattern = re.compile(select)
            ret.update({name:value for name, value in self.params.items() if pattern.match(name)})
        for cld in self._children.values():
            ret.update(cld.collect_params(select=select))
        return ret

    def _collect_params_with_prefix(self, prefix=''):
        if prefix:
            prefix += '.'
        ret = {prefix + key : val for key, val in self._reg_params.items()}
        for name, child in self._children.items():
            ret.update(child._collect_params_with_prefix(prefix + name))
        return ret

    def register_child(self, block, name=None):
        """Registers block as a child of self. :py:class:`Block` s assigned to self as
        attributes will be registered automatically."""
        if name is None:
            name = str(len(self._children))
        self._children[name] = block

    def register_forward_pre_hook(self, hook):
        r"""Registers a forward pre-hook on the block.
        The hook function is called immediately before :func:`forward`.
        It should not modify the input or output.
        Parameters
        ----------
        hook : callable
            The forward hook function of form `hook(block, input) -> None`.
        Returns
        -------
        :class:`mxnet.gluon.utils.HookHandle`
        """
        handle = HookHandle()
        handle.attach(self._forward_pre_hooks, hook)
        return handle

    def register_forward_hook(self, hook):
        r"""Registers a forward hook on the block.
        The hook function is called immediately after :func:`forward`.
        It should not modify the input or output.
        Parameters
        ----------
        hook : callable
            The forward hook function of form `hook(block, input, output) -> None`.
        Returns
        -------
        :class:`mxnet.gluon.utils.HookHandle`
        """
        handle = HookHandle()
        handle.attach(self._forward_hooks, hook)
        return handle

    def apply(self, fn):
        r"""Applies ``fn`` recursively to every child block as well as self.
        Parameters
        ----------
        fn : callable
            Function to be applied to each submodule, of form `fn(block)`.
        Returns
        -------
        this block
        """
        for cld in self._children.values():
            cld.apply(fn)
        fn(self)
        return self

    def initialize(self, init=anp.random.uniform, ctx=None, verbose=False,
                   force_reinit=False):
        """Initializes :py:class:`Parameter` s of this :py:class:`Block` and its children.
        Equivalent to ``block.collect_params().initialize(...)``
        Parameters
        ----------
        init : Initializer
            Global default Initializer to be used when :py:meth:`Parameter.init` is ``None``.
            Otherwise, :py:meth:`Parameter.init` takes precedence.
        ctx : Context or list of Context
            Keeps a copy of Parameters on one or many context(s).
        verbose : bool, default False
            Whether to verbosely print out details on initialization.
        force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.
        """
        self.collect_params().initialize(init, ctx, verbose, force_reinit)

    def hybridize(self, active=True, **kwargs):
        """ Please refer description of HybridBlock hybridize().
        """
        for cld in self._children.values():
            cld.hybridize(active, **kwargs)

    def cast(self, dtype):
        """Cast this Block to use another data type.
        Parameters
        ----------
        dtype : str or numpy.dtype
            The new data type.
        """
        for child in self._children.values():
            child.cast(dtype)
        for _, param in self.params.items():
            param.cast(dtype)

    def __call__(self, *args):
        """Calls forward. Only accepts positional arguments."""
        # for hook in self._forward_pre_hooks.values():
        #     hook(self, args)

        out = self.forward(*args)

        # for hook in self._forward_hooks.values():
        #     hook(self, args, out)
        # if _mx_npx.is_np_array():
        #     _check_all_np_ndarrays(out)
        return out

    def forward(self, *args):
        """Overrides to implement forward computation using :py:class:`NDArray`. Only
        accepts positional arguments.
        Parameters
        ----------
        *args : list of NDArray
            Input tensors.
        """
        raise NotImplementedError
        # pylint: disable= invalid-name

    def hybrid_forward(self, *args):
        return self(*args)
