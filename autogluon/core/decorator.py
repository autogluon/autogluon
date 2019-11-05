import copy
import logging
import argparse
import functools
from collections import OrderedDict
import numpy as np
import multiprocessing as mp
import ConfigSpace as CS

from .space import *
from .space import _add_hp, _add_cs, _rm_hp, _strip_config_space
from ..utils import EasyDict as ezdict
from ..utils.deprecate import make_deprecate

__all__ = ['autogluon_register_args', 'autogluon_object', 'autogluon_function',
           'sample_config', 'autogluon_register_dict']

logger = logging.getLogger(__name__)

def sample_config(args, config):
    args = copy.deepcopy(args)
    striped_keys = [k.split('.')[0] for k in config.keys()]
    if isinstance(args, (argparse.Namespace, argparse.ArgumentParser)):
        args_dict = vars(args)
    else:
        args_dict = args
    for k, v in args_dict.items():
        # handle different type of configurations
        if k in striped_keys:
            if isinstance(v, NestedSpace):
                sub_config = _strip_config_space(config, prefix=k)
                args_dict[k] = v.sample(**sub_config)
            else:
                if '.' in k: continue
                args_dict[k] = config[k]
        elif isinstance(v, AutoGluonObject):
            args_dict[k] = v.init()
    return args

class _autogluon_method(object):
    SEED = mp.Value('i', 0)
    LOCK = mp.Lock()
    def __init__(self, f):
        self.f = f
        self.args = ezdict()
        functools.update_wrapper(self, f)

    def __call__(self, args, config, **kwargs):
        new_config = copy.deepcopy(config)
        self._rand_seed()
        args = sample_config(args, new_config)

        output = self.f(args, **kwargs)
        if 'reporter' in kwargs and kwargs['reporter'] is not None:
            logger.debug('Reporter Done!')
            kwargs['reporter'](done=True)
        return output
 
    def register_args(self, default={}, **kwvars):
        if isinstance(default, (argparse.Namespace, argparse.ArgumentParser)):
            default = vars(default)
        self.kwvars = {}
        self.args = ezdict()
        self.args.update(default)
        self.update(**kwvars)

    def update(self, **kwargs):
        """For searcher support ConfigSpace
        """
        self.kwvars.update(kwargs)
        for k, v in self.kwvars.items():
            if isinstance(v, (NestedSpace)):
                self.args.update({k: v})
            elif isinstance(v, Space):
                hp = v.get_hp(name=k)
                self.args.update({k: hp.default_value})
            else:
                self.args.update({k: v})

    @property
    def cs(self):
        cs = CS.ConfigurationSpace()
        for k, v in self.kwvars.items():
            if isinstance(v, (Categorical, Sequence, Dict, AutoGluonObject)):
                _add_cs(cs, v.cs, k)
            elif isinstance(v, Space):
                hp = v.get_hp(name=k)
                _add_hp(cs, hp)
            else:
                _rm_hp(cs, k)
        return cs

    @property
    def kwspaces(self):
        """For RL searcher/controller
        """
        kw_spaces = OrderedDict()
        for k, v in self.kwvars.items():
            if isinstance(v, (AutoGluonObject, Sequence, Dict)):
                for sub_k, sub_v in v.kwspaces.items():
                    new_k = '{}.{}'.format(k, sub_k)
                    kw_spaces[new_k] = sub_v
            elif isinstance(v, Categorical):
                kw_spaces['{}.choice'.format(k)] = v
                for sub_k, sub_v in v.kwspaces.items():
                    new_k = '{}.{}'.format(k, sub_k)
                    kw_spaces[new_k] = sub_v
            elif isinstance(v, Space):
                kw_spaces[k] = v
        return kw_spaces

    def _rand_seed(self):
        _autogluon_method.SEED.value += 1
        np.random.seed(_autogluon_method.SEED.value)

    def __repr__(self):
        return repr(self.f)


def autogluon_register_args(default={}, **kwvars):
    """Decorator for customized training script, registering arguments or searchable spaces
    to the decorated function. The arguments should be python objects, autogluon objects
    (see :function:`autogluon.autogluon_object`_ .), or autogluon search spaces
    (:class:`autogluon.Int`, :class:`autogluon.Linear` ...).

    Example:
        >>> @autogluon_register_args(batch_size=10, lr=ag.Linear(0.01, 0.1))
        >>> def my_train(args):
        >>>     print('Batch size is {}, LR is {}'.format(args.batch_size, arg.lr))

    """
    kwvars['_default_config'] = default
    def registered_func(func):
        @_autogluon_method
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            return func(*args, **kwargs)

        default = kwvars['_default_config']
        wrapper_call.register_args(default=default, **kwvars)
        return wrapper_call

    return registered_func


def autogluon_function(**kwvars):
    """Register args or searchable spaces to the functions.

    Return:
        AutoGluonobject: a lazy init object, which allows distributed training.

    Example:
        >>> from gluoncv.model_zoo import get_model
        >>> 
        >>> @ag.autogluon_function(pretrained=ag.space.Categorical(True, False))
        >>> def cifar_resnet(pretrained):
        >>>     return get_model('cifar_resnet20_v1', pretrained=pretrained)
    """
    def registered_func(func):
        class autogluonobject(AutoGluonObject):
            @_autogluon_kwargs(**kwvars)
            def __init__(self, *args, **kwargs):
                self.func = func
                self.args = args
                self.kwargs = kwargs
                self._inited = False

            def sample(self, **nkwvars):
                # lazy initialization for passing config files
                self.kwargs.update(nkwvars)
                for k, v in self.kwargs.items():
                    if k in self.kwspaces and isinstance(self.kwspaces[k], Categorical):
                        self.kwargs[k] = self.kwspaces[k][v]
                        
                return self.func(*self.args, **self.kwargs)

        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            agobj = autogluonobject(*args, **kwargs)
            agobj.kwvars = agobj.__init__.kwvars
            return agobj
        return wrapper_call
    return registered_func

def autogluon_object(**kwvars):
    """Register args or searchable spaces to the class.

    Return:
        AutoGluonobject: a lazy init object, which allows distributed training.

    Example:
        >>> from gluoncv.model_zoo.cifarresnet import CIFARResNetV1, CIFARBasicBlockV1
        >>>
        >>> @autogluon_object(
        >>>     learning_rate=ag.space.Real(1e-4, 1e-1, log=True),
        >>>     wd=ag.space.Real(1e-4, 1e-1),
        >>> )
        >>> class Adam(optim.Adam):
        >>>     pass

    """
    def registered_class(Cls):
        class autogluonobject(AutoGluonObject):
            @_autogluon_kwargs(**kwvars)
            def __init__(self, *args, **kwargs):
                self._args = args
                self._kwargs = kwargs
                self._inited = False

            def sample(self, **nkwvars):
                kwargs = self._kwargs
                kwargs.update(nkwvars)
                kwspaces = autogluonobject.kwspaces
                for k, v in kwargs.items():
                    if k in kwspaces and isinstance(kwspaces[k], Categorical):
                        kwargs[k] = kwspaces[k][v]

                args = self._args
                return Cls(*args, **kwargs)

            def __repr__(self):
                return 'AutoGluonObject -- ' + Cls.__name__

        autogluonobject.kwvars = autogluonobject.__init__.kwvars
        return autogluonobject

    return registered_class

autogluon_register_dict = make_deprecate(autogluon_register_args, 'autogluon_register_dict')

def _autogluon_kwargs(**kwvars):
    def registered_func(func):
        kwspaces = OrderedDict()
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            kwvars.update(kwargs)
            for k, v in kwvars.items():
                if isinstance(v, NestedSpace):
                    kwspaces[k] = v
                    kwargs[k] = v
                elif isinstance(v, Space):
                    kwspaces[k] = v
                    hp = v.get_hp(name=k)
                    kwargs[k] = hp.default_value
                else:
                    kwargs[k] = v
            return func(*args, **kwargs)
        wrapper_call.kwspaces = kwspaces
        wrapper_call.kwvars = kwvars
        return wrapper_call
    return registered_func
