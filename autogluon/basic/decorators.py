import copy
import logging
import argparse
import functools
import numpy as np
import multiprocessing as mp
import ConfigSpace as CS

from ..searcher import Sample
from .space import *
from ..utils import EasyDict as ezdict

__all__ = ['autogluon_method', 'autogluon_kwargs', 'autogluon_object',
           'autogluon_function', 'autogluon_register_args',
           'autogluon_register_dict']

logger = logging.getLogger(__name__)

class autogluon_method(object):
    """Enable searcher to update default function args.
    """
    SEED = mp.Value('i', 0)
    LOCK = mp.Lock()
    def __init__(self, f):
        functools.update_wrapper(self, f)
        self.f = f

    def __call__(self, args, config, **kwargs):
        new_config = copy.deepcopy(config)
        self._rand_seed()
        striped_keys = [k.split('.')[0] for k in new_config.keys()]
        for k, v in args.items():
            # handle different type of configurations
            if k in striped_keys:
                if isinstance(v, AutoGluonObject):
                    sub_config = strip_cofing_space(new_config, prefix=k)
                    args[k] = v._lazy_init(**sub_config)
                elif isinstance(v, ListSpace):
                    sub_config = strip_cofing_space(new_config, prefix=k)
                    if len(sub_config) == 1:
                        args[k] = sub_config.pop(k)
                    else:
                        # nested space: List of AutoGluonobject
                        choice = sub_config.pop(k)
                        sub_config = strip_cofing_space(sub_config, prefix=str(choice))
                        args[k] = v[choice]._lazy_init(**sub_config)
                elif isinstance(new_config[k], Sample):
                    args[k] = new_config[k]()
                else:
                    if '.' in k:
                        continue
                    args[k] = new_config[k]

        self.f(args, **kwargs)
        if 'reporter' in kwargs and kwargs['reporter'] is not None:
            logger.debug('Reporter Done!')
            kwargs['reporter'](done=True)

    def _rand_seed(self):
        autogluon_method.SEED.value += 1
        np.random.seed(autogluon_method.SEED.value)

    def __repr__(self):
        return repr(self.f)

def autogluon_register_args(**kwvars):
    """Register default args or searchable spaces to the 'autogluon_method'
    """
    def registered_func(func):
        @autogluon_method
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            return func(*args, **kwargs)

        cs = CS.ConfigurationSpace()
        args = ezdict()
        for k, v in kwvars.items():
            if isinstance(v, ListSpace):
                sub_cs = v.get_config_space(name=k)
                cs.add_configuration_space(k, sub_cs, '.')
                args.update({k: v})
            elif isinstance(v, Space):
                hp = v.get_config_space(name=k)
                cs.add_hyperparameter(hp)
                args.update({k: hp.default_value})
            elif isinstance(v, AutoGluonObject):
                cs.add_configuration_space(k, v.cs, '.')
                args.update({k: v})
            else:
                args.update({k: v})
        wrapper_call.cs = cs
        wrapper_call.args = args
        logger.debug('Registering {}'.format(cs))
        return wrapper_call
    return registered_func

def autogluon_kwargs(**kwvars):
    """Decorating function and gather configspaces
    """
    def registered_func(func):
        cs = CS.ConfigurationSpace()
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            for k, v in kwvars.items():
                if k in kwargs.keys():
                    continue
                if isinstance(v, ListSpace):
                    kwargs[k] = v
                    if k not in cs.get_hyperparameter_names():
                        sub_cs = v.get_config_space(name=k)
                        cs.add_configuration_space('', sub_cs, '')
                elif isinstance(v, Space):
                    if k not in cs.get_hyperparameter_names():
                        hp = v.get_config_space(name=k)
                        cs.add_hyperparameter(hp)
                        kwargs[k] = hp.default_value
                    else:
                        kwargs[k] = cs.get_hyperparameter(k).default_value
                else:
                    kwargs[k] = kwvars[k]
            return func(*args, **kwargs)
        wrapper_call.cs = cs
        return wrapper_call
    return registered_func

def autogluon_function(**kwvars):
    def registered_func(func):
        class autogluonobject(AutoGluonObject):
            @autogluon_kwargs(**kwvars)
            def __init__(self, *args, **kwargs):
                self.func = func
                self.args = args
                self.kwargs = kwargs

            def _lazy_init(self, **kwvars):
                # lazy initialization for passing config files
                self.kwargs.update(kwvars)
                return self.func(*self.args, **self.kwargs)

        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            agobj = autogluonobject(*args, **kwargs)
            agobj.cs = agobj.__init__.cs
            return agobj
        return wrapper_call
    return registered_func

def autogluon_object(**kwvars):
    """Register args or searchable spaces to the class.
    AutoGluon object is a lazy init object, which allows distributed training.
    """
    def registered_class(Cls):
        #class cls(Cls):
        #    pass
        class autogluonobject(AutoGluonObject):
            @autogluon_kwargs(**kwvars)
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self._inited = False

            def _lazy_init(self, **kwvars):
                # lazy initialization for passing config files
                self._inited = True
                self.kwargs.update(kwvars)
                self._instance = Cls(*self.args, **self.kwargs)
                return self._instance

            def __repr__(self):
                return Cls.__repr__(self)

        autogluonobject.cs = autogluonobject.__init__.cs
        return autogluonobject

    return registered_class

def autogluon_register_dict(edict, **kwvars):
    """Similar to `autogluon_register_args`, but register default args from a extra dict or EasyDict,
    which is usually used in config files.
    """
    def registered_func(func):
        @autogluon_method
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            return func(*args, **kwargs)

        cs = CS.ConfigurationSpace()
        args = ezdict()
        args.update(edict)
        for k, v in kwvars.items():
            if isinstance(v, Space):
                hp = v.get_config_space(name=k)
                cs.add_hyperparameter(hp)
                args.update({k: hp.default_value})
            else:
                args.update({k: v})
        wrapper_call.cs = cs
        wrapper_call.args = args
        logger.debug('Registering {}'.format(cs))
        return wrapper_call
    return registered_func
