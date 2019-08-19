import copy
import logging
import argparse
import functools
import numpy as np
import multiprocessing as mp
import ConfigSpace as CS

from ..searcher import Sample
from .space import *

__all__ = ['autogluon_method', 'autogluon_register_args', 'autogluon_register_dict']

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
        def gen_config(seed, config):
            new_config = copy.deepcopy(config)
            np.random.seed(seed)
            for k, v in config.items():
                if isinstance(v, Sample):
                    new_config[k] = v()
            return new_config
        with autogluon_method.LOCK:
            autogluon_method.SEED.value += 1
            spec_config = gen_config(autogluon_method.SEED.value, config)
        if isinstance(args, (argparse.ArgumentParser, argparse.Namespace)):
            vars(args).update(spec_config)
        else:
            args.update(spec_config)
        self.f(args, **kwargs)
        if 'reporter' in kwargs and kwargs['reporter'] is not None:
            logger.debug('Reporter Done!')
            kwargs['reporter'](done=True)

    def __repr__(self):
        return repr(self.f)

def autogluon_kwargs(**kwvars):
    """
    """
    def registered_func(func):
        @functools.wraps(func)
        def wrapper_call(**kwargs):
            for k, w in kwargs.items():
                if k in kwvars.keys():
                    
            return func(**kwvars)
        return wrapper_call
    return registered_func

def autogluon_object(**kwvars):
    """Register args or searchable spaces to the class init method.
    """
    def registered_class(cls):
        init_method = cls.__dict__['__init__']
        @autogluon_kwargs(**kwvars)
        def wrapper_call(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper_call
    return registered_class


def autogluon_register_args(**kwvars):
    """Register default args or searchable spaces to the 'autogluon_method'
    """
    def registered_func(func):
        @autogluon_method
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            return func(*args, **kwargs)

        cs = CS.ConfigurationSpace()
        args = argparse.Namespace()
        args_dict = vars(args)
        for k, v in kwvars.items():
            if isinstance(v, Space):
                hp = get_config_space(k, v)
                cs.add_hyperparameter(hp)
                args_dict.update({k: hp.default_value})
            else:
                args_dict.update({k: v})
        wrapper_call.cs = cs
        wrapper_call.args = args
        logger.debug('Registering {}'.format(cs))
        return wrapper_call
    return registered_func


def autogluon_register_dict(edict, **kwvars):
    """Similar to `autogluon_register_args`, but register default args from a extra dict or EasyDict.
    """
    from easydict import EasyDict as ezdict
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
                hp = get_config_space(k, v)
                cs.add_hyperparameter(hp)
                args.update({k: hp.default_value})
            else:
                args.update({k: v})
        wrapper_call.cs = cs
        wrapper_call.args = args
        logger.debug('Registering {}'.format(cs))
        return wrapper_call
    return registered_func
