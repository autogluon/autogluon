import copy
import logging
import functools
import numpy as np
import multiprocessing as mp
import ConfigSpace as CS

from ..searcher import Sample
from .space import *

__all__ = ['autogluon_method', 'autogluon_register', 'gen_config']

logger = logging.getLogger(__name__)

class autogluon_method(object):
    SEED = mp.Value('i', 0)
    LOCK = mp.Lock()
    def __init__(self, f):
        functools.update_wrapper(self, f)
        self.f = f

    def __call__(self, args, config, **kwargs):
        with autogluon_method.LOCK:
            autogluon_method.SEED.value += 1
            spec_config = gen_config(autogluon_method.SEED.value, config)
        vars(args).update(spec_config)
        self.f(args, **kwargs)
        if 'reporter' in kwargs:
            logger.debug('Reporter Done!')
            kwargs['reporter'](done=True)

    def __repr__(self):
        return repr(self.f)

def autogluon_register(**kwvar):
    def registered_func(func):
        @autogluon_method
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            return func(*args, **kwargs)
        cs = CS.ConfigurationSpace()
        for k, v in kwvar.items():
            cs.add_hyperparameter(get_config_space(k, v)) 
        wrapper_call.cs = cs
        logger.debug('Registering {}'.format(cs))
        return wrapper_call
    return registered_func

def gen_config(seed, config):
    new_config = copy.deepcopy(config)
    np.random.seed(seed)
    for k, v in config.items():
        if isinstance(v, Sample):
            new_config[k] = v()
    return new_config
