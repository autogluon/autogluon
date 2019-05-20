import copy
import logging
import numpy as np
import multiprocessing as mp

from ..distribution import Sample

__all__ = ['autogluon_method', 'gen_config']

logger = logging.getLogger(__name__)

class autogluon_method(object):
    SEED = mp.Value('i', 0)
    LOCK = mp.Lock()
    def __init__(self, f):
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

def gen_config(seed, config):
    new_config = copy.deepcopy(config)
    np.random.seed(seed)
    for k, v in config.items():
        if isinstance(v, Sample):
            new_config[k] = v()
    return new_config
