import logging
import multiprocessing as mp
from ..distribution import gen_config

__all__ = ['autogluon_method']

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

        logger.debug('spec_config {}'.format(spec_config))
        vars(args).update(spec_config)
        self.f(args, **kwargs)
        if 'reporter' in kwargs:
            logger.debug('Reporter Done!')
            kwargs['reporter'](done=True)

    def __repr__(self):
        return repr(self.f)
