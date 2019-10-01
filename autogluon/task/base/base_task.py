import mxnet as mx
from ...scheduler import *

__all__ = ['BaseDataset', 'BaseTask']

schedulers = {
    'random': FIFOScheduler,
    'bayesian': FIFOScheduler,
    'hyperband': HyperbandScheduler,
    'rl': RLScheduler,
}

class BaseDataset(mx.gluon.data.Dataset):
    # put any sharable dataset methods here
    pass

class BaseTask(object):
    Dataset = BaseDataset
    @classmethod
    def run_fit(cls, train_fn, algorithm, scheduler_options):
        scheduler = schedulers[algorithm.lower()]
        cls.scheduler = scheduler(train_fn, **scheduler_options)
        #cls.scheduler.run()
        #cls.scheduler.join_tasks()
        #return ezdict({'best_config':scheduler_inst.get_best_config(),
        #               'best_reward':scheduler_inst.get_best_reward()})
        config = cls.scheduler.searcher.get_config()
        train_fn(train_fn.args, config, reporter=None)

    @classmethod
    def shut_down(cls):
        cls.scheduler.shutdown()
