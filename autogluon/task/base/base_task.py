import time
import collections
import mxnet as mx
from abc import abstractmethod
from ...scheduler import *

__all__ = ['BaseDataset', 'BaseTask']

Results = collections.namedtuple('Results', 'model reward config time metadata')

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
        start_time = time.time()
        # create scheduler and schedule tasks
        if isinstance(algorithm, str):
            scheduler = schedulers[algorithm.lower()]
        else:
            assert callable(algorithm)
            scheduler = algorithm
        cls.scheduler = scheduler(train_fn, **scheduler_options)
        cls.scheduler.run()
        cls.scheduler.join_tasks()
        # final fit
        best_reward = cls.scheduler.get_best_reward()
        best_config = cls.scheduler.get_best_config()
        args = train_fn.args
        args.final_fit = True
        #config = cls.scheduler.searcher.get_config()
        #model = train_fn(args, config, reporter=None)
        model = train_fn(args, best_config, reporter=None)
        total_time = time.time() - start_time
        cls.results = Results(model, best_reward, best_config, total_time, cls.scheduler.metadata)
        return cls.results

    @classmethod
    def shut_down(cls):
        cls.scheduler.shutdown()

    @classmethod
    @abstractmethod
    def fit(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def predict(cls, inputs):
        pass
