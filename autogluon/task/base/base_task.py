import time
import collections
import mxnet as mx
from abc import abstractmethod
from ...scheduler import *
from .base_predictor import *

__all__ = ['BaseDataset', 'BaseTask']

Results = collections.namedtuple('Results', 'model reward config time metadata')

schedulers = {
    'grid': FIFOScheduler,
    'random': FIFOScheduler,
    'skopt': FIFOScheduler,
    'hyperband': HyperbandScheduler,
    'rl': RLScheduler,
}

class BaseDataset(mx.gluon.data.Dataset):
    # put any sharable dataset methods here
    pass

class BaseTask(object):
    """BaseTask for AutoGluon applications
    """
    Dataset = BaseDataset
    @classmethod
    def run_fit(cls, train_fn, search_strategy, scheduler_options):
        start_time = time.time()
        # create scheduler and schedule tasks
        if isinstance(search_strategy, str):
            scheduler = schedulers[search_strategy.lower()]
        else:
            assert callable(search_strategy)
            scheduler = search_strategy
            scheduler_options['searcher'] = 'random'
        scheduler = scheduler(train_fn, **scheduler_options)
        scheduler.run()
        scheduler.join_jobs()
        # gather the best configuration
        best_reward = scheduler.get_best_reward()
        best_config = scheduler.get_best_config()
        args = train_fn.args
        args.final_fit = True
        # final fit
        results = scheduler.run_with_config(best_config)
        total_time = time.time() - start_time
        results.update(best_reward=best_reward, best_config=best_config,
                       total_time=total_time, metadata=scheduler.metadata,
                       training_history=scheduler.training_history,
                       config_history=scheduler.config_history)
        return results

    @classmethod
    @abstractmethod
    def fit(cls, *args, **kwargs):
        pass
