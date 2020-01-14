import time
import collections
import mxnet as mx
from abc import abstractmethod
from ...scheduler import *
from .base_predictor import *
from ...utils import in_ipynb

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
        plot_results = scheduler_options.pop('plot_results') \
                if 'plot_results' in scheduler_options else False
        scheduler = scheduler(train_fn, **scheduler_options)
        print('scheduler:', scheduler)
        scheduler.run()
        scheduler.join_jobs()
        # gather the best configuration
        best_reward = scheduler.get_best_reward()
        best_config = scheduler.get_best_config()
        args = train_fn.args
        args.final_fit = True
        results = scheduler.run_with_config(best_config)
        total_time = time.time() - start_time
        if plot_results or in_ipynb():
            plot_training_curves = scheduler_options['checkpoint'].replace('exp1.ag', 'plot_training_curves.png')
            scheduler.get_training_curves(filename=plot_training_curves, plot=True, use_legend=False)
        results.update(best_reward=best_reward, best_config=best_config,
                       total_time=total_time, metadata=scheduler.metadata,
                       training_history=scheduler.training_history,
                       config_history=scheduler.config_history,
                       reward_attr=scheduler._reward_attr,
                       args=args)
        return results

    @abstractmethod
    def fit(cls, *args, **kwargs):
        pass
