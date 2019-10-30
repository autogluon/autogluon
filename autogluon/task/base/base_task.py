import time
import collections
import mxnet as mx
from abc import abstractmethod
from ...scheduler import *

__all__ = ['BaseDataset', 'BaseTask']

Results = collections.namedtuple('Results', 'model reward config time metadata')

schedulers = {
    'grid': FIFOScheduler,
    'random': FIFOScheduler,
    'bayesopt': FIFOScheduler,
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
        # create scheduler and schedule tasks
        if isinstance(search_strategy, str):
            scheduler = schedulers[search_strategy.lower()]
        else:
            assert callable(search_strategy)
            scheduler = search_strategy
            scheduler_options['searcher'] = 'random'
        scheduler = scheduler(train_fn, **scheduler_options)
        scheduler.run()
        scheduler.join_tasks()
        # gather the best configuration
        best_reward = scheduler.get_best_reward()
        best_config = scheduler.get_best_config()
        args = train_fn.args
        args.final_fit = True
        # final fit
        return scheduler.run_with_config(best_config)

    @classmethod
    @abstractmethod
    def fit(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def predict(cls, inputs):
        """The task predict function given an input.
         Args:
            img: the input
         Example:
        """
        pass

    @classmethod
    @abstractmethod
    def evaluate(cls, dataset):
        """The task evaluation function given the test dataset.
         Args:
            dataset: test dataset
         Example:
            >>> from autogluon import ImageClassification as task
            >>> dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
            >>> test_reward = task.evaluate(dataset)
        """
        pass
