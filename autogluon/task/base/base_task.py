import time
import collections
import mxnet as mx
from abc import abstractmethod
from ...scheduler import *

__all__ = ['BaseDataset', 'BaseTask']

Results = collections.namedtuple('Results', 'model reward config time metadata')

schedulers = {
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
        start_time = time.time()
        # create scheduler and schedule tasks
        if isinstance(search_strategy, str):
            scheduler = schedulers[search_strategy.lower()]
        else:
            assert callable(search_strategy)
            scheduler = search_strategy
            scheduler_options['searcher'] = 'random'
        cls.scheduler = scheduler(train_fn, **scheduler_options)
        cls.scheduler.run()
        cls.scheduler.join_tasks()
        # final fit
        best_reward = cls.scheduler.get_best_reward()
        best_config = cls.scheduler.get_best_config()
        args = train_fn.args
        args.final_fit = True
        model = train_fn(args, best_config, reporter=None)
        total_time = time.time() - start_time
        cls.results = Results(model, best_reward, best_config, total_time, cls.scheduler.metadata)
        return cls.results

    @classmethod
    def get_training_curves(cls, filename=None, plot=False, use_legend=True):
        cls.scheduler.get_training_curves(filename=None, plot=False, use_legend=True)

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
