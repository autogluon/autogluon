import collections
import copy
import time
from abc import abstractmethod

import mxnet as mx

from ...scheduler import *
from ...utils import in_ipynb

__all__ = [
    'BaseDataset',
    'BaseTask',
    'compile_scheduler_options',
    'create_scheduler']

Results = collections.namedtuple('Results', 'model reward config time metadata')

schedulers = {
    'grid': FIFOScheduler,
    'random': FIFOScheduler,
    'skopt': FIFOScheduler,
    'hyperband': HyperbandScheduler,
    'rl': RLScheduler,
}


def create_scheduler(train_fn, search_strategy, scheduler_options):
    if isinstance(search_strategy, str):
        scheduler_cls = schedulers[search_strategy.lower()]
    else:
        assert callable(search_strategy)
        scheduler_cls = search_strategy
        scheduler_options = copy.copy(scheduler_options)
        scheduler_options['searcher'] = 'random'
    return scheduler_cls(train_fn, **scheduler_options)


class BaseDataset(mx.gluon.data.Dataset):
    # put any sharable dataset methods here
    pass


class BaseTask(object):
    """BaseTask for AutoGluon applications
    """
    Dataset = BaseDataset

    @classmethod
    def run_fit(cls, train_fn, search_strategy, scheduler_options,
                plot_results=False):
        start_time = time.time()
        # create scheduler and schedule tasks
        scheduler = create_scheduler(
            train_fn, search_strategy, scheduler_options)
        print('scheduler:', scheduler)
        scheduler.run()
        scheduler.join_jobs()
        # gather the best configuration
        best_reward = scheduler.get_best_reward()
        best_config = scheduler.get_best_config()
        args = train_fn.args
        args.final_fit = True
        if hasattr(args, 'epochs') and hasattr(args, 'final_fit_epochs'):
            args.epochs = args.final_fit_epochs
        results = scheduler.run_with_config(best_config)
        total_time = time.time() - start_time
        if plot_results or in_ipynb():
            plot_training_curves = scheduler_options['checkpoint'].replace('exp1.ag', 'plot_training_curves.png')
            scheduler.get_training_curves(filename=plot_training_curves, plot=True, use_legend=False)
        record_args = copy.deepcopy(args)
        results.update(best_reward=best_reward,
                       best_config=best_config,
                       total_time=total_time,
                       metadata=scheduler.metadata,
                       training_history=scheduler.training_history,
                       config_history=scheduler.config_history,
                       reward_attr=scheduler._reward_attr,
                       args=record_args)
        return results

    @classmethod
    @abstractmethod
    def fit(cls, *args, **kwargs):
        pass


# These search_strategies use HyperbandScheduler, along with certain
# searchers.
searcher_for_hyperband_strategy = {
    'hyperband': 'random'}


def compile_scheduler_options(
        search_strategy, nthreads_per_trial, ngpus_per_trial, checkpoint,
        num_trials, time_out, resume, visualizer, time_attr, reward_attr,
        search_options, dist_ip_addrs, epochs, **kwargs):
    assert isinstance(search_strategy, str)
    if visualizer is None:
        visualizer = 'none'
    if time_attr is None:
        time_attr = 'epoch'
    if reward_attr is None:
        reward_attr = 'accuracy'
    if search_options is None:
        search_options = dict()
    if epochs is None:
        epochs = 20
    scheduler_options = {
        'resource': {
            'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
        'checkpoint': checkpoint,
        'num_trials': num_trials,
        'time_out': time_out,
        'resume': resume,
        'visualizer': visualizer,
        'time_attr': time_attr,
        'reward_attr': reward_attr,
        'dist_ip_addrs': dist_ip_addrs,
        'searcher': search_strategy,
        'search_options': search_options,
        'delay_get_config': kwargs.get('delay_get_config', True)}
    searcher = searcher_for_hyperband_strategy.get(search_strategy)
    if searcher is not None:
        # Note: We can have grace_period=None in kwargs, or grace_period
        # missing in kwargs
        grace_period = kwargs.get('grace_period')
        if grace_period is None:
            grace_period = 1
        reduction_factor = kwargs.get('reduction_factor')
        if reduction_factor is None:
            reduction_factor = 3
        brackets = kwargs.get('brackets')
        if brackets is None:
            brackets = 1
        type = kwargs.get('type')
        if type is None:
            type = 'stopping'
        searcher_data = kwargs.get('searcher_data')
        if searcher_data is None:
            searcher_data = 'rungs'
        scheduler_options.update({
            'searcher': searcher,
            'max_t': epochs,
            'grace_period': grace_period,
            'reduction_factor': reduction_factor,
            'brackets': brackets,
            'type': type,
            'searcher_data': searcher_data})
    return scheduler_options
