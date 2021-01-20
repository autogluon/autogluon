import collections
import copy
import logging
import time
from abc import abstractmethod

from ...scheduler import *
from ...utils import in_ipynb, try_import_mxnet
from ...utils.utils import setup_compute

__all__ = [
    'BaseTask',
    'compile_scheduler_options',
    'compile_scheduler_options_v2',
    'create_scheduler']

Results = collections.namedtuple('Results', 'model reward config time metadata')

schedulers = {
    'grid': FIFOScheduler,
    'random': FIFOScheduler,
    'skopt': FIFOScheduler,
    'hyperband': HyperbandScheduler,
    'rl': RLScheduler,
    'bayesopt': FIFOScheduler,
    'bayesopt_hyperband': HyperbandScheduler}

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def create_scheduler(train_fn, search_strategy, scheduler_options):
    if isinstance(search_strategy, str):
        scheduler_cls = schedulers[search_strategy.lower()]
    else:
        assert callable(search_strategy)
        scheduler_cls = search_strategy
        scheduler_options = copy.copy(scheduler_options)
        scheduler_options['searcher'] = 'random'
    return scheduler_cls(train_fn, **scheduler_options)


class BaseTask(object):
    """BaseTask for AutoGluon applications
    """
    @property
    @staticmethod
    def Dataset():
        try_import_mxnet()
        from autogluon.mxnet.utils.dataset import BaseDataset
        return BaseDataset

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
        if results is None:
            logger.warning('No valid results obtained with best config, the result may not be useful...')
            results = {}
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
    'hyperband': 'random',
    'bayesopt_hyperband': 'bayesopt'}


def compile_scheduler_options(
        scheduler_options, search_strategy, search_options, nthreads_per_trial,
        ngpus_per_trial, checkpoint, num_trials, time_out, resume, visualizer,
        time_attr, reward_attr, dist_ip_addrs, epochs=None):
    """
    Updates a copy of scheduler_options (scheduler-specific options, can be
    empty) with general options. The result can be passed to __init__ of the
    scheduler.

    Special role of epochs for HyperbandScheduler: If the search_strategy
    involves HyperbandScheduler and epochs is given, then this value is
    copied to scheduler_options['max_t']. Pass epochs for applications
    where the time_attr is epoch, and epochs is the maximum number of
    epochs.

    :param scheduler_options:
    :param search_strategy:
    :param search_options:
    :param nthreads_per_trial:
    :param ngpus_per_trial:
    :param checkpoint:
    :param num_trials:
    :param time_out:
    :param resume:
    :param visualizer:
    :param time_attr:
    :param reward_attr:
    :param dist_ip_addrs:
    :param kwargs:
    :param epochs: See above. Optional
    :return: Copy of scheduler_options with updates

    """
    if scheduler_options is None:
        scheduler_options = dict()
    else:
        assert isinstance(scheduler_options, dict)
    assert isinstance(search_strategy, str)
    if search_options is None:
        search_options = dict()
    if visualizer is None:
        visualizer = 'none'
    if time_attr is None:
        time_attr = 'epoch'
    if reward_attr is None:
        reward_attr = 'accuracy'
    scheduler_options = copy.copy(scheduler_options)
    scheduler_options.update({
        'resource': {
            'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
        'searcher': search_strategy,
        'search_options': search_options,
        'checkpoint': checkpoint,
        'resume': resume,
        'num_trials': num_trials,
        'time_out': time_out,
        'reward_attr': reward_attr,
        'time_attr': time_attr,
        'visualizer': visualizer,
        'dist_ip_addrs': dist_ip_addrs})
    searcher = searcher_for_hyperband_strategy.get(search_strategy)
    if searcher is not None:
        scheduler_options['searcher'] = searcher
        if epochs is not None:
            scheduler_options['max_t'] = epochs
    return scheduler_options


# TODO: Migrate TextPrediction to use this version, delete old version
def compile_scheduler_options_v2(
        scheduler_options, nthreads_per_trial,
        ngpus_per_trial, num_trials, time_out, search_strategy=None, search_options=None, checkpoint=None, resume=False, visualizer=None,
        time_attr=None, reward_attr=None, dist_ip_addrs=None, epochs=None):
    """
    Updates a copy of scheduler_options (scheduler-specific options, can be
    empty) with general options. The result can be passed to __init__ of the
    scheduler.

    Special role of epochs for HyperbandScheduler: If the search_strategy
    involves HyperbandScheduler and epochs is given, then this value is
    copied to scheduler_options['max_t']. Pass epochs for applications
    where the time_attr is epoch, and epochs is the maximum number of
    epochs.

    :param scheduler_options:
    :param search_strategy:
    :param search_options:
    :param nthreads_per_trial:
    :param ngpus_per_trial:
    :param checkpoint:
    :param num_trials:
    :param time_out:
    :param resume:
    :param visualizer:
    :param time_attr:
    :param reward_attr:
    :param dist_ip_addrs:
    :param kwargs:
    :param epochs: See above. Optional
    :return: Copy of scheduler_options with updates

    """
    if scheduler_options is None:
        scheduler_options = dict()
    else:
        assert isinstance(scheduler_options, dict)
    if dist_ip_addrs is None:
        dist_ip_addrs = []
    if search_strategy is None:
        search_strategy = 'random'
    assert isinstance(search_strategy, str)
    if search_options is None:
        search_options = dict()
    if visualizer is None:
        visualizer = 'none'
    if time_attr is None:
        time_attr = 'epoch'
    if reward_attr is None:
        reward_attr = 'validation_performance'
    scheduler_params = {
        'resource': {
            'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
        'searcher': search_strategy,
        'search_options': search_options,
        'checkpoint': checkpoint,
        'resume': resume,
        'num_trials': num_trials,
        'time_out': time_out,
        'reward_attr': reward_attr,
        'time_attr': time_attr,
        'visualizer': visualizer,
        'dist_ip_addrs': dist_ip_addrs,
    }
    scheduler_params.update(copy.copy(scheduler_options))

    scheduler_params['resource']['num_cpus'], scheduler_params['resource']['num_gpus'] = setup_compute(
        nthreads_per_trial=scheduler_params['resource']['num_cpus'],
        ngpus_per_trial=scheduler_params['resource']['num_gpus'],
    )  # TODO: use 'auto' downstream

    searcher = searcher_for_hyperband_strategy.get(scheduler_params['searcher'])
    if searcher is not None:
        scheduler_params['searcher'] = searcher
        if epochs is not None:
            scheduler_params['max_t'] = epochs
    required_options = [
        'resource',
        'searcher',
        'search_options',
        'checkpoint',
        'resume',
        'num_trials',
        'time_out',
        'reward_attr',
        'time_attr',
        'visualizer',
        'dist_ip_addrs',
    ]
    missing_options = []
    for option in required_options:
        if option not in scheduler_params:
            missing_options.append(option)
    if missing_options:
        raise AssertionError(f'Missing required keys in scheduler_options: {missing_options}')
    return scheduler_params
