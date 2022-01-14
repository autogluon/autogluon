import copy
import logging
import time
from abc import abstractmethod

from ...scheduler.seq_scheduler import LocalSequentialScheduler
from ...utils import in_ipynb
from ...utils.utils import setup_compute

__all__ = [
    'BaseTask',
    'compile_scheduler_options_v2',
    'create_scheduler']

schedulers = {
    'local': LocalSequentialScheduler,
}

logger = logging.getLogger(__name__)


def create_scheduler(train_fn, search_space, scheduler, scheduler_options):
    if isinstance(scheduler, str):
        scheduler_cls = schedulers[scheduler.lower()]
    else:
        assert callable(scheduler)
        scheduler_cls = scheduler
        scheduler_options = copy.copy(scheduler_options)
    return scheduler_cls(train_fn, search_space=search_space, **scheduler_options)


# FIXME: REMOVE THIS, first GluonCV needs to stop depending on AG, as it imports this class
class BaseTask(object):
    """BaseTask for AutoGluon applications"""
    @classmethod
    def run_fit(cls, train_fn, search_space, search_strategy, scheduler_options,
                plot_results=False):
        start_time = time.time()
        # create scheduler and schedule tasks
        scheduler = create_scheduler(train_fn, search_space, search_strategy, scheduler_options)
        scheduler.run()
        scheduler.join_jobs()
        # gather the best configuration
        best_reward = scheduler.get_best_reward()
        best_config = scheduler.get_best_config()
        best_config_run = {**best_config}
        best_config_run['final_fit'] = True
        if hasattr(best_config_run, 'epochs') and hasattr(best_config_run, 'final_fit_epochs'):
            best_config_run['epochs'] = best_config_run['final_fit_epochs']
        scheduler_final = create_scheduler(train_fn, search_space, search_strategy, scheduler_options)
        results = scheduler_final.run_with_config(best_config_run)
        total_time = time.time() - start_time
        if plot_results or in_ipynb():
            plot_training_curves = scheduler_options['checkpoint'].replace('exp1.ag', 'plot_training_curves.png')
            scheduler.get_training_curves(filename=plot_training_curves, plot=True, use_legend=False)
        if results is None:
            logger.warning('No valid results obtained with best config, the result may not be useful...')
            results = {}
        results.update(best_reward=best_reward,
                       best_config=best_config,
                       total_time=total_time,
                       metadata=scheduler.metadata,
                       training_history=scheduler.training_history,
                       config_history=scheduler.config_history,
                       reward_attr=scheduler._reward_attr)
        return results

    @classmethod
    @abstractmethod
    def fit(cls, *args, **kwargs):
        pass


def compile_scheduler_options_v2(
        scheduler_options, nthreads_per_trial,
        ngpus_per_trial, num_trials, time_out, scheduler=None, search_strategy=None, search_options=None, checkpoint=None, resume=False, visualizer=None,
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
    :param scheduler:
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
    scheduler_options = copy.copy(scheduler_options)
    if dist_ip_addrs is None:
        dist_ip_addrs = []
    if search_strategy is None:
        search_strategy = 'random'
    if scheduler is None:
        scheduler = 'local'
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
        'scheduler': scheduler,
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
    resource = None
    if 'resource' in scheduler_options:
        scheduler_params['resource'].update(scheduler_options['resource'])
        resource = scheduler_params['resource'].copy()
    scheduler_params.update(scheduler_options)
    if resource:
        scheduler_params['resource'] = resource

    scheduler_params['resource']['num_cpus'], scheduler_params['resource']['num_gpus'] = setup_compute(
        nthreads_per_trial=scheduler_params['resource']['num_cpus'],
        ngpus_per_trial=scheduler_params['resource']['num_gpus'],
    )  # TODO: use 'auto' downstream

    required_options = [
        'resource',
        'scheduler',
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
