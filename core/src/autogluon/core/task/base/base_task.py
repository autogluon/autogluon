import copy

from ...scheduler import HyperbandScheduler, FIFOScheduler
from ...scheduler.seq_scheduler import LocalSequentialScheduler
from ...utils.utils import setup_compute

__all__ = [
    'compile_scheduler_options_v2']

schedulers = {
    'local': LocalSequentialScheduler,
    'fifo': FIFOScheduler,
    'hyperband_stopping': HyperbandScheduler,
    'hyperband_promotion': HyperbandScheduler,
}

# These search_strategies use HyperbandScheduler, along with certain
# searchers.
searcher_for_hyperband_strategy = {
    'hyperband': 'random',
    'bayesopt_hyperband': 'bayesopt'}


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

    searcher = searcher_for_hyperband_strategy.get(scheduler_params['searcher'])
    if searcher is not None:
        scheduler_params['searcher'] = searcher
        if epochs is not None:
            scheduler_params['max_t'] = epochs
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
