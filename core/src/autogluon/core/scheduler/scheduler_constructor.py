import logging

from ..task.base import compile_scheduler_options_v2
from ..task.base.base_task import schedulers
from ..utils.utils import setup_compute

logger = logging.getLogger()


_scheduler_presets = {
    'auto': {'searcher': 'random'},
    # 'grid': {'searcher': 'grid'},  # grid commented out as it isn't compatible with most default model search spaces
    'random': {'searcher': 'random'},
    'bayesopt': {'searcher': 'bayesopt'},
    # 'skopt': {'searcher': 'skopt'},  # TODO: Remove skopt? Is it worthwhile to keep as an option?
    # Don't include hyperband and bayesopt hyperband at present
}


def init_scheduler(hyperparameter_tune_kwargs, time_limit=None, num_trials=None, num_cpus=None, num_gpus=None, **kwargs):
    if hyperparameter_tune_kwargs is None:
        return None
    if isinstance(hyperparameter_tune_kwargs, str):
        hyperparameter_tune_kwargs = get_hyperparameter_tune_kwargs_preset(hyperparameter_tune_kwargs)
    if not isinstance(hyperparameter_tune_kwargs, dict):
        raise ValueError(f"hyperparameter_tune_kwargs must be of type str or dict, but is type: {type(hyperparameter_tune_kwargs)}")
    if 'searcher' not in hyperparameter_tune_kwargs:
        raise ValueError(f"Required key 'searcher' is not present in hyperparameter_tune_kwargs: {hyperparameter_tune_kwargs}")
    if num_trials is None:
        num_trials = 1000
    num_cpus, num_gpus = setup_compute(num_cpus, num_gpus)  # TODO: use 'auto' downstream

    scheduler_options = compile_scheduler_options_v2(
        scheduler_options=hyperparameter_tune_kwargs,
        nthreads_per_trial=num_cpus,
        ngpus_per_trial=num_gpus,
        num_trials=num_trials,
        time_out=time_limit,
        **kwargs,
    )
    if scheduler_options is None:
        return None

    scheduler_cls = schedulers[scheduler_options['searcher'].lower()]
    if scheduler_options['time_out'] is None:
        scheduler_options.pop('time_out', None)
    scheduler_options = (scheduler_cls, scheduler_options)  # wrap into tuple
    return scheduler_options


def get_hyperparameter_tune_kwargs_preset(preset: str):
    if preset not in _scheduler_presets:
        raise ValueError(f'Invalid hyperparameter_tune_kwargs preset value "{preset}". Valid presets: {list(_scheduler_presets.keys())}')
    return _scheduler_presets[preset].copy()
