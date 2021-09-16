import inspect
import logging

from ..task.base import compile_scheduler_options_v2
from ..task.base.base_task import schedulers

logger = logging.getLogger()


_scheduler_presets = {
    'auto': {'scheduler': 'local', 'searcher': 'bayesopt'},
    'random': {'scheduler': 'local', 'searcher': 'random'},
    'bayesopt': {'scheduler': 'local', 'searcher': 'bayesopt'},
    # Don't include hyperband and bayesopt hyperband at present
}


def scheduler_factory(
        hyperparameter_tune_kwargs,
        time_out: float = None,
        num_trials: int = None,
        nthreads_per_trial='all',
        ngpus_per_trial='all',
        **kwargs):
    """
    Constructs a scheduler via lazy initialization based on the input hyperparameter_tune_kwargs.
    The output will contain the scheduler class and init arguments except for the `train_fn` argument, which must be specified downstream.

    Parameters
    ----------
    hyperparameter_tune_kwargs : str or dict
        Hyperparameter tuning strategy and kwargs.
        If None, then hyperparameter tuning will not be performed.
        Valid preset values:
            'auto': Uses the 'bayesopt' preset.
            'random': Performs HPO via random search using local scheduler.
            'bayesopt': Performs HPO via bayesian optimization using local scheduler.
        For valid dictionary keys, refer to :class:`autogluon.core.scheduler.FIFOScheduler` documentation.
            The 'searcher' key is required when providing a dict.
            Some schedulers may have different valid keys.
    time_out : float, default = None
        Same as hyperparameter_tune_kwargs['time_out']. Ignored if specified in hyperparameter_tune_kwargs.
        At least one of time_out or num_trials must be specified.
    num_trials : int, default = None
        Same as hyperparameter_tune_kwargs['time_out']. Ignored if specified in hyperparameter_tune_kwargs.
        At least one of time_out or num_trials must be specified.
    nthreads_per_trial : str or int, default = 'all'
        Same as hyperparameter_tune_kwargs['nthreads_per_trial']. Ignored if specified in hyperparameter_tune_kwargs.
        Number of CPU threads to use per HPO trial.
        Valid str values:
            'all': Use all CPU threads.
            'auto': Keep value as 'auto' in output, must be updated downstream.
        If None, use all CPU threads as in 'all'.
    ngpus_per_trial : str or int, default = 'all'
        Same as hyperparameter_tune_kwargs['ngpus_per_trial']. Ignored if specified in hyperparameter_tune_kwargs.
        Number of GPUs to use per HPO trial.
        Valid str values:
            'all': Use all GPUs.
            'auto': Keep value as 'auto' in output, must be updated downstream.
        If None, use 0 GPUs.
    **kwargs :
        Kwargs to specify any other scheduler parameters.
        A kwarg will be ignored if also specified in hyperparameter_tune_kwargs.

    Returns
    -------
    scheduler_cls : class :class:`autogluon.core.scheduler.TaskScheduler`, scheduler_params : dict
        scheduler_cls is the class of the scheduler that will be constructed.
        scheduler_params is the key word parameter arguments to pass to the Scheeduler class constructor when initializing a Scheduler object.
        To actually construct a Scheduler object, call `scheduler_cls(train_fn, **scheduler_params)`
        By default in scheduler_params: time_attr='epoch', reward_attr='validation_performance'

    Examples
    --------
    >>> import numpy as np
    >>> import autogluon.core as ag
    >>> from autogluon.core.scheduler.scheduler_factory import scheduler_factory
    >>>
    >>>
    >>> @ag.args()
    >>> def train_fn(args, reporter):
    >>>     for e in range(args.epochs):
    >>>         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
    >>>         reporter(epoch=e+1, validation_performance=dummy_accuracy, lr=args.lr, wd=args.wd)
    >>>
    >>> hyperparameters = dict(
    >>>     lr=ag.space.Real(1e-3, 1e-2, log=True),
    >>>     wd=ag.space.Real(1e-3, 1e-2),
    >>>     epochs=10,
    >>> )
    >>>
    >>> scheduler_cls, scheduler_params = scheduler_factory('auto', num_trials=5)
    >>>
    >>> train_fn.register_args(**hyperparameters)  # Register search space
    >>> scheduler = scheduler_cls(train_fn, **scheduler_params)
    >>>
    >>> scheduler.run()
    >>> scheduler.join_jobs()
    >>> scheduler.get_training_curves(plot=True)
    """
    if hyperparameter_tune_kwargs is None:
        raise ValueError(f"hyperparameter_tune_kwargs cannot be None.")
    if isinstance(hyperparameter_tune_kwargs, str):
        hyperparameter_tune_kwargs = get_hyperparameter_tune_kwargs_preset(hyperparameter_tune_kwargs)
    if not isinstance(hyperparameter_tune_kwargs, dict):
        raise ValueError(f"hyperparameter_tune_kwargs must be of type str or dict, but is type: {type(hyperparameter_tune_kwargs)}")
    if 'scheduler' not in hyperparameter_tune_kwargs:
        raise ValueError(f"Required key 'scheduler' is not present in hyperparameter_tune_kwargs: {hyperparameter_tune_kwargs}")
    if 'searcher' not in hyperparameter_tune_kwargs:
        raise ValueError(f"Required key 'searcher' is not present in hyperparameter_tune_kwargs: {hyperparameter_tune_kwargs}")
    if num_trials is None and time_out is not None:
        num_trials = 1000

    scheduler_params = compile_scheduler_options_v2(
        scheduler_options=hyperparameter_tune_kwargs,
        nthreads_per_trial=nthreads_per_trial,
        ngpus_per_trial=ngpus_per_trial,
        num_trials=num_trials,
        time_out=time_out,
        **kwargs,
    )

    scheduler_cls = hyperparameter_tune_kwargs.get('scheduler', 'unknown')
    if isinstance(scheduler_cls, str):
        scheduler_cls = get_scheduler_from_preset(scheduler_cls)
    if not inspect.isclass(scheduler_cls):
        raise ValueError(f'scheduler_cls must be a class, but was instead: {scheduler_cls}')

    if scheduler_params['time_out'] is None:
        scheduler_params.pop('time_out', None)
    return scheduler_cls, scheduler_params


def get_scheduler_from_preset(scheduler_cls):
    scheduler_cls = scheduler_cls.lower()
    if scheduler_cls not in schedulers.keys():
        raise ValueError(f"Required key 'scheduler' in hyperparameter_tune_kwargs must be one of the "
                         f"values {schedulers.keys()}, but was instead: {scheduler_cls}")
    scheduler_cls = schedulers.get(scheduler_cls)
    return scheduler_cls


def get_hyperparameter_tune_kwargs_preset(preset: str):
    if preset not in _scheduler_presets:
        raise ValueError(f'Invalid hyperparameter_tune_kwargs preset value "{preset}". Valid presets: {list(_scheduler_presets.keys())}')
    return _scheduler_presets[preset].copy()
