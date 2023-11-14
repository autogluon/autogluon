import copy
import inspect
import logging

from ..utils.utils import setup_compute
from .seq_scheduler import LocalSequentialScheduler

logger = logging.getLogger(__name__)

schedulers = {
    "local": LocalSequentialScheduler,
}

_scheduler_presets = {
    "auto": {"scheduler": "local", "searcher": "local_random"},
    "local_random": {"scheduler": "local", "searcher": "local_random"},
    "random": {"scheduler": "local", "searcher": "random"},
}


def compile_scheduler_options(
    scheduler_options,
    nthreads_per_trial,
    ngpus_per_trial,
    num_trials,
    time_out,
    scheduler=None,
    search_strategy=None,
    search_options=None,
    checkpoint=None,
    resume=False,
    visualizer=None,
    time_attr=None,
    reward_attr=None,
    dist_ip_addrs=None,
    epochs=None,
):
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
        search_strategy = "random"
    if scheduler is None:
        scheduler = "local"
    assert isinstance(search_strategy, str)
    if search_options is None:
        search_options = dict()
    if visualizer is None:
        visualizer = "none"
    if time_attr is None:
        time_attr = "epoch"
    if reward_attr is None:
        reward_attr = "validation_performance"
    scheduler_params = {
        "resource": {"num_cpus": nthreads_per_trial, "num_gpus": ngpus_per_trial},
        "scheduler": scheduler,
        "searcher": search_strategy,
        "search_options": search_options,
        "checkpoint": checkpoint,
        "resume": resume,
        "num_trials": num_trials,
        "time_out": time_out,
        "reward_attr": reward_attr,
        "time_attr": time_attr,
        "visualizer": visualizer,
        "dist_ip_addrs": dist_ip_addrs,
    }
    resource = None
    if "resource" in scheduler_options:
        scheduler_params["resource"].update(scheduler_options["resource"])
        resource = scheduler_params["resource"].copy()
    scheduler_params.update(scheduler_options)
    if resource:
        scheduler_params["resource"] = resource

    scheduler_params["resource"]["num_cpus"], scheduler_params["resource"]["num_gpus"] = setup_compute(
        nthreads_per_trial=scheduler_params["resource"]["num_cpus"],
        ngpus_per_trial=scheduler_params["resource"]["num_gpus"],
    )  # TODO: use 'auto' downstream

    required_options = [
        "resource",
        "scheduler",
        "searcher",
        "search_options",
        "checkpoint",
        "resume",
        "num_trials",
        "time_out",
        "reward_attr",
        "time_attr",
        "visualizer",
        "dist_ip_addrs",
    ]
    missing_options = []
    for option in required_options:
        if option not in scheduler_params:
            missing_options.append(option)
    if missing_options:
        raise AssertionError(f"Missing required keys in scheduler_options: {missing_options}")
    return scheduler_params


def scheduler_factory(hyperparameter_tune_kwargs, time_out: float = None, num_trials: int = None, nthreads_per_trial="all", ngpus_per_trial="all", **kwargs):
    """
    Constructs a scheduler via lazy initialization based on the input hyperparameter_tune_kwargs.
    The output will contain the scheduler class and init arguments except for the `train_fn` argument, which must be specified downstream.

    Parameters
    ----------
    hyperparameter_tune_kwargs : str or dict
        Hyperparameter tuning strategy and kwargs.
        If None, then hyperparameter tuning will not be performed.
        Valid preset values:
            'auto': Uses the 'random' preset.
            'random': Performs HPO via random search using local scheduler.
        The 'searcher' key is required when providing a dict. Some schedulers may have different valid keys.
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
    """
    if hyperparameter_tune_kwargs is None:
        raise ValueError(f"hyperparameter_tune_kwargs cannot be None.")
    if isinstance(hyperparameter_tune_kwargs, str):
        hyperparameter_tune_kwargs = get_hyperparameter_tune_kwargs_preset(hyperparameter_tune_kwargs)
    if not isinstance(hyperparameter_tune_kwargs, dict):
        raise ValueError(f"hyperparameter_tune_kwargs must be of type str or dict, but is type: {type(hyperparameter_tune_kwargs)}")
    if "scheduler" not in hyperparameter_tune_kwargs:
        raise ValueError(f"Required key 'scheduler' is not present in hyperparameter_tune_kwargs: {hyperparameter_tune_kwargs}")
    if "searcher" not in hyperparameter_tune_kwargs:
        raise ValueError(f"Required key 'searcher' is not present in hyperparameter_tune_kwargs: {hyperparameter_tune_kwargs}")
    if num_trials is None and time_out is not None:
        num_trials = 1000

    scheduler_params = compile_scheduler_options(
        scheduler_options=hyperparameter_tune_kwargs,
        nthreads_per_trial=nthreads_per_trial,
        ngpus_per_trial=ngpus_per_trial,
        num_trials=num_trials,
        time_out=time_out,
        **kwargs,
    )

    scheduler_cls = hyperparameter_tune_kwargs.get("scheduler", "unknown")
    if isinstance(scheduler_cls, str):
        scheduler_cls = get_scheduler_from_preset(scheduler_cls)
    if not inspect.isclass(scheduler_cls):
        raise ValueError(f"scheduler_cls must be a class, but was instead: {scheduler_cls}")

    if scheduler_params["time_out"] is None:
        scheduler_params.pop("time_out", None)
    return scheduler_cls, scheduler_params


def get_scheduler_from_preset(scheduler_cls):
    scheduler_cls = scheduler_cls.lower()
    if scheduler_cls not in schedulers.keys():
        raise ValueError(
            f"Required key 'scheduler' in hyperparameter_tune_kwargs must be one of the " f"values {schedulers.keys()}, but was instead: {scheduler_cls}"
        )
    scheduler_cls = schedulers.get(scheduler_cls)
    return scheduler_cls


def get_hyperparameter_tune_kwargs_preset(preset: str):
    # TODO: re-enable bayesopt after it's been implemented
    if preset == "bayesopt":
        logger.warning(f"Bayesopt hyperparameter tuning is currently disabled. Will use random hyperparameter tuning instead.")
        preset = "random"
    if preset not in _scheduler_presets:
        raise ValueError(f'Invalid hyperparameter_tune_kwargs preset value "{preset}". Valid presets: {list(_scheduler_presets.keys())}')
    return _scheduler_presets[preset].copy()
