import logging
import os
import shutil

from autogluon.common import space as ag_space
from autogluon.common.utils.distribute_utils import DistributedContext
from autogluon.common.utils.try_import import try_import_ray

try_import_ray()  # try import ray before importing the remaining contents so we can give proper error messages
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import ray
from ray import air, tune
from ray.tune import ExperimentAnalysis, PlacementGroupFactory
from ray.tune.schedulers import TrialScheduler
from ray.tune.search import SearchAlgorithm, Searcher
from ray.tune.search.sample import Domain

from autogluon.common.utils.resource_utils import ResourceManager

from ..ray.resources_calculator import ResourceCalculator, ResourceCalculatorFactory
from .constants import MAX, MIN
from .exceptions import EmptySearchSpace
from .ray_tune_constants import SCHEDULER_PRESETS, SEARCHER_PRESETS
from .ray_tune_scheduler_factory import SchedulerFactory
from .ray_tune_searcher_factory import SearcherFactory
from .space_converter import RaySpaceConverterFactory

logger = logging.getLogger(__name__)


class RayTuneAdapter(ABC):
    """
    Abstract class to get module specific resource, and to update module specific arguments
    Instance of this class should be passed to `run` to provide custom behavior
    """

    presets = {
        "auto": {"searcher": "random", "scheduler": "FIFO"},
        "bayes": {"searcher": "bayes", "scheduler": "FIFO"},
        "random": {"searcher": "random", "scheduler": "FIFO"},
    }
    supported_searchers = []
    supported_schedulers = []

    def __init__(self):
        self.num_parallel_jobs = None
        self.cpu_per_job = None
        self.gpu_per_job = None
        self.resources_per_trial = None

    @property
    @abstractmethod
    def adapter_type(self):
        raise NotImplementedError

    def get_supported_searchers(self) -> list:
        """
        Some searchers requires reporting status within epochs or checkpointing in the middle of training.
        If the trainable doesn't support those functionality, provide supported_searchers here to warn users HPO might not work as expected.
        Returns a list of supported searchers
        """
        return self.supported_searchers

    def get_supported_schedulers(self) -> list:
        """
        Some schedulers requires reporting status within epochs or checkpointing in the middle of training.
        If the trainable doesn't support those functionality, provide supported_schedulers here to warn users HPO might not work as expected.
        Returns a list of supported schedulers
        """
        return self.supported_schedulers

    def check_user_provided_resources_per_trial(self, resources_per_trial: Optional[dict] = None):
        """Do checks or warnings on user provided resources here"""
        pass

    @abstractmethod
    def get_resource_calculator(self, **kwargs) -> ResourceCalculator:
        """Get resource calculator"""
        raise NotImplementedError

    def update_resource_info(self, resources_info: dict):
        """Get necessary info given resources_info"""
        self.num_parallel_jobs = resources_info.get("num_parallel_jobs", None)
        self.cpu_per_job = resources_info.get("cpu_per_job", None)
        self.gpu_per_job = resources_info.get("gpu_per_job", None)
        self.resources_per_trial = resources_info.get("resources_per_job", None)

    def get_resources_per_trial(
        self,
        total_resources: dict,
        num_samples: int,
        resources_per_trial: Optional[dict] = None,
        minimum_cpu_per_trial: int = 1,
        minimum_gpu_per_trial: float = 0.0,
        model_estimate_memory_usage: Optional[int] = None,
        **kwargs,
    ) -> Union[dict, PlacementGroupFactory]:
        """
        Calculate resources per trial if not specified by the user
        """
        self.check_user_provided_resources_per_trial(resources_per_trial)
        assert isinstance(minimum_cpu_per_trial, int) and minimum_cpu_per_trial >= 1, "minimum_cpu_per_trial must be a integer that is larger than 0"
        assert (
            isinstance(minimum_gpu_per_trial, (int, float)) and minimum_gpu_per_trial >= 0
        ), "minimum_gpu_per_trial must be an integer or float that is equal to or larger than 0"
        num_cpus = total_resources.get("num_cpus", ResourceManager.get_cpu_count_psutil())
        num_gpus = total_resources.get("num_gpus", 0)
        assert (
            num_gpus >= minimum_gpu_per_trial
        ), f"Total num_gpus available: {num_gpus} must be greater or equal to minimum_gpu_per_trial: {minimum_gpu_per_trial}"

        if minimum_gpu_per_trial > 0:
            resources_calculator = self.get_resource_calculator(num_gpus=num_gpus)
        else:
            resources_calculator = self.get_resource_calculator(num_gpus=0)
        resources_info = resources_calculator.get_resources_per_job(
            total_num_cpus=num_cpus,
            total_num_gpus=num_gpus,
            num_jobs=num_samples,
            minimum_cpu_per_job=minimum_cpu_per_trial,
            minimum_gpu_per_job=minimum_gpu_per_trial,
            model_estimate_memory_usage=model_estimate_memory_usage,
            user_resources_per_job=resources_per_trial,
            **kwargs,
        )

        self.update_resource_info(resources_info)
        return self.resources_per_trial

    @abstractmethod
    def trainable_args_update_method(trainable_args: dict) -> dict:
        """
        Update trainable_args being passed to tune.run with correct information for each trial.
        This can differ in forms for different predictor.
        """
        raise NotImplementedError


def run(
    trainable: Callable,
    trainable_args: dict,
    search_space: dict,
    hyperparameter_tune_kwargs: Union[dict, str],
    metric: str,
    mode: str,
    save_dir: str,
    ray_tune_adapter: RayTuneAdapter,
    trainable_is_parallel: bool = False,
    total_resources: Optional[dict] = dict(),
    minimum_cpu_per_trial: int = 1,
    minimum_gpu_per_trial: float = 0.0,
    model_estimate_memory_usage: Optional[int] = None,
    time_budget_s: Optional[float] = None,
    verbose: int = 1,
    tune_config_kwargs: Optional[dict] = None,
    run_config_kwargs: Optional[dict] = None,
) -> tune.ExperimentAnalysis:
    """
    Parse hyperparameter_tune_kwargs
    Init necessary objects, i.e. searcher, scheduler, and ray
    Translate search space
    Calculate resource per trial and update trainable_args accordingly
    Finally tune.run

    Parameters
    ----------
    trainable
        The function used to train the model.
    trainable_args
        Args passed to trainable.
    search_space
        Search space for HPO.
    hyperparameter_tune_kwargs
        User specified HPO options.
    metric
        Name of the monitored metric for HPO.
    mode
        Determines whether objective is minimizing or maximizing the metric.
        Must be one of [min, max].
    save_dir
        Directory to save ray tune results. Ray will write artifacts to save_dir/trial_dir/
        While HPO, ray will chdir to this directory. Therefore, it's important to provide dataset or model saving path as absolute path.
        After HPO, we change back to the original working directory.
        trial_dir name has been overwritten to be trial_id.
        To provide custom trial_dir, pass a custom function to `trial_dirname_creator` as kwargs.
        For example of creating the custom function, refer to `_trial_dirname_creator`.
    ray_tune_adapter
        Adapter to provide necessary custom info to ray tune.
    trainable_is_parallel
        Whether the trainable itself will use ray to run parallel job or not.
    total_resources
        Total resources can be used for HPO.
        If not specified, will use all the resources by default.
    minimum_cpu_per_trial
        Specify minimum number of cpu to use per trial.
    minimum_gpu_per_trial
        Specify minimum number of gpu to use per trial.
        Ray supports usage of fraction of gpu.
    model_estimate_memory_usage
        Provide optional estimate of the model memory usage.
        Calculation of the resources_per_trial might use this info to better distribute resources
    time_budget_s
        Time limit for the HPO.
    verbose
        0 = silent, 1 = only status updates, 2 = status and brief trial results, 3 = status and detailed trial results.
    tune_config_kwargs
        Additional args being passed to tune.TuneConfig https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.tune.tune_config.TuneConfig
    run_config_kwargs
        Additional args being passed to air.RunConfig https://docs.ray.io/en/latest/ray-air/package-ref.html#ray.air.config.RunConfig
    """
    assert mode in [MIN, MAX], f"mode {mode} is not a valid option. Options are {[MIN, MAX]}"
    if isinstance(hyperparameter_tune_kwargs, str):
        assert (
            hyperparameter_tune_kwargs in ray_tune_adapter.presets
        ), f"{hyperparameter_tune_kwargs} is not a valid option. Options are {ray_tune_adapter.presets.keys()}"
        hyperparameter_tune_kwargs = ray_tune_adapter.presets.get(hyperparameter_tune_kwargs)
    num_samples = hyperparameter_tune_kwargs.get("num_trials", None)
    if num_samples is None:
        num_samples = 1 if time_budget_s is None else 1000  # if both num_samples and time_budget_s are None, we only run 1 trial
    if not any(isinstance(search_space[hyperparam], (ag_space.Space, Domain)) for hyperparam in search_space):
        raise EmptySearchSpace
    search_space, default_hyperparameters = _convert_search_space(search_space)

    searcher = _get_searcher(
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        metric=metric,
        mode=mode,
        default_hyperparameters=default_hyperparameters,
        supported_searchers=ray_tune_adapter.get_supported_searchers(),
    )
    scheduler = _get_scheduler(hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, supported_schedulers=ray_tune_adapter.get_supported_schedulers())

    if not ray.is_initialized():
        if DistributedContext.is_distributed_mode():
            ray.init(address="auto")
        else:
            ray.init(
                log_to_driver=False,
                runtime_env={"env_vars": {"PL_DISABLE_FORK": "1"}},  # https://github.com/ray-project/ray/issues/28197
                logging_level=logging.ERROR,  # https://github.com/ray-project/ray/issues/29216
                **total_resources,
            )

    resources_per_trial = hyperparameter_tune_kwargs.get("resources_per_trial", None)
    resources_per_trial = ray_tune_adapter.get_resources_per_trial(
        total_resources=total_resources,
        num_samples=num_samples,
        resources_per_trial=resources_per_trial,
        minimum_gpu_per_trial=minimum_gpu_per_trial,
        minimum_cpu_per_trial=minimum_cpu_per_trial,
        model_estimate_memory_usage=model_estimate_memory_usage,
        wrap_resources_per_job_into_placement_group=trainable_is_parallel,
    )
    resources_per_trial = _validate_resources_per_trial(resources_per_trial)
    logger.debug(f"resources_per_trial to be dispatched by ray tune: {resources_per_trial}")
    ray_tune_adapter.resources_per_trial = resources_per_trial
    trainable_args = ray_tune_adapter.trainable_args_update_method(trainable_args)

    original_path = os.getcwd()
    save_dir = os.path.normpath(save_dir)
    if tune_config_kwargs is None:
        tune_config_kwargs = dict()
    if run_config_kwargs is None:
        run_config_kwargs = dict()
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(trainable, **trainable_args), resources_per_trial),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples, search_alg=searcher, scheduler=scheduler, metric=metric, mode=mode, time_budget_s=time_budget_s, **tune_config_kwargs
        ),
        run_config=air.RunConfig(name=os.path.basename(save_dir), local_dir=os.path.dirname(save_dir), verbose=verbose, **run_config_kwargs),
        _tuner_kwargs={"trial_name_creator": _trial_name_creator, "trial_dirname_creator": _trial_dirname_creator},
    )
    results = tuner.fit()

    os.chdir(original_path)  # go back to the original directory to avoid relative path being broken
    return results._experiment_analysis


def cleanup_trials(save_dir: str, trials_to_keep: Optional[List[str]]):
    """
    Cleanup trial artifacts and keep trials as specified

    Parameters
    ----------
    save_dir
        The path to the root of all the saved trials.
        This should be the same `save_dir` as you passed to `run`
    trials_to_keep
        List of trials to keep.
        Provide the dir name to the trial.
        This should be the same as trial_id if you didn't provide custom `trial_dirname_creator`
    """
    directories = [dir for dir in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, dir))]
    for directory in directories:
        if directory not in trials_to_keep:
            shutil.rmtree(os.path.join(save_dir, directory))


def cleanup_checkpoints(save_dir):
    """
    Cleanup trial artifacts and keep trials as specified

    Parameters
    ----------
    save_dir
        The path to the root of all the saved checkpoints.
        This should be the path of a specific trial.
    """
    directories = [dir for dir in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, dir)) and dir.startswith("checkpoint")]
    for directory in directories:
        shutil.rmtree(os.path.join(save_dir, directory))


def _trial_name_creator(trial):
    return trial.trial_id


def _trial_dirname_creator(trial):
    return trial.trial_id


def _validate_resources_per_trial(resources_per_trial):
    if isinstance(resources_per_trial, dict):
        if "num_cpus" in resources_per_trial:
            resources_per_trial["cpu"] = resources_per_trial.pop("num_cpus")
        if "num_gpus" in resources_per_trial:
            resources_per_trial["gpu"] = resources_per_trial.pop("num_gpus")
    return resources_per_trial


def _convert_search_space(search_space: dict):
    """Convert the search space to Ray Tune search space if it's AG search space"""
    tune_search_space = search_space.copy()
    default_hyperparameters = dict()
    for hyperparameters, space in search_space.items():
        if isinstance(space, ag_space.Space):
            tune_search_space[hyperparameters] = RaySpaceConverterFactory.get_space_converter(space.__class__.__name__).convert(space)
            default_hyperparameters[hyperparameters] = space.default
    default_hyperparameters = default_hyperparameters
    if len(default_hyperparameters) == 0:
        # hyperopt + ray have trouble taking in empty default_hyperparameters
        default_hyperparameters = None
    else:
        default_hyperparameters = [default_hyperparameters]
    return tune_search_space, default_hyperparameters


def _get_searcher(
    hyperparameter_tune_kwargs: dict,
    metric: str,
    mode: str,
    default_hyperparameters: Optional[List[dict]] = None,
    supported_searchers: Optional[List[str]] = None,
):
    """Initialize searcher object"""
    searcher = hyperparameter_tune_kwargs.get("searcher")
    user_init_args = hyperparameter_tune_kwargs.get("searcher_init_args", dict())
    if isinstance(searcher, str):
        assert searcher in SEARCHER_PRESETS, f"{searcher} is not a valid option. Options are {SEARCHER_PRESETS.keys()}"
        # Check supported schedulers for str input
        if supported_searchers is not None:
            if searcher not in supported_searchers:
                logger.warning(f"{searcher} is not supported yet. Using it might behave unexpected. Supported options are {supported_searchers}")
        searcher = SearcherFactory.get_searcher(
            searcher_name=searcher,
            user_init_args=user_init_args,
            metric=metric,
            mode=mode,
            points_to_evaluate=default_hyperparameters,
        )
    assert isinstance(searcher, (SearchAlgorithm, Searcher)) and searcher.__class__ in SEARCHER_PRESETS.values()
    # Check supported schedulers for obj input
    if supported_searchers is not None:
        supported_searchers_cls = [SEARCHER_PRESETS[searchers] for searchers in supported_searchers]
        if searcher.__class__ not in supported_searchers_cls:
            logger.warning(f"{searcher.__class__} is not supported yet. Using it might behave unexpected. Supported options are {supported_searchers_cls}")
    return searcher


def _get_scheduler(hyperparameter_tune_kwargs: dict, supported_schedulers: Optional[List[str]] = None):
    """Initialize scheduler object"""
    scheduler = hyperparameter_tune_kwargs.get("scheduler")
    user_init_args = hyperparameter_tune_kwargs.get("scheduler_init_args", dict())
    if isinstance(scheduler, str):
        assert scheduler in SCHEDULER_PRESETS, f"{scheduler} is not a valid option. Options are {SCHEDULER_PRESETS.keys()}"
        # Check supported schedulers for str input
        if supported_schedulers is not None:
            if scheduler not in supported_schedulers:
                logger.warning(f"{scheduler} is not supported yet. Using it might behave unexpected. Supported options are {supported_schedulers}")
        scheduler = SchedulerFactory.get_scheduler(
            scheduler_name=scheduler,
            user_init_args=user_init_args,
        )
    assert isinstance(scheduler, TrialScheduler) and scheduler.__class__ in SCHEDULER_PRESETS.values()
    # Check supported schedulers for obj input
    if supported_schedulers is not None:
        supported_schedulers_cls = [SCHEDULER_PRESETS[scheduler] for scheduler in supported_schedulers]
        if scheduler.__class__ not in supported_schedulers_cls:
            logger.warning(f"{scheduler.__class__} is not supported yet. Using it might behave unexpected. Supported options are {supported_schedulers_cls}")
    return scheduler


class TabularRayTuneAdapter(RayTuneAdapter):
    supported_searchers = ["random", "bayes"]
    supported_schedulers = ["FIFO"]

    @property
    def adapter_type(self):
        return "tabular"

    def get_resource_calculator(self, num_gpus, **kwargs) -> ResourceCalculator:
        return ResourceCalculatorFactory.get_resource_calculator(calculator_type="cpu" if num_gpus == 0 else "gpu")

    def trainable_args_update_method(self, trainable_args: dict) -> dict:
        if isinstance(self.resources_per_trial, dict):
            trainable_args["fit_kwargs"]["num_cpus"] = self.resources_per_trial.get("cpu", 1)
            trainable_args["fit_kwargs"]["num_gpus"] = self.resources_per_trial.get("gpu", 0)
        elif isinstance(self.resources_per_trial, tune.PlacementGroupFactory):
            required_resources = self.resources_per_trial.required_resources
            trainable_args["fit_kwargs"]["num_cpus"] = required_resources.get("CPU", 1)
            trainable_args["fit_kwargs"]["num_gpus"] = required_resources.get("GPU", 0)
        return trainable_args


class AutommRayTuneAdapter(RayTuneAdapter):
    supported_searchers = ["random", "bayes"]
    supported_schedulers = ["FIFO", "ASHA"]

    def __init__(self):
        super().__init__()

    @property
    def adapter_type(self):
        return "automm"

    def check_user_provided_resources_per_trial(self, resources_per_trial: Optional[dict] = None):
        if resources_per_trial is not None:
            # We do not support a single trial running on multiple GPUs without ray_lightning for now.
            num_gpus = resources_per_trial.get("num_gpus", None)
            if num_gpus is not None and num_gpus > 1:
                resources_per_trial["num_gpus"] = 1
                logger.warning("We do not support a single trial running on multiple GPUs yet")
            return resources_per_trial

    def get_resource_calculator(self, num_gpus: float):
        return ResourceCalculatorFactory.get_resource_calculator(calculator_type="cpu" if num_gpus == 0 else "non_parallel_gpu")

    def trainable_args_update_method(self, trainable_args: dict) -> dict:
        trainable_args["hyperparameters"]["env.num_gpus"] = self.gpu_per_job
        trainable_args["hyperparameters"]["env.num_workers"] = self.cpu_per_job

        return trainable_args


class TimeSeriesRayTuneAdapter(TabularRayTuneAdapter):
    supported_searchers = ["random", "bayes"]
    supported_schedulers = ["FIFO"]

    @property
    def adapter_type(self):
        return "timeseries"


class RayTuneAdapterFactory:
    __supported_adapters = [
        TabularRayTuneAdapter,
        TimeSeriesRayTuneAdapter,
        AutommRayTuneAdapter,
    ]

    __type_to_adapter = {cls().adapter_type: cls for cls in __supported_adapters}

    @staticmethod
    def get_adapter(adapter_type: str) -> RayTuneAdapter:
        """Return the executor"""
        assert adapter_type in RayTuneAdapterFactory.__type_to_adapter, f"{adapter_type} not supported"
        return RayTuneAdapterFactory.__type_to_adapter[adapter_type]
