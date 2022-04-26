import logging
import math
import os
import psutil

from abc import ABC, abstractmethod
from typing import Optional, Callable, Union, List

from .. import Space
from ..utils.try_import import try_import_ray

try_import_ray()
import ray

from ray import tune
from ray.tune import PlacementGroupFactory
from ray.tune.schedulers import TrialScheduler, FIFOScheduler, AsyncHyperBandScheduler
from ray.tune.suggest import SearchAlgorithm, Searcher
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray_lightning.tune import get_tune_resources



logger = logging.getLogger(__name__)

MIN = 'min'
MAX = 'max'

searcher_presets = {
    'random': BasicVariantGenerator,
    'bayes': HyperOptSearch,
}
scheduler_presets = {
    'FIFO': FIFOScheduler,
    'ASHA': AsyncHyperBandScheduler,
}

class EmptySearchSpace(Exception):
    pass


class RayTuneResourcesCalculator(ABC):
    
    @staticmethod
    @abstractmethod
    def get_resources_per_trial(total_resources: dict, num_samples: int, **kwargs) -> Union[dict, PlacementGroupFactory]:
        """
        Calculate resources per trial if not specified by the user
        """
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def resources_per_trial_update_method(trainable_args: dict, resources_per_trial: Union[dict, PlacementGroupFactory]) -> dict:
        """
        Update trainable_args being passed to tune.run with correct resources for each trial. This can differ in forms for different predictor
        """
        raise NotImplementedError


def run(
    trainable: Callable,
    trainable_args: dict,
    search_space: dict,
    hyperparameter_tune_kwargs: dict,
    metric: str,
    mode: str,  # must be one of [min, max]
    save_dir: str,  # directory to save results. Ray will write artifacts to save_dir/trial_dir/
    resource_calculator: RayTuneResourcesCalculator,
    total_resources: Optional(dict) = dict(),
    minimum_cpu_per_trial: int = 1,
    minimum_gpu_per_trial: float = 1.0,
    model_estimate_memroy_usage: Optional(int) = None,
    time_budget_s: Optional(float) = None,
    supported_searchers: Optional(List[str]) = None,
    supported_schedulers: Optional(List[str]) = None,
    verbose: int = 1,  # 0 = silent, 1 = only status updates, 2 = status and brief trial results, 3 = status and detailed trial results.
    **kwargs  # additional args being passed to tune.run
    ) -> tune.ExperimentAnalysis:
    """
    Parse hyperparameter_tune_kwargs
    Init necessary objects, i.e. searcher, scheduler, and ray
    Translate search space
    Calculate resource per trial
    Finally tune.run
    local_dir/experiment_name/trial_dir
    """
    assert mode in [MIN, MAX]
    num_samples = hyperparameter_tune_kwargs.get('num_trials', hyperparameter_tune_kwargs.get('num_samples', None))
    if num_samples is None:
        num_samples = 1 if time_budget_s is None else 1000  # if both num_samples and time_budget_s are None, we only run 1 trial
    search_space = _convert_search_space(search_space)
    if not search_space:
        raise EmptySearchSpace
    searcher = _get_searcher(hyperparameter_tune_kwargs, metric, mode, supported_searchers=supported_searchers)
    scheduler = _get_scheduler(hyperparameter_tune_kwargs, metric, mode, supported_schedulers=supported_schedulers)
    resources_per_trial = hyperparameter_tune_kwargs.get('resources_per_trial', None)
    if resources_per_trial is None:
        resources_per_trial = resource_calculator.get_resources_per_trial(
            total_resources=total_resources,
            num_samples=num_samples,
            minimum_gpu_per_trial=minimum_gpu_per_trial,
            minimum_cpu_per_trial=minimum_cpu_per_trial,
            model_estimate_memroy_usage=model_estimate_memroy_usage
        )
    trainable_args = resource_calculator.resources_per_trial_update_method(trainable_args, resources_per_trial)
    tune_kwargs = _get_default_tune_kwargs()
    tune_kwargs.update(kwargs)

    original_path = os.getcwd()
    if 'original_path' not in trainable_args:
        trainable_args['original_path'] = original_path
    
    if not ray.is_initialized():
        ray.init(**total_resources)
    analysis = tune.run(
        tune.with_parameters(trainable, **trainable_args),
        config=search_space,
        num_samples=num_samples,
        search_alg=searcher,
        scheduler=scheduler,
        metric=metric,
        mode=mode,
        time_budget_s=time_budget_s,
        resources_per_trial=resources_per_trial,
        verbose=verbose,
        local_dir=original_path,
        name=save_dir,
        trial_dirname_creator=_trial_dirname_creator,
        trial_name_creator=_trial_name_creator,
        **tune_kwargs
    )
    return analysis


def cleanup_trials(experiment_dir: str, trials_to_keep: Optional(List[str])):
    """Cleanup trial artifacts and keep trials as specified"""
    pass


def _trial_name_creator(trial):
    return trial.trial_id


def _trial_dirname_creator(trial):
    return trial.trial_id


def _get_default_tune_kwargs():
    kwargs = dict(
        raise_on_failed_trial=False,
        trial_name_creator=_trial_name_creator,
        trial_dirname_creator=_trial_dirname_creator,
    )
    return kwargs


def _convert_search_space(search_space: dict):
    """Convert the search space to Ray Tune search space if it's AG search space"""
    tune_search_space = search_space.copy()
    for hyperparmaeter, space in search_space.items():
        if isinstance(space, Space):
            tune_search_space[hyperparmaeter] = space.convert_to_ray()
    return tune_search_space


def _get_searcher(hyperparameter_tune_kwargs: dict, metric: str, mode: str, supported_searchers: Optional(List[str])=None):
    """Initialize searcher object"""
    searcher = hyperparameter_tune_kwargs.get('searcher')
    user_init_args = hyperparameter_tune_kwargs.get('searcher_init_args', dict())
    if isinstance(searcher, str):
        assert searcher in searcher_presets, f'{searcher} is not a valid option. Options are {searcher_presets.keys()}'
        # Check supported schedulers for str input
        if supported_searchers is not None:
            if searcher not in supported_searchers:
                logger.warning(f'{searcher} is not supported yet. Using it might behave unexpected. Supported options are {supported_searchers}')
        searcher_cls = searcher_presets.get(searcher)
        init_args = dict()
        if searcher_cls == HyperOptSearch:
            init_args = dict(metirc=metric, mode=mode)
            init_args.update(user_init_args)
        searcher = searcher_cls(**init_args)
    assert isinstance(searcher, (SearchAlgorithm, Searcher)) and searcher.__class__ in searcher_presets.values()
    # Check supported schedulers for obj input
    if supported_searchers is not None:
        supported_searchers_cls = [scheduler_presets[scheduler] for scheduler in supported_searchers]
        if searcher.__class__ in supported_searchers_cls:
            logger.warning(f'{searcher.__class__} is not supported yet. Using it might behave unexpected. Supported options are {supported_searchers_cls}')
    return searcher


def _get_scheduler(hyperparameter_tune_kwargs: dict, metric: str, mode: str, supported_schedulers: Optional(List[str])=None):
    """Initialize scheduler object"""
    scheduler = hyperparameter_tune_kwargs.get('scheduler')
    user_init_args = hyperparameter_tune_kwargs.get('scheduler_init_args', dict())
    if isinstance(scheduler, str):
        assert scheduler in scheduler_presets, f'{scheduler} is not a valid option. Options are {scheduler_presets.keys()}'
        # Check supported schedulers for str input
        if supported_schedulers is not None:
            if scheduler not in supported_schedulers:
                logger.warning(f'{scheduler} is not supported yet. Using it might behave unexpected. Supported options are {supported_schedulers}')
        scheduler_cls = scheduler_presets.get(scheduler)
        init_args = dict()
        if scheduler_cls == AsyncHyperBandScheduler:
            init_args = dict(metric=metric, mode=mode, max_t=9999)
            init_args.update(user_init_args)
        scheduler = scheduler_cls(**init_args)
    assert isinstance(scheduler, TrialScheduler) and scheduler.__class__ in scheduler_presets.values()
    # Check supported schedulers for obj input
    if supported_schedulers is not None:
        supported_schedulers_cls = [scheduler_presets[scheduler] for scheduler in supported_schedulers]
        if scheduler.__class__ in supported_schedulers_cls:
            logger.warning(f'{scheduler.__class__} is not supported yet. Using it might behave unexpected. Supported options are {supported_schedulers_cls}')
    return scheduler
    
    
class TabularRayTuneResourcesCalculator(RayTuneResourcesCalculator):
    
    @staticmethod
    def get_resources_per_trial(
        total_resources: dict,
        num_samples: int,
        minimum_cpu_per_trial: int = 1,
        minimum_gpu_per_trial: float = 1.0,
        model_estimate_memroy_usage: Optional(int) = None,
        **kwargs,
        ) -> Union[dict, PlacementGroupFactory]:
        assert isinstance(minimum_cpu_per_trial, int) and minimum_cpu_per_trial >= 1, 'minimum_cpu_per_trial must be a interger that is larger than 0'
        num_cpus = total_resources.get('num_cpus', psutil.cpu_count())
        num_gpus = total_resources.get('num_gpus', 0)
        cpu_per_job = max(minimum_cpu_per_trial, int(num_cpus // num_samples))
        gpu_per_job = 0
        max_jobs_in_parallel_memory = num_samples

        if model_estimate_memroy_usage is not None:
            mem_available = psutil.virtual_memory().available
            # calculate how many jobs can run in parallel given memory available
            max_jobs_in_parallel_memory = max(minimum_cpu_per_trial, int(mem_available // model_estimate_memroy_usage))
        num_parallel_jobs = min(num_samples, num_cpus // cpu_per_job, max_jobs_in_parallel_memory)
        cpu_per_job = max(minimum_cpu_per_trial, int(num_cpus // num_parallel_jobs))  # update cpu_per_job in case memory is not enough and can use more cores for each job
        resources = dict(cpu=cpu_per_job)
        logger.log(20, f"Will run {num_parallel_jobs} jobs in parallel given number of cpu cores, memory avaialable, and user specification")

        if num_gpus > 0:
            gpu_per_job = max(minimum_gpu_per_trial, num_gpus / num_parallel_jobs)
        resources = dict(cpu=cpu_per_job, gpu=gpu_per_job)
        
        return resources
    
    @staticmethod
    def resources_per_trial_update_method(trainable_args: dict, resources_per_trial: Union[dict, PlacementGroupFactory]) -> dict:
        assert isinstance(resources_per_trial, dict)
        trainable_args['fit_kwargs']['num_cpus'] = resources_per_trial.get('cpu', 1)
        trainable_args['fit_kwargs']['num_gpus'] = resources_per_trial.get('gpu', 0)
        return trainable_args
    
    
class AutommRayTuneResourcesCalculator(RayTuneResourcesCalculator):
    
    @staticmethod
    def get_resources_per_trial(
        total_resources: dict,
        num_samples: int,
        minimum_cpu_per_trial: int = 1,
        minimum_gpu_per_trial: float = 1.0,
        model_estimate_memroy_usage: Optional(int) = None,
        **kwargs,
        ) -> Union[dict, PlacementGroupFactory]:
        # Ray Tune requires 1 additional CPU per trial to use for the Trainable driver. 
        # So the actual number of cpu resources each trial requires is num_workers * num_cpus_per_worker + 1
        # Each ray worker will reserve 1 gpu
        # The num_workers in ray stands for worker process to train the model
        # The num_workers in AutoMM stands for worker process to load data
        assert isinstance(minimum_cpu_per_trial, int) and minimum_cpu_per_trial >= 1, 'minimum_cpu_per_trial must be an interger that is larger than 0'
        num_cpus = total_resources.get('num_cpus', psutil.cpu_count())
        num_gpus = total_resources.get('num_gpus', 0)
        num_cpus = (num_cpus - num_parallel_jobs)  # reserve cpus for the master process
        cpu_per_job = None
        gpu_per_job = None
        num_workers = None
        if num_gpus > 0:
            gpu_per_job = max(int(minimum_gpu_per_trial), num_gpus // num_samples)
            num_workers = gpu_per_job  # each worker uses 1 gpu
            num_parallel_jobs = min(num_samples, num_gpus // gpu_per_job)
            assert num_cpus > 0
            cpu_per_job = max(minimum_cpu_per_trial, num_cpus // num_parallel_jobs)
            cpu_per_worker = max(1, cpu_per_job // num_workers)
        else:
            # TODO: for cpu case, is it better to have more workers or more cpus per worker?
            cpu_per_job = max(minimum_cpu_per_trial, num_cpus // num_samples)
            num_workers = cpu_per_job
            cpu_per_worker = 1
        resources_per_trial = get_tune_resources(
            num_workers=num_workers,
            num_cpus_per_worker=cpu_per_worker,
            use_gpu=(gpu_per_job is not None)
        )
        return resources_per_trial
        
    @staticmethod
    def resources_per_trial_update_method(trainable_args: dict, resources_per_trial: Union[dict, PlacementGroupFactory]) -> dict:
        assert isinstance(resources_per_trial, PlacementGroupFactory)
        trainable_args['hyperparameters']['env.num_gpus'] = resources_per_trial.required_resources.get('GPU', 0)
        trainable_args['hyperparameters']['env.num_workers'] = resources_per_trial.required_resources.get('CPU') - 1  # 1 cpu reserved for master process, which doesn't do training
        trainable_args['hyperparameters']['env.num_nodes'] = 1  # num_nodes is not needed by ray lightning. Setting it to default, which is 1
        return trainable_args

# def _get_resources_per_trial(
#     total_resources,
#     num_samples,
#     minimum_cpu_per_trial=1,
#     minimum_gpu_per_trial=1.0,  # ray allows to use partial gpu, but we are not likely to allow 0.01 gpu per trial for example
#     model_estimate_memroy_usage=None,
#     ):
#     """
#     Calculate resources per trial if not specified by the user
#     """
#     assert isinstance(minimum_cpu_per_trial, int) and minimum_cpu_per_trial >= 1, 'minimum_cpu_per_trial must be a interger that is larger than 0'
#     num_cpus = total_resources.get('num_cpus')
#     num_gpus = total_resources.get('num_gpus')
#     cpu_per_job = max(minimum_cpu_per_trial, int(num_cpus // num_samples))
#     gpu_per_job = 0
#     max_jobs_in_parallel_memory = num_samples

#     if model_estimate_memroy_usage is not None:
#         mem_available = psutil.virtual_memory().available
#         # calculate how many jobs can run in parallel given memory available
#         max_jobs_in_parallel_memory = max(minimum_cpu_per_trial, int(mem_available // model_estimate_memroy_usage))
#     num_parallel_jobs = min(num_samples, num_cpus // cpu_per_job, max_jobs_in_parallel_memory)
#     cpu_per_job = max(minimum_cpu_per_trial, int(num_cpus // num_parallel_jobs))  # update cpu_per_job in case memory is not enough and can use more cores for each job
#     resources = dict(cpu=cpu_per_job)
#     logger.log(20, f"Will run {num_parallel_jobs} jobs in parallel given number of cpu cores, memory avaialable, and user specification")

#     batches = math.ceil(num_samples / num_parallel_jobs)
#     if num_gpus > 0:
#         gpu_per_job = max(minimum_gpu_per_trial, num_gpus / num_parallel_jobs)
#     resources = dict(cpu=cpu_per_job, gpu=gpu_per_job)
    
#     return resources