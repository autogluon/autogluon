import logging
import os
import psutil
import shutil

from abc import ABC, abstractmethod
from typing import Optional, Callable, Union, List

from .. import Space
from ..utils.try_import import try_import_ray

try_import_ray()
import ray

from ray import tune
from ray_lightning import RayPlugin
from ray.tune import PlacementGroupFactory
from ray.tune.schedulers import TrialScheduler, FIFOScheduler, AsyncHyperBandScheduler, PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
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
    # Not support PBT/PB2 for now: https://github.com/ray-project/ray_lightning/issues/145
    # 'PBT' : PopulationBasedTraining,
    # 'PB2': PB2,
}

tabular_supported_searchers = ['random', 'bayes']
tabular_supported_schedulers = ['FIFO']


class EmptySearchSpace(Exception):
    pass


class RayTuneAdapter(ABC):
    """
    Abstract class to get module specific resource, and to update module specific arguments
    Instance of this class should be passed to `run` to provide custom behavior
    """
    
    def __init__(self):
        self.num_parallel_jobs = None
        self.cpu_per_job = None
        self.gpu_per_job = None
        self.resources_per_trial = None
    
    @abstractmethod
    def get_resources_per_trial(total_resources: dict, num_samples: int, **kwargs) -> Union[dict, PlacementGroupFactory]:
        """
        Calculate resources per trial if not specified by the user
        """
        raise NotImplementedError
    
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
    hyperparameter_tune_kwargs: dict,
    metric: str,
    mode: str,  # must be one of [min, max]
    save_dir: str,  # directory to save results. Ray will write artifacts to save_dir/trial_dir/
    ray_tune_adapter: RayTuneAdapter,
    total_resources: Optional[dict] = dict(),
    minimum_cpu_per_trial: int = 1,
    minimum_gpu_per_trial: float = 1.0,
    model_estimate_memory_usage: Optional[int] = None,
    time_budget_s: Optional[float] = None,
    supported_searchers: Optional[List[str]] = None,
    supported_schedulers: Optional[List[str]] = None,
    verbose: int = 1,  # 0 = silent, 1 = only status updates, 2 = status and brief trial results, 3 = status and detailed trial results.
    **kwargs  # additional args being passed to tune.run
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
        Directory to save results. Ray will write artifacts to save_dir/trial_dir/
        trial_dir name has been overwritten to be trial_id.
        To provide custom trial_dir, pass a custom function to `trial_dirname_creator` as kwargs.
        For example of creating the custom function, refer to `_trial_dirname_creator`.
    ray_tune_adapter
        Adapter to provide necessary custom info to ray tune.
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
    supported_searchers
        Some searchers requires reporting status within epochs or checkpointing in the middle of training.
        If the trainable doesn't support those functionality, provide supported_searchers here to warn users HPO might not work as expected.
    supported_schedulers
        Some schedulers requires reporting status within epochs or checkpointing in the middle of training.
        If the trainable doesn't support those functionality, provide supported_sschedulers here to warn users HPO might not work as expected.
    verbose
        0 = silent, 1 = only status updates, 2 = status and brief trial results, 3 = status and detailed trial results.
    **kwargs
        Additional args being passed to tune.run
    """
    assert mode in [MIN, MAX], f'mode is {mode}'
    num_samples = hyperparameter_tune_kwargs.get('num_trials', hyperparameter_tune_kwargs.get('num_samples', None))
    if num_samples is None:
        num_samples = 1 if time_budget_s is None else 1000  # if both num_samples and time_budget_s are None, we only run 1 trial
    if not search_space:
        raise EmptySearchSpace
    searcher = _get_searcher(hyperparameter_tune_kwargs, metric, mode, supported_searchers=supported_searchers)
    scheduler = _get_scheduler(hyperparameter_tune_kwargs, supported_schedulers=supported_schedulers)
    search_space = _convert_search_space(search_space, searcher)

    if not ray.is_initialized():
        ray.init(log_to_driver=False, **total_resources)

    resources_per_trial = hyperparameter_tune_kwargs.get('resources_per_trial', None)
    if resources_per_trial is None:
        resources_per_trial = ray_tune_adapter.get_resources_per_trial(
            total_resources=total_resources,
            num_samples=num_samples,
            minimum_gpu_per_trial=minimum_gpu_per_trial,
            minimum_cpu_per_trial=minimum_cpu_per_trial,
            model_estimate_memory_usage=model_estimate_memory_usage
        )
    else:
        if isinstance(ray_tune_adapter, AutommRayTuneAdapter):
            # Ray Lightning provides a way to get the resources_per_trial because of the complexity of head process and worker process.
            # It's non-trivial to let the user to specify it. Hence we disable such option
            logger.warning('AutoMM does not support customized resources_per_trial. We will calculate it for you instead.')
            resources_per_trial = ray_tune_adapter.get_resources_per_trial(
                total_resources=total_resources,
                num_samples=num_samples,
                minimum_gpu_per_trial=minimum_gpu_per_trial,
                minimum_cpu_per_trial=minimum_cpu_per_trial,
                model_estimate_memory_usage=model_estimate_memory_usage
            )
    _validate_resources_per_trial(resources_per_trial)
    trainable_args = ray_tune_adapter.trainable_args_update_method(trainable_args)
    tune_kwargs = _get_default_tune_kwargs()
    tune_kwargs.update(kwargs)
    
    original_path = os.getcwd()
    if 'original_path' not in trainable_args:
        trainable_args['original_path'] = original_path
    
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
        **tune_kwargs
    )
    ray.shutdown()
    return analysis


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


def _validate_resources_per_trial(resources_per_trial):
    if isinstance(resources_per_trial, dict):
        if 'num_cpus' in resources_per_trial:
            resources_per_trial['cpu'] = resources_per_trial.pop('num_cpus')
        if 'num_gpus' in resources_per_trial:
            resources_per_trial['gpu'] = resources_per_trial.pop('num_gpus')


def _convert_search_space(search_space: dict, searcher: Union[SearchAlgorithm, Searcher]):
    """Convert the search space to Ray Tune search space if it's AG search space"""
    from ray.tune.sample import Categorical
    tune_search_space = search_space.copy()
    for hyperparmaeter, space in search_space.items():
        if isinstance(space, Space):
            tune_search_space[hyperparmaeter] = space.convert_to_ray()
        # This is a hack
        # nested list are not correctly converted to hyperopt space by ray https://github.com/ray-project/ray/issues/24050
        # convert list to tuple instead. models need to check for tuple and convert it back to list
        # TODO: remove the hack once ray releases new version containing the fix
        # TODO: remove searcher parameter passed to this function once the fix is out
        if isinstance(tune_search_space[hyperparmaeter], Categorical) and isinstance(searcher, HyperOptSearch):
            cat_list_tuple = [tuple(cat) if isinstance(cat, list) else cat
                              for cat in tune_search_space[hyperparmaeter].categories]
            tune_search_space[hyperparmaeter] = tune.choice(cat_list_tuple)
    return tune_search_space


def _get_searcher(hyperparameter_tune_kwargs: dict, metric: str, mode: str, supported_searchers: Optional[List[str]]=None):
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
            init_args = dict(metric=metric, mode=mode)
            init_args.update(user_init_args)
        searcher = searcher_cls(**init_args)
    assert isinstance(searcher, (SearchAlgorithm, Searcher)) and searcher.__class__ in searcher_presets.values()
    # Check supported schedulers for obj input
    if supported_searchers is not None:
        supported_searchers_cls = [scheduler_presets[scheduler] for scheduler in supported_searchers]
        if searcher.__class__ not in supported_searchers_cls:
            logger.warning(f'{searcher.__class__} is not supported yet. Using it might behave unexpected. Supported options are {supported_searchers_cls}')
    return searcher


def _get_scheduler(hyperparameter_tune_kwargs: dict, supported_schedulers: Optional[List[str]]=None):
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
            init_args = dict(max_t=9999)
        init_args.update(user_init_args)
        scheduler = scheduler_cls(**init_args)
    assert isinstance(scheduler, TrialScheduler) and scheduler.__class__ in scheduler_presets.values()
    # Check supported schedulers for obj input
    if supported_schedulers is not None:
        supported_schedulers_cls = [scheduler_presets[scheduler] for scheduler in supported_schedulers]
        if scheduler.__class__ not in supported_schedulers_cls:
            logger.warning(f'{scheduler.__class__} is not supported yet. Using it might behave unexpected. Supported options are {supported_schedulers_cls}')
    return scheduler
    
    
class TabularRayTuneAdapter(RayTuneAdapter):
    
    def get_resources_per_trial(
        self,
        total_resources: dict,
        num_samples: int,
        minimum_cpu_per_trial: int = 1,
        minimum_gpu_per_trial: float = 1.0,
        model_estimate_memory_usage: Optional[int] = None,
        **kwargs,
        ) -> Union[dict, PlacementGroupFactory]:
        assert isinstance(minimum_cpu_per_trial, int) and minimum_cpu_per_trial >= 1, 'minimum_cpu_per_trial must be a interger that is larger than 0'
        num_cpus = total_resources.get('num_cpus', psutil.cpu_count())
        num_gpus = total_resources.get('num_gpus', 0)
        cpu_per_job = max(minimum_cpu_per_trial, int(num_cpus // num_samples))
        gpu_per_job = 0
        max_jobs_in_parallel_memory = num_samples

        if model_estimate_memory_usage is not None:
            mem_available = psutil.virtual_memory().available
            # calculate how many jobs can run in parallel given memory available
            max_jobs_in_parallel_memory = max(1, int(mem_available // model_estimate_memory_usage))
        num_parallel_jobs = min(num_samples, num_cpus // cpu_per_job, max_jobs_in_parallel_memory)
        cpu_per_job = max(minimum_cpu_per_trial, int(num_cpus // num_parallel_jobs))  # update cpu_per_job in case memory is not enough and can use more cores for each job
        resources = dict(cpu=cpu_per_job)

        if num_gpus > 0:
            gpu_per_job = max(minimum_gpu_per_trial, num_gpus / num_parallel_jobs)
        resources = dict(cpu=cpu_per_job, gpu=gpu_per_job)
        
        self.num_parallel_jobs = num_parallel_jobs
        self.cpu_per_job = cpu_per_job
        self.gpu_per_job = gpu_per_job
        self.resources_per_trial = resources
        
        return resources
    
    def trainable_args_update_method(self, trainable_args: dict) -> dict:
        trainable_args['fit_kwargs']['num_cpus'] = self.resources_per_trial.get('cpu', 1)
        trainable_args['fit_kwargs']['num_gpus'] = self.resources_per_trial.get('gpu', 0)
        return trainable_args
    
    
class AutommRayTuneAdapter(RayTuneAdapter):
    
    def __init__(self):
        super().__init__()
        self.num_workers = None
        self.cpu_per_worker = None
    
    def get_resources_per_trial(
        self,
        total_resources: dict,
        num_samples: int,
        minimum_cpu_per_trial: int = 1,
        minimum_gpu_per_trial: float = 1.0,
        model_estimate_memory_usage: Optional[int] = None,
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
        cpu_per_job = None
        gpu_per_job = None
        num_workers = None
        if num_gpus > 0:
            gpu_per_job = max(int(minimum_gpu_per_trial), num_gpus // num_samples)
            num_workers = gpu_per_job  # each worker uses 1 gpu
            num_parallel_jobs = min(num_samples, num_gpus // gpu_per_job)
            num_cpus = (num_cpus - num_parallel_jobs)  # reserve cpus for the master process
            assert num_cpus > 0
            cpu_per_job = max(minimum_cpu_per_trial, num_cpus // num_parallel_jobs)
            cpu_per_worker = max(1, cpu_per_job // num_workers)
        else:
            # TODO: for cpu case, is it better to have more workers or more cpus per worker?
            cpu_per_job = max(minimum_cpu_per_trial, num_cpus // num_samples)
            num_workers = max(minimum_cpu_per_trial, cpu_per_job - 1)  # 1 cpu for master process
            cpu_per_worker = 1
        resources_per_trial = get_tune_resources(
            num_workers=num_workers,
            num_cpus_per_worker=cpu_per_worker,
            use_gpu=(gpu_per_job is not None)
        )
        
        self.num_parallel_jobs = num_parallel_jobs
        self.cpu_per_job = cpu_per_job
        self.gpu_per_job = gpu_per_job
        self.resources_per_trial = resources_per_trial
        self.num_workers = num_workers
        self.cpu_per_worker = cpu_per_worker
        
        return resources_per_trial
        
    def trainable_args_update_method(self, trainable_args: dict) -> dict:
        trainable_args['hyperparameters']['env.num_gpus'] = self.gpu_per_job
        trainable_args['hyperparameters']['env.num_workers'] = self.cpu_per_job
        trainable_args['hyperparameters']['env.num_nodes'] = 1  # num_nodes is not needed by ray lightning. Setting it to default, which is 1
        trainable_args['_ray_lightning_plugin'] = RayPlugin(
                                                    num_workers=self.num_workers,
                                                    num_cpus_per_worker=self.cpu_per_worker,
                                                    use_gpu=self.gpu_per_job is not None,
                                                )
        return trainable_args
