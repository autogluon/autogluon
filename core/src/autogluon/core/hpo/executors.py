from __future__ import annotations

import copy
import logging
import os
import time

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, Callable

from .constants import RAY_BACKEND, CUSTOM_BACKEND
from .exceptions import EmptySearchSpace
from .. import Space
from ..ray.resources_calculator import ResourceCalculator
from ..scheduler.scheduler_factory import scheduler_factory

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import AbstractModel


logger = logging.getLogger(__name__)


class HpoExecutor(ABC):
    """Interface for abstract model to executor HPO runs"""
    
    def __init__(self):
        self.resources = None
        self.search_space = None
        self._time_limit = None
    
    @property
    @abstractmethod
    def executor_type(self):
        """Type of the executor"""
        raise NotImplementedError
    
    @property
    def time_limit(self):
        """Time limit for the hpo experiment"""
        return self._time_limit
    
    @time_limit.setter
    def time_limit(self, value):
        self._time_limit = value

    @abstractmethod
    def initialize(
            self,
            hyperparameter_tune_kwargs: Union[str, dict],
            default_num_trials: Optional[int] = None,
            time_limit: Optional[float] = None
        ):
        """
        Parse `hyperparameter_tune_kwargs` and initialize the executor
        
        Parameters
        ----------
        hyperparameter_tune_kwargs
            User specified parameters to perform HPO experiment
        default_num_trials
            If user didn't provide `num_trials` in `hyperparameter_tune_kwargs`, will use this value instead.
            If None and user didn't provide `num_trials`, will do 1000 trials when there is no time limit, and 1 trial if there is time limit.
        time_limit
            Time limit provided to the experiment.
        """
        raise NotImplementedError
    
    def register_resources(self, initialized_model: AbstractModel):
        """
        Register total resources used for the experiment
        
        Parameters
        ----------
        initialized_model
            The model that will be performed HPO. This model MUST be initialized
        """
        user_cpu_count = initialized_model._get_child_aux_val(key='num_cpus', default=None)
        user_gpu_count = initialized_model._get_child_aux_val(key='num_gpus', default=None)
        resource_kwargs = initialized_model._preprocess_fit_resources()

        num_gpus = ResourceCalculator.get_total_gpu_count(
            user_specified_num_gpus=user_gpu_count,
            model_default_num_gpus=resource_kwargs.get('num_gpus', 0),
        )
        num_cpus = ResourceCalculator.get_total_cpu_count(
            user_specified_num_cpus=user_cpu_count,
            model_default_num_cpus=resource_kwargs.get('num_cpus', 0),
        )

        self.resources = dict(num_gpus=num_gpus, num_cpus=num_cpus)
    
    @abstractmethod
    def validate_search_space(
            self,
            search_space: dict,
            model_name: str,
        ):
        """
        Validate the search space for the experiment
        
        Parameters
        ----------
        search_space
            Search space provided to the experiment.
        model_name
            The model name of the hpo experiment is tuning. This is only used for logging purpose
        """
        raise NotImplementedError
    
    @abstractmethod
    def execute(self, **kwargs):
        """Execute the experiment"""
        raise NotImplementedError
    
    @abstractmethod
    def report(
            self,
            reporter: 'LocalReporter',
            **kwargs
        ):
        """Report result of the experiment to the reporter. If no reporter needed, pass in None"""
        raise NotImplementedError
    
    @abstractmethod
    def get_hpo_results(
            self,
            model_name: str,
            model_path_root: str,
            **kwargs
        ):
        """
        Retrieve hpo results
        
        Parameters
        ----------
        model_name
            The model name of the hpo experiment is tuning.
        model_path_root
            Root path of the model
        """
        raise NotImplementedError


class RayHpoExecutor(HpoExecutor):
    """Implementation of HpoExecutor Interface, where ray tune is used as the backend"""
    
    custom_to_ray_preset_map = {
        'auto': {'scheduler': 'FIFO', 'searcher': 'bayes'},
        'local_random': {'scheduler': 'FIFO', 'searcher': 'random'},
        'random': {'scheduler': 'FIFO', 'searcher': 'random'},
    }
    custom_to_ray_scheduler_preset_map = {
        'local': 'FIFO',
    }
    custom_to_ray_searcher_preset_map = {
        'local_random': 'random',
        'random': 'random',
        'auto': 'random',
    }
    
    def __init__(self):
        self.resources = None
        self.hyperparameter_tune_kwargs = None
        self.analysis = None
    
    @property
    def executor_type(self):
        return RAY_BACKEND
    
    def initialize(self, hyperparameter_tune_kwargs, default_num_trials=None, time_limit=None):
        self.time_limit = time_limit
        if isinstance(hyperparameter_tune_kwargs, dict):
            hyperparameter_tune_kwargs = hyperparameter_tune_kwargs.copy()
        if isinstance(hyperparameter_tune_kwargs, str):
            hyperparameter_tune_kwargs: dict = self.custom_to_ray_preset_map[hyperparameter_tune_kwargs].copy()
        hyperparameter_tune_kwargs['scheduler'] = self.custom_to_ray_scheduler_preset_map.get(
            hyperparameter_tune_kwargs['scheduler'],
            hyperparameter_tune_kwargs['scheduler']
        )
        hyperparameter_tune_kwargs['searcher'] = self.custom_to_ray_searcher_preset_map.get(
            hyperparameter_tune_kwargs['searcher'],
            hyperparameter_tune_kwargs['searcher']
        )
        if 'num_trials' not in hyperparameter_tune_kwargs and default_num_trials is not None:
            hyperparameter_tune_kwargs['num_trials'] = default_num_trials
        self.hyperparameter_tune_kwargs = hyperparameter_tune_kwargs
        
    def validate_search_space(self, search_space, model_name):
        from ray.tune.sample import Domain
        if not any(isinstance(search_space[hyperparam], (Space, Domain)) for hyperparam in search_space):
            logger.warning(f"\tNo hyperparameter search space specified for {model_name}. Skipping HPO. "
                           f"Will train one model based on the provided hyperparameters.")
            raise EmptySearchSpace
        self.search_space = search_space
        logger.log(15, f"\tHyperparameter search space for {model_name}: ")
        for hyperparam in search_space:
                if isinstance(search_space[hyperparam], (Space, Domain)):
                    logger.log(15, f"{hyperparam}:   {search_space[hyperparam]}")
                    
    def execute(
            self,
            *,  # Force kwargs to avoid bugs
            model_trial: Callable,
            train_fn_kwargs: dict,
            directory: str,
            minimum_cpu_per_trial: int,
            minimum_gpu_per_trial: float,
            model_estimate_memory_usage: float,
            adapter_type: str,
        ):
        """
        Execute ray hpo experiment
        
        Parameters
        ----------
        model_trial: Callable
            A function conducting individual trial
        train_fn_kwargs: dict
            A dict containing kwargs passed to model_trial
        directory: str
            Directory to save ray tune results. Ray will write artifacts to directory/trial_dir/
            While HPO, ray will chdir to this directory. Therefore, it's important to provide dataset or model saving path as absolute path.
            After HPO, we change back to the original working directory.
        minimum_cpu_per_trial: int
            Minimum number of cpu required to perform a trial. Must be >= 1
        minimum_gpu_per_trial: float
            Minimum number of gpu required to perform a trial. You are allowed to provide fractional gpu with a float.
            If not needed, provide 0
        adapter_type: str
            Type of adapter used by ray hpo experiment.
            Adapters are used to provide custom info or behavior that's module specific to the ray hpo experiment.
            For more info, please refer to `autogluon/core/hpo/ray_hpo`
            Valid values are ['tabular', 'timeseries', 'automm', 'automm_ray_lightning']
        """
        from .ray_hpo import (
            run,
            RayTuneAdapterFactory
        )
        analysis = run(
            trainable=model_trial,
            trainable_args=train_fn_kwargs,
            search_space=self.search_space,
            hyperparameter_tune_kwargs=self.hyperparameter_tune_kwargs,
            metric='validation_performance',
            mode='max',
            save_dir=directory,
            ray_tune_adapter=RayTuneAdapterFactory.get_adapter(adapter_type)(),
            total_resources=self.resources,
            minimum_cpu_per_trial=minimum_cpu_per_trial,
            minimum_gpu_per_trial=minimum_gpu_per_trial,
            model_estimate_memory_usage=model_estimate_memory_usage,
            time_budget_s=self.time_limit,
            verbose=0,
        )
        self.analysis = analysis
        
    def report(self, reporter, **kwargs):
        from ray import tune
        tune.report(**kwargs)
        
    def get_hpo_results(self, model_name, model_path_root, **kwargs):
        assert self.analysis is not None, 'Call `execute()` before `get_hpo_results()`'
        hpo_models = {}
        for trial, details in self.analysis.results.items():
            validation_performance = details.get('validation_performance', None)
            # when equals to -inf, trial finished with TimeLimitExceeded exception and didn't finish at least 1 epoch
            if validation_performance is None or validation_performance == float('-inf'):
                continue
            trial_id = details.get('trial_id')
            file_id = trial_id  # unique identifier to files from this trial
            trial_model_name = model_name + os.path.sep + file_id
            trial_model_path = model_path_root + trial_model_name + os.path.sep
            hpo_models[trial_model_name] = dict(
                path=trial_model_path
            )

            hpo_models[trial_model_name] = dict(
                path=trial_model_path,
                val_score=validation_performance,
                trial=trial,
                hyperparameters= details['config'],
            )

        return hpo_models, self.analysis


class CustomHpoExecutor(HpoExecutor):
    """Implementation of HpoExecutor Interface, where our custom logic is used as the backend"""
    
    def __init__(self):
        self.scheduler_options = None
        self.scheduler = None
        
    @property
    def executor_type(self):
        return CUSTOM_BACKEND
    
    @property
    def time_limit(self):
        return self._time_limit
    
    @time_limit.setter
    def time_limit(self, value):
        assert self.scheduler_options is not None
        self.scheduler_options[1]['time_out'] = value
        self._time_limit = value
    
    def initialize(self, hyperparameter_tune_kwargs, default_num_trials=None, time_limit=None):
        if not isinstance(hyperparameter_tune_kwargs, tuple):
            if isinstance(hyperparameter_tune_kwargs, dict):
                hyperparameter_tune_kwargs = hyperparameter_tune_kwargs.copy()
            num_trials = default_num_trials  # This will be ignored if hyperparameter_tune_kwargs contains num_trials
            if default_num_trials is None:
                num_trials = 1 if time_limit is None else 1000
            hyperparameter_tune_kwargs = scheduler_factory(hyperparameter_tune_kwargs, num_trials=num_trials, nthreads_per_trial='auto', ngpus_per_trial='auto')
            hyperparameter_tune_kwargs = copy.deepcopy(hyperparameter_tune_kwargs)
            if 'time_out' not in hyperparameter_tune_kwargs[1]:
                hyperparameter_tune_kwargs[1]['time_out'] = time_limit
            time_limit = hyperparameter_tune_kwargs[1]['time_out']
        self.scheduler_options = hyperparameter_tune_kwargs
        self.time_limit = time_limit
    
    def register_resources(self, initialized_model):
        assert self.scheduler_options is not None, 'Call `initialize()` before register resources'
        super().register_resources(initialized_model)
        self.scheduler_options[1]['resources'] = self.resources
        
    def validate_search_space(self, search_space, model_name):
        if not any(isinstance(search_space[hyperparam], Space) for hyperparam in search_space):
            logger.warning(f"\tNo hyperparameter search space specified for {model_name}. Skipping HPO. "
                           f"Will train one model based on the provided hyperparameters.")
            raise EmptySearchSpace
        self.search_space = search_space
        logger.log(15, f"\tHyperparameter search space for {model_name}: ")
        for hyperparam in search_space:
                if isinstance(search_space[hyperparam], Space):
                    logger.log(15, f"{hyperparam}:   {search_space[hyperparam]}")
                    
    def execute(self, model_trial, train_fn_kwargs, **kwargs):
        assert self.scheduler_options is not None, 'Call `initialize()` before execute'
        scheduler_cls, scheduler_params = self.scheduler_options  # Unpack tuple
        if scheduler_cls is None or scheduler_params is None:
            raise ValueError("scheduler_cls and scheduler_params cannot be None for hyperparameter tuning")
        train_fn_kwargs['fit_kwargs'].update(scheduler_params['resource'].copy())
        scheduler = scheduler_cls(model_trial, search_space=self.search_space, train_fn_kwargs=train_fn_kwargs, **scheduler_params)
        self.scheduler = scheduler
        
        scheduler.run()
        scheduler.join_jobs()
    
    def report(self, reporter, **kwargs):
        assert reporter is not None
        reporter(**kwargs)   
        
    def get_hpo_results(self, model_name, model_path_root, time_start, **kwargs):
        assert self.scheduler is not None, 'Call `execute()` before `get_hpo_results()`'
        # Store results / models from this HPO run:
        best_hp = self.scheduler.get_best_config()  # best_hp only contains searchable stuff
        hpo_results = {
            'best_reward': self.scheduler.get_best_reward(),
            'best_config': best_hp,
            'total_time': time.time() - time_start,
            'metadata': self.scheduler.metadata,
            'training_history': self.scheduler.training_history,
            'config_history': self.scheduler.config_history,
            'reward_attr': self.scheduler._reward_attr,
        }

        hpo_models = {}  # stores all the model names and file paths to model objects created during this HPO run.
        hpo_model_performances = {}
        for trial in sorted(hpo_results['config_history'].keys()):
            # TODO: ignore models which were killed early by scheduler (eg. in Hyperband). How to ID these?
            file_id = f"T{trial+1}"  # unique identifier to files from this trial
            trial_model_name = model_name + os.path.sep + file_id
            trial_model_path = model_path_root + trial_model_name + os.path.sep
            trial_reward = self.scheduler.searcher.get_reward(hpo_results['config_history'][trial])

            hpo_models[trial_model_name] = dict(
                path=trial_model_path,
                val_score=trial_reward,
                trial=trial,
                hyperparameters=hpo_results['config_history'][trial]
            )

            hpo_model_performances[trial_model_name] = trial_reward
        
        hpo_results['hpo_model_performances'] = hpo_model_performances

        logger.log(15, "Time for %s model HPO: %s" % (model_name, str(hpo_results['total_time'])))
        logger.log(15, "Best hyperparameter configuration for %s model: " % model_name)
        logger.log(15, str(best_hp))

        return hpo_models, hpo_results


class HpoExecutorFactory:
    
    __supported_executors = [
        RayHpoExecutor,
        CustomHpoExecutor,
    ]
    
    __type_to_executor = {cls().executor_type: cls for cls in __supported_executors}

    @staticmethod
    def get_hpo_executor(hpo_executor: str) -> HpoExecutor:
        """Return the executor"""
        assert hpo_executor in HpoExecutorFactory.__type_to_executor, f'{hpo_executor} not supported'
        return HpoExecutorFactory.__type_to_executor[hpo_executor]
