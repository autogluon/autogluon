import copy
import logging
import os
import time

from abc import ABC, abstractmethod

from .constants import RAY_BACKEND, CUSTOM_BACKEND
from .exceptions import EmptySearchSpace
from .. import Space
from ..scheduler.scheduler_factory import scheduler_factory


logger = logging.getLogger(__name__)


class HpoExecutor(ABC):
    """Interface for abstract model to executor HPO runs"""
    
    def __init__(self):
        self.search_space = None
        self.time_limit = None
    
    @property
    @abstractmethod
    def executor_type(self):
        raise NotImplementedError

    @abstractmethod
    def initialize(self, hyperparameter_tune_kwargs, default_num_trials=None, time_limit=None):
        raise NotImplementedError
    
    @abstractmethod
    def register_resources(self, resources):
        raise NotImplementedError
    
    @abstractmethod
    def validate_search_space(self, search_space, model_name):
        raise NotImplementedError
    
    @abstractmethod
    def execute(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def report(self, reporter, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def get_hpo_results(self, model_name, model_path_root, **kwargs):
        raise NotImplementedError


class RayHpoExecutor(HpoExecutor):
    
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
            hyperparameter_tune_kwargs = self.custom_to_ray_preset_map[hyperparameter_tune_kwargs]
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
        return time_limit
    
    def register_resources(self, resources):
        self.resources = resources
        
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
            model_trial,
            train_fn_kwargs,
            directory,
            minimum_cpu_per_trial,
            minimum_gpu_per_trial,
            model_estimate_memory_usage
        ):
        from .ray_hpo import (
            run,
            TabularRayTuneAdapter
        )
        analysis = run(
            trainable=model_trial,
            trainable_args=train_fn_kwargs,
            search_space=self.search_space,
            hyperparameter_tune_kwargs=self.hyperparameter_tune_kwargs,
            metric='validation_performance',
            mode='max',
            save_dir=directory,
            ray_tune_adapter=TabularRayTuneAdapter(),
            total_resources=self.resources,
            minimum_cpu_per_trial=minimum_cpu_per_trial,
            minimum_gpu_per_trial=minimum_gpu_per_trial,
            model_estimate_memory_usage=model_estimate_memory_usage,
            time_budget_s=self.time_limit
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
            hpo_models[trial_model_name] = trial_model_path

        return hpo_models, self.analysis


class CustomHpoExecutor(HpoExecutor):
    
    def __init__(self):
        self.scheduler_options = None
        self.scheduler = None
        
    @property
    def executor_type(self):
        return CUSTOM_BACKEND
    
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
        
        return time_limit
    
    def register_resources(self, resources):
        assert self.scheduler_options is not None, 'Call `initialize()` before register resources'
        self.scheduler_options[1]['resources'] = resources
        
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
        print(scheduler_params)
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

            hpo_models[trial_model_name] = trial_model_path
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
