from __future__ import annotations

import copy
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from autogluon.common import space
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.s3_utils import is_s3_url

from ..ray.resources_calculator import ResourceCalculator
from ..scheduler.scheduler_factory import scheduler_factory
from ..utils.savers import save_pkl
from .constants import CUSTOM_BACKEND, RAY_BACKEND
from .exceptions import EmptySearchSpace

if TYPE_CHECKING:
    from ..models import AbstractModel


logger = logging.getLogger(__name__)


class HpoExecutor(ABC):
    """Interface for abstract model to executor HPO runs"""

    def __init__(self):
        self.hyperparameter_tune_kwargs = None
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
    def initialize(self, hyperparameter_tune_kwargs: Union[str, dict], default_num_trials: Optional[int] = None, time_limit: Optional[float] = None):
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

    def register_resources(self, initialized_model: AbstractModel, num_cpus: int, num_gpus: Union[int, float], k_fold: Optional[int] = None, **kwargs):
        """
        Register total resources used for the experiment, and calculate resources per trial if user specified.
        User specified resources per trial will be validated against total resources and minimum resources required, and respected directly if legit.
        When HPO with bagging, user could specify resources per fold as well as resources per trial.
            Resources per fold will be checked against total resources, minimum resources required, and resources per trial (if specified).
            Resources per fold will have higher priority than resources per trial, and the corresponding resources per trial will be calculated accordingly.
        When no user specified resources present, we try to maximize trials running in parallel while respecting the minimum resources required.

        Parameters
        ----------
        initialized_model
            The model that will be performed HPO. This model MUST be initialized.
        num_cpus
            Total number of cpus available for the experiment.
        num_gpus
            Total number of gpus available for the experiment.
        k_fold
            Number of folds if bagging. Used to check if an individual trial is a bagged model.
        kwargs
            Any additional parameters being passed to `AbstractModel.hyperparameter_tune()`
            This function will pass these parameters to initialized model to get estimation of memory usage
        """
        minimum_model_resources = initialized_model.get_minimum_resources(is_gpu_available=(num_gpus > 0))
        minimum_model_num_cpus = minimum_model_resources.get("num_cpus", 1)
        minimum_model_num_gpus = minimum_model_resources.get("num_gpus", 0)
        initialized_model_params = initialized_model.get_params()

        if "hyperparameters" in initialized_model_params and "ag_args_fit" in initialized_model_params["hyperparameters"]:
            user_specified_trial_num_cpus = initialized_model_params["hyperparameters"]["ag_args_fit"].get("num_cpus", None)
            user_specified_trial_num_gpus = initialized_model_params["hyperparameters"]["ag_args_fit"].get("num_gpus", None)
            if user_specified_trial_num_cpus is not None or user_specified_trial_num_gpus is not None:
                num_trials_in_parallel_with_gpu = math.inf
                if user_specified_trial_num_cpus is None:
                    # If user didn't specify cpu per trial, we find the min based on gpu
                    num_trials_in_parallel_with_gpu = num_gpus // user_specified_trial_num_gpus
                    user_specified_trial_num_cpus = num_cpus // num_trials_in_parallel_with_gpu  # keep gpus per trial int to avoid complexity
                num_trials_in_parallel_with_cpu = math.inf
                if user_specified_trial_num_gpus is None:
                    # If user didn't specify gpu per trial, we find the min based on cpu
                    num_trials_in_parallel_with_cpu = num_cpus // user_specified_trial_num_cpus
                    user_specified_trial_num_gpus = num_gpus // num_trials_in_parallel_with_cpu  # keep gpus per trial int to avoid complexity
                assert (
                    user_specified_trial_num_cpus <= num_cpus
                ), f"Detected trial level cpu requirement = {user_specified_trial_num_cpus} > total cpu granted to AG predictor = {num_cpus}"
                assert (
                    user_specified_trial_num_cpus >= minimum_model_num_cpus
                ), f"The trial requires minimum cpu {minimum_model_num_cpus}, but you only specified {user_specified_trial_num_cpus}"
                assert (
                    user_specified_trial_num_gpus <= num_gpus
                ), f"Detected trial level gpu requirement = {user_specified_trial_num_gpus} > total gpu granted to AG predictor = {num_gpus}"
                assert (
                    user_specified_trial_num_gpus >= minimum_model_num_gpus
                ), f"The trial requires minimum gpu {minimum_model_num_gpus}, but you only specified {user_specified_trial_num_gpus}"

                # Custom backend should set its total resource to be resources_per_trial
                self.hyperparameter_tune_kwargs["resources_per_trial"] = {"num_cpus": user_specified_trial_num_cpus, "num_gpus": user_specified_trial_num_gpus}

        model_base = initialized_model._get_model_base()
        if model_base != initialized_model:
            # This is an ensemble model
            total_num_cpus_per_trial = num_cpus
            total_num_gpus_per_trial = num_gpus
            if "resources_per_trial" in self.hyperparameter_tune_kwargs:
                resources_per_trial = self.hyperparameter_tune_kwargs["resources_per_trial"]
                total_num_cpus_per_trial = resources_per_trial.get("num_cpus")
                total_num_gpus_per_trial = resources_per_trial.get("num_gpus")
            user_specified_fold_resources = model_base._user_params_aux
            user_specified_fold_num_cpus = user_specified_fold_resources.get("num_cpus", None)  # We shouldn't always use it
            user_specified_fold_num_gpus = user_specified_fold_resources.get("num_gpus", None)
            if user_specified_fold_num_cpus is not None or user_specified_fold_num_gpus is not None:
                num_folds_in_parallel_with_cpu = math.inf
                if minimum_model_num_cpus > 0:
                    num_folds_in_parallel_with_cpu = total_num_cpus_per_trial // minimum_model_num_cpus
                if user_specified_fold_num_cpus is not None:
                    assert (
                        user_specified_fold_num_cpus <= total_num_cpus_per_trial
                    ), f"Detected fold level cpu requirement = {user_specified_fold_num_cpus} > total cpu granted to AG predictor per trial= {total_num_cpus_per_trial}"
                    assert (
                        user_specified_fold_num_cpus >= minimum_model_num_cpus
                    ), f"The model requires minimum cpu {minimum_model_num_cpus}, but you only specified {user_specified_fold_num_cpus}"
                    num_folds_in_parallel_with_cpu = total_num_cpus_per_trial // user_specified_fold_num_cpus
                num_folds_in_parallel_with_gpu = math.inf
                if minimum_model_num_gpus > 0:
                    num_folds_in_parallel_with_gpu = total_num_gpus_per_trial // minimum_model_num_gpus
                if user_specified_fold_num_gpus is not None:
                    assert (
                        user_specified_fold_num_gpus <= total_num_gpus_per_trial
                    ), f"Detected fold level gpu requirement = {user_specified_fold_num_gpus} > total gpu granted to AG predictor per trial = {total_num_gpus_per_trial}"
                    assert (
                        user_specified_fold_num_gpus >= minimum_model_num_gpus
                    ), f"The model requires minimum gpu {minimum_model_num_gpus}, but you only specified {user_specified_fold_num_gpus}"
                    if minimum_model_num_gpus > 0:
                        num_folds_in_parallel_with_gpu = total_num_gpus_per_trial // user_specified_fold_num_gpus
                num_folds_in_parallel = min(num_folds_in_parallel_with_cpu, num_folds_in_parallel_with_gpu)

                cpu_per_trial = user_specified_fold_num_cpus * min(k_fold, num_folds_in_parallel)
                gpu_per_trial = user_specified_fold_num_gpus * min(k_fold, num_folds_in_parallel)

                # Custom backend should set its total resource to be resources_per_trial
                self.hyperparameter_tune_kwargs["resources_per_trial"] = {"num_cpus": cpu_per_trial, "num_gpus": gpu_per_trial}
        if "resources_per_trial" not in self.hyperparameter_tune_kwargs:
            # User didn't provide any requirements
            num_jobs_in_parallel_with_mem = math.inf

            if initialized_model.estimate_memory_usage is not None:
                model_estimate_memory_usage = initialized_model.estimate_memory_usage(**kwargs)
                total_memory_available = ResourceManager.get_available_virtual_mem()
                num_jobs_in_parallel_with_mem = total_memory_available // model_estimate_memory_usage

            num_jobs_in_parallel_with_cpu = num_cpus // minimum_model_num_cpus
            num_jobs_in_parallel_with_gpu = math.inf
            if minimum_model_num_gpus > 0:
                num_jobs_in_parallel_with_gpu = num_gpus // minimum_model_num_gpus
            num_jobs_in_parallel = min(num_jobs_in_parallel_with_mem, num_jobs_in_parallel_with_cpu, num_jobs_in_parallel_with_gpu)
            if k_fold is not None and k_fold > 0:
                max_models = self.hyperparameter_tune_kwargs.get("num_trials", math.inf) * k_fold
                num_jobs_in_parallel = min(num_jobs_in_parallel, max_models)
            system_num_cpu = ResourceManager.get_cpu_count()
            system_num_gpu = ResourceManager.get_gpu_count()
            if model_base != initialized_model:
                # bagged model
                if num_jobs_in_parallel // k_fold < 1:
                    # We can only train 1 trial in parallel
                    num_trials_in_parallel = 1
                else:
                    num_trials_in_parallel = num_jobs_in_parallel // k_fold
                if self.executor_type == "custom":
                    # custom backend runs sequentially
                    num_trials_in_parallel = 1
                cpu_per_trial = int(num_cpus // num_trials_in_parallel)
                gpu_per_trial = num_gpus // num_trials_in_parallel
            else:
                num_trials = self.hyperparameter_tune_kwargs.get("num_trials", math.inf)
                if self.executor_type == "custom":
                    # custom backend runs sequentially
                    num_jobs_in_parallel = 1
                cpu_per_trial = int(num_cpus // min(num_jobs_in_parallel, num_trials))
                gpu_per_trial = num_gpus / min(num_jobs_in_parallel, num_trials)
            # In distributed setting, a single trial could be scheduled with resources that's more than a single node causing hanging
            # Force it to be less than the current node. This works under the assumption that all nodes are of the same type
            cpu_per_trial = min(cpu_per_trial, system_num_cpu)
            gpu_per_trial = min(gpu_per_trial, system_num_gpu)

            self.hyperparameter_tune_kwargs["resources_per_trial"] = {"num_cpus": cpu_per_trial, "num_gpus": gpu_per_trial}

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

    def prepare_data(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, path_prefix: str) -> Tuple[str, str]:
        """
        Prepare data as pickle files for hpo trials.
        If path_prefix is a s3 url, will store to s3. Otherwise, store in local disk

        Parameters
        ----------
        X: pd.DataFrame
            Training data
        y: pd.Series
            Training label
        X_val: pd.DataFrame
            Validation data
        y_val: pd.Series
            Validation label
        path_prefix: str
            path prefix to store the data artifacts

        Return
        ------
        Tuple[str, str]:
            Path to both the training and validation data
        """

        def save_data(data: Any, path_prefix: str, filename: str) -> str:
            if is_s3_url(path_prefix):
                path = path_prefix + filename if path_prefix.endswith("/") else path_prefix + f"/{filename}"
            else:
                path = os.path.join(path_prefix, filename)
            save_pkl.save(path=path, object=data, verbose=False)
            return path

        dataset_train_filename = "dataset_train.pkl"
        dataset_val_filename = "dataset_val.pkl"
        train_path = save_data(data=(X, y), path_prefix=path_prefix, filename=dataset_train_filename)
        val_path = save_data(data=(X_val, y_val), path_prefix=path_prefix, filename=dataset_val_filename)

        return train_path, val_path

    @abstractmethod
    def execute(self, **kwargs):
        """Execute the experiment"""
        raise NotImplementedError

    @abstractmethod
    def report(self, reporter: "LocalReporter", **kwargs):
        """Report result of the experiment to the reporter. If no reporter needed, pass in None"""
        raise NotImplementedError

    @abstractmethod
    def get_hpo_results(self, model_name: str, model_path_root: str, **kwargs):
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
        "auto": {"scheduler": "FIFO", "searcher": "bayes"},
        "local_random": {"scheduler": "FIFO", "searcher": "random"},
        "distributed_random": {"scheduler": "FIFO", "searcher": "random"},
        "random": {"scheduler": "FIFO", "searcher": "random"},
    }
    custom_to_ray_scheduler_preset_map = {
        "local": "FIFO",
        "distributed": "FIFO",
    }
    custom_to_ray_searcher_preset_map = {
        "local_random": "random",
        "distributed_random": "random",
        "random": "random",
        "auto": "bayes",
    }

    def __init__(self):
        super().__init__()
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
        hyperparameter_tune_kwargs["scheduler"] = self.custom_to_ray_scheduler_preset_map.get(
            hyperparameter_tune_kwargs["scheduler"], hyperparameter_tune_kwargs["scheduler"]
        )
        hyperparameter_tune_kwargs["searcher"] = self.custom_to_ray_searcher_preset_map.get(
            hyperparameter_tune_kwargs["searcher"], hyperparameter_tune_kwargs["searcher"]
        )
        if "num_trials" not in hyperparameter_tune_kwargs and default_num_trials is not None:
            hyperparameter_tune_kwargs["num_trials"] = default_num_trials
        self.hyperparameter_tune_kwargs = copy.deepcopy(hyperparameter_tune_kwargs)

    def validate_search_space(self, search_space, model_name):
        from ray.tune.search.sample import Domain

        if not any(isinstance(search_space[hyperparam], (space.Space, Domain)) for hyperparam in search_space):
            logger.warning(
                f"\tNo hyperparameter search space specified for {model_name}. Skipping HPO. " f"Will train one model based on the provided hyperparameters."
            )
            raise EmptySearchSpace
        self.search_space = search_space
        logger.log(15, f"\tHyperparameter search space for {model_name}: ")
        for hyperparam in search_space:
            if isinstance(search_space[hyperparam], (space.Space, Domain)):
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
        trainable_is_parallel: bool = False,
        tune_config_kwargs: Optional[Dict[str, Any]] = None,
        run_config_kwargs: Optional[Dict[str, Any]] = None,
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
            Valid values are ['tabular', 'timeseries', 'automm']
        trainable_is_parallel
            Whether the trainable itself will use ray to run parallel job or not.
        tune_config_kwargs
            Additional args being passed to tune.TuneConfig https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html#ray-tune-tuneconfig
        run_config_kwargs
            Additional args being passed to air.RunConfig https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray.train.RunConfig
        """
        from .ray_hpo import RayTuneAdapterFactory, run

        # Disable tensorboard logging to avoid layer warning
        # TODO: remove this when ray tune fix ray tune pass tuple to hyperopt issue
        os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
        analysis = run(
            trainable=model_trial,
            trainable_args=train_fn_kwargs,
            search_space=self.search_space,
            hyperparameter_tune_kwargs=self.hyperparameter_tune_kwargs,
            metric="validation_performance",
            mode="max",
            save_dir=directory,
            ray_tune_adapter=RayTuneAdapterFactory.get_adapter(adapter_type)(),
            trainable_is_parallel=trainable_is_parallel,
            total_resources=self.resources,
            minimum_cpu_per_trial=minimum_cpu_per_trial,
            minimum_gpu_per_trial=minimum_gpu_per_trial,
            model_estimate_memory_usage=model_estimate_memory_usage,
            time_budget_s=self.time_limit,
            verbose=0,
            tune_config_kwargs=tune_config_kwargs,
            run_config_kwargs=run_config_kwargs,
        )
        os.environ.pop("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", None)
        self.analysis = analysis

    def report(self, reporter, **kwargs):
        from ray.air import session

        session.report(kwargs)

    def get_hpo_results(self, model_name, model_path_root, **kwargs):
        assert self.analysis is not None, "Call `execute()` before `get_hpo_results()`"
        hpo_models = {}
        for trial, details in self.analysis.results.items():
            validation_performance = details.get("validation_performance", None)
            # when equals to -inf, trial finished with TimeLimitExceeded exception and didn't finish at least 1 epoch
            if validation_performance is None or validation_performance == float("-inf"):
                continue
            trial_id = details.get("trial_id")
            file_id = trial_id  # unique identifier to files from this trial
            trial_model_name = os.path.join(model_name, file_id)
            trial_model_path = os.path.join(model_path_root, trial_model_name)
            hpo_models[trial_model_name] = dict(path=trial_model_path)

            hpo_models[trial_model_name] = dict(
                path=trial_model_path,
                val_score=validation_performance,
                trial=trial,
                hyperparameters=details["config"],
            )

        return hpo_models, self.analysis


class CustomHpoExecutor(HpoExecutor):
    """Implementation of HpoExecutor Interface, where our custom logic is used as the backend"""

    def __init__(self):
        super().__init__()
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
        self.scheduler_options[1]["time_out"] = value
        self._time_limit = value

    def initialize(self, hyperparameter_tune_kwargs, default_num_trials=None, time_limit=None):
        if not isinstance(hyperparameter_tune_kwargs, tuple):
            if isinstance(hyperparameter_tune_kwargs, dict):
                hyperparameter_tune_kwargs = copy.deepcopy(hyperparameter_tune_kwargs)
                self.hyperparameter_tune_kwargs = hyperparameter_tune_kwargs
            num_trials = default_num_trials  # This will be ignored if hyperparameter_tune_kwargs contains num_trials
            if default_num_trials is None:
                num_trials = 1 if time_limit is None else 1000
            hyperparameter_tune_kwargs = scheduler_factory(hyperparameter_tune_kwargs, num_trials=num_trials, nthreads_per_trial="auto", ngpus_per_trial="auto")
            hyperparameter_tune_kwargs = copy.deepcopy(hyperparameter_tune_kwargs)
            if "time_out" not in hyperparameter_tune_kwargs[1]:
                hyperparameter_tune_kwargs[1]["time_out"] = time_limit
            time_limit = hyperparameter_tune_kwargs[1]["time_out"]
        self.scheduler_options = hyperparameter_tune_kwargs
        self.time_limit = time_limit
        if self.hyperparameter_tune_kwargs is None:
            self.hyperparameter_tune_kwargs = {}

    def register_resources(self, initialized_model, **kwargs):
        assert self.scheduler_options is not None, "Call `initialize()` before register resources"
        super().register_resources(initialized_model, **kwargs)
        if self.hyperparameter_tune_kwargs.get("resources_per_trial", None) is not None:
            # Custom backend only run trials sequentially
            self.scheduler_options[1]["resource"] = self.hyperparameter_tune_kwargs["resources_per_trial"]
        logger.debug(f"custom backend resource: {self.resources}, per trial resource: {self.hyperparameter_tune_kwargs}")

    def validate_search_space(self, search_space, model_name):
        if not any(isinstance(search_space[hyperparam], space.Space) for hyperparam in search_space):
            logger.warning(
                f"\tNo hyperparameter search space specified for {model_name}. Skipping HPO. " f"Will train one model based on the provided hyperparameters."
            )
            raise EmptySearchSpace
        self.search_space = search_space
        logger.log(15, f"\tHyperparameter search space for {model_name}: ")
        for hyperparam in search_space:
            if isinstance(search_space[hyperparam], space.Space):
                logger.log(15, f"{hyperparam}:   {search_space[hyperparam]}")

    def execute(self, model_trial, train_fn_kwargs, **kwargs):
        assert self.scheduler_options is not None, "Call `initialize()` before execute"
        scheduler_cls, scheduler_params = self.scheduler_options  # Unpack tuple
        if scheduler_cls is None or scheduler_params is None:
            raise ValueError("scheduler_cls and scheduler_params cannot be None for hyperparameter tuning")
        train_fn_kwargs["fit_kwargs"].update(scheduler_params["resource"].copy())
        scheduler = scheduler_cls(model_trial, search_space=self.search_space, train_fn_kwargs=train_fn_kwargs, **scheduler_params)
        self.scheduler = scheduler

        scheduler.run()
        scheduler.join_jobs()

    def report(self, reporter, **kwargs):
        assert reporter is not None
        reporter(**kwargs)

    def get_hpo_results(self, model_name, model_path_root, time_start, **kwargs):
        assert self.scheduler is not None, "Call `execute()` before `get_hpo_results()`"
        # Store results / models from this HPO run:
        best_hp = self.scheduler.get_best_config()  # best_hp only contains searchable stuff
        hpo_results = {
            "best_reward": self.scheduler.get_best_reward(),
            "best_config": best_hp,
            "total_time": time.time() - time_start,
            "metadata": self.scheduler.metadata,
            "training_history": self.scheduler.training_history,
            "config_history": self.scheduler.config_history,
            "reward_attr": self.scheduler._reward_attr,
        }

        hpo_models = {}  # stores all the model names and file paths to model objects created during this HPO run.
        hpo_model_performances = {}
        for trial in sorted(hpo_results["config_history"].keys()):
            # TODO: ignore models which were killed early by scheduler (eg. in Hyperband). How to ID these?
            file_id = f"T{trial+1}"  # unique identifier to files from this trial
            trial_model_name = os.path.join(model_name, file_id)
            trial_model_path = os.path.join(model_path_root, trial_model_name)
            trial_reward = self.scheduler.searcher.get_reward(hpo_results["config_history"][trial])
            if trial_reward is None or trial_reward == float("-inf"):
                continue
            hpo_models[trial_model_name] = dict(
                path=trial_model_path, val_score=trial_reward, trial=trial, hyperparameters=hpo_results["config_history"][trial]
            )

            hpo_model_performances[trial_model_name] = trial_reward

        hpo_results["hpo_model_performances"] = hpo_model_performances

        logger.log(15, "Time for %s model HPO: %s" % (model_name, str(hpo_results["total_time"])))
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
        assert hpo_executor in HpoExecutorFactory.__type_to_executor, f"{hpo_executor} not supported"
        return HpoExecutorFactory.__type_to_executor[hpo_executor]
