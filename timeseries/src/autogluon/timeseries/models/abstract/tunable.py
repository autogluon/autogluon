from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any

from typing_extensions import Self

from autogluon.common.savers import save_pkl
from autogluon.common.utils.distribute_utils import DistributedContext
from autogluon.common.utils.log_utils import DuplicateFilter
from autogluon.common.utils.try_import import try_import_ray
from autogluon.core.hpo.constants import CUSTOM_BACKEND, RAY_BACKEND
from autogluon.core.hpo.exceptions import EmptySearchSpace
from autogluon.core.hpo.executors import HpoExecutor, HpoExecutorFactory, RayHpoExecutor
from autogluon.core.models import Tunable
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.utils.warning_filters import disable_stdout, warning_filter

from .model_trial import model_trial, skip_hpo

logger = logging.getLogger(__name__)
dup_filter = DuplicateFilter()
logger.addFilter(dup_filter)


class TimeSeriesTunable(Tunable, ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.name: str
        self.path: str
        self.path_root: str

    def hyperparameter_tune(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame | None,
        val_splitter: Any = None,
        default_num_trials: int | None = 1,
        refit_every_n_windows: int | None = 1,
        hyperparameter_tune_kwargs: str | dict = "auto",
        time_limit: float | None = None,
    ) -> tuple[dict[str, Any], Any]:
        hpo_executor = self._get_default_hpo_executor()
        hpo_executor.initialize(
            hyperparameter_tune_kwargs, default_num_trials=default_num_trials, time_limit=time_limit
        )

        # we use k_fold=1 to circumvent autogluon.core logic to manage resources during parallelization
        # of different folds
        # FIXME: we pass in self which currently does not inherit from AbstractModel
        hpo_executor.register_resources(self, k_fold=1, **self._get_system_resources())  # type: ignore

        time_start = time.time()
        logger.debug(f"\tStarting hyperparameter tuning for {self.name}")
        search_space = self._get_search_space()

        try:
            hpo_executor.validate_search_space(search_space, self.name)
        except EmptySearchSpace:
            return skip_hpo(self, train_data, val_data, time_limit=hpo_executor.time_limit)

        train_path, val_path = self._save_with_data(train_data, val_data)

        train_fn_kwargs = self._get_hpo_train_fn_kwargs(
            model_cls=self.__class__,
            init_params=self.get_params(),
            time_start=time_start,
            time_limit=hpo_executor.time_limit,
            fit_kwargs=dict(
                val_splitter=val_splitter,
                refit_every_n_windows=refit_every_n_windows,
            ),
            train_path=train_path,
            val_path=val_path,
            hpo_executor=hpo_executor,
        )

        minimum_resources = self.get_minimum_resources(is_gpu_available=self._is_gpu_available())
        hpo_context = disable_stdout if isinstance(hpo_executor, RayHpoExecutor) else nullcontext

        minimum_cpu_per_trial = minimum_resources.get("num_cpus", 1)
        if not isinstance(minimum_cpu_per_trial, int):
            logger.warning(
                f"Minimum number of CPUs per trial for {self.name} is not an integer. "
                f"Setting to 1. Minimum number of CPUs per trial: {minimum_cpu_per_trial}"
            )
            minimum_cpu_per_trial = 1

        with hpo_context(), warning_filter():  # prevent Ray from outputting its results to stdout with print
            hpo_executor.execute(
                model_trial=model_trial,
                train_fn_kwargs=train_fn_kwargs,
                directory=self.path,
                minimum_cpu_per_trial=minimum_cpu_per_trial,
                minimum_gpu_per_trial=minimum_resources.get("num_gpus", 0),
                model_estimate_memory_usage=None,  # type: ignore
                adapter_type="timeseries",
            )

            assert self.path_root is not None
            hpo_models, analysis = hpo_executor.get_hpo_results(
                model_name=self.name,
                model_path_root=self.path_root,
                time_start=time_start,
            )

        return hpo_models, analysis

    def _get_default_hpo_executor(self) -> HpoExecutor:
        backend = (
            self._get_model_base()._get_hpo_backend()
        )  # If ensemble, will use the base model to determine backend
        if backend == RAY_BACKEND:
            try:
                try_import_ray()
            except Exception as e:
                warning_msg = f"Will use custom hpo logic because ray import failed. Reason: {str(e)}"
                dup_filter.attach_filter_targets(warning_msg)
                logger.warning(warning_msg)
                backend = CUSTOM_BACKEND
        hpo_executor = HpoExecutorFactory.get_hpo_executor(backend)()  # type: ignore
        return hpo_executor

    def _get_hpo_backend(self) -> str:
        """Choose which backend("ray" or "custom") to use for hpo"""
        if DistributedContext.is_distributed_mode():
            return RAY_BACKEND
        return CUSTOM_BACKEND

    def _get_hpo_train_fn_kwargs(self, **train_fn_kwargs) -> dict:
        """Update kwargs passed to model_trial depending on the model configuration.

        These kwargs need to be updated, for example, by MultiWindowBacktestingModel.
        """
        return train_fn_kwargs

    def estimate_memory_usage(self, *args, **kwargs) -> float | None:
        """Return the estimated memory usage of the model. None if memory usage cannot be
        estimated.
        """
        return None

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
        }

    def _save_with_data(
        self, train_data: TimeSeriesDataFrame, val_data: TimeSeriesDataFrame | None
    ) -> tuple[str, str]:
        self.path = os.path.abspath(self.path)
        self.path_root = self.path.rsplit(self.name, 1)[0]

        dataset_train_filename = "dataset_train.pkl"
        train_path = os.path.join(self.path, dataset_train_filename)
        save_pkl.save(path=train_path, object=train_data)

        dataset_val_filename = "dataset_val.pkl"
        val_path = os.path.join(self.path, dataset_val_filename)
        save_pkl.save(path=val_path, object=val_data)
        return train_path, val_path

    @abstractmethod
    def _get_model_base(self) -> Self:
        pass

    @abstractmethod
    def _is_gpu_available(self) -> bool:
        pass

    @abstractmethod
    def _get_search_space(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return a clean copy of constructor parameters that can be used to
        clone the current model.
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_system_resources() -> dict[str, Any]:
        pass
