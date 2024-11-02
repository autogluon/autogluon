import logging
from dataclasses import dataclass
from typing import Callable
import time
from autogluon.core.models import AbstractModel, StackerEnsembleModel
from autogluon.common.utils.resource_utils import get_resource_manager

logger = logging.getLogger(__name__)

DEFAULT_REMOTE_KWARGS = dict(max_calls=1, retry_exceptions=False, max_retries=0)


@dataclass(kw_only=True)
class ModelResources:
    """Resource allocation for a model."""

    num_gpus_for_fold_worker: int
    num_cpus_for_fold_worker: int
    num_gpus_for_model_worker: int
    num_cpus_for_model_worker: int
    total_num_cpus: int
    total_num_gpus: int


# TODO: test if `not isinstance(_model, StackerEnsembleModel)` makes non-bagged run to use num_cpus many for model-worker
# TODO: cluster-wide memory management is not implemented so far and some memory checks might use memory values from the wrong node
class DistributedFitManager:
    """Tracks how many resources are used when scheduling jobs in a distributed setting.

    Distributed Fit
    ---------------
    We use ray to start a model-worker with a certain number of CPUs and GPUs.
    The worker then starts new fold-workers to fit each fold of a model.
    Alternatively, the model-worker uses the given resources to perform a single fit.

    We must pass GPU resources to a model-worker if the model has `refit_folds is True`
    as the refit_folds call happens in the model-worker.

    For full parallelization, we require the following:
        - GPUs
            - refit_folds is True: `num_gpus` + `num_bag_folds` * `num_bag_sets` * `num_gpus`
            - refit_folds is False: `num_bag_folds` * `num_bag_sets` * `num_gpus`
        - CPUs:
            - model with bagging: 1 + `num_cpus` * `num_bag_folds` * `num_bag_sets`
            - model without bagging: `num_cpus`

    Parameters
    ----------
    models_to_fit: list[AbstractModel]
        The models that shall be fitted in a distributed manner.
    func: callable
        The fit function to distribute.
    func_kwargs: dict, default=None
        Additional kwargs to pass to the function.
    func_put_kwargs: dict, default=None
        Additional kwargs to pass to the function, where the values are put into the object store.
    num_cpus : int | str
        Total number of CPUs available in the cluster (or `auto`).
    num_gpus : int | str
        Total number of GPUs available in the cluster (or `auto`).
    num_splits : int
        Number of training splits/bags for a model.
    """

    job_refs_to_allocated_resources: dict[str, ModelResources] = {}

    def __init__(
        self,
        *,
        models_to_fit: list[AbstractModel],
        func: Callable,
        func_kwargs: dict,
        func_put_kwargs: dict,
        num_cpus: int | str,
        num_gpus: int | str,
        num_splits: int,
    ):
        self.num_splits = num_splits

        # Resource tracking
        if isinstance(num_cpus, str):
            num_cpus = get_resource_manager().get_cpu_count()
        if isinstance(num_gpus, str):
            num_gpus = get_resource_manager().get_gpu_count()
        self.total_num_cpus = num_cpus
        self.total_num_gpus = num_gpus
        self.available_num_cpus = num_cpus
        self.available_num_gpus = num_gpus

        # Job tracking
        self.models_to_schedule = models_to_fit[:]

        # Init remote function
        import ray

        self.remote_func = ray.remote(**DEFAULT_REMOTE_KWARGS)(func)
        self.job_kwargs = dict()
        for key, value in func_kwargs.items():
            self.job_kwargs[key] = value
        for key, value in func_put_kwargs.items():
            self.job_kwargs[key] = ray.put(value)
        self.func_put_kwargs = func_put_kwargs

    def schedule_jobs(self) -> list[str]:
        """Yield the next model to schedule."""
        import ray

        models_to_schedule_later = []
        job_refs = []
        for model in self.models_to_schedule:
            model_resources = self.get_resources_for_model(model=model)

            is_sufficient, reason = self.check_sufficient_resources(resources=model_resources)
            if not is_sufficient:
                logger.log(0, f"Delay scheduling model {model.name}: {reason}.")
                models_to_schedule_later.append(model)
                continue

            job_ref = self.remote_func.options(
                num_cpus=model_resources.num_cpus_for_model_worker, num_gpus=model_resources.num_gpus_for_model_worker
            ).remote(model=ray.put(model), **self.job_kwargs)
            job_refs.append(job_ref)

            logger.log(20, f"Scheduled model training for {model.name}"
                           f"\n\tAllocated {model_resources.total_num_cpus} CPUs and {model_resources.total_num_gpus} GPUs"
                           f"\n\tRay{job_ref}")
            self.allocate_resources(job_ref=job_ref, resources=model_resources)
            time.sleep(0.1)

        self.models_to_schedule = models_to_schedule_later
        return job_refs

    def check_sufficient_resources(self, *, resources: ModelResources) -> tuple[bool, str | None]:
        """Determine if there are enough resources to scheduling fitting another model."""

        # Allow for oversubscribing to 10% of the CPUs due to scheduling overhead.
        if self.available_num_cpus + (self.total_num_cpus // 10) <= resources.total_num_cpus:
            return False, "not enough CPUs free."

        # All models need at least one CPU but not all a GPU
        # Avoid scheduling a model if there are not at least 50% of the required GPUs available
        if (resources.total_num_gpus > 0) and (self.available_num_gpus <= (resources.total_num_gpus // 2)):
            return False, "not enough GPUs free."

        return True, None

    def get_resources_for_model(self, *, model: AbstractModel) -> ModelResources:
        """Estimate the resources required to fit a model."""

        num_gpus_for_fold_worker = getattr(model, "model_base", model)._user_params_aux.get("num_gpus", 0)
        num_cpus_for_fold_worker = getattr(model, "model_base", model)._user_params_aux.get("num_cpus", 1)

        if (not isinstance(model, StackerEnsembleModel)) or model._user_params.get("use_child_oof", False):
            # Only one fold is fit, so we need to use all fit resources for the model-worker
            num_cpus_for_model_worker = num_cpus_for_fold_worker
            num_gpus_for_model_worker = num_gpus_for_fold_worker
            num_cpus_for_fold_worker = 0
            num_gpus_for_fold_worker = 0
        else:
            # If refit_folds is True, we need to pass GPU resources to the model-worker
            num_gpus_for_model_worker = (
                num_gpus_for_fold_worker
                if ((num_gpus_for_fold_worker > 0) and model._user_params.get("refit_folds", False))
                else 0
            )
            num_cpus_for_model_worker = 1

        total_num_cpus = num_cpus_for_model_worker + num_cpus_for_fold_worker * self.num_splits
        total_num_gpus = num_gpus_for_model_worker + num_gpus_for_fold_worker * self.num_splits

        return ModelResources(
            num_gpus_for_fold_worker=num_gpus_for_fold_worker,
            num_cpus_for_fold_worker=num_cpus_for_fold_worker,
            num_gpus_for_model_worker=num_gpus_for_model_worker,
            num_cpus_for_model_worker=num_cpus_for_model_worker,
            total_num_cpus=total_num_cpus,
            total_num_gpus=total_num_gpus,
        )

    def allocate_resources(self, *, job_ref: str, resources: ModelResources) -> None:
        """Allocate resources for a model fit."""

        self.available_num_cpus -= resources.total_num_cpus
        self.available_num_gpus -= resources.total_num_gpus
        self.job_refs_to_allocated_resources[job_ref] = resources

    def deallocate_resources(self, *, job_ref: str) -> None:
        """Deallocate resources for a model fit."""

        resources = self.job_refs_to_allocated_resources.pop(job_ref)
        self.available_num_cpus += resources.total_num_cpus
        self.available_num_gpus += resources.total_num_gpus

    def clean_up_ray(self, *, unfinished_job_refs: list[str]) -> None:
        """Try to clean up ray object store."""
        import ray

        # TODO: determine how to supress error messages from cancelling jobs.
        for f in unfinished_job_refs:
            ray.cancel(f)

        ray.internal.free(object_refs=[self.job_kwargs[key] for key in self.func_put_kwargs])
        for key in self.func_put_kwargs:
            del self.job_kwargs[key]
