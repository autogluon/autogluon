from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal
import time
from autogluon.core.models import AbstractModel, StackerEnsembleModel
from autogluon.common.utils.resource_utils import get_resource_manager
import math

logger = logging.getLogger(__name__)

DEFAULT_REMOTE_KWARGS = dict(max_calls=1, retry_exceptions=False, max_retries=0)


@dataclass
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
    mode: {"fit", "refit"}
        The mode to use for fitting the models.
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
    num_splits : int | None, default=None
        Number of training splits/bags for a model. Required if mode='fit'.
    get_model_attribute_func : callable, default=None
        Function to get an attribute for a model. Required if mode='refit'.
    """

    def __init__(
        self,
        *,
        mode: Literal["fit", "refit"],
        func: Callable,
        func_kwargs: dict,
        func_put_kwargs: dict,
        num_cpus: int | str,
        num_gpus: int | str,
        num_splits: int | None = None,
        get_model_attribute_func: Callable | None = None,
    ):
        import ray

        self.mode = mode
        self.num_splits = num_splits
        self.get_model_attribute_func = get_model_attribute_func

        if self.mode == "fit":
            assert num_splits is not None, "num_splits must be set for mode='fit'."
        if self.mode == "refit":
            assert get_model_attribute_func is not None, "get_model_attribute_func must be set for mode='refit'."

        # Resource tracking
        if isinstance(num_cpus, str):
            num_cpus = get_resource_manager().get_cpu_count()
        if isinstance(num_gpus, str):
            num_gpus = get_resource_manager().get_gpu_count()
        self.total_num_cpus = num_cpus
        self.total_num_gpus = num_gpus
        self.available_num_cpus = num_cpus
        self.available_num_gpus = num_gpus

        # Detect max resources a job could use on some node
        # FIXME: deprecated call without alternative?
        self.max_cpu_resources_per_node = int(max([n["CPU"] for n in ray.state.total_resources_per_node().values()]))

        # Job tracking
        self.job_refs_to_allocated_resources: dict[str, ModelResources] = {}
        self.models_to_schedule: list[AbstractModel] | list[str] = []

        # Init remote function
        self.remote_func = ray.remote(**DEFAULT_REMOTE_KWARGS)(func)
        self.job_kwargs = dict()
        for key, value in func_kwargs.items():
            self.job_kwargs[key] = value
        for key, value in func_put_kwargs.items():
            self.job_kwargs[key] = ray.put(value)
        self.func_put_kwargs = func_put_kwargs

    def schedule_jobs(self, *, models_to_fit: list[AbstractModel] | list[str] | None = None) -> list[str]:
        """Schedule model training.

        This function must be first called with `models_to_fit is not None` and then with `models_to_fit is None`.
        Whereby the first call initializes the list of models to fit and subsequent calls schedule the remaining jobs.

        models_to_fit: list[AbstractModel] | list[str] | None, default=None
            The models that shall be fitted in a distributed manner.
        """
        import ray

        if models_to_fit is not None:
            models_to_schedule = models_to_fit
        else:
            models_to_schedule = self.models_to_schedule

        models_to_schedule_later = []
        job_refs = []
        for model in models_to_schedule:
            if isinstance(model, AbstractModel):
                # FIXME: Prefer to avoid this hack, but doing so would require some refactoring.
                #  Basically, we want to avoid passing `model` when it is initialized.
                #  Rather, it is better to pass the cls and kwargs to initialize the model downstream as late as possible.
                model = prepare_model_resources_for_fit(
                    model=model,
                    num_cpus=1,
                    num_gpus=0,
                    total_num_cpus=self.total_num_cpus,
                    total_num_gpus=self.total_num_gpus,
                )
            model_resources = self.get_resources_for_model(model=model)
            model_name = model if self.mode == "refit" else model.name

            # FIXME: Keep track of memory usage estimates, use it to limit # of parallel model fits
            is_sufficient, reason = self.check_sufficient_resources(resources=model_resources)
            if not is_sufficient:
                if len(job_refs) + len(self.job_refs_to_allocated_resources) == 0:
                    logger.log(
                        20,
                        "DISTRIBUTED WARNING: Insufficient total resources for training a model fully distributed parallel. "
                        "Consider disabling distributed training. "
                        "Forcing to train one model anyhow, but this will lead to inefficient parallelization.",
                    )

                    # Ray's nested calls will keep blocking GPUs and thus create a deadlock if all GPUs are allocated to the model-worker and
                    # none can be used by the fold-worker.
                    if (
                        model_resources.num_gpus_for_model_worker + model_resources.num_gpus_for_fold_worker
                    ) > self.total_num_gpus:
                        raise ValueError(
                            "DISTRIBUTED ERROR: Insufficient number of GPUs to train any model, even in a non-parallel setting. "
                            "This is likely the results of requiring more GPUs than available to distribute the training. "
                            "Ray does not support freeing GPU resources for nested calls with GPUs. "
                            "Thus, we need at least twice the amount of GPUs needed to fit one model."
                        )
                else:
                    if (
                        model_resources.num_gpus_for_model_worker + model_resources.num_gpus_for_fold_worker
                    ) > self.total_num_gpus:
                        logger.log(
                            40,
                            f"DISTRIBUTED WARNING: Delay scheduling model {model_name}: "
                            "Insufficient number of GPUs to train any model, even in a non-parallel setting. "
                            "This is likely the results of requiring more GPUs than available to distribute the training. "
                            "Ray does not support freeing GPU resources for nested calls with GPUs. "
                            "Thus, we need at least twice the amount of GPUs needed to fit one model.",
                        )

                    logger.log(0, f"Delay scheduling model {model_name}: {reason}.")
                    models_to_schedule_later.append(model)
                    continue

            job_ref = self.remote_func.options(
                num_cpus=model_resources.num_cpus_for_model_worker, num_gpus=model_resources.num_gpus_for_model_worker
            ).remote(model=ray.put(model) if self.mode in ["fit"] else model, **self.job_kwargs)
            job_refs.append(job_ref)
            self.allocate_resources(job_ref=job_ref, resources=model_resources)

            logger.log(
                20,
                f"Scheduled model {self.mode} for {model_name}. {len(self.job_refs_to_allocated_resources)} jobs are running."
                f"\n\tAllocated {'' if is_sufficient else 'UP TO '}{model_resources.total_num_cpus} CPUs and {model_resources.total_num_gpus} GPUs"
                f"\n\tRay{job_ref}",
            )
            time.sleep(0.1)

        self.models_to_schedule = models_to_schedule_later
        return job_refs

    def check_sufficient_resources(self, *, resources: ModelResources) -> tuple[bool, str | None]:
        """Determine if there are enough resources to scheduling fitting another model."""

        # Allow for oversubscribing to 10% of the CPUs due to scheduling overhead.
        if self.available_num_cpus + (self.total_num_cpus // 10) < resources.total_num_cpus:
            return False, "not enough CPUs free."

        # All models need at least one CPU but not all a GPU
        # Avoid scheduling a model if there are not at least 50% of the required GPUs available
        if (resources.total_num_gpus > 0) and (self.available_num_gpus < math.ceil(resources.total_num_gpus / 2)):
            return False, "not enough GPUs free."

        return True, None

    def get_resources_for_model(self, *, model: AbstractModel | str) -> ModelResources:
        if self.mode == "fit":
            # model is AbstractModel
            return self.get_resources_for_model_fit(model=model)
        elif self.mode == "refit":
            # model is str
            return self.get_resources_for_model_refit(model=model)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_resources_for_model_refit(self, model: str) -> ModelResources:
        """Estimate the resources required to fit a model."""

        # FIXME: We should synchronize this with the actual values used downstream.
        #  Currently we are specifying it here and downstream separately, which gives the opportunity for them to not be matching.
        #  This should be fixed.
        # FIXME: Should this use fit_num_cpus_child or fit_num_cpus ?
        num_gpus_for_fold_worker = self.get_model_attribute_func(model=model, attribute="fit_num_gpus_child")
        num_cpus_for_fold_worker = self.get_model_attribute_func(model=model, attribute="fit_num_cpus_child")
        num_cpus_for_fold_worker = (
            num_cpus_for_fold_worker if num_cpus_for_fold_worker is not None else min(self.max_cpu_resources_per_node, self.total_num_cpus)
        )
        num_gpus_for_fold_worker = num_gpus_for_fold_worker if num_gpus_for_fold_worker is not None else 0

        num_gpus_for_model_worker = (
            1 if self.get_model_attribute_func(model=model, attribute="refit_full_requires_gpu") else 0
        )
        return ModelResources(
            num_gpus_for_fold_worker=num_gpus_for_fold_worker,
            num_cpus_for_fold_worker=num_cpus_for_fold_worker,
            num_gpus_for_model_worker=num_gpus_for_model_worker,
            num_cpus_for_model_worker=1,
            # num_cpus_for_model_worker is freed once the nested ray call is done
            total_num_cpus=num_cpus_for_fold_worker,
            total_num_gpus=num_gpus_for_model_worker + num_gpus_for_fold_worker,
        )

    def get_resources_for_model_fit(self, *, model: AbstractModel) -> ModelResources:
        """Estimate the resources required to fit a model."""

        if "num_cpus" not in getattr(model, "model_base", model)._user_params_aux:
            logger.warning(
                f"DISTRIBUTED WARNING: Model {model.name} does not specify the number of resources to use! "
                "Assuming that the model will use all available node resources, which can heavily impact the performance of distributed training."
            )

        # Fallback if required information are not given at this point, we can only assume that the model will use all available CPU resources of a node.
        num_cpus_for_fold_worker = getattr(model, "model_base", model)._user_params_aux.get(
            "num_cpus", self.max_cpu_resources_per_node
        )
        # As we have no information about if the model needs GPUs, we assume that it does not need any.
        num_gpus_for_fold_worker = getattr(model, "model_base", model)._user_params_aux.get("num_gpus", 0)

        if (not isinstance(model, StackerEnsembleModel)) or model._user_params.get("use_child_oof", False):
            # Only one fold is fit, so we need to use all fit resources for the model-worker
            num_cpus_for_model_worker = num_cpus_for_fold_worker
            num_gpus_for_model_worker = num_gpus_for_fold_worker
            num_cpus_for_fold_worker = 0
            num_gpus_for_fold_worker = 0
            total_num_cpus = num_cpus_for_model_worker
        else:
            # If refit_folds is True, we need to pass GPU resources to the model-worker
            num_gpus_for_model_worker = (
                num_gpus_for_fold_worker
                if ((num_gpus_for_fold_worker > 0) and model._user_params.get("refit_folds", False))
                else 0
            )
            num_cpus_for_model_worker = 1

            # num_cpus_for_model_worker is freed once the nested ray call is done
            total_num_cpus = num_cpus_for_fold_worker * self.num_splits

        return ModelResources(
            num_gpus_for_fold_worker=num_gpus_for_fold_worker,
            num_cpus_for_fold_worker=num_cpus_for_fold_worker,
            num_gpus_for_model_worker=num_gpus_for_model_worker,
            num_cpus_for_model_worker=num_cpus_for_model_worker,
            total_num_cpus=total_num_cpus,
            total_num_gpus=num_gpus_for_model_worker + num_gpus_for_fold_worker * self.num_splits,
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

    def clean_unfinished_job_refs(self, *, unfinished_job_refs: list[str] | None = None):
        import ray

        # TODO: determine how to suppress error messages from cancelling jobs.
        if unfinished_job_refs is not None:
            for f in unfinished_job_refs:
                ray.cancel(f, force=True)

    def clean_job_state(self, *, unfinished_job_refs: list[str] | None = None) -> None:
        """Clean up state of manager."""
        self.job_refs_to_allocated_resources = {}
        self.models_to_schedule = []
        self.available_num_cpus = self.total_num_cpus
        self.available_num_gpus = self.total_num_gpus
        self.clean_unfinished_job_refs(unfinished_job_refs=unfinished_job_refs)

    def clean_up_ray(self, *, unfinished_job_refs: list[str] | None = None) -> None:
        """Try to clean up ray object store."""
        import ray

        self.clean_unfinished_job_refs(unfinished_job_refs=unfinished_job_refs)

        ray.internal.free(object_refs=[self.job_kwargs[key] for key in self.func_put_kwargs])
        for key in self.func_put_kwargs:
            del self.job_kwargs[key]


# TODO: make this logic be good.
def prepare_model_resources_for_fit(
    *,
    model: AbstractModel,
    total_num_cpus: int,
    total_num_gpus: int,
    num_cpus: int = 1,
    num_gpus: float = 0,
    num_cpus_worker: int = 1,
    num_gpus_worker: float = 0,
) -> AbstractModel:
    """Allocate each model resources for fitting. (This is currently an in-place operation!)

    We allocate resources by setting the _user_params_aux of a model.

    """

    if num_cpus == 0:
        # Non-bagged mode
        num_cpus = num_cpus_worker
        num_gpus = num_gpus_worker

    upa = getattr(model, "model_base", model)._user_params_aux
    if "num_cpus" not in upa:
        getattr(model, "model_base", model)._user_params_aux["num_cpus"] = num_cpus
    if "num_gpus" not in upa:
        getattr(model, "model_base", model)._user_params_aux["num_gpus"] = num_gpus

    return model
