from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Literal
import time
from autogluon.core.models import AbstractModel, BaggedEnsembleModel
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
        X,
        y,
        problem_type="infer",
        num_classes="infer",
        total_mem: int | None | str = "auto",
        max_mem_frac: float = 0.8,
    ):
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.num_classes = num_classes
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
        self.extra_num_cpus = self.total_num_cpus // 10  # FIXME: Maybe remove, allows for oversubscribing
        if isinstance(total_mem, str) and total_mem == "auto":
            # FIXME: when None should be infinite, implement
            total_mem = get_resource_manager().get_available_virtual_mem()
        # FIXME: Should this be passed during init or re-evaluated for each schedule call?
        self.total_mem = total_mem
        self.available_mem = self.total_mem
        self.max_mem_frac = max_mem_frac
        self.max_mem = self.total_mem * self.max_mem_frac

        # Detect max resources a job could use on some node
        # FIXME: deprecated call without alternative?
        self.max_cpu_resources_per_node = int(max([n["CPU"] for n in ray.state.total_resources_per_node().values()]))

        # Job tracking
        self.job_refs_to_allocated_resources: dict[str, ModelResources] = {}
        self.job_refs_to_model_name: dict[str, str] = {}
        self.job_refs_to_model_memory_estimate: dict[str, int] = {}
        self.model_child_mem_estimate_cache: dict[str, int] = {}
        self.models_to_schedule: list[AbstractModel] | list[str] = []

        # Init remote function
        self.remote_func = ray.remote(**DEFAULT_REMOTE_KWARGS)(func)
        self.job_kwargs = dict()
        for key, value in func_kwargs.items():
            self.job_kwargs[key] = value
        for key, value in func_put_kwargs.items():
            self.job_kwargs[key] = ray.put(value)
        self.func_put_kwargs = func_put_kwargs

        logger.log(20, f"MEM AVAILABLE: {self.total_mem * 1e-9:.2f} GB")
        logger.log(20, f"MEM MAX: {self.max_mem * 1e-9:.2f} GB")
        logger.log(20, f"MEM AVAILABLE PER CORE: {self.total_mem_per_core*1e-9:.2f} GB")
        logger.log(20, f"MEM MAX PER CORE: {self.max_mem_per_core*1e-9:.2f} GB")

    @property
    def available_num_cpus_virtual(self):
        return self.available_num_cpus + self.extra_num_cpus

    def num_children_model(self, model: AbstractModel) -> int:
        if (not isinstance(model, BaggedEnsembleModel)) or model._user_params.get("use_child_oof", False):
            return 1
        else:
            return self.num_splits

    @property
    def max_mem_per_core(self):
        return self.max_mem / self.total_num_cpus

    @property
    def total_mem_per_core(self):
        return self.total_mem / self.total_num_cpus

    # # FIXME: What if only 1 model to schedule? In this case we should use all resources for the 1 model.
    # # FIXME: Record memory estimate for efficency per model
    # #  Make mem estimates fast
    # #  Don't mutate models ever
    # #  Mutate num_cpus user args in worker process or even better just pass num_cpus as fit kwarg
    # #  Remove memory estimate for model from total_estimate once job completes
    # # FIXME: Don't give LightGBM/XGBoost/CatBoost more CPUs than they want, cap them at 50% of resources no matter what
    # def prepare_resource_allocation(self, *, models_to_fit: list[AbstractModel] | list[str] | None = None) -> list[str]:
    #     max_mem_per_core = self.max_mem_per_core
    #
    #     if models_to_fit is not None:
    #         models_to_schedule = models_to_fit
    #     else:
    #         models_to_schedule = self.models_to_schedule
    #
    #     models_to_schedule_later = []
    #     job_refs = []
    #     for model in models_to_schedule:
    #         ts = time.time()
    #         model_child_memory_estimate = self.get_memory_estimate_for_model_child(model=model)
    #         te = time.time()
    #         logger.log(20, f"{te - ts:.2f}s\tMEM ESTIMATE TIME {model.name}")
    #         num_cpus_per_child_safe = model_child_memory_estimate / max_mem_per_core
    #
    #         logger.log(20, f"Safe CPUs per child: {math.ceil(num_cpus_per_child_safe)} ({num_cpus_per_child_safe:.2f}): {model.name}")
    #         num_cpus_per_child_safe = max(math.ceil(num_cpus_per_child_safe), 1)
    #
    #         # FIXME: If enough memory for only fitting 1 child, then do it but without fitting parallel! Give bag same resources and model
    #         # FIXME: If enough memory for only fitting 2 child, then do it but only fit 2 at a time, give bag 2x reosurces as child.
    #         # FIXME: This does in-place mutation of model, don't do this!
    #         model = prepare_model_resources_for_fit(
    #             model=model,
    #             num_cpus=num_cpus_per_child_safe,
    #             num_gpus=0,
    #             total_num_cpus=self.total_num_cpus,
    #             total_num_gpus=self.total_num_gpus,
    #         )
    #
    #         # FIXME: is_sufficient must satisfy memory!
    #
    #         model_resources = self.get_resources_for_model(model=model)
    #         model_name = model if self.mode == "refit" else model.name
    #
    #         # FIXME: Keep track of memory usage estimates, use it to limit # of parallel model fits
    #         is_sufficient, reason = self.check_sufficient_resources(resources=model_resources)
    #         if not is_sufficient:
    #             if len(job_refs) + len(self.job_refs_to_allocated_resources) == 0:
    #                 logger.log(
    #                     20,
    #                     "DISTRIBUTED WARNING: Insufficient total resources for training a model fully distributed parallel. "
    #                     "Consider disabling distributed training. "
    #                     "Forcing to train one model anyhow, but this will lead to inefficient parallelization.",
    #                 )
    #
    #                 # Ray's nested calls will keep blocking GPUs and thus create a deadlock if all GPUs are allocated to the model-worker and
    #                 # none can be used by the fold-worker.
    #                 if (
    #                         model_resources.num_gpus_for_model_worker + model_resources.num_gpus_for_fold_worker
    #                 ) > self.total_num_gpus:
    #                     raise ValueError(
    #                         "DISTRIBUTED ERROR: Insufficient number of GPUs to train any model, even in a non-parallel setting. "
    #                         "This is likely the results of requiring more GPUs than available to distribute the training. "
    #                         "Ray does not support freeing GPU resources for nested calls with GPUs. "
    #                         "Thus, we need at least twice the amount of GPUs needed to fit one model."
    #                     )
    #             else:
    #                 if (
    #                         model_resources.num_gpus_for_model_worker + model_resources.num_gpus_for_fold_worker
    #                 ) > self.total_num_gpus:
    #                     logger.log(
    #                         40,
    #                         f"DISTRIBUTED WARNING: Delay scheduling model {model_name}: "
    #                         "Insufficient number of GPUs to train any model, even in a non-parallel setting. "
    #                         "This is likely the results of requiring more GPUs than available to distribute the training. "
    #                         "Ray does not support freeing GPU resources for nested calls with GPUs. "
    #                         "Thus, we need at least twice the amount of GPUs needed to fit one model.",
    #                     )
    #
    #                 logger.log(0, f"Delay scheduling model {model_name}: {reason}.")
    #                 models_to_schedule_later.append(model)
    #                 continue
    #
    #         model_memory_estimate = self.get_memory_estimate_for_model(model=model, mem_usage_child=model_child_memory_estimate)  # FIXME: num_child parallel
    #         job_ref = model_name
    #         job_refs.append(job_ref)
    #         self.allocate_resources(job_ref=job_ref, resources=model_resources, model_name=model_name, model_memory_estimate=model_memory_estimate)
    #         logger.log(
    #             20,
    #             f"Scheduled model {self.mode} for {model_name}: "
    #             f"allocated {'' if is_sufficient else 'UP TO '}{model_resources.total_num_cpus} CPUs and {model_resources.total_num_gpus} GPUs. "
    #             f"{len(self.job_refs_to_allocated_resources)} jobs are running."
    #             f"\n\t{self.total_num_cpus - self.available_num_cpus}/{self.total_num_cpus} Allocated CPUS\n"
    #             f"\t{(self.total_mem - self.available_mem)*1e-9:.1f}/{self.total_mem*1e-9:.1f} GB Allocated Memory"
    #         )
    #
    #     self.models_to_schedule = models_to_schedule_later
    #     return job_refs

    # FIXME: We can lazily execute this logic. We first come up with a plan, then we schedule the workers. This allows for optimal CPU utilization.
    # FIXME: Use available memory to re-calculate per-core values, will more effectively use memory.
    def schedule_jobs(self, *, models_to_fit: list[AbstractModel] | list[str] | None = None) -> list[str]:
        """Schedule model training.

        This function must be first called with `models_to_fit is not None` and then with `models_to_fit is None`.
        Whereby the first call initializes the list of models to fit and subsequent calls schedule the remaining jobs.

        models_to_fit: list[AbstractModel] | list[str] | None, default=None
            The models that shall be fitted in a distributed manner.
        """
        import ray

        max_mem_per_core = self.max_mem_per_core

        if models_to_fit is not None:
            models_to_schedule = models_to_fit
        else:
            models_to_schedule = self.models_to_schedule

        models_to_schedule_later = []
        job_refs = []
        for model in models_to_schedule:
            model_name = model if self.mode == "refit" else model.name
            if self.available_num_cpus_virtual < 1:
                logger.log(20, f"Delay scheduling model {model_name}: CPUs are fully allocated")
                models_to_schedule_later.append(model)
                continue
            num_children = self.num_children_model(model=model)
            if model_name in self.model_child_mem_estimate_cache:
                model_child_memory_estimate = self.model_child_mem_estimate_cache[model_name]
            else:
                ts = time.time()
                model_child_memory_estimate = self.get_memory_estimate_for_model_child(model=model)
                te = time.time()
                logger.log(20, f"{te - ts:.2f}s\tMEM ESTIMATE TIME {model.name}")
                self.model_child_mem_estimate_cache[model_name] = model_child_memory_estimate
            num_cpus_per_child_safe = model_child_memory_estimate / max_mem_per_core

            logger.log(20, f"Safe CPUs per child: {math.ceil(num_cpus_per_child_safe)} ({num_cpus_per_child_safe:.2f}): {model.name}")
            num_cpus_per_child_safe = max(math.ceil(num_cpus_per_child_safe), 1)

            if model_child_memory_estimate > self.max_mem:
                logger.log(20, f"Insufficient total memory to fit {model_name} for even a single fold. Skipping.")
                continue

            num_cpus_avail = self.available_num_cpus_virtual

            # FIXME: Adjust num_cpus_per_child if fitting fewer folds in parallel?
            # FIXME: Not accurate, due to oversubscription, this is dangerous for memory...
            max_safe_children = math.floor(num_cpus_avail / num_cpus_per_child_safe)

            safe_children = max(min(max_safe_children, num_children), 0)

            if safe_children < num_children:
                # FIXME: Make this better, do real successive halving rather than this hack code that only works for 8 or fewer
                if safe_children > 4:
                    safe_children = 4
                elif safe_children > 2:
                    safe_children = 2
                elif safe_children > 1:
                    safe_children = 1
                else:
                    safe_children = 0
                    # skip model
                    pass

                logger.log(20, f"\t{max_safe_children} MAX SAFE CHILDREN\t| {model.name}")

            if safe_children == 0:
                # FIXME: level 15 logging for release
                logger.log(20, f"Delay scheduling model {model_name}: No safe children able to be fit.")
                models_to_schedule_later.append(model)
                continue

            model_memory_estimate = self.get_memory_estimate_for_model(model=model, mem_usage_child=model_child_memory_estimate, num_children=safe_children)

            if safe_children < num_children:
                if (num_children * model_child_memory_estimate) < self.max_mem:
                    # try to wait to schedule later when all folds can be fit in parallel
                    logger.log(
                        20,
                        f"Delay scheduling model {model_name}: Currently can safely fit {safe_children} folds in parallel, "
                        f"waiting to be able to fit all {num_children} folds in parallel."
                    )
                    models_to_schedule_later.append(model)
                    continue
                else:
                    logger.log(
                        20,
                        f"NOTE: {model_name} is too large to ever fit all {num_children} folds in parallel. Fitting {safe_children} folds in parallel..."
                    )
                    # Will never be able to fit all children in parallel because it would use too much memory
                    # TODO: Figure out best option here, for now we train them immediately
                    #  but it may be better to deprioritize them in favor of lower memory usage models

            if isinstance(model, AbstractModel):
                # FIXME: This does in-place mutation of model, don't do this!
                # FIXME: Prefer to avoid this hack, but doing so would require some refactoring.
                #  Basically, we want to avoid passing `model` when it is initialized.
                #  Rather, it is better to pass the cls and kwargs to initialize the model downstream as late as possible.
                model = prepare_model_resources_for_fit(
                    model=model,
                    num_cpus=num_cpus_per_child_safe,
                    num_gpus=0,
                    num_parallel=safe_children,
                    num_children=num_children,
                    total_num_cpus=self.total_num_cpus,
                    total_num_gpus=self.total_num_gpus,
                )
            model_resources = self.get_resources_for_model(model=model)

            # FIXME: Keep track of memory usage estimates, use it to limit # of parallel model fits
            # FIXME: Should this logic ever trigger given the above new logic that checks for `safe_children`
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
            self.allocate_resources(job_ref=job_ref, resources=model_resources, model_name=model_name, model_memory_estimate=model_memory_estimate)

            logger.log(
                20,
                f"Scheduled model {self.mode} for {model_name}: "
                f"allocated {'' if is_sufficient else 'UP TO '}{model_resources.total_num_cpus} CPUs and {model_resources.total_num_gpus} GPUs. | "
                f"{len(self.job_refs_to_allocated_resources)} jobs are running."
                f"\n\tUsing {model_resources.num_cpus_for_fold_worker if num_children != 1 else model_resources.num_cpus_for_model_worker} CPUs per fold for {num_children} folds, fitting {safe_children} folds in parallel"
                f"\n\t{self.total_num_cpus - self.available_num_cpus}/{self.total_num_cpus} Allocated CPUS\n"
                f"\t{(self.total_mem - self.available_mem) * 1e-9:.1f}/{self.total_mem * 1e-9:.1f} GB Allocated Memory",
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

    def get_memory_estimate_for_model_child(self, *, model: AbstractModel) -> int:
        X = self.X  # FIXME: HACK
        y = self.y  # FIXME: HACK

        if hasattr(model, "model_base") and isinstance(model.model_base, AbstractModel) and hasattr(model.model_base, "estimate_memory_usage_static"):
            # FIXME: Don't use `_user_params`
            model_base_hyperparameters = model.model_base._user_params
            # FIXME: need `hyperparameters`, `num_classes`, `problem_type`
            alternate = model.model_base.estimate_memory_usage_static(
                X=X,
                y=y,
                hyperparameters=model_base_hyperparameters,
                problem_type=self.problem_type,
                num_classes=self.num_classes,
            )
            # print(f"ALTERNATE: {alternate}")
            return alternate

        model_clone = copy.deepcopy(model)

        model_clone.initialize(
            X=X,
            y=y,
            # total_resources=kwargs["total_resources"],
            # **kwargs["fit_kwargs"],
        )
        model_clone.model_base.initialize(
            X=X,
            y=y,
            # total_resources=kwargs["total_resources"],
            # **kwargs["fit_kwargs"],
        )
        mem_usage_child = model_clone.estimate_memory_usage_child(
            X=X,
            y=y,
            # total_resources=kwargs["total_resources"],
            # **kwargs["fit_kwargs"],
        )
        # print(f"EXPENSIVE: {mem_usage_child}")

        return mem_usage_child

    def get_memory_estimate_for_model(self, *, model: AbstractModel, mem_usage_child: int = None, num_children: int = None) -> float:
        if num_children is None:
            num_children = self.num_children_model(model)
        if mem_usage_child is None:
            mem_usage_child = self.get_memory_estimate_for_model_child(model=model)
        mem_usage_bag = mem_usage_child * num_children
        mem_usage_child_mb = mem_usage_child * 1e-6
        mem_usage_bag_mb = mem_usage_child_mb * num_children

        logger.log(20, f"\t{mem_usage_bag_mb:.0f} MB (per bag)\t| {mem_usage_child_mb:.0f} MB (per child)\t| {num_children} children\t| {model.name}")
        return mem_usage_bag

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

        if (not isinstance(model, BaggedEnsembleModel)) or model._user_params.get("use_child_oof", False):
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
            total_num_cpus = model._user_params_aux["num_cpus"]

        return ModelResources(
            num_gpus_for_fold_worker=num_gpus_for_fold_worker,
            num_cpus_for_fold_worker=num_cpus_for_fold_worker,
            num_gpus_for_model_worker=num_gpus_for_model_worker,
            num_cpus_for_model_worker=num_cpus_for_model_worker,
            total_num_cpus=total_num_cpus,
            total_num_gpus=num_gpus_for_model_worker + num_gpus_for_fold_worker * self.num_splits,
        )

    def allocate_resources(self, *, job_ref: str, resources: ModelResources, model_memory_estimate: int, model_name: str = None) -> None:
        """Allocate resources for a model fit."""

        self.available_num_cpus -= resources.total_num_cpus
        self.available_num_gpus -= resources.total_num_gpus
        self.available_mem -= model_memory_estimate
        self.job_refs_to_allocated_resources[job_ref] = resources
        self.job_refs_to_model_name[job_ref] = model_name
        self.job_refs_to_model_memory_estimate[job_ref] = model_memory_estimate

    def deallocate_resources(self, *, job_ref: str) -> None:
        """Deallocate resources for a model fit."""

        resources = self.job_refs_to_allocated_resources.pop(job_ref)
        model_name = self.job_refs_to_model_name.pop(job_ref)
        self.available_num_cpus += resources.total_num_cpus
        self.available_num_gpus += resources.total_num_gpus
        model_memory_estimate = self.job_refs_to_model_memory_estimate.pop(job_ref)
        self.available_mem += model_memory_estimate
        self.model_child_mem_estimate_cache.pop(model_name)

    def clean_unfinished_job_refs(self, *, unfinished_job_refs: list[str] | None = None):
        import ray

        # TODO: determine how to suppress error messages from cancelling jobs.
        if unfinished_job_refs is not None and len(unfinished_job_refs) > 0:
            model_names = [self.job_refs_to_model_name[f] for f in unfinished_job_refs]
            logger.log(20, f"Cancelling {len(model_names)} jobs: {model_names}")
            for f in unfinished_job_refs:
                ray.cancel(f, force=True)

    def clean_job_state(self, *, unfinished_job_refs: list[str] | None = None) -> None:
        """Clean up state of manager."""
        self.job_refs_to_allocated_resources = {}
        self.job_refs_to_model_name = {}
        self.job_refs_to_model_memory_estimate = {}
        self.model_child_mem_estimate_cache = {}
        self.models_to_schedule = []
        self.available_num_cpus = self.total_num_cpus
        self.available_num_gpus = self.total_num_gpus
        self.available_mem = self.total_mem
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
    num_parallel: int = None,
    num_children: int = 1,
) -> AbstractModel:
    """Allocate each model resources for fitting. (This is currently an in-place operation!)

    We allocate resources by setting the _user_params_aux of a model.

    """
    if num_parallel is None:
        num_parallel = num_children
    assert num_parallel <= num_children

    if num_cpus == 0:
        # Non-bagged mode
        num_cpus = num_cpus_worker
        num_gpus = num_gpus_worker
    else:
        num_cpus_parent = num_cpus * num_parallel
        num_gpus_parent = num_gpus * num_parallel

        model_aux = model._user_params_aux
        if "num_cpus" not in model_aux:
            model_aux["num_cpus"] = num_cpus_parent
        if "num_gpus" not in model_aux:
            model_aux["num_gpus"] = num_gpus_parent

    upa = getattr(model, "model_base", model)._user_params_aux
    if "num_cpus" not in upa:
        getattr(model, "model_base", model)._user_params_aux["num_cpus"] = num_cpus
    if "num_gpus" not in upa:
        getattr(model, "model_base", model)._user_params_aux["num_gpus"] = num_gpus

    return model
