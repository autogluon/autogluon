from __future__ import annotations

import copy
import json
import logging
import math
import os
import pickle
import time
import traceback
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union, TYPE_CHECKING

import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series

from autogluon.common.utils.lite import disable_if_lite_mode
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.s3_utils import download_s3_folder, s3_path_to_bucket_prefix, upload_s3_folder
from autogluon.common.utils.try_import import try_import_ray
from autogluon.common.utils.distribute_utils import DistributedContext
from autogluon.common.utils.log_utils import reset_logger_for_remote_call

from ...pseudolabeling.pseudolabeling import assert_pseudo_column_match
from ...ray.resources_calculator import ResourceCalculatorFactory
from ...utils.exceptions import AutoGluonException, NoGPUError, NoValidFeatures, NoStackFeatures, NotValidStacker, InsufficientTime, NotEnoughCudaMemoryError, NotEnoughMemoryError, TimeLimitExceeded
from ..abstract.abstract_model import AbstractModel

if TYPE_CHECKING:
    from .bagged_ensemble_model import BaggedEnsembleModel

logger = logging.getLogger(__name__)

TEXT_MODEL = "TextPredictorModel"
IMAGE_MODEL = "ImagePredictorModel"
TABULAR_TORCH_MODEL = "TabularNeuralNetModel"
TABULAR_FASTAI_MODEL = "NNFastAiTabularModel"


class AbstractFoldFittingStrategy:
    @abstractmethod
    def schedule_fold_model_fit(self, fold_ctx):
        """
        Schedule fold model training.
        By design this part is supposed to be 'lazy' evaluator,
        no actual training is performed here.
        Distributed fitters will handle jobs scheduling here.
        """

    @abstractmethod
    def after_all_folds_scheduled(self):
        """
        Method is called when all the folds are scheduled.
        Local fitters will perform training here.
        Distributed fitters will handle job handles and results retrieval here.
        """

    @abstractmethod
    def _fit(self, model_base, time_start_fold, time_limit_fold, fold_ctx, kwargs):
        """
        Method is called when a fold is ready to be fit
        """


class FoldFittingStrategy(AbstractFoldFittingStrategy):
    """
    Provides some default implementation for AbstractFoldFittingStrategy

    Parameters
    ----------
        model_base: AbstractModel
            The template for the folds of model to be trained on.
        model_base_kwargs: dict
            kwargs required to initialize the model base when training the model.
        bagged_ensemble_model : BaggedEnsembleModel
            The ensemble model that holds all the trained folds.
        X : DataFrame
            The training data the model will be trained on.
        y: Series
            The labels of the training data.
        sample_weight:
            The sample weight of the training data.
        time_limit: float
            Approximately how long(in seconds) the fold fitting should be run for.
            If None, no time-constraint will be enforced allowing the folds to fully train.
        time_start: float
            Time starts to fit the BaggedEnsembleModel.
        models: list
            List of models that will be trained.
        oof_pred_proba: ndarray
            Out of folds predict probabilities that are already calculated.
        oof_pred_model_repeats: ndarray,
            Number of repeats the out of folds predict probabilities has been done.
        save_folds: bool,
            Whether to save the folds to disk.
        time_limit_fold_ratio: float, default=0.8
            The ratio of max time limit for each fold.
            If the estimated time required exceeds this ratio, will raise TimeLimitExceed error
    Attributes
    ----------
        X : DataFrame
            The training data the model will be trained on.
        y: Series
            The labels of the training data.
        sample_weight:
            The sample weight of the training data.
        time_limit: float
            Approximately how long(in seconds) the fold fitting should be run for.
            If None, no time-constraint will be enforced allowing the folds to fully train.
        time_start: float
            Time starts to fit the BaggedEnsembleModel.
        models: list
            List of models that will be trained.
        oof_pred_proba: ndarray
            Out of folds predict probabilities that are already calculated.
        oof_pred_model_repeats: ndarray,
            Number of repeats the out of folds predict probabilities has been done.
        jobs: list
            List of jobs that will be scheduled.
        save_folds: bool,
            Whether to save the folds to disk.
        time_limit_fold_ratio: float
            The ratio of max time limit for each fold.
    """

    def __init__(
        self,
        model_base,
        model_base_kwargs,
        bagged_ensemble_model: "BaggedEnsembleModel",
        X: DataFrame,
        y: Series,
        X_pseudo: DataFrame,
        y_pseudo: Series,
        sample_weight,
        time_limit: float,
        time_start: float,
        models: list,
        oof_pred_proba: ndarray,
        oof_pred_model_repeats: ndarray,
        save_folds: bool,
        num_cpus: int,
        num_gpus: Union[int, float],
        time_limit_fold_ratio=0.8,
        **kwargs,
    ):
        self.model_base = model_base
        self.model_base_kwargs = model_base_kwargs
        self.X = X
        self.y = y
        self.X_pseudo = X_pseudo
        self.y_pseudo = y_pseudo
        self.sample_weight = sample_weight
        self.time_limit = time_limit
        self.time_start = time_start
        self.models = models
        self.oof_pred_proba = oof_pred_proba
        self.oof_pred_model_repeats = oof_pred_model_repeats
        self.bagged_ensemble_model = bagged_ensemble_model
        self.jobs = []
        self.save_folds = save_folds
        self.time_limit_fold_ratio = time_limit_fold_ratio
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        logger.debug(f"Upper level total_num_cpus, num_gpus {self.num_cpus} | {self.num_gpus}")
        self._validate_user_specified_resources()
        if not isinstance(self.num_cpus, int):
            raise TypeError(f"`num_cpus` must be an int! Found: {type(num_cpus)} | Value: {self.num_cpus}")

    def schedule_fold_model_fit(self, fold_ctx):
        raise NotImplementedError

    def after_all_folds_scheduled(self):
        raise NotImplementedError

    def _validate_user_specified_resources(self):
        # User specified value through ag_args_fit means they want this individual model to use this amount of resources
        user_ensemble_resources = None
        user_resources_per_job = None
        # initialize the model base to get necessary info for estimating memory usage and getting resources
        self._initialized_model_base = copy.deepcopy(self.model_base)
        self._initialized_model_base.initialize(X=self.X, y=self.y, **self.model_base_kwargs)
        user_cpu_per_job = self._initialized_model_base._get_child_aux_val(key="num_cpus", default=None)
        user_gpu_per_job = self._initialized_model_base._get_child_aux_val(key="num_gpus", default=None)
        minimum_model_resources = self._initialized_model_base.get_minimum_resources(
            is_gpu_available=(self.num_gpus > 0),
        )
        minimum_model_num_cpus = minimum_model_resources.get("num_cpus", 1)
        minimum_model_num_gpus = minimum_model_resources.get("num_gpus", 0)
        logger.debug(f"minimum_model_resources: {minimum_model_resources}")
        logger.debug(f"user_cpu_per_job, user_gpu_per_job {user_cpu_per_job} | {user_gpu_per_job}")
        user_ensemble_cpu = self.bagged_ensemble_model._user_params_aux.get("num_cpus", None)
        user_ensemble_gpu = self.bagged_ensemble_model._user_params_aux.get("num_gpus", None)
        logger.debug(f"user_ensemble_cpu, user_ensemble_gpu {user_ensemble_cpu} | {user_ensemble_gpu}")
        if user_ensemble_cpu is not None or user_ensemble_gpu is not None:
            user_ensemble_resources = dict()
        if user_ensemble_cpu is not None:
            assert user_ensemble_cpu <= self.num_cpus, f"Detected ensemble cpu requirement = {user_ensemble_cpu} > total cpu granted = {self.num_cpus}"
            assert (
                user_ensemble_cpu >= minimum_model_num_cpus
            ), f"Detected ensenble cpu requirement = {user_ensemble_cpu} < minimum cpu required by the model = {minimum_model_num_cpus}"
            user_ensemble_resources["num_cpus"] = user_ensemble_cpu
            self.num_cpus = user_ensemble_cpu
        if user_ensemble_gpu is not None:
            assert user_ensemble_gpu <= self.num_gpus, f"Detected ensemble gpu requirement = {user_ensemble_gpu} > total gpu granted = {self.num_gpus}"
            assert (
                user_ensemble_gpu >= minimum_model_num_gpus
            ), f"Detected ensenble gpu requirement = {user_ensemble_cpu} < minimum gpu required by the model = {minimum_model_num_gpus}"
            user_ensemble_resources["num_gpus"] = user_ensemble_gpu
            self.num_gpus = user_ensemble_gpu
        if user_cpu_per_job is not None or user_gpu_per_job is not None:
            user_resources_per_job = dict()
        if user_cpu_per_job is not None:
            assert (
                user_cpu_per_job <= self.num_cpus
            ), f"Detected model level cpu requirement = {user_cpu_per_job} > total cpu granted to the bagged model = {self.num_cpus}"
            assert (
                user_cpu_per_job >= minimum_model_num_cpus
            ), f"Detected model level cpu requirement = {user_cpu_per_job} < minimum cpu required by the model = {minimum_model_num_cpus}"
            user_resources_per_job["num_cpus"] = user_cpu_per_job
        if user_gpu_per_job is not None:
            assert (
                user_gpu_per_job <= self.num_gpus
            ), f"Detected model level gpu requirement = {user_gpu_per_job} > total gpu granted to the bagged model = {self.num_gpus}"
            assert (
                user_gpu_per_job >= minimum_model_num_gpus
            ), f"Detected model level gpu requirement = {user_gpu_per_job} < minimum gpu required by the model = {minimum_model_num_gpus}"
            user_resources_per_job["num_gpus"] = user_gpu_per_job
        self.user_ensemble_resources = user_ensemble_resources
        self.user_resources_per_job = user_resources_per_job

    def _get_fold_time_limit(self, fold_ctx):
        _, folds_finished, folds_left, folds_to_fit, _, _, _ = self._get_fold_properties(fold_ctx)
        time_elapsed = time.time() - self.time_start
        if self.time_limit is not None:
            time_left = self.time_limit - time_elapsed
            required_time_per_fold = time_left / folds_left
            time_limit_fold = required_time_per_fold * self.time_limit_fold_ratio
            if folds_finished > 0:
                expected_time_required = time_elapsed * folds_to_fit / folds_finished
                expected_remaining_time_required = expected_time_required * folds_left / folds_to_fit
                if expected_remaining_time_required > time_left:
                    raise TimeLimitExceeded
            if time_left <= 0:
                raise TimeLimitExceeded
        else:
            time_limit_fold = None
        return time_limit_fold

    def _update_bagged_ensemble(self, fold_model, pred_proba, fold_ctx):
        _, val_index = fold_ctx["fold"]
        model_to_append = fold_model
        if not self.save_folds:
            fold_model.model = None
        if self.bagged_ensemble_model.low_memory:
            self.bagged_ensemble_model.save_child(fold_model, verbose=False)
            model_to_append = fold_model.name
        self.models.append(model_to_append)
        self.oof_pred_proba[val_index] += pred_proba
        self.oof_pred_model_repeats[val_index] += 1
        self.bagged_ensemble_model._add_child_times_to_bag(model=fold_model)
        self.bagged_ensemble_model._add_child_num_cpus(num_cpus=fold_model.fit_num_cpus)
        self.bagged_ensemble_model._add_child_num_gpus(num_gpus=fold_model.fit_num_gpus)

    def _predict_oof(self, fold_model: AbstractModel, fold_ctx) -> Tuple[AbstractModel, ndarray]:
        fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix, _ = self._get_fold_properties(fold_ctx)
        _, val_index = fold
        X_val_fold = self.X.iloc[val_index, :]
        y_val_fold = self.y.iloc[val_index]
        # Check to avoid unnecessarily predicting and saving a model
        # when an Exception is going to be raised later
        if self.time_limit is not None:
            if not is_last_fold:
                time_elapsed = time.time() - self.time_start
                time_left = self.time_limit - time_elapsed
                expected_time_required = time_elapsed * folds_to_fit / (folds_finished + 1)
                expected_remaining_time_required = expected_time_required * (folds_left - 1) / folds_to_fit
                if expected_remaining_time_required > time_left:
                    raise TimeLimitExceeded
        y_pred_proba = fold_model.predict_proba(X_val_fold, record_time=True)
        fold_model.val_score = fold_model.score_with_y_pred_proba(y=y_val_fold, y_pred_proba=y_pred_proba)
        fold_model.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
        if not self.bagged_ensemble_model.params.get("save_bag_folds", True):
            fold_model.model = None
        return fold_model, y_pred_proba

    @staticmethod
    def _get_fold_properties(fold_ctx):
        fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix, random_seed = [
            fold_ctx[f] for f in ["fold", "folds_finished", "folds_left", "folds_to_fit", "is_last_fold", "model_name_suffix", "random_seed"]
        ]
        return fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix, random_seed


class SequentialLocalFoldFittingStrategy(FoldFittingStrategy):
    """
    This strategy fits the folds locally in a sequence.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        total_num_cpus = self.num_cpus
        total_num_gpus = self.num_gpus

        default_num_cpus, default_num_gpus = self._initialized_model_base._get_default_resources()
        if self.user_resources_per_job is None:
            fit_num_cpus, fit_num_gpus = default_num_cpus, default_num_gpus
        else:
            fit_num_cpus = self.user_resources_per_job.get("num_cpus", default_num_cpus)
            fit_num_gpus = self.user_resources_per_job.get("num_gpus", default_num_gpus)

        # ensure that we never use more resources than the total system resources provided
        fit_num_cpus = min(fit_num_cpus, total_num_cpus)
        fit_num_gpus = min(fit_num_gpus, total_num_gpus)

        assert fit_num_cpus >= 1
        assert fit_num_gpus >= 0

        # TODO: v1.5: Fix the below, need to consistently define what `get_minimum_resources` and `get_default_resources` mean.
        #  Currently SequentialLocal will use default resources to define the resources to fit the model
        #  But this differs from ParallelLocal which can use more than default resources if num_jobs=1 (pseudo sequential)
        #  This means ParallelLocal can use all logical cores to fit 1 model even if the model's default specifies only physical cores.
        #  Currently I think that the above code is the more correct code, as it respects `get_default_resources`
        #  TL;DR: Align logic between parallel and sequential so when num_jobs=1, they both do the same thing in terms of num_cpus and num_gpus during fit.
        # model_min_resources = self._initialized_model_base.get_minimum_resources(is_gpu_available=(self.num_gpus > 0))
        # resources_calculator = ResourceCalculatorFactory.get_resource_calculator(calculator_type="cpu" if self.num_gpus == 0 else "gpu")
        # # use minimum resource to control number of jobs running in parallel
        # min_cpu_based_on_model = model_min_resources.get("num_cpus", 1)
        # min_gpu_based_on_model = model_min_resources.get("num_gpus", 0)
        #
        # get_resources_per_job_args = dict(
        #     total_num_cpus=self.num_cpus,
        #     total_num_gpus=self.num_gpus,
        #     num_jobs=1,
        #     minimum_cpu_per_job=max(self.num_cpus, min_cpu_based_on_model),
        #     minimum_gpu_per_job=max(self.num_gpus, min_gpu_based_on_model),
        #     user_resources_per_job=self.user_resources_per_job,
        # )
        # if self.user_resources_per_job is not None:
        #     get_resources_per_job_args["minimum_cpu_per_job"] = min_cpu_based_on_model
        #     get_resources_per_job_args["minimum_gpu_per_job"] = min_gpu_based_on_model
        #
        # resources_info = resources_calculator.get_resources_per_job(**get_resources_per_job_args)

        self.resources = {"num_cpus": fit_num_cpus, "num_gpus": fit_num_gpus}

    def schedule_fold_model_fit(self, fold_ctx):
        self.jobs.append(fold_ctx)

    def after_all_folds_scheduled(self):
        for job in self.jobs:
            self._fit_fold_model(job)

    def _fit_fold_model(self, fold_ctx):
        time_start_fold = time.time()
        time_limit_fold = self._get_fold_time_limit(fold_ctx)
        fold_model = self._fit(self.model_base, time_start_fold, time_limit_fold, fold_ctx, self.model_base_kwargs)
        fold_model, pred_proba = self._predict_oof(fold_model, fold_ctx)
        self._update_bagged_ensemble(fold_model, pred_proba, fold_ctx)

    def _fit(self, model_base, time_start_fold, time_limit_fold, fold_ctx, kwargs):
        fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix, random_seed = self._get_fold_properties(fold_ctx)
        train_index, val_index = fold
        X_fold, X_val_fold = self.X.iloc[train_index, :], self.X.iloc[val_index, :]
        y_fold, y_val_fold = self.y.iloc[train_index], self.y.iloc[val_index]
        fold_model = copy.deepcopy(model_base)
        fold_model.name = f"{fold_model.name}{model_name_suffix}"
        fold_model.set_contexts(os.path.join(self.bagged_ensemble_model.path, fold_model.name))
        kwargs_fold = kwargs.copy()
        is_pseudo = self.X_pseudo is not None and self.y_pseudo is not None
        if self.sample_weight is not None:
            kwargs_fold["sample_weight"] = self.sample_weight[train_index]
            kwargs_fold["sample_weight_val"] = self.sample_weight[val_index]

            if is_pseudo:
                # TODO: Add support for sample_weight when pseudo is present
                raise Exception("Sample weights given, but not used due to pseudo labelled data being given.")
            else:
                kwargs_fold["sample_weight"] = self.sample_weight[train_index]
                kwargs_fold["sample_weight_val"] = self.sample_weight[val_index]

        if random_seed is not None:
            kwargs_fold["random_seed"] = random_seed

        if is_pseudo:
            logger.log(15, f"{len(self.X_pseudo)} extra rows of pseudolabeled data added to training set for {fold_model.name}")
            assert_pseudo_column_match(X=X_fold, X_pseudo=self.X_pseudo)
            X_fold = pd.concat([X_fold, self.X_pseudo], axis=0, ignore_index=True)
            y_fold = pd.concat([y_fold, self.y_pseudo], axis=0, ignore_index=True)

        num_cpus = self.num_cpus
        num_gpus = self.num_gpus
        if self.user_resources_per_job is not None:
            num_cpus = min(self.num_cpus, self.user_resources_per_job.get("num_cpus", math.inf))
            num_gpus = min(self.num_gpus, self.user_resources_per_job.get("num_gpus", math.inf))
        fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, num_cpus=num_cpus, num_gpus=num_gpus, **kwargs_fold)
        fold_model.fit_time = time.time() - time_start_fold
        return fold_model


def _ray_fit(
    *,
    model_base: AbstractModel,
    bagged_ensemble_model_path: str,
    X: Union[str, pd.DataFrame],
    y: Union[str, pd.DataFrame],
    X_pseudo: Union[str, pd.DataFrame],
    y_pseudo: Union[str, pd.DataFrame],
    task_id: int,
    fold_ctx: Dict[str, Any],
    assignments: Dict[int, List[int]],
    time_limit_fold: float,
    save_bag_folds: bool,
    resources: Dict[str, Any],
    kwargs_fold: Dict[str, Any],
    head_node_id: str,
    model_sync_path: Optional[str] = None,
):
    import ray  # ray must be present
    import torch
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # torch.cuda.set_device(0)
    gpu_ids = assignments.get(task_id, [])
    if gpu_ids:
        # Set CUDA_VISIBLE_DEVICES to the assigned GPU IDs
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        logger.debug(f"Set CUDA_VISIBLE_DEVICES to {gpu_ids}")

    reset_logger_for_remote_call(verbosity=kwargs_fold.get("verbosity",2))

    node_id = ray.get_runtime_context().get_node_id()
    is_head_node = node_id == head_node_id
    logger.debug(f"head node: {is_head_node}")
    logger.debug(f"executing fold on node {node_id}")
    logger.log(10, "ray worker training")

    # DEBUG: Show GPU assignment
    try:
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        num_gpus = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device() if torch.cuda.is_available() else "N/A"
        print(f"[GPU DEBUG] CUDA_VISIBLE_DEVICES={visible_gpus}, Torch sees {num_gpus} GPUs, Using GPU {current_gpu}", flush=True)
    except Exception as e:
        print(f"[GPU DEBUG] Could not get GPU info: {e}", flush=True)
    time_start_fold = time.time()
    fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix, _ = FoldFittingStrategy._get_fold_properties(fold_ctx)
    train_index, val_index = fold
    fold_model = copy.deepcopy(model_base)
    fold_model.name = f"{fold_model.name}{model_name_suffix}"
    fold_model_local_save_path = os.path.join(bagged_ensemble_model_path, fold_model.name)
    fold_model.set_contexts(fold_model_local_save_path)
    if type(X) == str and type(y) == str:
        with open(X, "rb") as X_f, open(y, "rb") as y_f:
            X = pickle.load(X_f)
            y = pickle.load(y_f)
    is_pseudo = False
    if X_pseudo is not None and y_pseudo is not None:
        if type(X_pseudo) == str and type(y_pseudo) == str:
            with open(X_pseudo, "rb") as X_pseudo_f, open(y_pseudo, "rb") as y_pseudo_f:
                X_pseudo = pickle.load(X_pseudo_f)
                y_pseudo = pickle.load(y_pseudo_f)
        is_pseudo = True

    X_fold, X_val_fold = X.iloc[train_index, :], X.iloc[val_index, :]
    y_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    if is_pseudo:
        logger.log(15, f"{len(X_pseudo)} extra rows of pseudolabeled data added to training set for {fold_model.name}")
        X_fold = pd.concat([X_fold, X_pseudo], axis=0, ignore_index=True)
        y_fold = pd.concat([y_fold, y_pseudo], axis=0, ignore_index=True)
    try:
        fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **resources, **kwargs_fold)

        time_train_end_fold = time.time()
        fold_model.fit_time = time_train_end_fold - time_start_fold
        fold_model, pred_proba = _ray_predict_oof(
            fold_model=fold_model,
            X_val_fold=X_val_fold,
            y_val_fold=y_val_fold,
            num_cpus=resources["num_cpus"],
            save_bag_folds=save_bag_folds,
        )
        save_path = fold_model.save()
    except (AutoGluonException, ImportError, MemoryError) as e:
        e = encode_exception(e)
        return {
            "status": "expected_error",
            "error": e,
        }

    if model_sync_path is not None and not is_head_node:
        model_sync_path = model_sync_path + f"{fold_model.name}/"  # s3 path hence need "/" as the saperator
        bucket, prefix = s3_path_to_bucket_prefix(model_sync_path)
        upload_s3_folder(bucket=bucket, prefix=prefix, folder_to_upload=save_path, verbose=False)
    return fold_model.name, pred_proba, time_start_fold, time_train_end_fold, fold_model.predict_time, fold_model.predict_1_time, fold_model.predict_n_size, fold_model.fit_num_cpus, fold_model.fit_num_gpus


def _ray_predict_oof(fold_model: AbstractModel, X_val_fold: pd.DataFrame, y_val_fold: pd.Series, num_cpus: int = -1, save_bag_folds: bool = True) -> tuple[AbstractModel, ndarray]:
    y_pred_proba = fold_model.predict_proba(X_val_fold, record_time=True, num_cpus=num_cpus)
    fold_model.val_score = fold_model.score_with_y_pred_proba(y=y_val_fold, y_pred_proba=y_pred_proba)
    fold_model.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
    if not save_bag_folds:
        fold_model.model = None
    return fold_model, y_pred_proba


class ParallelFoldFittingStrategy(FoldFittingStrategy):
    """
    An implementation of FoldFittingStrategy to train multiple folds in parallel.
    Folds are spread to cpu/gpu cores by ray tasks.
    Large data are stored in ray object store, which minimizes memory usage and unessary serializations.
    Trained models are saved to disk within each ray task.

    Parameters
    ----------
        num_folds_parallel: int
            Number of folds to train in parallel at once.
            Consider lower this parameter if encounter out of memory issue.
        max_memory_usage_ratio: float, default=0.8
            The ratio of max memory usage for parallel folding.
            If the estimated usage exceeds this ratio, will fall back to sequential folding.
        model_sync_path: Optional[str], default=None
            The path to be used for workers to upload model artifacts and for headers to download
            Currently supports providing a s3 path.
            If None, model artifacts will be saved locally meaning no sync is required
    Attributes
    ----------
        num_cpus: int
            Number of cpu cores available.
        num_gpus: int
            Number of gpus available.
        num_parallel_jobs: int
            Number of parallel jobs to be executed once.
        max_memory_usage_ratio: float
            The ratio of max memory usage for parallel folding.
        time_start_fit: float
            The time of the first model starts training.
        time_end_fit: float
            The time of the last model finishes training.
        fit_time: float
            The amount of time used to fit all folds.
        predict_time: float
            The amount of time used to do out of folds predictions for all folds.
    """

    def __init__(self, *, num_jobs: int, num_folds_parallel: int, max_memory_usage_ratio: float = 0.8, model_sync_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.ray = try_import_ray()
        self.max_memory_usage_ratio = max_memory_usage_ratio
        self.model_sync_path = model_sync_path
        self.time_start_fit = None
        self.time_end_fit = None
        self.fit_time = 0
        self.predict_time = 0
        self.predict_1_time = None
        self.predict_n_size_lst = None
        self.fit_num_cpus = None
        self.fit_num_gpus = None
        # max_calls to guarantee release of gpu resource
        self._ray_fit = self.ray.remote(max_calls=1)(_ray_fit)
        self.mem_est_model = self._initialized_model_base.estimate_memory_usage(X=self.X)
        self.mem_est_data = self._estimate_data_memory_usage()
        self.mem_available = ResourceManager.get_available_virtual_mem()
        num_folds_parallel = self.folds_to_fit_in_parallel_with_mem(user_specified_num_folds_parallel=num_folds_parallel)
        self._pseudo_sequential: bool = num_folds_parallel == 1
        self.resources, self.resources_model, self.batches, self.num_parallel_jobs = self._get_resource_suggestions(
            num_jobs=num_jobs, user_specified_num_folds_parallel=num_folds_parallel, user_resources_per_job=self.user_resources_per_job
        )

    def mem_est_proportion_per_fold(self):
        return (self.mem_est_model + self.mem_est_data) / self.mem_available

    @disable_if_lite_mode(ret=1)
    def folds_to_fit_in_parallel_with_mem(self, user_specified_num_folds_parallel: int) -> int:
        """Check if the memory is sufficient to do parallel training"""
        mem_available = self.mem_available
        # Train 1 fold at least as the estimation might be off
        mem_est_total = self.mem_est_model + self.mem_est_data
        mem_proportion_per_fold = mem_est_total / mem_available

        model_max_memory_usage_ratio = self._initialized_model_base.params_aux.get("max_memory_usage_ratio", 1)
        max_memory_usage_ratio = self.max_memory_usage_ratio * model_max_memory_usage_ratio

        folds_to_train_with_mem_valid = mem_available / mem_est_total * max_memory_usage_ratio
        max_folds_to_train_with_mem = max(1, int(folds_to_train_with_mem_valid))
        if max_folds_to_train_with_mem == 1:
            self._initialized_model_base._validate_fit_memory_usage(approx_mem_size_req=mem_est_total, available_mem=mem_available)
        num_folds_parallel = user_specified_num_folds_parallel
        if max_folds_to_train_with_mem < user_specified_num_folds_parallel:
            # If memory is not sufficient to train num_folds_parallel, reduce to max power of 2 folds that's smaller than folds_can_be_fit_in_parallel.
            num_folds_parallel = int(math.pow(2, math.floor((math.log10(max_folds_to_train_with_mem) / math.log10(2)))))
            logger.log(
                30,
                f"\tMemory not enough to fit {user_specified_num_folds_parallel} folds in parallel. "
                f"Will train {num_folds_parallel} folds in parallel instead (Estimated {mem_proportion_per_fold*100:.2f}% memory usage per fold, "
                f"{num_folds_parallel*mem_proportion_per_fold*100:.2f}%/{max_memory_usage_ratio*100:.2f}% total).",
            )
        return num_folds_parallel

    def _estimate_data_memory_usage(self):
        X_mem = get_approximate_df_mem_usage(self.X).sum()
        y_mem = get_approximate_df_mem_usage(self.y.to_frame()).sum()
        return X_mem + y_mem

    def schedule_fold_model_fit(self, fold_ctx):
        self.jobs.append(fold_ctx)

    def _get_ray_init_args(self) -> Dict[str, Any]:
        """
        Get the arguments needed to init ray runtime.
        This could differ in different context, i.e. distributed vs local
        """
        return dict(address="auto", logging_level=logging.ERROR, log_to_driver=False)

    def _process_fold_results(self, finished, unfinished, fold_ctx):
        try:
            out = self.ray.get(finished)
            if isinstance(out, dict):
                # TODO: Improve the structure of this logic for better logging
                # TODO: Also do this for HPO w/ Ray
                assert "status" in out
                assert out["status"] == "expected_error"
                err_dict = out["error"]
                err = decode_exception(err_dict)
                raise err
            else:
                fold_model, pred_proba, time_start_fit, time_end_fit, predict_time, predict_1_time, predict_n_size, fit_num_cpus, fit_num_gpus = out
            assert fold_ctx is not None
            self._update_bagged_ensemble(
                fold_model=fold_model,
                pred_proba=pred_proba,
                time_start_fit=time_start_fit,
                time_end_fit=time_end_fit,
                predict_time=predict_time,
                predict_1_time=predict_1_time,
                predict_n_size=predict_n_size,
                fit_num_cpus=fit_num_cpus,
                fit_num_gpus=fit_num_gpus,
                fold_ctx=fold_ctx,
            )
            model_sync_path = None
            if self.model_sync_path is not None:
                model_sync_path: str = self.model_sync_path + fold_model
                if not model_sync_path.endswith("/"):
                    model_sync_path += "/"
            self.sync_model_artifact(local_path=os.path.join(self.bagged_ensemble_model.path, fold_model), model_sync_path=model_sync_path)
        except TimeLimitExceeded:
            # Terminate all ray tasks because a fold failed
            self.terminate_all_unfinished_tasks(unfinished)
            raise TimeLimitExceeded
        # NotEnoughMemoryError is an autogluon custom error,
        # it predict memory usage before hand
        # MemoryError is the actual python memory error if the process failed
        except (NotEnoughMemoryError, MemoryError):
            error_msg = "Consider decreasing folds trained in parallel by passing num_folds_parallel to ag_args_ensemble when calling `predictor.fit`."
            logger.warning(error_msg)
            # Terminate all ray tasks because a fold failed
            self.terminate_all_unfinished_tasks(unfinished)
            raise NotEnoughMemoryError
        except Exception as e:
            processed_exception = self._parse_ray_error(e)
            # Terminate all ray tasks because a fold failed
            self.terminate_all_unfinished_tasks(unfinished)
            raise processed_exception

    def _update_bagged_ensemble_times(self):
        self.fit_time = 0
        if self.time_start_fit and self.time_end_fit:
            self.fit_time = self.time_end_fit - self.time_start_fit
        self.bagged_ensemble_model._add_parallel_child_times(fit_time=self.fit_time, predict_time=self.predict_time, predict_1_time=self.predict_1_time)
        self.bagged_ensemble_model._add_predict_n_size(predict_n_size_lst=self.predict_n_size_lst)

    def _update_bagged_ensemble_child_resources(self):
        for child_num_cpus in self.fit_num_cpus:
            self.bagged_ensemble_model._add_child_num_cpus(num_cpus=child_num_cpus)
        for child_num_gpus in self.fit_num_gpus:
            self.bagged_ensemble_model._add_child_num_gpus(num_gpus=child_num_gpus)

    def _run_parallel(self, X, y, X_pseudo, y_pseudo, model_base_ref, time_limit_fold, head_node_id):
        job_refs = []
        job_fold_map = {}
        gpu_assignments = {}

        # spread the task
        for task_id, job in enumerate(self.jobs):
            fold_ctx = job
            ref = self._fit(
                model_base_ref=model_base_ref,
                X_ref=X,
                y_ref=y,
                X_pseudo_ref=X_pseudo,
                y_pseudo_ref=y_pseudo,
                time_limit_fold=time_limit_fold,
                task_id=task_id,
                fold_ctx=fold_ctx,
                gpu_assignments = gpu_assignments,
                resources=self.resources,
                resources_model=self.resources_model,
                head_node_id=head_node_id,
                kwargs=self.model_base_kwargs,
            )
            job_fold_map[ref] = fold_ctx
            job_refs.append(ref)

        # update ensemble whenever a model return
        unfinished = job_refs
        while unfinished:
            finished, unfinished = self.ray.wait(unfinished, num_returns=1)
            finished = finished[0]
            fold_ctx = job_fold_map.get(finished, None)
            self._process_fold_results(finished, unfinished, fold_ctx)

        self._update_bagged_ensemble_child_resources()
        self._update_bagged_ensemble_times()

    def _run_pseudo_sequential(self, X, y, X_pseudo, y_pseudo, model_base_ref, time_limit_fold, head_node_id):
        """
        A pseudo sequential runner using ray. The advantage of this is related to memory management in Python.
        As each fold is executed in its own subprocess, the memory state of the main process is clean and does
        not rely on the unreliable garbage collector of Python. In contrast to `SequentialLocalFoldFittingStrategy`,
        this fold fitting strategy will not leak memory across fits.

        Moreover, compared to just running the default `_run_parallel`, this code also has a lower worst case memory
        overhead. Here, at most, we have the overhead of one fold. In the case of `_run_parallel` the asynchronous
        processing of the fold results can result in having up to k-1 fold overhead at the same time. Furthermore,
        a job could start fitting a model while the results are processed; resulting in the fit running out of memory
        due to the overhead of processing and storing the result.
        """
        for job in self.jobs:
            fold_ctx = job
            ref = self._fit(
                model_base_ref=model_base_ref,
                X_ref=X,
                y_ref=y,
                X_pseudo_ref=X_pseudo,
                y_pseudo_ref=y_pseudo,
                time_limit_fold=time_limit_fold,
                fold_ctx=fold_ctx,
                resources=self.resources,
                resources_model=self.resources_model,
                head_node_id=head_node_id,
                kwargs=self.model_base_kwargs,
            )

            finished, unfinished = self.ray.wait([ref], num_returns=1)
            self._process_fold_results(finished[0], unfinished, fold_ctx)

        self._update_bagged_ensemble_times()

    def _calculate_gpu_assignment(self, gpu_assignments: Dict, task_id: int, gpus_per_task: int, total_gpus: int):
        if total_gpus == 0:
            logger.debug(f"No GPUs available, CPU-only mode for task {task_id}")
            gpu_assignments[task_id] = []
            return gpu_assignments   
        if gpus_per_task >= 1:
            gpu_id = task_id * gpus_per_task
            assigned_gpus = []
            for i in range(gpus_per_task):
                assigned_gpus.append(gpu_id + i % total_gpus)
            gpu_assignments[task_id] = assigned_gpus
        else:
            gpu_id = task_id % total_gpus
            gpu_assignments[task_id] = [gpu_id]
        return gpu_assignments

    def after_all_folds_scheduled(self):
        if not self.ray.is_initialized():
            ray_init_args = self._get_ray_init_args()
            self.ray.init(**ray_init_args)
        # See what the ray args are
        head_node_id = self.ray.get_runtime_context().get_node_id()
        logger.debug(f"Dispatching folds on node {head_node_id}")

        # prepare shared data
        X, y, X_pseudo, y_pseudo = self._prepare_data()
        model_base_ref = self.ray.put(self.model_base)
        time_limit_fold = self._get_fold_time_limit()

        if self._pseudo_sequential:
            logger.log(
                30,
                f"\t\tSwitching to pseudo sequential ParallelFoldFittingStrategy to avoid Python memory leakage.\n"
                f"\t\tOverrule this behavior by setting fold_fitting_strategy to 'sequential_local' in ag_args_ensemble when when calling `predictor.fit`",
            )
            self._run_pseudo_sequential(X, y, X_pseudo, y_pseudo, model_base_ref, time_limit_fold, head_node_id)
        else:
            self._run_parallel(X, y, X_pseudo, y_pseudo, model_base_ref, time_limit_fold, head_node_id)

    def terminate_all_unfinished_tasks(self, unfinished_tasks):
        # Cancel everyone else, forcefully, and drain to observe their cancellations
        for task in unfinished_tasks:
            try:
                self.ray.cancel(task, force=True)
            except Exception:
                pass

        for task in unfinished_tasks:
            try:
                _ = self.ray.get(task)
            except self.ray.exceptions.TaskCancelledError:
                pass
            except Exception:
                # If something else failed while we were cancelling, ignore here
                pass

    def _fit(
        self,
        *,
        model_base_ref,
        X_ref,
        y_ref,
        X_pseudo_ref,
        y_pseudo_ref,
        time_limit_fold: float,
        task_id: int,
        fold_ctx: dict,
        gpu_assignments: dict,
        resources: dict,
        head_node_id: str,
        kwargs: dict,
        resources_model: dict = None,
    ):
        if resources_model is None:
            resources_model = resources
        fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix, random_seed = self._get_fold_properties(fold_ctx)
        train_index, val_index = fold
        fold_ctx_ref = self.ray.put(fold_ctx)
        save_bag_folds = self.save_folds
        kwargs_fold = kwargs.copy()
        is_pseudo = X_pseudo_ref is not None and y_pseudo_ref is not None
        if self.sample_weight is not None:
            if is_pseudo:
                # TODO: Add support for sample_weight when pseudo is present
                raise Exception("Sample weights given, but not used due to pseudo labelled data being given.")
            else:
                kwargs_fold["sample_weight"] = self.sample_weight[train_index]
                kwargs_fold["sample_weight_val"] = self.sample_weight[val_index]
        if random_seed is not None:
            kwargs_fold["random_seed"] = random_seed
        pg = self.ray.util.get_current_placement_group()
        gpu_assignments = self._calculate_gpu_assignment(gpu_assignments=gpu_assignments, task_id=task_id, gpus_per_task=int(resources["num_gpus"]), total_gpus=self.num_gpus)
        return self._ray_fit.options(
            **resources, scheduling_strategy=self.ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(placement_group=pg)
        ).remote(
            model_base=model_base_ref,
            bagged_ensemble_model_path=self.bagged_ensemble_model.path,
            X=X_ref,
            y=y_ref,
            X_pseudo=X_pseudo_ref,
            y_pseudo=y_pseudo_ref,
            task_id=task_id,
            fold_ctx=fold_ctx_ref,
            assignments=gpu_assignments,
            time_limit_fold=time_limit_fold,
            save_bag_folds=save_bag_folds,
            resources=resources_model,
            kwargs_fold=kwargs_fold,
            head_node_id=head_node_id,
            model_sync_path=self.model_sync_path,
        )

    def _update_bagged_ensemble(self, fold_model, pred_proba, time_start_fit, time_end_fit, predict_time, predict_1_time, predict_n_size, fit_num_cpus, fit_num_gpus, fold_ctx):
        _, val_index = fold_ctx["fold"]
        self.models.append(fold_model)
        self.oof_pred_proba[val_index] += pred_proba
        self.oof_pred_model_repeats[val_index] += 1
        if self.time_start_fit:
            self.time_start_fit = min(time_start_fit, self.time_start_fit)
        else:
            self.time_start_fit = time_start_fit
        if self.time_end_fit:
            self.time_end_fit = max(time_end_fit, self.time_end_fit)
        else:
            self.time_end_fit = time_end_fit
        self.predict_time += predict_time
        if predict_1_time is not None:
            if self.predict_1_time is None:
                self.predict_1_time = 0
            self.predict_1_time += predict_1_time
        if self.predict_n_size_lst is None:
            self.predict_n_size_lst = []
        self.predict_n_size_lst.append(predict_n_size)
        if self.fit_num_cpus is None:
            self.fit_num_cpus = []
        self.fit_num_cpus.append(fit_num_cpus)
        if self.fit_num_gpus is None:
            self.fit_num_gpus = []
        self.fit_num_gpus.append(fit_num_gpus)

    def _get_fold_time_limit(self):
        time_elapsed = time.time() - self.time_start
        if self.time_limit is not None:
            time_left = self.time_limit - time_elapsed
            required_time_per_fold = time_left / self.batches
            time_limit_fold = required_time_per_fold * self.time_limit_fold_ratio
            if time_left <= 0:
                raise TimeLimitExceeded
        else:
            time_limit_fold = None
        return time_limit_fold

    def _get_resource_suggestions(self, num_jobs: int, user_specified_num_folds_parallel: int, user_resources_per_job: dict) -> Tuple[dict, dict, int, int]:
        """
        Get resources per job, number of total batches, and number of jobs running in parallel for a single batch
        based on total number of jobs, user specified number of jobs to be run in parallel, and user specified resources per job.
        When user specified resources per job, will validate and force this value if legit.
        Otherwise, will try to run as many jobs in parallel as possible respecting the minimum resources required per job.
        """
        user_specified_num_folds_parallel = min(num_jobs, user_specified_num_folds_parallel)
        model_min_resources = self._initialized_model_base.get_minimum_resources(is_gpu_available=(self.num_gpus > 0))
        resources_calculator = ResourceCalculatorFactory.get_resource_calculator(calculator_type="cpu" if self.num_gpus == 0 else "gpu")
        # use minimum resource to control number of jobs running in parallel
        min_cpu_per_job_based_on_num_folds_parallel = self.num_cpus // user_specified_num_folds_parallel
        min_gpu_per_job_based_on_num_folds_parallel = self.num_gpus / user_specified_num_folds_parallel
        min_cpu_based_on_model = model_min_resources.get("num_cpus", 1)
        min_gpu_based_on_model = model_min_resources.get("num_gpus", 0)

        get_resources_per_job_args = dict(
            total_num_cpus=self.num_cpus,
            total_num_gpus=self.num_gpus,
            num_jobs=num_jobs,
            minimum_cpu_per_job=max(min_cpu_per_job_based_on_num_folds_parallel, min_cpu_based_on_model),
            minimum_gpu_per_job=max(min_gpu_per_job_based_on_num_folds_parallel, min_gpu_based_on_model),
            user_resources_per_job=user_resources_per_job,
        )
        if user_resources_per_job is not None:
            get_resources_per_job_args["minimum_cpu_per_job"] = min_cpu_based_on_model
            get_resources_per_job_args["minimum_gpu_per_job"] = min_gpu_based_on_model

        resources_info = resources_calculator.get_resources_per_job(**get_resources_per_job_args)
        resources = resources_info.get("resources_per_job")
        if "num_gpus" not in resources:
            resources["num_gpus"] = 0
        num_parallel_jobs = resources_info.get("num_parallel_jobs")
        batches = resources_info.get("batches")

        # renname key to match ray job requirement
        resources["num_cpus"] = resources.pop("cpu")
        num_gpus = resources.pop("gpu", None)
        if num_gpus is not None and num_gpus > 0:
            resources["num_gpus"] = num_gpus

        num_cpus_model, num_gpus_model = self._initialized_model_base._get_default_resources()
        resources_model = dict(
            num_cpus=num_cpus_model,
            num_gpus=num_gpus_model,
        )

        if resources["num_cpus"] < resources_model["num_cpus"]:
            resources_model["num_cpus"] = resources["num_cpus"]
        if resources["num_gpus"] < resources_model["num_gpus"]:
            resources_model["num_gpus"] = resources["num_gpus"]
        if user_resources_per_job is not None:
            if "num_cpus" in user_resources_per_job:
                resources_model["num_cpus"] = resources["num_cpus"]
            if "num_gpus" in user_resources_per_job:
                resources_model["num_gpus"] = resources["num_gpus"]

        assert resources_model["num_cpus"] <= resources["num_cpus"]
        assert resources_model["num_gpus"] <= resources["num_gpus"]

        return resources, resources_model, batches, num_parallel_jobs

    def _prepare_data(self, in_mem=True):
        X_pseudo = None
        y_pseudo = None
        if in_mem:
            X = self.ray.put(self.X)
            y = self.ray.put(self.y)
            if self.X_pseudo is not None and self.y_pseudo is not None:
                X_pseudo = self.ray.put(self.X_pseudo)
                y_pseudo = self.ray.put(self.y_pseudo)
        else:
            X = "X.pkl"
            y = "y.pkl"
            utils = "utils"
            X = os.path.join(self.bagged_ensemble_model.path, utils, X)
            y = os.path.join(self.bagged_ensemble_model.path, utils, y)
            with open(X, "wb") as X_f, open(y, "wb") as y_f:
                pickle.dump(self.X, X_f)
                pickle.dump(self.y, y_f)
            if self.X_pseudo is not None and self.y_pseudo is not None:
                X_pseudo = "X_pseudo.pkl"
                y_pseudo = "y_pseudo.pkl"
                X_pseudo = os.path.join(self.bagged_ensemble_model.path, utils, X_pseudo)
                y_pseudo = os.path.join(self.bagged_ensemble_model.path, utils, y_pseudo)
        return X, y, X_pseudo, y_pseudo

    def _parse_ray_error(self, e):
        error = str(e).lower()
        if "cuda" in error and ("out of memory" in error or "alloc" in error):
            default_error_msg = (
                "If none working, use sequential folding by passing SequentialLocalFoldFittingStrategy to ag_args_ensemble "
                "when calling `predictor.fit` and try again."
            )
            # FIXME: Avoid hardcoding model names.
            if self.model_base.__class__.__name__ in [TEXT_MODEL, IMAGE_MODEL]:
                error_msg = (
                    f"Out of CUDA memory while training "
                    f"{self.model_base.__class__.__name__}. "
                    f"Consider decreasing batch size in hyperparameter and try again.\n"
                    f"Alternatively, decrease folds trained in parallel by passing num_folds_parallel "
                    f"to ag_args_ensemble when calling `predictor.fit` if you have multiple GPUs and try again"
                )
                logger.warning(error_msg)
            # FIXME: Avoid hardcoding model names.
            elif self.model_base.__class__.__name__ in [TABULAR_TORCH_MODEL, TABULAR_FASTAI_MODEL]:
                error_msg = (
                    f"Out of CUDA memory while training {self.model_base.__class__.__name__}. "
                    f"Consider decreasing batch size in hyperparameter and try again.\n"
                    f"Alternatively, decrease folds trained in parallel by passing num_folds_parallel "
                    f"to ag_args_ensemble when calling `predictor.fit` and try again"
                )
                logger.warning(error_msg)
            else:
                error_msg = (
                    f"Out of CUDA memory while training "
                    f"{self.model_base.__class__.__name__}. "
                    f"Consider decreasing folds trained in parallel by passing "
                    f"num_folds_parallel to ag_args_ensemble when calling `predictor.fit` "
                    f"and try again."
                )
                logger.warning(error_msg)
            logger.warning(default_error_msg)
            e = NotEnoughCudaMemoryError
        return e

    def sync_model_artifact(self, local_path: str, model_sync_path: str):
        """
        Sync model artifacts being uploaded to `model_sync_path` to `local_path`
        This method is expected to be called on the head node in the cluster to collect model artifacts after training

        Parameters
        ----------
        local_path: str
            local path to download artifacts
        model_sync_path: str
            remote path to download model artifacts from
        """
        self._sync_model_artifact(local_path=local_path, model_sync_path=model_sync_path)

    def _sync_model_artifact(self, **kwargs):
        pass


class ParallelLocalFoldFittingStrategy(ParallelFoldFittingStrategy):
    def _get_ray_init_args(self):
        ray_init_args = dict(log_to_driver=False, logging_level=logging.ERROR, num_cpus=self.num_cpus)
        if self.num_gpus > 0:
            ray_init_args["num_gpus"] = self.num_gpus
        return ray_init_args


class ParallelDistributedFoldFittingStrategy(ParallelFoldFittingStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Append bag model name in the path, only use when sync path is required.
        if not DistributedContext.is_shared_network_file_system():
            self.model_sync_path = self.model_sync_path + os.path.basename(os.path.normpath(self.bagged_ensemble_model.path)) + "/"

    def _sync_model_artifact(self, local_path, model_sync_path):
        if DistributedContext.is_shared_network_file_system():
            # Not need to sync model artifacts in a shared file system.
            return

        bucket, path = s3_path_to_bucket_prefix(model_sync_path)
        download_s3_folder(bucket=bucket, prefix=path, local_path=local_path, error_if_exists=False, verbose=False)


def _json_safe(x: Any) -> Any:
    try:
        json.dumps(x)  # fast path
        return x
    except Exception:
        return repr(x)


def encode_exception(e: BaseException) -> dict[str, Any]:
    return {
        "exc_type": e.__class__.__name__,
        "message": str(e),
        "args": [_json_safe(a) for a in getattr(e, "args", ())],
        "attrs": {k: _json_safe(v) for k, v in getattr(e, "__dict__", {}).items()},
        "remote_traceback": traceback.format_exc(),
    }


class UnknownRemoteException(RuntimeError):
    def __init__(self, exc_type: str, message: str):
        super().__init__(f"{exc_type}: {message}")
        self.exc_type = exc_type


EXPECTED_EXC_LST = [
    AutoGluonException,
    NoGPUError,
    NoValidFeatures,
    NoStackFeatures,
    NotValidStacker,
    InsufficientTime,
    NotEnoughCudaMemoryError,
    NotEnoughMemoryError,
    TimeLimitExceeded,
    MemoryError,
    ImportError,
]
EXPECTED_EXC_REGISTRY: Mapping[str, Type[BaseException]] = {
    err_cls.__name__: err_cls for err_cls in EXPECTED_EXC_LST
}


def decode_exception(payload: Dict[str, Any],
                     registry: Mapping[str, Type[BaseException]] = EXPECTED_EXC_REGISTRY
                     ) -> BaseException:
    name = payload.get("exc_type", "Exception")
    args = payload.get("args", [])
    attrs = payload.get("attrs", {}) or {}
    msg = payload.get("message", "")
    tb_str = payload.get("remote_traceback")

    cls = registry.get(name)
    if cls is None:
        # If it's not registered, wrap as UnknownRemoteException but keep context
        ex = UnknownRemoteException(name, msg)
        ex.remote_traceback = tb_str
        ex.remote_attrs = attrs
        return ex

    # Try normal construction with original args; fall back to message-only
    try:
        ex = cls(*args)
    except Exception:
        ex = cls(msg)

    # Restore extra attributes (best-effort)
    for k, v in attrs.items():
        try:
            setattr(ex, k, v)
        except Exception:
            pass

    # Attach remote traceback string for debugging
    try:
        setattr(ex, "remote_traceback", tb_str)
    except Exception:
        pass
    return ex
