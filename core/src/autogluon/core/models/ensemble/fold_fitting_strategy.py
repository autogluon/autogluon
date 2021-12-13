import copy
import logging
import os
import time
import pandas as pd
import pickle
import psutil
import math
from abc import abstractmethod

from numpy import ndarray
from pandas import DataFrame, Series

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage

from ...utils.exceptions import TimeLimitExceeded, NotEnoughMemoryError, NotEnoughCudaMemoryError
from ...utils import get_gpu_count
from ...utils.try_import import try_import_ray
from ...scheduler.resource import get_cpu_count

logger = logging.getLogger(__name__)

TEXT_MODEL = 'TextPredictorModel'
IMAGE_MODEL = 'ImagePredictorModel'
TABULAR_MXNET_MODEL = 'TabularNeuralNetModel'
TABULAR_FASTAI_MODEL = 'NNFastAiTabularModel'


class AbstractFoldFittingStrategy():

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


class LocalFoldFittingStrategy(AbstractFoldFittingStrategy):
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

    def __init__(self, model_base, model_base_kwargs, bagged_ensemble_model,
                 X: DataFrame, y: Series, X_pseudo: DataFrame, y_pseudo: Series,
                 sample_weight, time_limit: float, time_start: float,
                 models: list, oof_pred_proba: ndarray, oof_pred_model_repeats: ndarray,
                 save_folds: bool, time_limit_fold_ratio=0.8):
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

    def schedule_fold_model_fit(self, fold_ctx):
        raise NotImplementedError

    def after_all_folds_scheduled(self):
        raise NotImplementedError

    def _get_fold_time_limit(self, fold_ctx):
        _, folds_finished, folds_left, folds_to_fit, _, _ = self._get_fold_properties(fold_ctx)
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
        _, val_index = fold_ctx['fold']
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

    def _predict_oof(self, fold_model, fold_ctx):
        time_train_end_fold = time.time()
        fold, folds_finished, folds_left, \
            folds_to_fit, is_last_fold, model_name_suffix = self._get_fold_properties(fold_ctx)
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
        pred_proba = fold_model.predict_proba(X_val_fold)
        fold_model.predict_time = time.time() - time_train_end_fold
        fold_model.val_score = fold_model.score_with_y_pred_proba(y=y_val_fold,
                                                                  y_pred_proba=pred_proba)
        fold_model.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
        if not self.bagged_ensemble_model.params.get('save_bag_folds', True):
            fold_model.model = None
        return fold_model, pred_proba

    @staticmethod
    def _get_fold_properties(fold_ctx):
        fold, folds_finished, folds_left, \
            folds_to_fit, is_last_fold, model_name_suffix = [
                fold_ctx[f] for f in ['fold', 'folds_finished', 'folds_left', 'folds_to_fit', 'is_last_fold', 'model_name_suffix']]
        return fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix


class SequentialLocalFoldFittingStrategy(LocalFoldFittingStrategy):
    """
    This strategy fits the folds locally in a sequence.
    """
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
        fold, folds_finished, folds_left, folds_to_fit, is_last_fold, model_name_suffix = self._get_fold_properties(fold_ctx)
        train_index, val_index = fold
        X_fold, X_val_fold = self.X.iloc[train_index, :], self.X.iloc[val_index, :]
        y_fold, y_val_fold = self.y.iloc[train_index], self.y.iloc[val_index]
        fold_model = copy.deepcopy(model_base)
        fold_model.name = f'{fold_model.name}{model_name_suffix}'
        fold_model.set_contexts(self.bagged_ensemble_model.path + fold_model.name + os.path.sep)
        kwargs_fold = kwargs.copy()
        is_pseudo = self.X_pseudo is not None and self.y_pseudo is not None
        if self.sample_weight is not None:
            kwargs_fold['sample_weight'] = self.sample_weight[train_index]
            kwargs_fold['sample_weight_val'] = self.sample_weight[val_index]

            if is_pseudo:
                # TODO: Add support for sample_weight when pseudo is present
                raise Exception('Sample weights given, but not used due to pseudo labelled data being given.')
            else:
                kwargs_fold['sample_weight'] = self.sample_weight[train_index]
                kwargs_fold['sample_weight_val'] = self.sample_weight[val_index]

        if is_pseudo:
            logger.log(15, f'{len(self.X_pseudo)} extra rows of pseudolabeled data added to training set for {fold_model.name}')
            X_fold = pd.concat([X_fold, self.X_pseudo], axis=0, ignore_index=True)
            y_fold = pd.concat([y_fold, self.y_pseudo], axis=0, ignore_index=True)

        fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **kwargs_fold)
        fold_model.fit_time = time.time() - time_start_fold
        return fold_model


def _ray_fit(model_base, bagged_ensemble_model_path,
             X, y, X_pseudo, y_pseudo,
             fold_ctx, time_limit_fold, num_cpus,
             save_bag_folds, kwargs_fold):
    logger.log(10, 'ray worker training')
    time_start_fold = time.time()
    fold, folds_finished, folds_left, \
        folds_to_fit, is_last_fold, \
        model_name_suffix = LocalFoldFittingStrategy._get_fold_properties(fold_ctx)
    train_index, val_index = fold
    fold_model = copy.deepcopy(model_base)
    fold_model.name = f'{fold_model.name}{model_name_suffix}'
    fold_model.set_contexts(bagged_ensemble_model_path + fold_model.name + os.path.sep)
    if type(X) == str and type(y) == str:
        with open(X, 'rb') as X_f, open(y, 'rb') as y_f:
            X = pickle.load(X_f)
            y = pickle.load(y_f)
    is_pseudo = False
    if X_pseudo and y_pseudo:
        if type(X_pseudo) == str and type(y_pseudo) == str:
            with open(X_pseudo, 'rb') as X_pseudo_f, open(y_pseudo, 'rb') as y_pseudo_f:
                X_pseudo = pickle.load(X_pseudo_f)
                y_pseudo = pickle.load(y_pseudo_f)
        is_pseudo = True

    X_fold, X_val_fold = X.iloc[train_index, :], X.iloc[val_index, :]
    y_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    if is_pseudo:
            logger.log(15, f'{len(X_pseudo)} extra rows of pseudolabeled data added to training set for {fold_model.name}')
            X_fold = pd.concat([X_fold, X_pseudo], axis=0, ignore_index=True)
            y_fold = pd.concat([y_fold, y_pseudo], axis=0, ignore_index=True)
    fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold,
                   time_limit=time_limit_fold, num_cpus=num_cpus, **kwargs_fold)
    time_train_end_fold = time.time()
    fold_model.fit_time = time_train_end_fold - time_start_fold
    fold_model, pred_proba = _ray_predict_oof(fold_model, X_val_fold, y_val_fold,
                                              time_train_end_fold, num_cpus, save_bag_folds)
    fold_model.save()
    return fold_model.name, pred_proba, time_start_fold, \
        time_train_end_fold, fold_model.predict_time


def _ray_predict_oof(fold_model, X_val_fold, y_val_fold, time_train_end_fold,
                     num_cpus=-1, save_bag_folds=True):
    pred_proba = fold_model.predict_proba(X_val_fold, num_cpus=num_cpus)
    time_pred_end_fold = time.time()
    fold_model.predict_time = time_pred_end_fold - time_train_end_fold
    fold_model.val_score = fold_model.score_with_y_pred_proba(y=y_val_fold,
                                                              y_pred_proba=pred_proba)
    fold_model.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
    if not save_bag_folds:
        fold_model.model = None
    return fold_model, pred_proba


class ParallelLocalFoldFittingStrategy(LocalFoldFittingStrategy):
    """
    An implementation of LocalFoldFittingStrategy to train multiple folds in parallel.
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
    def __init__(self, num_folds_parallel: int, max_memory_usage_ratio=0.8, **kwargs):
        super().__init__(**kwargs)
        self.ray = try_import_ray()
        self.num_cpus = get_cpu_count()
        self.num_gpus = None
        self.num_parallel_jobs = num_folds_parallel
        self.max_memory_usage_ratio = min(max_memory_usage_ratio, 1.0)
        self.time_start_fit = None
        self.time_end_fit = None
        self.fit_time = 0
        self.predict_time = 0
        # max_calls to guarantee release of gpu resource
        self._ray_fit = self.ray.remote(max_calls=1)(_ray_fit)
        # initialize the model base to get necessary info for estimating memory usage
        self._initialized_model_base = copy.deepcopy(self.model_base)
        self._initialized_model_base.initialize(X=self.X, y=self.y, **self.model_base_kwargs)

    def is_mem_sufficient(self, num_jobs):
        '''Check if the memory is sufficient to do parallel training'''
        model_mem_est = self._initialized_model_base.estimate_memory_usage(X=self.X)
        _, _, num_parallel_jobs = self._get_resource_suggestions(num_jobs)
        total_model_mem_est = num_parallel_jobs * model_mem_est
        data_mem_est = self._estimate_data_memory_usage()
        total_data_mem_est = num_parallel_jobs * data_mem_est
        mem_available = psutil.virtual_memory().available
        return (mem_available * self.max_memory_usage_ratio) > (total_model_mem_est + total_data_mem_est)

    def _estimate_data_memory_usage(self):
        X_mem = get_approximate_df_mem_usage(self.X).sum()
        y_mem = get_approximate_df_mem_usage(self.y.to_frame()).sum()
        return X_mem + y_mem

    def _get_gpu_count(self):
        if not self.num_gpus:
            _user_gpu_count = self.model_base.get_params().get('hyperparameters', {}). \
                get('ag_args_fit', {}).get('num_gpus', 0)
            model_base_class_name = self.model_base.__class__.__name__
            # If user didn't specify num gpus and we are training text or image models,
            # use all gpus.
            if model_base_class_name in [TEXT_MODEL, IMAGE_MODEL] and _user_gpu_count == 0:
                _user_gpu_count = get_gpu_count()
            self.num_gpus = min(_user_gpu_count, get_gpu_count())

    def schedule_fold_model_fit(self, fold_ctx):
        self._get_gpu_count()
        if not self.ray.is_initialized():
            if self.num_gpus > 0:
                self.ray.init(log_to_driver=False, num_gpus=self.num_gpus)
            else:
                self.ray.init(log_to_driver=False)
        self.jobs.append(fold_ctx)

    def after_all_folds_scheduled(self):
        num_jobs = len(self.jobs)
        resources, _, _ = self._get_resource_suggestions(num_jobs)
        job_refs = []
        job_fold_map = {}
        # prepare shared data
        X, y, X_pseudo, y_pseudo = self._prepare_data()
        model_base_ref = self.ray.put(self.model_base)
        time_limit_fold = self._get_fold_time_limit(num_jobs)
        # spread the task
        for job in self.jobs:
            fold_ctx = job
            ref = self._fit(model_base_ref, X, y, X_pseudo, y_pseudo, time_limit_fold, fold_ctx, resources, self.model_base_kwargs)
            job_fold_map[ref] = fold_ctx
            job_refs.append(ref)

        # update ensemble whenver a model return
        unfinished = job_refs
        while unfinished:
            finished, unfinished = self.ray.wait(unfinished, num_returns=1)
            finished = finished[0]
            try:
                fold_model, pred_proba, time_start_fit, \
                    time_end_fit, predict_time = self.ray.get(finished)
                fold_ctx = job_fold_map.get(finished, None)
                assert fold_ctx is not None
                self._update_bagged_ensemble(fold_model, pred_proba, time_start_fit,
                                             time_end_fit, predict_time, fold_ctx)
            except TimeLimitExceeded:
                # Terminate all ray tasks because a fold failed
                self.ray.shutdown()
                raise TimeLimitExceeded
            # NotEnoughMemoryError is an autogluon custom error,
            # it predict memory usage before hand
            # MemoryError is the actual python memory error if the process failed
            except (NotEnoughMemoryError, MemoryError):
                error_msg = 'Consider decrease folds trained in parallel \
                             by passing num_fold_parallel to ag_args_ensemble \
                             when calling tabular.fit.\n\
                             If none working, use sequential folding by passing \
                             SequentialLocalFoldFittingStrategy to ag_args_ensemble \
                             when calling tabular.fit and try again.'
                logger.warning(error_msg)
                # Terminate all ray tasks because a fold failed
                self.ray.shutdown()
                raise NotEnoughMemoryError
            except Exception as e:
                processed_exception = self._parse_ray_error(e)
                # Terminate all ray tasks because a fold failed
                self.ray.shutdown()
                raise processed_exception
        self.fit_time = 0
        if self.time_start_fit and self.time_end_fit:
            self.fit_time = self.time_end_fit - self.time_start_fit
        self.bagged_ensemble_model._add_parallel_child_times(self.fit_time, self.predict_time)

    def _fit(self, model_base_ref, X_ref, y_ref, X_pseudo_ref, y_pseudo_ref, time_limit_fold, fold_ctx, resources, kwargs):
        fold, folds_finished, folds_left, \
            folds_to_fit, is_last_fold, \
            model_name_suffix = self._get_fold_properties(fold_ctx)
        train_index, val_index = fold
        fold_ctx_ref = self.ray.put(fold_ctx)
        save_bag_folds = self.bagged_ensemble_model.params.get('save_bag_folds', True)
        kwargs_fold = kwargs.copy()
        is_pseudo = X_pseudo_ref is not None and y_pseudo_ref is not None
        if self.sample_weight is not None:
            if is_pseudo:
                # TODO: Add support for sample_weight when pseudo is present
                raise Exception('Sample weights given, but not used due to pseudo labelled data being given.')
            else:
                kwargs_fold['sample_weight'] = self.sample_weight[train_index]
                kwargs_fold['sample_weight_val'] = self.sample_weight[val_index]
        num_cpus = resources.get('num_cpus', 0)
        return self._ray_fit.options(**resources) \
            .remote(model_base_ref, self.bagged_ensemble_model.path,
                    X_ref, y_ref, X_pseudo_ref, y_pseudo_ref, fold_ctx_ref, time_limit_fold, num_cpus,
                    save_bag_folds, kwargs_fold)

    def _update_bagged_ensemble(self, fold_model, pred_proba, time_start_fit,
                                time_end_fit, predict_time, fold_ctx):
        _, val_index = fold_ctx['fold']
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

    def _get_fold_time_limit(self, num_jobs):
        _, batches, _ = self._get_resource_suggestions(num_jobs)
        time_elapsed = time.time() - self.time_start
        if self.time_limit is not None:
            time_left = self.time_limit - time_elapsed
            required_time_per_fold = time_left / batches
            time_limit_fold = required_time_per_fold * self.time_limit_fold_ratio
            if time_left <= 0:
                raise TimeLimitExceeded
        else:
            time_limit_fold = None
        return time_limit_fold

    def _get_resource_suggestions(self, num_jobs):
        # TODO: We need to handle user provide custom num_cpus
        cpu_per_job = max(1, int(self.num_cpus // self.num_parallel_jobs))
        gpu_per_job = 0
        resources = dict(num_cpus=cpu_per_job)
        num_parallel_jobs = min(self.num_parallel_jobs, self.num_cpus // cpu_per_job)
        self.num_parallel_jobs = num_parallel_jobs
        batches = math.ceil(num_jobs / num_parallel_jobs)
        if self.num_gpus and self.num_gpus > 0:
            gpu_per_job = self.num_gpus / self.num_parallel_jobs
            # For Image and Text model,
            # we always guarantee a task has at least one full gpu to use
            # FIXME: Avoid hardcoding model names.
            if self.model_base.__class__.__name__ in [TEXT_MODEL, IMAGE_MODEL]:
                gpu_per_job = max(1, math.ceil(gpu_per_job))
            resources = dict(num_cpus=cpu_per_job, num_gpus=gpu_per_job)
        return resources, batches, num_parallel_jobs

    def _prepare_data(self, in_mem=True):
        X_pseudo = None
        y_pseudo = None
        if in_mem:
            X = self.ray.put(self.X)
            y = self.ray.put(self.y)
            if self.X_pseudo and self.y_pseudo:
                X_pseudo = self.ray.put(self.X_pseudo)
                y_pseudo = self.ray.put(self.y_pseudo)
        else:
            X = 'X.pkl'
            y = 'y.pkl'
            utils = 'utils'
            X = os.path.join(self.bagged_ensemble_model.path, utils, X)
            y = os.path.join(self.bagged_ensemble_model.path, utils, y)
            with open(X, 'wb') as X_f, open(y, 'wb') as y_f:
                pickle.dump(self.X, X_f)
                pickle.dump(self.y, y_f)
            if self.X_pseudo and self.y_pseudo:
                X_pseudo = 'X_pseudo.pkl'
                y_pseudo = 'y_pseudo.pkl'
                X_pseudo = os.path.join(self.bagged_ensemble_model.path, utils, X_pseudo)
                y_pseudo = os.path.join(self.bagged_ensemble_model.path, utils, y_pseudo)
        return X, y, X_pseudo, y_pseudo

    def _parse_ray_error(self, e):
        error = str(e).lower()
        if 'cuda' in error and ('out of memory' in error or 'alloc' in error):
            default_error_msg = 'If none working, use sequential folding by passing \
                         SequentialLocalFoldFittingStrategy to ag_args_ensemble \
                         when calling tabular.fit and try again.'
            # FIXME: Avoid hardcoding model names.
            if self.model_base.__class__.__name__ in [TEXT_MODEL, IMAGE_MODEL]:
                error_msg = f'Out of CUDA memory while training \
                            {self.model_base.__class__.__name__}. \
                            Consider decrease batch size in hyperparameter and try again.\n\
                            Or decrease folds trained in parallel by passing num_fold_parallel \
                            to ag_args_ensemble when calling tabular.fit if you have multiple \
                            gpus and try again'
                logger.warning(error_msg)
            # FIXME: Avoid hardcoding model names.
            elif self.model_base.__class__.__name__ in [TABULAR_MXNET_MODEL, TABULAR_FASTAI_MODEL]:
                error_msg = f'Out of CUDA memory while training \
                            {self.model_base.__class__.__name__}. \
                            Consider decrease batch size in hyperparameter and try again.\n\
                            Or decrease folds trained in parallel by passing num_fold_parallel \
                            to ag_args_ensemble when calling tabular.fit and try again'
                logger.warning(error_msg)
            else:
                error_msg = f'Out of CUDA memory while training \
                            {self.model_base.__class__.__name__}. \
                            Consider decrease folds trained in parallel by passing \
                            num_fold_parallel to ag_args_ensemble when calling tabular.fit \
                            and try again'
                logger.warning(error_msg)
            logger.warning(default_error_msg)
            e = NotEnoughCudaMemoryError
        return e
