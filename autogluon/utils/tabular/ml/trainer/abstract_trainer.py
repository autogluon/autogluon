import copy, time, traceback, logging
import os
from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from collections import defaultdict

from ..constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS
from ...utils.loaders import load_pkl
from ...utils.savers import save_pkl
from ...utils.exceptions import TimeLimitExceeded, NotEnoughMemoryError
from ..utils import get_pred_from_proba, dd_list, generate_train_test_split, combine_pred_and_true
from ..models.abstract.abstract_model import AbstractModel
from ...metrics import accuracy, log_loss, root_mean_squared_error, scorer_expects_y_pred
from ..models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from ..trainer.model_presets.presets import get_preset_stacker_model
from ..models.ensemble.stacker_ensemble_model import StackerEnsembleModel
from ..models.ensemble.weighted_ensemble_model import WeightedEnsembleModel
from ..trainer.model_presets.presets_distill import get_preset_models_distillation

logger = logging.getLogger(__name__)


# TODO: Try to optimize for log loss at level 0 for stacking, only optimize for objective func at later levels or in aux models. Might work better.
# FIXME: Below is major defect!
#  Weird interaction for metrics like AUC during bagging.
#  If kfold = 5, scores are 0.9, 0.85, 0.8, 0.75, and 0.7, the score is not 0.8! It is much lower because probs are combined together and AUC is recalculated
#  Do we want this to happen? Should we calculate score by 5 separate scores and then averaging instead?

# TODO: Add post-fit cleanup function which loads all models and saves them after removing unnecessary variables such as oof_pred_probas to optimize load times and space usage
#  Trainer will not be able to be fit further after this operation is done, but it will be able to predict.
# TODO: Dynamic model loading for ensemble models during prediction, only load more models if prediction is uncertain. This dynamically reduces inference time.
# TODO: Try midstack Semi-Supervised. Just take final models and re-train them, use bagged preds for SS rows. This would be very cheap and easy to try.
class AbstractTrainer:
    trainer_file_name = 'trainer.pkl'

    def __init__(self, path: str, problem_type: str, scheduler_options=None, objective_func=None, stopping_metric=None,
                 num_classes=None, low_memory=False, feature_types_metadata=None, kfolds=0, n_repeats=1,
                 stack_ensemble_levels=0, time_limit=None, save_data=False, verbosity=2):
        self.path = path
        self.problem_type = problem_type
        if feature_types_metadata is None:
            feature_types_metadata = {}
        self.feature_types_metadata = feature_types_metadata
        self.save_data = save_data
        self.verbosity = verbosity
        if objective_func is not None:
            self.objective_func = objective_func
        elif self.problem_type == BINARY:
            self.objective_func = accuracy
        elif self.problem_type == MULTICLASS:
            self.objective_func = accuracy
        else:
            self.objective_func = root_mean_squared_error

        # stopping_metric is used to early stop all models except for aux models.
        if stopping_metric is not None:
            self.stopping_metric = stopping_metric
        elif self.objective_func.name == 'roc_auc':
            self.stopping_metric = log_loss
        else:
            self.stopping_metric = self.objective_func

        self.objective_func_expects_y_pred = scorer_expects_y_pred(scorer=self.objective_func)
        logger.log(25, "AutoGluon will gauge predictive performance using evaluation metric: %s" % self.objective_func.name)
        if not self.objective_func_expects_y_pred:
            logger.log(25, "This metric expects predicted probabilities rather than predicted class labels, so you'll need to use predict_proba() instead of predict()")

        logger.log(20, "To change this, specify the eval_metric argument of fit()")
        logger.log(25, "AutoGluon will early stop models using evaluation metric: %s" % self.stopping_metric.name)  # TODO: stopping_metric is likely not used during HPO, fix this
        self.num_classes = num_classes
        self.feature_prune = False # will be set to True if feature-pruning is turned on.
        self.low_memory = low_memory
        self.bagged_mode = True if kfolds >= 2 else False
        if self.bagged_mode:
            self.kfolds = kfolds  # int number of folds to do model bagging, < 2 means disabled
            self.stack_ensemble_levels = stack_ensemble_levels
            self.stack_mode = True if self.stack_ensemble_levels >= 1 else False
            self.n_repeats = n_repeats
        else:
            self.kfolds = 0
            self.stack_ensemble_levels = 0
            self.stack_mode = False
            self.n_repeats = 1

        self.hyperparameters = {}  # TODO: This is currently required for fetching stacking layer models. Consider incorporating more elegantly

        # self.models_level_all['core'][0] # Includes base models
        # self.models_level_all['core'][1] # Stacker level 1
        # self.models_level_all['aux1'][1] # Stacker level 1 aux models, such as weighted_ensemble
        # self.models_level_all['core'][2] # Stacker level 2
        self.models_level = defaultdict(dd_list)
        self.models_level_hpo = defaultdict(dd_list)  # stores additional models produced during HPO

        self.model_best = None
        self.model_best_core = None

        self.model_performance = {}
        self.model_paths = {}
        self.model_types = {}  # Outer type, can be BaggedEnsemble, StackEnsemble (Type that is able to load the model)
        self.model_types_inner = {}  # Inner type, if Ensemble then it is the type of the inner model (May not be able to load with this type)
        self.model_fit_times = {}
        self.model_pred_times = {}
        self.models = {}
        self.reset_paths = False

        self.hpo_results = {}  # Stores summary of HPO process
        # Scheduler attributes:
        if scheduler_options is not None:
            self.scheduler_func = scheduler_options[0]  # unpack tuple
            self.scheduler_options = scheduler_options[1]
        else:
            self.scheduler_func = None
            self.scheduler_options = None

        self.time_limit = time_limit
        if self.time_limit is None:
            self.time_limit = 1e7
            self.ignore_time_limit = True
        else:
            self.ignore_time_limit = False
        self.time_train_start = None
        self.time_train_level_start = None
        self.time_limit_per_level = self.time_limit / (self.stack_ensemble_levels + 1)

        self.num_rows_train = None
        self.num_cols_train = None

        self.is_data_saved = False
        self.normalize_predprobs = False # whether or not probabilistic predictions may need to be renormalized (eg. distillation with BINARY -> REGRESSION)

    # path_root is the directory containing learner.pkl
    @property
    def path_root(self):
        return self.path.rsplit(os.path.sep, maxsplit=2)[0] + os.path.sep

    @property
    def path_utils(self):
        return self.path_root + 'utils' + os.path.sep

    @property
    def path_data(self):
        return self.path_utils + 'data' + os.path.sep

    def load_X_train(self):
        path = self.path_data + 'X_train.pkl'
        return load_pkl.load(path=path)

    def load_X_val(self):
        path = self.path_data + 'X_val.pkl'
        return load_pkl.load(path=path)

    def load_y_train(self):
        path = self.path_data + 'y_train.pkl'
        return load_pkl.load(path=path)

    def load_y_val(self):
        path = self.path_data + 'y_val.pkl'
        return load_pkl.load(path=path)

    def save_X_train(self, X, verbose=True):
        path = self.path_data + 'X_train.pkl'
        save_pkl.save(path=path, object=X, verbose=verbose)

    def save_X_val(self, X, verbose=True):
        path = self.path_data + 'X_val.pkl'
        save_pkl.save(path=path, object=X, verbose=verbose)

    def save_y_train(self, y, verbose=True):
        path = self.path_data + 'y_train.pkl'
        save_pkl.save(path=path, object=y, verbose=verbose)

    def save_y_val(self, y, verbose=True):
        path = self.path_data + 'y_val.pkl'
        save_pkl.save(path=path, object=y, verbose=verbose)

    def get_model_names_all(self):
        model_names = []
        for stack_name in self.models_level.keys():
            model_names += self.get_model_names(stack_name)
        return model_names

    def get_model_names(self, stack_name):
        model_names = []
        levels = np.sort(list(self.models_level[stack_name].keys()))
        for level in levels:
            model_names += self.models_level[stack_name][level]
        return model_names

    def get_max_level(self, stack_name: str):
        try:
            return np.sort(list(self.models_level[stack_name].keys()))[-1]
        except IndexError:
            return -1

    def get_max_level_all(self):
        max_level = 0
        for stack_name in self.models_level.keys():
            max_level = max(max_level, self.get_max_level(stack_name))
        return max_level

    def get_models(self, hyperparameters, hyperparameter_tune=False, **kwargs):
        raise NotImplementedError

    def get_model_level(self, model_name):
        for stack_name in self.models_level.keys():
            for level in self.models_level[stack_name].keys():
                if model_name in self.models_level[stack_name][level]:
                    return level
        raise ValueError('Model' + str(model_name) + 'does not exist in trainer.')

    def set_contexts(self, path_context):
        self.path, self.model_paths = self.create_contexts(path_context)

    def create_contexts(self, path_context):
        path = path_context
        model_paths = copy.deepcopy(self.model_paths)
        for model in self.model_paths:
            prev_path = self.model_paths[model]
            model_local_path = prev_path.split(self.path, 1)[1]
            new_path = path + model_local_path
            model_paths[model] = new_path

        return path, model_paths

    def train(self, X_train, y_train, X_test=None, y_test=None, hyperparameter_tune=True, feature_prune=False, holdout_frac=0.1, hyperparameters=None):
        raise NotImplementedError

    def train_single(self, X_train, y_train, X_test, y_test, model, kfolds=None, k_fold_start=0, k_fold_end=None, n_repeats=None, n_repeat_start=0, level=0, time_limit=None):
        if kfolds is None:
            kfolds = self.kfolds
        if n_repeats is None:
            n_repeats = self.n_repeats
        if model.feature_types_metadata is None:
            model.feature_types_metadata = self.feature_types_metadata  # TODO: move this into model creation process?
        model_fit_kwargs = {}
        if self.scheduler_options is not None:
            model_fit_kwargs = {'verbosity': self.verbosity,
                                'num_cpus': self.scheduler_options['resource']['num_cpus'],
                                'num_gpus': self.scheduler_options['resource']['num_gpus']}  # Additional configurations for model.fit
        if self.bagged_mode or isinstance(model, WeightedEnsembleModel):
            model.fit(X=X_train, y=y_train, k_fold=kfolds, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, compute_base_preds=False, time_limit=time_limit, **model_fit_kwargs)
        else:
            model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, time_limit=time_limit, **model_fit_kwargs)
        return model

    def train_and_save(self, X_train, y_train, X_test, y_test, model: AbstractModel, stack_name='core', kfolds=None, k_fold_start=0, k_fold_end=None, n_repeats=None, n_repeat_start=0, level=0, time_limit=None):
        fit_start_time = time.time()
        model_names_trained = []
        try:
            if time_limit is not None:
                if time_limit <= 0:
                    logging.log(15, 'Skipping ' + str(model.name) + ' due to lack of time remaining.')
                    return model_names_trained
                time_left_total = self.time_limit - (fit_start_time - self.time_train_start)
                logging.log(20, 'Fitting model: ' + str(model.name) + ' ...' + ' Training model for up to ' + str(round(time_limit, 2)) + 's of the ' + str(round(time_left_total, 2)) + 's of remaining time.')
            else:
                logging.log(20, 'Fitting model: ' + str(model.name) + ' ...')
            model = self.train_single(X_train, y_train, X_test, y_test, model, kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, level=level, time_limit=time_limit)
            fit_end_time = time.time()
            if isinstance(model, BaggedEnsembleModel):
                if model.bagged_mode or isinstance(model, WeightedEnsembleModel):
                    score = model.score_with_oof(y=y_train)
                else:
                    score = np.nan
            else:
                score = model.score(X=X_test, y=y_test)
            pred_end_time = time.time()
            if model.fit_time is None:
                model.fit_time = fit_end_time - fit_start_time
            if model.predict_time is None:
                if np.isnan(score):
                    model.predict_time = np.nan
                else:
                    model.predict_time = pred_end_time - fit_end_time
            self.save_model(model=model)
        except TimeLimitExceeded:
            logger.log(20, '\tTime limit exceeded... Skipping %s.' % model.name)
            # logger.log(20, '\tTime wasted: ' + str(time.time() - fit_start_time))
            del model
        except NotEnoughMemoryError:
            logger.warning('\tNot enough memory to train %s... Skipping this model.' % model.name)
            del model
        except Exception as err:
            if self.verbosity >= 1:
                traceback.print_tb(err.__traceback__)
            logger.exception('Warning: Exception caused %s to fail during training... Skipping this model.' % model.name)
            logger.log(20, err)
            del model
        else:
            self.add_model(model=model, stack_name=stack_name, level=level, score=score)
            model_names_trained.append(model.name)
            if self.low_memory:
                del model
        return model_names_trained

    def add_model(self, model: AbstractModel, stack_name: str, level: int, score):
        stack_loc = self.models_level[stack_name]  # TODO: Consider removing, have train_multi handle this
        self.model_performance[model.name] = score
        self.model_paths[model.name] = model.path
        self.model_types[model.name] = type(model)
        if isinstance(model, BaggedEnsembleModel):
            self.model_types_inner[model.name] = model._child_type
        else:
            self.model_types_inner[model.name] = type(model)
        if not np.isnan(score):
            logger.log(20, '\t' + str(round(score, 4)) + '\t = Validation ' + self.objective_func.name + ' score')
        if not np.isnan(model.fit_time):
            logger.log(20, '\t' + str(round(model.fit_time, 2)) + 's' + '\t = Training runtime')
        if not np.isnan(model.predict_time):
            logger.log(20, '\t' + str(round(model.predict_time, 2)) + 's' + '\t = Validation runtime')
        # TODO: Add to HPO
        self.model_fit_times[model.name] = model.fit_time
        self.model_pred_times[model.name] = model.predict_time
        if model.is_valid():
            if model.name not in stack_loc[level]:
                stack_loc[level].append(model.name)
            if self.model_best_core is None:
                self.model_best_core = model.name
            else:
                best_score = self.model_performance[self.model_best_core]
                cur_score = self.model_performance[model.name]
                if cur_score > best_score:
                    # new best core model
                    self.model_best_core = model.name
        if self.low_memory:
            del model

    def train_single_full(self, X_train, y_train, X_test, y_test, model: AbstractModel, feature_prune=False,
                          hyperparameter_tune=True, stack_name='core', kfolds=None, k_fold_start=0, k_fold_end=None, n_repeats=None, n_repeat_start=0, level=0, time_limit=None):
        if (n_repeat_start == 0) and (k_fold_start == 0):
            model.feature_types_metadata = self.feature_types_metadata  # TODO: Don't set feature_types_metadata here
        if feature_prune:
            if n_repeat_start != 0:
                raise ValueError('n_repeat_start must be 0 to feature_prune, value = ' + str(n_repeat_start))
            elif k_fold_start != 0:
                raise ValueError('k_fold_start must be 0 to feature_prune, value = ' + str(k_fold_start))
            self.autotune(X_train=X_train, X_holdout=X_test, y_train=y_train, y_holdout=y_test, model_base=model)  # TODO: Update to use CV instead of holdout
        if hyperparameter_tune:
            if self.scheduler_func is None or self.scheduler_options is None:
                raise ValueError("scheduler_options cannot be None when hyperparameter_tune = True")
            if n_repeat_start != 0:
                raise ValueError('n_repeat_start must be 0 to hyperparameter_tune, value = ' + str(n_repeat_start))
            elif k_fold_start != 0:
                raise ValueError('k_fold_start must be 0 to hyperparameter_tune, value = ' + str(k_fold_start))
            # hpo_models (dict): keys = model_names, values = model_paths
            try:  # TODO: Make exception handling more robust? Return successful HPO models?
                if isinstance(model, BaggedEnsembleModel):
                    hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X=X_train, y=y_train, k_fold=kfolds, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
                else:
                    if (X_test is None) or (y_test is None):
                        X_train, X_test, y_train, y_test = generate_train_test_split(X_train, y_train, problem_type=self.problem_type, test_size=0.2)  # TODO: Adjust test_size, perhaps user specified?
                    hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X_train=X_train, X_test=X_test,
                        Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
            except Exception as err:
                if self.verbosity >= 1:
                    traceback.print_tb(err.__traceback__)
                logger.exception('Warning: Exception caused ' + model.name + ' to fail during hyperparameter tuning... Skipping this model.')
                logger.debug(err)
                del model
                model_names_trained = []
            else:
                model_names_trained = list(sorted(hpo_models.keys()))
                self.models_level_hpo[stack_name][level] += model_names_trained
                self.model_paths.update(hpo_models)
                self.model_performance.update(hpo_model_performances)
                self.hpo_results[model.name] = hpo_results
                self.model_types.update({name: type(model) for name in model_names_trained})
                if isinstance(model, BaggedEnsembleModel):
                    self.model_types_inner.update({name: model._child_type for name in model_names_trained})
                else:
                    self.model_types_inner.update({name: type(model) for name in model_names_trained})
        else:
            model_names_trained = self.train_and_save(X_train, y_train, X_test, y_test, model, stack_name=stack_name, kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, level=level, time_limit=time_limit)
        self.save()
        return model_names_trained

    # TODO: How to deal with models that fail during this? They have trained valid models before, but should we still use those models or remove the entire model? Currently we still use models.
    # TODO: Time allowance can be made better by only using time taken during final model training and not during HPO and feature pruning.
    # TODO: Time allowance not accurate if running from fit_continue
    # Takes trained bagged ensemble models and fits additional k-fold bags.
    def train_multi_repeats(self, X_train, y_train, X_test, y_test, models, kfolds, n_repeats, n_repeat_start=1, stack_name='core', level=0, time_limit=None):
        models_valid = models
        models_valid_next = []
        repeats_completed = 0
        time_start = time.time()
        for n in range(n_repeat_start, n_repeats):
            if time_limit is not None:
                time_start_repeat = time.time()
                time_left = time_limit - (time_start_repeat - time_start)
                if n == n_repeat_start:
                    time_required = self.time_limit_per_level * 0.575  # Require slightly over 50% to be safe
                else:
                    time_required = (time_start_repeat - time_start) / repeats_completed * (0.575/0.425)
                if time_left < time_required:
                    logger.log(15, 'Not enough time left to finish repeated k-fold bagging, stopping early ...')
                    break
            logger.log(20, 'Repeating k-fold bagging: ' + str(n+1) + '/' + str(n_repeats))
            for i, model in enumerate(models_valid):
                if isinstance(model, str):
                    model = self.load_model(model)
                if time_limit is None:
                    time_left = None
                else:
                    time_start_model = time.time()
                    time_left = time_limit - (time_start_model - time_start)
                models_valid_next += self.train_single_full(X_train, y_train, X_test, y_test, model, hyperparameter_tune=False, feature_prune=False, stack_name=stack_name, kfolds=kfolds, k_fold_start=0, k_fold_end=None, n_repeats=n+1, n_repeat_start=n, level=level, time_limit=time_left)
            models_valid = copy.deepcopy(models_valid_next)
            models_valid_next = []
            repeats_completed += 1
        logger.log(20, 'Completed ' + str(n_repeat_start + repeats_completed) + '/' + str(n_repeats) + ' k-fold bagging repeats ...')
        return models_valid

    def train_multi_initial(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], kfolds, n_repeats, hyperparameter_tune=True, feature_prune=False, stack_name='core', level=0, time_limit=None):
        stack_loc = self.models_level[stack_name]

        model_names_trained = []
        model_names_trained_hpo = []
        models_valid = models
        if kfolds == 0:
            models_valid = self.train_multi_fold(X_train, y_train, X_test, y_test, models_valid, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, stack_name=stack_name,
                                                          kfolds=kfolds, level=level, time_limit=time_limit)
        else:
            k_fold_start = 0
            if hyperparameter_tune or feature_prune:
                time_start = time.time()
                models_valid = self.train_multi_fold(X_train, y_train, X_test, y_test, models_valid, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, stack_name=stack_name,
                                                     kfolds=kfolds, k_fold_start=0, k_fold_end=1, n_repeats=n_repeats, n_repeat_start=0, level=level, time_limit=time_limit)
                k_fold_start = 1
                if time_limit is not None:
                    time_limit = time_limit - (time.time() - time_start)

            models_valid = self.train_multi_fold(X_train, y_train, X_test, y_test, models_valid, hyperparameter_tune=False, feature_prune=False, stack_name=stack_name,
                                                 kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=kfolds, n_repeats=n_repeats, n_repeat_start=0, level=level, time_limit=time_limit)

        if hyperparameter_tune:
            model_names_trained_hpo += models_valid
        else:
            model_names_trained += models_valid

        stack_loc[level] += model_names_trained_hpo  # Update model list with (potentially empty) list of new models created during HPO
        model_names_trained += model_names_trained_hpo
        unique_names = []
        for item in stack_loc[level]:
            if item not in unique_names: unique_names.append(item)
        stack_loc[level] = unique_names  # make unique and preserve order
        return model_names_trained

    # TODO: Ban KNN from being a Stacker model outside of aux. Will need to ensemble select on all stack layers ensemble selector to make it work
    # TODO: Robert dataset, LightGBM is super good but RF and KNN take all the time away from it on 1h despite being much worse
    # TODO: Add time_limit_per_model
    def train_multi_fold(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False, stack_name='core', kfolds=None, k_fold_start=0, k_fold_end=None, n_repeats=None, n_repeat_start=0, level=0, time_limit=None):
        models_valid = []
        time_start = time.time()
        for i, model in enumerate(models):
            if isinstance(model, str):
                model = self.load_model(model)
            elif self.low_memory:
                model = copy.deepcopy(model)
            # TODO: Only update scores when finished, only update model as part of final models if finished!
            if time_limit is None:
                time_left = None
            else:
                time_start_model = time.time()
                time_left = time_limit - (time_start_model - time_start)
            model_name_trained_lst = self.train_single_full(X_train, y_train, X_test, y_test, model, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, stack_name=stack_name,
                                                            kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=k_fold_end,
                                                            n_repeats=n_repeats, n_repeat_start=n_repeat_start, level=level, time_limit=time_left)

            if self.low_memory:
                del model
            models_valid += model_name_trained_lst

        return models_valid

    def train_multi(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False, stack_name='core', kfolds=None, n_repeats=None, n_repeat_start=0, level=0, time_limit=None):
        if kfolds is None:
            kfolds = self.kfolds
        if n_repeats is None:
            n_repeats = self.n_repeats
        if (kfolds == 0) and (n_repeats != 1):
            raise ValueError('n_repeats must be 1 when kfolds is 0, values: (%s, %s)' % (n_repeats, kfolds))
        if time_limit is None:
            n_repeats_initial = n_repeats
        else:
            n_repeats_initial = 1
        if n_repeat_start == 0:
            time_start = time.time()
            model_names_trained = self.train_multi_initial(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, kfolds=kfolds, n_repeats=n_repeats_initial, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                                                           stack_name=stack_name, level=level, time_limit=time_limit)
            n_repeat_start = n_repeats_initial
            if time_limit is not None:
                time_limit = time_limit - (time.time() - time_start)
        else:
            model_names_trained = models
        if (n_repeats > 1) and self.bagged_mode and (n_repeat_start < n_repeats):
            model_names_trained = self.train_multi_repeats(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=model_names_trained,
                                                           kfolds=kfolds, n_repeats=n_repeats, n_repeat_start=n_repeat_start, stack_name=stack_name, level=level, time_limit=time_limit)
        return model_names_trained

    def train_multi_and_ensemble(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False):
        if self.save_data and not self.is_data_saved:
            self.save_X_train(X_train)
            self.save_y_train(y_train)
            if X_test is not None:
                self.save_X_val(X_test)
                if y_test is not None:
                    self.save_y_val(y_test)
            self.is_data_saved = True

        self.num_rows_train = len(X_train)
        if X_test is not None:
            self.num_rows_train += len(X_test)
        self.num_cols_train = len(list(X_train.columns))
        self.time_train_start = time.time()
        self.train_multi_levels(X_train, y_train, X_test, y_test, models=models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, level_start=0, level_end=self.stack_ensemble_levels)
        if len(self.get_model_names_all()) == 0:
            raise ValueError('AutoGluon did not successfully train any models')

    def train_multi_levels(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False, level_start=0, level_end=0):
        for level in range(max(0, level_start), level_end + 1):
            self.time_train_level_start = time.time()
            self.time_limit_per_level = (self.time_limit - (self.time_train_level_start - self.time_train_start)) / (level_end + 1 - level)
            if self.ignore_time_limit:
                time_limit_core = None
                time_limit_aux = None
            else:
                time_limit_core = self.time_limit_per_level
                time_limit_aux = max(self.time_limit_per_level * 0.1, min(self.time_limit, 360))  # Allows aux to go over time_limit, but only by a small amount
            if level == 0:
                self.stack_new_level(X=X_train, y=y_train, X_test=X_test, y_test=y_test, models=models, level=level, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, time_limit_core=time_limit_core, time_limit_aux=time_limit_aux)
            else:
                self.stack_new_level(X=X_train, y=y_train, X_test=X_test, y_test=y_test, level=level, time_limit_core=time_limit_core, time_limit_aux=time_limit_aux)

        self.save()

    def stack_new_level(self, X, y, X_test=None, y_test=None, level=0, models=None, hyperparameter_tune=False, feature_prune=False, time_limit_core=None, time_limit_aux=None):
        self.stack_new_level_core(X=X, y=y, X_test=X_test, y_test=y_test, models=models, level=level, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, time_limit=time_limit_core)
        if self.bagged_mode:
            self.stack_new_level_aux(X=X, y=y, level=level+1, time_limit=time_limit_aux)
        else:
            self.stack_new_level_aux(X=X_test, y=y_test, fit=False, level=level+1, time_limit=time_limit_aux)

    def stack_new_level_core(self, X, y, X_test=None, y_test=None, models=None, level=1, stack_name='core', kfolds=None, n_repeats=None, hyperparameter_tune=False, feature_prune=False, time_limit=None):
        use_orig_features = True
        if models is None:
            models = self.get_models(self.hyperparameters, level=level)
        if kfolds is None:
            kfolds = self.kfolds
        if n_repeats is None:
            n_repeats = self.n_repeats

        if self.bagged_mode:
            if level == 0:
                (base_model_names, base_model_paths, base_model_types) = ([], {}, {})
            elif level > 0:
                base_model_names, base_model_paths, base_model_types = self.get_models_info(model_names=self.models_level['core'][level - 1])
                if len(base_model_names) == 0:
                    logger.log(20, 'No base models to train on, skipping stack level...')
                    return
            else:
                raise AssertionError('Stack level cannot be negative! level = %s' % level)
            models = [
                StackerEnsembleModel(path=self.path, name=model.name + '_STACKER_l' + str(level), model_base=model, base_model_names=base_model_names,
                                     base_model_paths_dict=base_model_paths, base_model_types_dict=base_model_types, use_orig_features=use_orig_features,
                                     num_classes=self.num_classes, random_state=level)
                for model in models]
        X_train_init = self.get_inputs_to_stacker(X, level_start=0, level_end=level, fit=True)
        if X_test is not None:
            X_test = self.get_inputs_to_stacker(X_test, level_start=0, level_end=level, fit=False)

        return self.train_multi(X_train=X_train_init, y_train=y, X_test=X_test, y_test=y_test, models=models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, level=level, stack_name=stack_name, kfolds=kfolds, n_repeats=n_repeats, time_limit=time_limit)

    def stack_new_level_aux(self, X, y, level, fit=True, time_limit=None):
        stack_name = 'aux1'
        X_train_stack_preds = self.get_inputs_to_stacker(X, level_start=0, level_end=level, fit=fit)
        self.generate_weighted_ensemble(X=X_train_stack_preds, y=y, level=level, kfolds=0, n_repeats=1, stack_name=stack_name, time_limit=time_limit)

    def generate_weighted_ensemble(self, X, y, level, kfolds=0, n_repeats=1, stack_name=None, hyperparameters=None, time_limit=None, name_suffix=''):
        if len(self.models_level['core'][level-1]) == 0:
            logger.log(20, 'No base models to train on, skipping weighted ensemble...')
            return
        weighted_ensemble_model = WeightedEnsembleModel(path=self.path, name='weighted_ensemble_' + name_suffix + 'k' + str(kfolds) + '_l' + str(level), base_model_names=self.models_level['core'][level-1],
                                                        base_model_paths_dict=self.model_paths, base_model_types_dict=self.model_types, base_model_types_inner_dict=self.model_types_inner, base_model_performances_dict=self.model_performance, hyperparameters=hyperparameters,
                                                        objective_func=self.objective_func, num_classes=self.num_classes, random_state=level)

        self.train_multi(X_train=X, y_train=y, X_test=None, y_test=None, models=[weighted_ensemble_model], kfolds=kfolds, n_repeats=n_repeats, hyperparameter_tune=False, feature_prune=False, stack_name=stack_name, level=level, time_limit=time_limit)
        if weighted_ensemble_model.name in self.get_model_names_all():
            if self.model_best is None:
                self.model_best = weighted_ensemble_model.name
            else:
                best_score = self.model_performance[self.model_best]
                cur_score = self.model_performance[weighted_ensemble_model.name]
                if cur_score > best_score:
                    # new best model
                    self.model_best = weighted_ensemble_model.name
        return weighted_ensemble_model.name

    def generate_stack_log_reg(self, X, y, level, kfolds=0, stack_name=None):
        base_model_names, base_model_paths, base_model_types = self.get_models_info(model_names=self.models_level['core'][level-1])
        stacker_model_lr = get_preset_stacker_model(path=self.path, problem_type=self.problem_type, objective_func=self.objective_func, num_classes=self.num_classes)
        name_new = stacker_model_lr.name + '_STACKER_k' + str(kfolds) + '_l' + str(level)

        stacker_model_lr = StackerEnsembleModel(path=self.path, name=name_new, model_base=stacker_model_lr, base_model_names=base_model_names, base_model_paths_dict=base_model_paths, base_model_types_dict=base_model_types,
                                                use_orig_features=False,
                                                num_classes=self.num_classes, random_state=level)

        return self.train_multi(X_train=X, y_train=y, X_test=None, y_test=None, models=[stacker_model_lr], hyperparameter_tune=False, feature_prune=False, stack_name=stack_name, kfolds=kfolds, level=level)

    def predict(self, X, model=None):
        if model is not None:
            return self.predict_model(X, model)
        elif self.model_best is not None:
            return self.predict_model(X, self.model_best)
        elif self.model_best_core is not None:
            return self.predict_model(X, self.model_best_core)
        else:
            raise Exception('Trainer has no fit models to predict with.')

    def predict_proba(self, X, model=None):
        if model is not None:
            return self.predict_proba_model(X, model)
        elif self.model_best is not None:
            return self.predict_proba_model(X, self.model_best)
        elif self.model_best_core is not None:
            return self.predict_proba_model(X, self.model_best_core)
        else:
            raise Exception('Trainer has no fit models to predict with.')

    def predict_model(self, X, model, level_start=0):
        if isinstance(model, str):
            model = self.load_model(model)
        X = self.get_inputs_to_model(model=model, X=X, level_start=level_start, fit=False)
        return model.predict(X=X, preprocess=False)

    def predict_proba_model(self, X, model, level_start=0):
        if isinstance(model, str):
            model = self.load_model(model)
        X = self.get_inputs_to_model(model=model, X=X, level_start=level_start, fit=False)
        EPS = 1e-8 # predicted probabilities can be at most this confident if we normalize predicted probabilities
        if not self.normalize_predprobs:
            return model.predict_proba(X=X, preprocess=False)
        elif self.problem_type == MULTICLASS:
           y_predproba = model.predict_proba(X=X, preprocess=False)
           most_negative_rowvals = np.clip(np.min(y_predproba, axis=1), a_min=None, a_max=0)
           y_predproba = y_predproba - most_negative_rowvals[:,None] # ensure nonnegative rows
           y_predproba = np.clip(y_predproba, a_min = EPS, a_max = None) # ensure no zeros
           return y_predproba / y_predproba.sum(axis=1, keepdims=1) # renormalize
        elif self.problem_type == BINARY:
            y_predproba = model.predict_proba(X=X, preprocess=False)
            min_y = np.min(y_predproba)
            max_y = np.max(y_predproba)
            if min_y < EPS or max_y > 1-EPS: # remap predicted probs to line that goes through: (min_y, EPS), (max_y, 1-EPS)
                y_predproba =  EPS + ((1-2*EPS)/(max_y-min_y)) * (y_predproba - min_y)
            return y_predproba
        return model.predict_proba(X=X, preprocess=False)

    def get_inputs_to_model(self, model, X, level_start, fit=False, preprocess=True):
        if isinstance(model, str):
            model = self.load_model(model)
        model_level = self.get_model_level(model.name)
        if model_level >= 1:
            X = self.get_inputs_to_stacker(X=X, level_start=level_start, level_end=model_level-1, fit=fit)
            X = model.preprocess(X, fit=fit, preprocess=preprocess)
        else:
            if preprocess:
                X = model.preprocess(X)
        return X

    def score(self, X, y, model=None):
        if self.objective_func_expects_y_pred:
            y_pred_ensemble = self.predict(X=X, model=model)
            return self.objective_func(y, y_pred_ensemble)
        else:
            y_pred_proba_ensemble = self.predict_proba(X=X, model=model)
            return self.objective_func(y, y_pred_proba_ensemble)

    def score_with_y_pred_proba(self, y, y_pred_proba):
        if self.objective_func_expects_y_pred:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            return self.objective_func(y, y_pred)
        else:
            return self.objective_func(y, y_pred_proba)

    def autotune(self, X_train, X_holdout, y_train, y_holdout, model_base: AbstractModel):
        model_base.feature_prune(X_train, X_holdout, y_train, y_holdout)

    def pred_proba_predictions(self, models, X_test):
        preds = []
        for model in models:
            if isinstance(model, str):
                model = self.load_model(model)
            model_pred = model.predict_proba(X_test)
            preds.append(model_pred)
        return preds

    def get_inputs_to_stacker(self, X, level_start, level_end, y_pred_probas=None, fit=False):
        if level_start > level_end:
            raise AssertionError('level_start cannot be greater than level end:' + str(level_start) + ', ' + str(level_end))
        if (level_start == 0) and (level_end == 0):
            return X
        if fit:
            if level_start >= 1:
                dummy_stacker_start = self._get_dummy_stacker(level=level_start, use_orig_features=True)
                cols_to_drop = dummy_stacker_start.stack_columns
                X = X.drop(cols_to_drop, axis=1)
            dummy_stacker = self._get_dummy_stacker(level=level_end, use_orig_features=True)
            X = dummy_stacker.preprocess(X=X, preprocess=False, fit=True, compute_base_preds=True)
        elif y_pred_probas is not None:
            dummy_stacker = self._get_dummy_stacker(level=level_end, use_orig_features=True)
            X_stacker = dummy_stacker.pred_probas_to_df(pred_proba=y_pred_probas)
            if dummy_stacker.use_orig_features:
                if level_start >= 1:
                    dummy_stacker_start = self._get_dummy_stacker(level=level_start, use_orig_features=True)
                    cols_to_drop = dummy_stacker_start.stack_columns
                    X = X.drop(cols_to_drop, axis=1)
                X = pd.concat([X_stacker, X], axis=1)
            else:
                X = X_stacker
        else:
            dummy_stackers = {}
            for level in range(level_start, level_end+1):
                if level >= 1:
                    dummy_stackers[level] = self._get_dummy_stacker(level=level, use_orig_features=True)
            for level in range(level_start, level_end):
                if level >= 1:
                    cols_to_drop = dummy_stackers[level].stack_columns
                else:
                    cols_to_drop = []
                X = dummy_stackers[level+1].preprocess(X=X, preprocess=False, fit=False, compute_base_preds=True)
                if len(cols_to_drop) > 0:
                    X = X.drop(cols_to_drop, axis=1)
        return X

    # TODO: add compress support for non-bagged models
    def compress(self, X=None, y=None, models=None):
        if X is None:
            X = self.load_X_train()
        if y is None:
            y = self.load_y_train()
        if models is None:
            models = self.get_model_names_all()

        models_compressed = {}
        model_levels = defaultdict(dd_list)
        ignore_models = []
        ignore_stack_names = ['compressed']
        for stack_name in ignore_stack_names:
            ignore_models += self.get_model_names(stack_name)  # get_model_names returns [] if stack_name does not exist
        for model_name in models:
            model = self.load_model(model_name)
            if isinstance(model, WeightedEnsembleModel) or model_name in ignore_models:
                continue
            model_level = self.get_model_level(model_name)
            model_levels['compressed'][model_level] += [model_name]
            model_compressed = model.convert_to_compressed_template()
            models_compressed[model_name] = model_compressed
        levels = sorted(model_levels['compressed'].keys())
        models_trained_full = []
        for level in levels:
            models_level = model_levels['compressed'][level]
            models_level = [models_compressed[model_name] for model_name in models_level]
            models_trained = self.stack_new_level_core(X=X, y=y, models=models_level, level=level, stack_name='compressed', hyperparameter_tune=False, feature_prune=False, kfolds=0, n_repeats=1)
            models_trained_full += models_trained
        return models_trained_full

    def distill(self, X=None, y=None):
        if X is None:
            X = self.load_X_train()
        if y is None:
            y = self.load_y_train()

        model_best = self.load_model(self.model_best)
        models_distill = get_preset_models_distillation(path=self.path, problem_type=self.problem_type,
                                                        objective_func=self.objective_func,
                                                        stopping_metric=self.stopping_metric,
                                                        num_classes=self.num_classes,
                                                        hyperparameters=self.hyperparameters)
        if not self.bagged_mode:
            raise NotImplementedError
        if self.problem_type == MULTICLASS:
            distillation_type = 'pure' # 'mixed' 'oof' 'pure' # TODO: remove, for prototyping only
            if distillation_type == 'oof': # OOF distillation
                print("using ensemble OOF preds for distillation")
                y_distill = pd.DataFrame(model_best.oof_pred_proba)
            elif distillation_type == 'pure': # Try and replicate ensemble exactly
                print("using augmentation+pure ensemble in-fold preds for distillation")
                X = self.augment_data_preserve_joint(X)
                y_distill = pd.DataFrame(model_best.predict_proba(X))
            else:
                print("using mixed ensemble In-Fold preds + true_y for distillation")
                y_distill = model_best.predict_proba(X)
                y_distill = pd.DataFrame(combine_pred_and_true(y_distill, y, upweight_factor = 0.01))
            og_bagged_mode = self.bagged_mode
            og_verbosity = self.verbosity
            self.bagged_mode = False # turn off bagging
            self.verbosity = 4
            X_train, X_test, y_train, y_test = generate_train_test_split(X, y_distill, problem_type=SOFTCLASS, test_size=0.1)
            for model in models_distill:
                model_distill = self.train_single_full(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model=model,
                                                           hyperparameter_tune=False, stack_name='distill')
            self.bagged_mode = og_bagged_mode
            self.verbosity = og_verbosity
        else: # Binary/regression:
            y_distill = pd.Series(model_best.oof_pred_proba)
            # X_train, X_test, y_train, y_test = generate_train_test_split(X, y_distill, problem_type=REGRESSION, test_size=0.1)  # TODO: Do stratified for binary/multiclass!
            # self.bagged_mode = False
            # self.stack_new_level_core(X=X_train, y=y_train, X_test=X_test, y_test=y_test, models=models_distill, level=0, stack_name='distill', hyperparameter_tune=False, feature_prune=False)
            # self.bagged_mode = True
            self.stack_new_level_core(X=X, y=y_distill, models=models_distill, level=0,
                                      stack_name='distilled', hyperparameter_tune=False,
                                      feature_prune=False)
            # TODO: Do stratified for binary/multiclass, folds are not aligned!
            models_trained = self.stack_new_level_core(X=X, y=y_distill, models=models_distill, level=0, stack_name='distilled', hyperparameter_tune=False, feature_prune=False)
            self.compress(X=X, y=y_distill, models=models_trained)

        self.save()


    def augment_distill(self, X=None, y=None, num_augmented_samples=50000, time_limits=None):
        """ Distillation of best model into single models.
            num_augmented_samples: higher values will take longer, but likely improved distillation performance.
            time_limits: only controls time-limit for each model, not overall time-limit (to ensure every model type gets fair chance)
        """
        EPS_bin2regress = 0.01 # truncate predicted probabilities to [EPS, 1-EPS] when converting binary problems -> regression
        og_bagged_mode = self.bagged_mode
        og_verbosity = self.verbosity
        self.bagged_mode = False # turn off bagging
        self.verbosity = 4 # high verbosity for debugging  # TODO: remove
        if X is None:
            X = self.load_X_train()
        if y is None:
            y = self.load_y_train() # TODO: doesn't appear anywhere?

        X_aug = self.augment_data_preserve_joint(X, num_augmented_samples)
        y_aug = self.predict_proba(X_aug)
        X_train, X_test, y_train, y_test = generate_train_test_split(X, y, problem_type=self.problem_type, test_size=0.2)
        if self.problem_type == MULTICLASS:
            y_aug = pd.DataFrame(y_aug)
            # y_train = convertToOneHot?? # TODO
            # y_test = convertToOneHot?? # TODO
            self.normalize_predprobs = True
        else:
            y_aug = pd.Series(y_aug)
            if self.problem_type == BINARY:
                min_pred = 0.0
                max_pred = 1.0
                y_train = EPS_bin2regress + ((1-2*EPS_bin2regress)/(max_pred-min_pred)) * (y_train - min_pred)
                y_test = EPS_bin2regress + ((1-2*EPS_bin2regress)/(max_pred-min_pred)) * (y_test - min_pred)
                y_aug = EPS_bin2regress + ((1-2*EPS_bin2regress)/(max_pred-min_pred)) * (y_aug - min_pred)
                self.normalize_predprobs = True

        X_train = pd.concat([X_train, X_aug])
        X_train.reset_index(drop=True, inplace=True)
        y_train = pd.concat([y_train, y_aug])
        y_train.reset_index(drop=True, inplace=True)
        models_distill = get_preset_models_distillation(path=self.path, problem_type=self.problem_type,
                                                        objective_func=self.objective_func, stopping_metric=self.stopping_metric,
                                                        num_classes=self.num_classes, hyperparameters=self.hyperparameters)
        for model in models_distill:
            print("Distilling with model: %s ..." % str(model.name))
            model = self.train_single_full(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model=model,
                                                   hyperparameter_tune=False, stack_name='distill', time_limit=time_limits)
            trainer.model_performance[model.name] = self.score(X_test, y_test, model=model.name) # measure original metric on validation data
        # reset trainer to old state:
        self.bagged_mode = og_bagged_mode
        self.verbosity = og_verbosity
        self.save()


    # TODO: experimental code.
    def augment_data_preserve_joint(self, X, num_augmented_samples = 10000, frac_feature_perturb=0.1, continuous_feature_noise = 0.05):
        """ Generates synthetic datapoints for learning to mimic teacher model in distillation.
            num_augmented_samples: number of additional augmented data points to return
            frac_feature_perturb: fraction of features perturbed in each data point. Set smaller to ensure augmented samples remain closer to real data.
            continuous_feature_noise: we noise numeric features by this factor times their std-dev.
            These data are NOT marginally sampled, rather we replace randomly selected subset of features for each datapoint. Larger subset -> augmented data is more different than original.
        """
        if len(X) >= num_augmented_samples:
            print("No data augmentation performed since training data is large enough.")
            return X
        if frac_feature_perturb > 1.0:
            raise ValueError("frac_feature_perturb must be <= 1")
        print("Augmenting training data with synthetic samples for distillation...")
        num_feature_perturb = max(1, int(frac_feature_perturb*len(X.columns)))
        num_augmented_samples = num_augmented_samples - len(X)
        X_aug = pd.concat([X.iloc[[0]]]*num_augmented_samples)
        X_aug.reset_index(drop=True, inplace=True)
        continuous_types = ['float','int', 'datetime']
        continuous_featnames = [] # these features will have shuffled values with added noise
        for contype in continuous_types:
            if contype in self.feature_types_metadata:
                continuous_featnames += self.feature_types_metadata[contype]

        for i in range(num_augmented_samples): # hot-deck sample some features per datapoint
            og_ind = i % len(X)
            augdata_i = X.iloc[og_ind].copy()
            cols_toperturb = np.random.choice(list(X.columns), size=num_feature_perturb, replace=False)
            for feature in cols_toperturb:
                feature_data = X[feature]
                augdata_i[feature] = feature_data.sample(n=1).values[0]
            X_aug.iloc[i] = augdata_i

        for feature in X.columns:
            if feature in continuous_featnames:
                feature_data = X[feature]
                aug_data = X_aug[feature]
                noise = np.random.normal(scale=np.std(feature_data)*continuous_feature_noise, size=num_augmented_samples)
                aug_data = aug_data + noise
                X_aug[feature] = pd.Series(aug_data, index=X_aug.index)

        X_aug.drop_duplicates(keep='first', inplace=True)
        print("Augmented training dataset has %s datapoints" % X_aug.shape[0])
        return X_aug

    # TODO: experimental code.
    def augment_data_hotdeck(self, X, num_augmented_samples = 50000, continuous_feature_noise = 0.1):
        """ Generates synthetic datapoints for learning to mimic teacher model in distillation.
            num_augmented_samples: number of total augmented data points to return (we add extra points to training set until this number is reached).
            continuous_feature_noise: we noise numeric features by this factor times their std-dev.
            These data are independent samples from the marginal distribution of each feature
        """
        if len(X) > num_augmented_samples:
            print("No data augmentation performed since training data is large enough.")
            return X
        num_augmented_samples = num_augmented_samples - len(X)

        X_aug = pd.concat([X.iloc[[0]]]*num_augmented_samples)
        X_aug.reset_index(drop=True, inplace=True)
        continuous_types = ['float','int', 'datetime']
        continuous_featnames = [] # these features will have shuffled values with added noise
        for contype in continuous_types:
            if contype in self.feature_types_metadata:
                continuous_featnames += self.feature_types_metadata[contype]
        for feature in X.columns:
            feature_data = X[feature]
            new_feature_data = feature_data.sample(n=num_augmented_samples, replace=True)
            new_feature_data.reset_index(drop=True, inplace=True)
            if feature in continuous_featnames:
                noise = np.random.normal(scale=np.std(feature_data)*continuous_feature_noise, size=num_augmented_samples)
                new_feature_data = new_feature_data + noise
            X_aug[feature] = pd.Series(new_feature_data, index=X_aug.index)
        X_aug.drop_duplicates(keep='first', inplace=True)
        # print(X_aug)
        X_aug = pd.concat([X_aug, X])
        X_aug.reset_index(drop=True, inplace=True)
        print("Augmented training dataset has %s datapoints" % X_aug.shape[0])
        return X_aug

    # TODO: experimental code.
    def augment_munge(self, X, num_augmented_samples = 50000, frac_feature_perturb=0.1, continuous_feature_noise = 0.05):
        """ Generates synthetic datapoints for learning to mimic teacher model in distillation.
            num_augmented_samples: number of total augmented data points to return (we add extra points to training set until this number is reached)
            frac_feature_perturb: fraction of features perturbed in each data point. Set smaller to ensure augmented samples remain closer to real data.
            continuous_feature_noise: we noise numeric features by this factor times their std-dev.
            These data are NOT marginally sampled, rather we replace randomly selected subset of features for each datapoint. Larger subset -> augmented data is more different than original.
        """
        if len(X) >= num_augmented_samples:
            print("No data augmentation performed since training data is large enough.")
            return X
        if frac_feature_perturb > 1.0:
            raise ValueError("frac_feature_perturb must be <= 1")
        print("Augmenting training data with synthetic samples for distillation...")
        num_feature_perturb = max(1, int(frac_feature_perturb*len(X.columns)))
        num_augmented_samples = num_augmented_samples - len(X)
        X_aug = pd.concat([X.iloc[[0]]]*num_augmented_samples)
        X_aug.reset_index(drop=True, inplace=True)
        continuous_types = ['float','int', 'datetime']
        continuous_featnames = [] # these features will have shuffled values with added noise
        for contype in continuous_types:
            if contype in self.feature_types_metadata:
                continuous_featnames += self.feature_types_metadata[contype]

        for i in range(num_augmented_samples): # hot-deck sample some features per datapoint
            og_ind = i % len(X)
            augdata_i = X.iloc[og_ind].copy()
            cols_toperturb = np.random.choice(list(X.columns), size=num_feature_perturb, replace=False)
            for feature in cols_toperturb:
                feature_data = X[feature]
                augdata_i[feature] = feature_data.sample(n=1).values[0]
            X_aug.iloc[i] = augdata_i

        for feature in X.columns:
            if feature in continuous_featnames:
                feature_data = X[feature]
                aug_data = X_aug[feature]
                noise = np.random.normal(scale=np.std(feature_data)*continuous_feature_noise, size=num_augmented_samples)
                aug_data = aug_data + noise
                X_aug[feature] = pd.Series(aug_data, index=X_aug.index)

        X_aug.drop_duplicates(keep='first', inplace=True)
        # print(X_aug)
        X_aug = pd.concat([X_aug, X])
        X_aug.reset_index(drop=True, inplace=True)
        print("Augmented training dataset has %s datapoints" % X_aug.shape[0])
        return X_aug

    # TODO: experimental code.
    def augment_trade(self, X = None):
        if X is None:
            X = self.load_X_train()
        # Convert X to numbers:
        continuous_types = ['float','int', 'datetime']
        continuous_featnames = [] # these features will have shuffled values with added noise
        for contype in continuous_types:
            if contype in self.feature_types_metadata:
                continuous_featnames += self.feature_types_metadata[contype]
        # Convert categoricals to int:
        feature_levels = {}
        for feature in X.columns:
            if feature not in continuous_featnames:
                feature_levels[feature] = {}
                feature_vals = X[feature].copy()
                feat_categories = sorted(list(feature_vals.unique()))
                for j in range(len(feat_categories)):
                    feat_category_j = feat_categories[j]
                    feature_levels[feature][feat_category_j] = j
                X.loc[:,feature] = pd.Series(feature_vals.map(feature_levels[feature]), index = X.index)
            feature_data = X[feature]
        # Save X:
        X.to_csv("data4trade.csv", index=False)
        import pickle
        pickle.dump(continuous_featnames, open("continuous_features.p", "wb") )
        """
        # Script to reload from this file for training TRADE generative model:
        import pandas as pd
        import pickle
        X = pd.read_csv("data4trade.csv")
        continuous_featnames = pickle.load( open( "continuous_features.p", "rb" ) )
        # Num categories for a categorical feature FEAT: len(X[FEAT].unique())
        num_categories = {}
        for feat in X.columns:
            if feat not in continuous_featnames:
                num_categories[feat] = len(X[feat].unique())
        X = X.to_numpy()
        """

    def best_single_model(self, stack_name, stack_level):
        """ Returns name of best single (compressed) model in this trainer object.

            Examples:
                To get get best single (compressed) model:
                    trainer.best_single_model('compressed', 0)  # TODO: does not work because compressed models have no validation score.
                To get best single (distilled) model:
                    trainer.best_single_model('distill', 0)
        """
        single_models = self.models_level[stack_name][stack_level]
        perfs = [self.model_performance[m] for m in single_models]
        return single_models[perfs.index(max(perfs))]

    def save_model(self, model):
        if self.low_memory:
            model.save()
        else:
            self.models[model.name] = model

    def save(self):
        save_pkl.save(path=self.path + self.trainer_file_name, object=self)

    def load_models_into_memory(self, model_names=None):
        if model_names is None:
            model_names = self.get_model_names_all()
        models = []
        for model_name in model_names:
            model = self.load_model(model_name)
            self.models[model.name] = model
            models.append(model)

        for model in models:
            if isinstance(model, StackerEnsembleModel):
                for base_model_name in model.base_model_names:
                    if base_model_name not in model.base_models_dict.keys():
                        if base_model_name in self.models.keys():
                            model.base_models_dict[base_model_name] = self.models[base_model_name]
            if isinstance(model, BaggedEnsembleModel):
                for fold, fold_model in enumerate(model.models):
                    if isinstance(fold_model, str):
                        model.models[fold] = model.load_child(fold_model)

    def load_model(self, model_name: str) -> AbstractModel:
        if model_name in self.models.keys():
            return self.models[model_name]
        else:
            return self.model_types[model_name].load(path=self.model_paths[model_name], reset_paths=self.reset_paths)

    def _get_dummy_stacker(self, level, use_orig_features=True):
        model_names = self.models_level['core'][level-1]
        base_models_dict = {}
        for model_name in model_names:
            if model_name in self.models.keys():
                base_models_dict[model_name] = self.models[model_name]
        dummy_stacker = StackerEnsembleModel(
            path='', name='',
            model_base=AbstractModel(path='', name='', problem_type=self.problem_type, objective_func=self.objective_func),
            base_model_names=model_names, base_models_dict=base_models_dict, base_model_paths_dict=self.model_paths,
            base_model_types_dict=self.model_types, use_orig_features=use_orig_features, num_classes=self.num_classes, random_state=level
        )
        return dummy_stacker

    def get_models_info(self, model_names):
        model_names = copy.deepcopy(model_names)
        model_paths = {model_name: self.model_paths[model_name] for model_name in model_names}
        model_types = {model_name: self.model_types[model_name] for model_name in model_names}
        return model_names, model_paths, model_types

    # TODO: Add pred_time_val_full (Will be incorrect unless graph representation is added)
    def leaderboard(self):
        model_names = self.get_model_names_all()
        score_val = []
        fit_time = []
        pred_time_val = []
        stack_level = []
        for model_name in model_names:
            score_val.append(self.model_performance.get(model_name))
            fit_time.append(self.model_fit_times.get(model_name))
            pred_time_val.append(self.model_pred_times.get(model_name))
            stack_level.append(self.get_model_level(model_name))
        df = pd.DataFrame(data={
            'model': model_names,
            'score_val': score_val,
            'fit_time': fit_time,
            'pred_time_val': pred_time_val,
            'stack_level': stack_level,
        })
        df_sorted = df.sort_values(by=['score_val', 'model'], ascending=False)
        return df_sorted

    def info(self):
        model_count = len(self.get_model_names_all())
        if self.model_best is not None:
            best_model = self.model_best
        else:
            best_model = self.model_best_core
        best_model_score_val = self.model_performance.get(best_model)
        # fit_time = None
        num_bagging_folds = self.kfolds
        max_stack_level = self.get_max_level('core')
        best_model_stack_level = self.get_model_level(best_model)
        problem_type = self.problem_type
        objective_func = self.objective_func.name
        time_train_start = self.time_train_start
        num_rows_train = self.num_rows_train
        num_cols_train = self.num_cols_train
        num_classes = self.num_classes
        # TODO:
        #  Disk size of models
        #  Raw feature count
        #  HPO time
        #  Bag time
        #  Feature prune time
        #  Exception count / models failed count
        #  True model count (models * kfold)
        #  AutoGluon version fit on
        #  Max memory usage
        #  CPU count used / GPU count used

        info = {
            'model_count': model_count,
            'best_model': best_model,
            'best_model_score_val': best_model_score_val,
            'num_bagging_folds': num_bagging_folds,
            'max_stack_level': max_stack_level,
            'best_model_stack_level': best_model_stack_level,
            'problem_type': problem_type,
            'objective_func': objective_func,
            'time_train_start': time_train_start,
            'num_rows_train': num_rows_train,
            'num_cols_train': num_cols_train,
            'num_classes': num_classes,
        }

        return info

    @classmethod
    def load(cls, path, reset_paths=False):
        load_path = path + cls.trainer_file_name
        if not reset_paths:
            return load_pkl.load(path=load_path)
        else:
            obj = load_pkl.load(path=load_path)
            obj.set_contexts(path)
            obj.reset_paths = reset_paths
            return obj
