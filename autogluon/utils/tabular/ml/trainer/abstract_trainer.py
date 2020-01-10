import copy, time, traceback, logging
from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from collections import defaultdict
from sklearn.model_selection import train_test_split

from ..constants import BINARY, MULTICLASS, REGRESSION
from ...utils.loaders import load_pkl
from ...utils.savers import save_pkl
from ...utils.exceptions import TimeLimitExceeded, NotEnoughMemoryError
from ..utils import get_pred_from_proba, dd_list
from ..models.abstract.abstract_model import AbstractModel
from ..tuning.feature_pruner import FeaturePruner
from ...metrics import accuracy, root_mean_squared_error, scorer_expects_y_pred
from ..models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from ..trainer.model_presets.presets import get_preset_stacker_model
from ..models.ensemble.stacker_ensemble_model import StackerEnsembleModel
from ..models.ensemble.weighted_ensemble_model import WeightedEnsembleModel

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

    def __init__(self, path: str, problem_type: str, scheduler_options=None, objective_func=None,
                 num_classes=None, low_memory=False, feature_types_metadata={}, kfolds=0, n_repeats=1,
                 stack_ensemble_levels=0, time_limit=None, save_data=False, verbosity=2):
        self.path = path
        self.problem_type = problem_type
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

        self.objective_func_expects_y_pred = scorer_expects_y_pred(scorer=self.objective_func)
        logger.log(25, "AutoGluon will gauge predictive performance using evaluation metric: %s" % self.objective_func.name)
        if not self.objective_func_expects_y_pred:
            logger.log(25, "This metric expects predicted probabilities rather than predicted class labels, so you'll need to use predict_proba() instead of predict()")

        logger.log(20, "To change this, specify the eval_metric argument of fit()")
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

    # path_root is the directory containing learner.pkl
    @property
    def path_root(self):
        return self.path.rsplit('/', maxsplit=2)[0] + '/'

    @property
    def path_utils(self):
        return self.path_root + 'utils/'

    @property
    def path_data(self):
        return self.path_utils + 'data/'

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

    def get_models(self, hyperparameters, hyperparameter_tune=False):
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

    def generate_train_test_split(self, X: DataFrame, y: Series, test_size: float = 0.1, random_state=42) -> (DataFrame, DataFrame, Series, Series):
        if (test_size <= 0.0) or (test_size >= 1.0):
            raise ValueError("fraction of data to hold-out must be specified between 0 and 1")
        if self.problem_type == REGRESSION:
            stratify = None
        else:
            stratify = y

        # TODO: Enable stratified split when y class would result in 0 samples in test.
        #  One approach: extract low frequency classes from X/y, add back (1-test_size)% to X_train, y_train, rest to X_test
        #  Essentially stratify the high frequency classes, random the low frequency (While ensuring at least 1 example stays for each low frequency in train!)
        #  Alternatively, don't test low frequency at all, trust it to work in train set. Risky, but highest quality for predictions.
        X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=test_size, shuffle=True, random_state=random_state, stratify=stratify)
        y_train = pd.Series(y_train, index=X_train.index)
        y_test = pd.Series(y_test, index=X_test.index)

        return X_train, X_test, y_train, y_test

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
        if self.bagged_mode or (type(model) == WeightedEnsembleModel):
            model.fit(X=X_train, y=y_train, k_fold=kfolds, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, compute_base_preds=False, time_limit=time_limit, **model_fit_kwargs)
        else:
            model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, time_limit=time_limit, **model_fit_kwargs)
        return model

    def train_and_save(self, X_train, y_train, X_test, y_test, model: AbstractModel, stack_name='core', kfolds=None, k_fold_start=0, k_fold_end=None, n_repeats=None, n_repeat_start=0, level=0, ignore_time_limit=False):
        stack_loc = self.models_level[stack_name]  # TODO: Consider removing, have train_multi handle this
        fit_start_time = time.time()
        model_names_trained = []
        try:
            if not ignore_time_limit:
                time_left = self.time_limit_per_level - (fit_start_time - self.time_train_level_start)
                if time_left < 0:
                    logging.log(15, 'Skipping ' + str(model.name) + ' due to lack of time remaining.')
                    return model_names_trained
                time_left_total = self.time_limit - (fit_start_time - self.time_train_start)
                logging.log(20, 'Fitting model: ' + str(model.name) + ' ...' + ' Training model for up to ' + str(round(time_left, 2)) + 's of the ' + str(round(time_left_total, 2)) + 's of remaining time.')
            else:
                time_left = None
                logging.log(20, 'Fitting model: ' + str(model.name) + ' ...')
            model = self.train_single(X_train, y_train, X_test, y_test, model, kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, level=level, time_limit=time_left)
            fit_end_time = time.time()
            if type(model) in [BaggedEnsembleModel, StackerEnsembleModel, WeightedEnsembleModel]:
                score = model.score_with_oof(y=y_train)
            else:
                score = model.score(X=X_test, y=y_test)
            pred_end_time = time.time()
        except TimeLimitExceeded as err:
            logger.log(20, '\tTime limit exceeded... Skipping ' + model.name + '.')
            # logger.log(20, '\tTime wasted: ' + str(time.time() - fit_start_time))
            del model
        except NotEnoughMemoryError as err:
            logger.warning('\tNot enough memory to train model... Skipping ' + model.name + '.')
            del model
        except Exception as err:
            if self.verbosity >= 1:
                traceback.print_tb(err.__traceback__)
            logger.exception('Warning: Exception caused ' +str(model.name)+' to fail during training... Skipping this model.')
            logger.debug(err)
            del model
        else:
            self.model_performance[model.name] = score
            self.model_paths[model.name] = model.path
            self.model_types[model.name] = type(model)
            if type(model) in [BaggedEnsembleModel, StackerEnsembleModel, WeightedEnsembleModel]:
                self.model_types_inner[model.name] = model._child_type
            else:
                self.model_types_inner[model.name] = type(model)
            logger.log(20, '\t' + str(round(fit_end_time - fit_start_time, 2))+'s' + '\t = Training runtime')
            logger.log(20, '\t' + str(round(score, 4)) + '\t = Validation ' + self.objective_func.name + ' score')
            logger.log(15, '\tEvaluation runtime of '+str(model.name)+ ' = '+str(round(pred_end_time - fit_end_time, 2))+' s')
            # TODO: Should model have fit-time/pred-time information?
            # TODO: Add to HPO
            model_fit_time = fit_end_time - fit_start_time
            model_pred_time = pred_end_time - fit_end_time
            if n_repeat_start > 0:
                self.model_fit_times[model.name] += model_fit_time
                self.model_pred_times[model.name] += model_pred_time
            else:
                self.model_fit_times[model.name] = model_fit_time
                self.model_pred_times[model.name] = model_pred_time
            self.save_model(model=model)
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

            model_names_trained.append(model.name)
            if self.low_memory:
                del model
        return model_names_trained

    def train_single_full(self, X_train, y_train, X_test, y_test, model: AbstractModel, feature_prune=False, 
                          hyperparameter_tune=True, stack_name='core', kfolds=None, k_fold_start=0, k_fold_end=None, n_repeats=None, n_repeat_start=0, level=0, ignore_time_limit=False):
        if n_repeat_start == 0:
            model.feature_types_metadata = self.feature_types_metadata  # TODO: Don't set feature_types_metadata here
        if feature_prune:
            if n_repeat_start != 0:
                raise ValueError('n_repeat_start must be 0 to feature_prune, value = ' + str(n_repeat_start))
            self.autotune(X_train=X_train, X_holdout=X_test, y_train=y_train, y_holdout=y_test, model_base=model)  # TODO: Update to use CV instead of holdout
        if hyperparameter_tune:
            if self.scheduler_func is None or self.scheduler_options is None:
                raise ValueError("scheduler_options cannot be None when hyperparameter_tune = True")
            if n_repeat_start != 0:
                raise ValueError('n_repeat_start must be 0 to hyperparameter_tune, value = ' + str(n_repeat_start))
            # hpo_models (dict): keys = model_names, values = model_paths
            try:  # TODO: Make exception handling more robust? Return successful HPO models?
                if type(model) in [BaggedEnsembleModel, StackerEnsembleModel, WeightedEnsembleModel]:
                    hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X=X_train, y=y_train, k_fold=kfolds, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
                else:
                    if (X_test is None) or (y_test is None):
                        X_train, X_test, y_train, y_test = self.generate_train_test_split(X_train, y_train, test_size=0.2)  # TODO: Adjust test_size, perhaps user specified?
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
                if type(model) in [BaggedEnsembleModel, StackerEnsembleModel, WeightedEnsembleModel]:
                    self.model_types_inner.update({name: model._child_type for name in model_names_trained})
                else:
                    self.model_types_inner.update({name: type(model) for name in model_names_trained})
        else:
            model_names_trained = self.train_and_save(X_train, y_train, X_test, y_test, model, stack_name=stack_name, kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, level=level, ignore_time_limit=ignore_time_limit)
        self.save()
        return model_names_trained

    # TODO: How to deal with models that fail during this? They have trained valid models before, but should we still use those models or remove the entire model? Currently we still use models.
    # TODO: Time allowance can be made better by only using time taken during final model training and not during HPO and feature pruning.
    # TODO: Time allowance not accurate if running from fit_continue
    # Takes trained bagged ensemble models and fits additional k-fold bags.
    def train_multi_repeats(self, X_train, y_train, X_test, y_test, models, stack_name='core', kfolds=None, n_repeats=None, n_repeat_start=1, level=0, ignore_time_limit=False):
        if n_repeats is None:
            n_repeats = self.n_repeats
        models_valid = models
        models_valid_next = []
        start_time = time.time()
        repeats_completed = 0
        for n in range(n_repeat_start, n_repeats):
            if not ignore_time_limit:
                repeat_start_time = time.time()
                time_left = self.time_limit_per_level - (repeat_start_time - self.time_train_level_start)
                if n == n_repeat_start:
                    time_required = self.time_limit_per_level * 0.575  # Require slightly over 50% to be safe
                else:
                    time_required = (repeat_start_time - start_time) / repeats_completed * (0.575/0.425)
                if time_left < time_required:
                    logger.log(15, 'Not enough time left to finish repeated k-fold bagging, stopping early ...')
                    break
            logger.log(20, 'Repeating k-fold bagging: ' + str(n+1) + '/' + str(n_repeats))
            for i, model in enumerate(models_valid):
                if type(model) == str:
                    model = self.load_model(model)
                models_valid_next += self.train_single_full(X_train, y_train, X_test, y_test, model, hyperparameter_tune=False, feature_prune=False, stack_name=stack_name, kfolds=kfolds, k_fold_start=0, k_fold_end=None, n_repeats=n+1, n_repeat_start=n, level=level, ignore_time_limit=ignore_time_limit)
            models_valid = copy.deepcopy(models_valid_next)
            models_valid_next = []
            repeats_completed += 1
        logger.log(20, 'Completed ' + str(n_repeat_start + repeats_completed) + '/' + str(n_repeats) + ' k-fold bagging repeats ...')
        return models_valid

    def train_multi_initial(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False, stack_name='core', kfolds=None, n_repeats=None, level=0, ignore_time_limit=False):
        if kfolds is None:
            kfolds = self.kfolds
        stack_loc = self.models_level[stack_name]

        model_names_trained = []
        model_names_trained_hpo = []
        models_valid = models
        if kfolds == 0:
            models_valid = self.train_multi_fold(X_train, y_train, X_test, y_test, models_valid, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, stack_name=stack_name,
                                                          kfolds=kfolds, level=level, ignore_time_limit=ignore_time_limit)
        else:
            k_fold_start = 0
            if hyperparameter_tune or feature_prune:
                models_valid = self.train_multi_fold(X_train, y_train, X_test, y_test, models_valid, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, stack_name=stack_name,
                                                     kfolds=kfolds, k_fold_start=0, k_fold_end=1, n_repeats=n_repeats, n_repeat_start=0, level=level, ignore_time_limit=ignore_time_limit)
                k_fold_start = 1
            models_valid = self.train_multi_fold(X_train, y_train, X_test, y_test, models_valid, hyperparameter_tune=False, feature_prune=False, stack_name=stack_name,
                                                 kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=kfolds, n_repeats=n_repeats, n_repeat_start=0, level=level, ignore_time_limit=ignore_time_limit)

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

    def train_multi_fold(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False, stack_name='core', kfolds=None, k_fold_start=0, k_fold_end=None, n_repeats=None, n_repeat_start=0, level=0, ignore_time_limit=False):
        models_valid = []
        for i, model in enumerate(models):
            if type(model) == str:
                model = self.load_model(model)
            elif self.low_memory:
                model = copy.deepcopy(model)
            # TODO: Only update scores when finished, only update model as part of final models if finished!
            model_name_trained_lst = self.train_single_full(X_train, y_train, X_test, y_test, model, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, stack_name=stack_name,
                                                            kfolds=kfolds, k_fold_start=k_fold_start, k_fold_end=k_fold_end,
                                                            n_repeats=n_repeats, n_repeat_start=n_repeat_start, level=level, ignore_time_limit=ignore_time_limit)

            if self.low_memory:
                del model
            models_valid += model_name_trained_lst

        return models_valid

    def train_multi(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False, stack_name='core', kfolds=None, n_repeats=None, n_repeat_start=0, level=0, ignore_time_limit=False):
        if n_repeats is None:
            n_repeats = self.n_repeats
        if ignore_time_limit:
            n_repeats_initial = n_repeats
        else:
            n_repeats_initial = 1
        if n_repeat_start == 0:
            model_names_trained = self.train_multi_initial(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                                                           stack_name=stack_name, kfolds=kfolds, n_repeats=n_repeats_initial, level=level, ignore_time_limit=ignore_time_limit)
            n_repeat_start = n_repeats_initial
        else:
            model_names_trained = models
        if (n_repeats > 1) and self.bagged_mode and (n_repeat_start < n_repeats):
            model_names_trained = self.train_multi_repeats(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=model_names_trained,
                                                           stack_name=stack_name, kfolds=kfolds, n_repeats=n_repeats, n_repeat_start=n_repeat_start, level=level, ignore_time_limit=ignore_time_limit)
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
            if level == 0:
                self.stack_new_level(X=X_train, y=y_train, X_test=X_test, y_test=y_test, models=models, level=level, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, ignore_time_limit=self.ignore_time_limit)
            else:
                self.stack_new_level(X=X_train, y=y_train, X_test=X_test, y_test=y_test, level=level, ignore_time_limit=self.ignore_time_limit)

        self.save()

        # TODO: Select best weighted ensemble given L2 can be much worse than L1 when dealing with time limitation

    def stack_new_level(self, X, y, X_test=None, y_test=None, level=0, models=None, hyperparameter_tune=False, feature_prune=False, ignore_time_limit=True):
        self.stack_new_level_core(X=X, y=y, X_test=X_test, y_test=y_test, models=models, level=level, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, ignore_time_limit=ignore_time_limit)
        if self.bagged_mode:
            self.stack_new_level_aux(X=X, y=y, level=level+1, ignore_time_limit=ignore_time_limit)
        else:
            self.stack_new_level_aux(X=X_test, y=y_test, fit=False, level=level+1, ignore_time_limit=ignore_time_limit)

    def stack_new_level_core(self, X, y, X_test=None, y_test=None, models=None, level=1, hyperparameter_tune=False, feature_prune=False, ignore_time_limit=True):
        use_orig_features = True
        if models is None:
            models = self.get_models(self.hyperparameters)

        if self.bagged_mode:
            if level == 0:
                (base_model_names, base_model_paths, base_model_types) = ([], {}, {})
            elif level > 0:
                base_model_names, base_model_paths, base_model_types = self.get_models_info(model_names=self.models_level['core'][level - 1])
                if len(base_model_names) == 0:
                    logger.log(20, 'No base models to train on, skipping stack level...')
                    return
            models = [
                StackerEnsembleModel(path=self.path, name=model.name + '_STACKER_l' + str(level), model_base=model, base_model_names=base_model_names,
                                     base_model_paths_dict=base_model_paths, base_model_types_dict=base_model_types, use_orig_features=use_orig_features,
                                     num_classes=self.num_classes, random_state=level)
                for model in models]
        X_train_init = self.get_inputs_to_stacker(X, level_start=0, level_end=level, fit=True)
        if X_test is not None:
            X_test = self.get_inputs_to_stacker(X_test, level_start=0, level_end=level, fit=False)

        self.train_multi(X_train=X_train_init, y_train=y, X_test=X_test, y_test=y_test, models=models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, level=level, ignore_time_limit=ignore_time_limit)

    def stack_new_level_aux(self, X, y, level, fit=True, ignore_time_limit=True):
        stack_name = 'aux1'
        X_train_stack_preds = self.get_inputs_to_stacker(X, level_start=0, level_end=level, fit=fit)
        self.generate_weighted_ensemble(X=X_train_stack_preds, y=y, level=level, k_fold=0, stack_name=stack_name, ignore_time_limit=True)

    def generate_weighted_ensemble(self, X, y, level, k_fold=0, stack_name=None, hyperparameters=None, ignore_time_limit=False, name_suffix=''):
        if len(self.models_level['core'][level-1]) == 0:
            logger.log(20, 'No base models to train on, skipping weighted ensemble...')
            return
        weighted_ensemble_model = WeightedEnsembleModel(path=self.path, name='weighted_ensemble_' + name_suffix + 'k' + str(k_fold) + '_l' + str(level), base_model_names=self.models_level['core'][level-1],
                                                        base_model_paths_dict=self.model_paths, base_model_types_dict=self.model_types, base_model_types_inner_dict=self.model_types_inner, base_model_performances_dict=self.model_performance, hyperparameters=hyperparameters,
                                                        num_classes=self.num_classes, random_state=level)

        self.train_multi(X_train=X, y_train=y, X_test=None, y_test=None, models=[weighted_ensemble_model], hyperparameter_tune=False, feature_prune=False, stack_name=stack_name, kfolds=k_fold, level=level, ignore_time_limit=ignore_time_limit)
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

    def generate_stack_log_reg(self, X, y, level, k_fold=0, stack_name=None):
        base_model_names, base_model_paths, base_model_types = self.get_models_info(model_names=self.models_level['core'][level-1])
        stacker_model_lr = get_preset_stacker_model(path=self.path, problem_type=self.problem_type, objective_func=self.objective_func, num_classes=self.num_classes)
        name_new = stacker_model_lr.name + '_STACKER_k' + str(k_fold) + '_l' + str(level)

        stacker_model_lr = StackerEnsembleModel(path=self.path, name=name_new, model_base=stacker_model_lr, base_model_names=base_model_names, base_model_paths_dict=base_model_paths, base_model_types_dict=base_model_types,
                                                use_orig_features=False,
                                                num_classes=self.num_classes, random_state=level)

        self.train_multi(X_train=X, y_train=y, X_test=None, y_test=None, models=[stacker_model_lr], hyperparameter_tune=False, feature_prune=False, stack_name=stack_name, kfolds=k_fold, level=level)

    def predict(self, X):
        if self.model_best is not None:
            return self.predict_model(X, self.model_best)
        elif self.model_best_core is not None:
            return self.predict_model(X, self.model_best_core)
        else:
            raise Exception('Trainer has no fit models to predict with.')

    def predict_proba(self, X):
        if self.model_best is not None:
            return self.predict_proba_model(X, self.model_best)
        elif self.model_best_core is not None:
            return self.predict_proba_model(X, self.model_best_core)
        else:
            raise Exception('Trainer has no fit models to predict with.')

    def predict_model(self, X, model, level_start=0):
        if type(model) == str:
            model = self.load_model(model)
        X = self.get_inputs_to_model(model=model, X=X, level_start=level_start, fit=False)
        return model.predict(X=X, preprocess=False)

    def predict_proba_model(self, X, model, level_start=0):
        if type(model) == str:
            model = self.load_model(model)
        X = self.get_inputs_to_model(model=model, X=X, level_start=level_start, fit=False)
        return model.predict_proba(X=X, preprocess=False)

    def get_inputs_to_model(self, model, X, level_start, fit=False, preprocess=True):
        if type(model) == str:
            model = self.load_model(model)
        model_level = self.get_model_level(model.name)
        if model_level >= 1:
            X = self.get_inputs_to_stacker(X=X, level_start=level_start, level_end=model_level-1, fit=fit)
            X = model.preprocess(X, fit=fit, preprocess=preprocess)
        else:
            if preprocess:
                X = model.preprocess(X)
        return X

    def score(self, X, y):
        if self.objective_func_expects_y_pred:
            y_pred_ensemble = self.predict(X=X)
            return self.objective_func(y, y_pred_ensemble)
        else:
            y_pred_proba_ensemble = self.predict_proba(X=X)
            return self.objective_func(y, y_pred_proba_ensemble)

    def score_with_y_pred_proba(self, y, y_pred_proba):
        if self.objective_func_expects_y_pred:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            return self.objective_func(y, y_pred)
        else:
            return self.objective_func(y, y_pred_proba)

    def autotune(self, X_train, X_holdout, y_train, y_holdout, model_base: AbstractModel):
        feature_pruner = FeaturePruner(model_base=model_base)
        X_train, X_test, y_train, y_test = self.generate_train_test_split(X_train, y_train)
        feature_pruner.tune(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_holdout=X_holdout, y_holdout=y_holdout)
        features_to_keep = feature_pruner.features_in_iter[feature_pruner.best_iteration]
        logger.debug(str(features_to_keep))
        model_base.features = features_to_keep
        # autotune.evaluate()

    def pred_proba_predictions(self, models, X_test):
        preds = []
        for model in models:
            if type(model) is str:
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

    def save_model(self, model):
        if self.low_memory:
            model.save()
        else:
            self.models[model.name] = model

    def save(self):
        save_pkl.save(path=self.path + self.trainer_file_name, object=self)

    def load_model(self, model_name: str) -> AbstractModel:
        if self.low_memory:
            return self.model_types[model_name].load(path=self.model_paths[model_name], reset_paths=self.reset_paths)
        else:
            return self.models[model_name]

    def _get_dummy_stacker(self, level, use_orig_features=True):
        model_names = self.models_level['core'][level-1]
        dummy_stacker = StackerEnsembleModel(
            path='', name='',
            model_base=AbstractModel(path='', name='', model=None, problem_type=self.problem_type, objective_func=self.objective_func),
            base_model_names=model_names, base_model_paths_dict=self.model_paths,
            base_model_types_dict=self.model_types, use_orig_features=use_orig_features, num_classes=self.num_classes, random_state=level
        )
        return dummy_stacker

    def get_models_info(self, model_names):
        model_names = copy.deepcopy(model_names)
        model_paths = {model_name: self.model_paths[model_name] for model_name in model_names}
        model_types = {model_name: self.model_types[model_name] for model_name in model_names}
        return model_names, model_paths, model_types

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
