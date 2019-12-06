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
from ..utils import get_pred_from_proba
from ..models.abstract.abstract_model import AbstractModel
from ..tuning.feature_pruner import FeaturePruner
from ..models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from ...metrics import accuracy, root_mean_squared_error, scorer_expects_y_pred
from ..models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from ..trainer.model_presets.presets import get_preset_stacker_model
from ..models.ensemble.stacker_ensemble_model import StackerEnsembleModel
from ..models.ensemble.weighted_ensemble_model import WeightedEnsembleModel

logger = logging.getLogger(__name__)

# TODO: Add post-fit cleanup function which loads all models and saves them after removing unnecessary variables such as oof_pred_probas to optimize load times and space usage
#  Trainer will not be able to be fit further after this operation is done, but it will be able to predict.
# TODO: Dynamic model loading for ensemble models during prediction, only load more models if prediction is uncertain. This dynamically reduces inference time.
# TODO: Try midstack Semi-Supervised. Just take final models and re-train them, use bagged preds for SS rows. This would be very cheap and easy to try.
class AbstractTrainer:
    trainer_file_name = 'trainer.pkl'

    def __init__(self, path: str, problem_type: str, scheduler_options=None, objective_func=None,
                 num_classes=None, low_memory=False, feature_types_metadata={}, kfolds=0, 
                 stack_ensemble_levels=0, verbosity=2):
        self.path = path
        self.problem_type = problem_type
        self.feature_types_metadata = feature_types_metadata
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
        else:
            self.kfolds = 0
            self.stack_ensemble_levels = 0
            self.stack_mode = False

        self.hyperparameters = {}  # TODO: This is currently required for fetching stacking layer models. Consider incorporating more elegantly

        # self.models_level[0] # Includes base models
        # self.models_level[1] # Stacker level 1, includes weighted ensembles of level 0 (base)
        # self.models_level[2] # Stacker level 2, includes weighted ensembles of level 1
        self.models_level = defaultdict(list)
        self.models_level_auxiliary = defaultdict(list)

        self.model_best = None
        self.model_best_core = None

        self.model_performance = {}
        self.model_paths = {}
        self.model_types = {}
        self.model_fit_times = {}
        self.model_pred_times = {}
        self.models = {}
        self.model_weights = None
        self.reset_paths = False
        # Things stored
        self.hpo_results = {}  # Stores summary of HPO process
        self.hpo_model_names = defaultdict(list)  # stores additional models produced during HPO
        # Scheduler attributes:
        if scheduler_options is not None:
            self.scheduler_func = scheduler_options[0]  # unpack tuple
            self.scheduler_options = scheduler_options[1]
        else:
            self.scheduler_func = None
            self.scheduler_options = None
        # nthreads_per_trial = self.scheduler_options['resource']['num_cpus']
        # ngpus_per_trial = self.scheduler_options['resource']['num_gpus']

    @property
    def model_names(self):
        return self.model_names_core + self.model_names_aux

    @property
    def model_names_core(self):
        model_names = []
        levels = np.sort(list(self.models_level.keys()))
        for level in levels:
            model_names += self.models_level[level]
        return model_names

    @property
    def model_names_aux(self):
        model_names = []
        levels = np.sort(list(self.models_level_auxiliary.keys()))
        for level in levels:
            model_names += self.models_level_auxiliary[level]
        return model_names

    @property
    def max_level(self):
        try:
            return np.sort(list(self.models_level.keys()))[-1]
        except IndexError:
            return -1

    @property
    def max_level_auxiliary(self):
        try:
            return np.sort(list(self.models_level_auxiliary.keys()))[-1]
        except IndexError:
            return -1

    def get_models(self, hyperparameters, hyperparameter_tune=False):
        raise NotImplementedError

    def get_model_level(self, model_name):
        for level in self.models_level.keys():
            if model_name in self.models_level[level]:
                return level
        for level in self.models_level_auxiliary.keys():
            if model_name in self.models_level_auxiliary[level]:
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

    def train_single(self, X_train, y_train, X_test, y_test, model, kfolds=None, level=0):
        if kfolds is None:
            kfolds = self.kfolds
        logging.log(20, 'Fitting model: '+str(model.name)+' ...')
        model.feature_types_metadata = self.feature_types_metadata  # TODO: move this into model creation process?
        model_fit_kwargs = {}
        if self.scheduler_options is not None:
            model_fit_kwargs = {'verbosity': self.verbosity, 
                                'num_cpus': self.scheduler_options['resource']['num_cpus'],
                                'num_gpus': self.scheduler_options['resource']['num_gpus']}  # Additional configurations for model.fit
        if self.bagged_mode or (type(model) == WeightedEnsembleModel):
            if type(model) not in [BaggedEnsembleModel, StackerEnsembleModel, WeightedEnsembleModel]:
                model = BaggedEnsembleModel(path=model.path[:-(len(model.name) + 1)], name=model.name + '_BAGGED' + '_l' + str(level), model_base=model)
            model.fit(X=X_train, y=y_train, k_fold=kfolds, random_state=level, compute_base_preds=False, **model_fit_kwargs)
        else:
            model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, **model_fit_kwargs)
        return model

    def train_and_save(self, X_train, y_train, X_test, y_test, model: AbstractModel, stack_loc=None, kfolds=None, level=0):
        if stack_loc is None:
            stack_loc = self.models_level
        try:
            fit_start_time = time.time()
            model = self.train_single(X_train, y_train, X_test, y_test, model, kfolds=kfolds, level=level)
            fit_end_time = time.time()
            if type(model) in [BaggedEnsembleModel, StackerEnsembleModel, WeightedEnsembleModel]:
                score = model.score_with_y_pred_proba(y=y_train, y_pred_proba=model.oof_pred_proba)
            else:
                score = model.score(X=X_test, y=y_test)
            pred_end_time = time.time()
        except Exception as err:
            traceback.print_tb(err.__traceback__)
            logger.exception('Warning: Exception caused ' +str(model.name)+' to fail during training... Skipping this model.')
            logger.debug(err)
            del model
        else:
            stack_loc[level].append(model.name)
            self.model_performance[model.name] = score
            self.model_paths[model.name] = model.path
            self.model_types[model.name] = type(model)
            logger.log(15, 'Performance of ' +str(model.name)+': '+str(score))
            logger.log(15, 'Training runtime of '+str(model.name)+ ' = '+str(fit_end_time - fit_start_time)+' s')
            logger.log(15, 'Evaluation runtime of '+str(model.name)+ ' = '+str(pred_end_time - fit_end_time)+' s')
            # TODO: Should model have fit-time/pred-time information?
            # TODO: Add to HPO
            self.model_fit_times[model.name] = fit_end_time - fit_start_time
            self.model_pred_times[model.name] = pred_end_time - fit_end_time
            self.save_model(model=model)
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
                          hyperparameter_tune=True, stack_loc=None, kfolds=None, level=0):
        model.feature_types_metadata = self.feature_types_metadata  # TODO: Don't set feature_types_metadata here
        if feature_prune:
            self.autotune(X_train=X_train, X_holdout=X_test, y_train=y_train, y_holdout=y_test, model_base=model)  # TODO: Update to use CV instead of holdout
        if hyperparameter_tune:
            if self.scheduler_func is None or self.scheduler_options is None:
                raise ValueError("scheduler_options cannot be None when hyperparameter_tune = True")
            if (X_test is None) or (y_test is None):
                X_train, X_test, y_train, y_test = self.generate_train_test_split(X_train, y_train, test_size=0.2)  # TODO: Adjust test_size, perhaps user specified?
            # Moved split into lightGBM. TODO: need to do same for other models that use their own splits as well. Old code was:  model.hyperparameter_tune(pd.concat([X_train, X_test], ignore_index=True), pd.concat([y_train, y_test], ignore_index=True))
            # hpo_models (dict): keys = model_names, values = model_paths
            try:  # TODO: Make exception handling more robust? Return successful HPO models?
                hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X_train=X_train, X_test=X_test,
                    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
            except Exception as err:
                traceback.print_tb(err.__traceback__)
                logger.exception('Warning: Exception caused ' + model.name + ' to fail during hyperparameter tuning... Skipping this model.')
                logger.debug(err)
                del model
            else:
                self.hpo_model_names[level] += list(sorted(hpo_models.keys()))
                self.model_paths.update(hpo_models)
                self.model_performance.update(hpo_model_performances)
                self.hpo_results[model.name] = hpo_results
                self.model_types.update({name: type(model) for name in sorted(hpo_models.keys())})
        else:
            self.train_and_save(X_train, y_train, X_test, y_test, model, stack_loc=stack_loc, kfolds=kfolds, level=level)
        self.save()

    def train_multi(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False, stack_loc=None, kfolds=None, level=0):
        for i, model in enumerate(models):
            self.train_single_full(X_train, y_train, X_test, y_test, model, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune, stack_loc=stack_loc, kfolds=kfolds, level=level)
        if self.bagged_mode:  # TODO: Maybe toggle this based on if we have sufficient time left in our time budget after HPO
            # TODO: Maybe generate weighted_ensemble prior to bagging, and only bag models which were given weight in the initial weighted_ensemble
            for i, hpo_model_name in enumerate(self.hpo_model_names[level]):
                model_hpo = self.load_model(hpo_model_name)
                if type(model_hpo) == TabularNeuralNetModel:  # TODO: Remove this after fixing TabularNeuralNetModel
                    model_hpo = model_hpo.create_unfit_copy()
                model_bagged = BaggedEnsembleModel(path=model_hpo.path[:-(len(model_hpo.name) + 1)], name=model_hpo.name + '_' + str(i) + '_BAGGED' + '_l' + str(level), model_base=model_hpo)
                # TODO: Throws exception on Neural Network since trained object is not pickle-able. Fix this to enable bagging for NN by creating new base model in BaggedEnsembleModel with trained model's hyperparams
                self.train_and_save(X_train, y_train, X_test, y_test, model_bagged, stack_loc=stack_loc, kfolds=kfolds, level=level)
                self.save()
        else:
            self.models_level[level] += self.hpo_model_names[level]  # Update model list with (potentially empty) list of new models created during HPO
        unique_names = []
        for item in self.models_level[level]:
            if item not in unique_names: unique_names.append(item)
        self.models_level[level] = unique_names  # make unique and preserve order

    def train_multi_and_ensemble(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False):
        self.train_multi(X_train, y_train, X_test, y_test, models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
        for model_name in self.models_level[0]:
            if model_name not in self.model_performance:
                model = self.load_model(model_name)
                self.model_performance[model_name] = model.score(X_test, y_test)
            logger.log(15, "Performance of %s model: %s" % (model_name, self.model_performance[model_name]))
        if len(self.models_level[0]) == 0:
            raise ValueError('AutoGluon did not successfully train any models')

        if self.bagged_mode:
            self.stack_new_level_aux(X=X_train, y=y_train, level=1)
        else:
            stack_loc = self.models_level_auxiliary
            X_test_preds = self.get_inputs_to_stacker(X=X_test, level_start=0, level_end=1, fit=False)
            self.generate_weighted_ensemble(X=X_test_preds, y=y_test, level=1, stack_loc=stack_loc)

        if self.stack_mode:
            for level in range(1, self.stack_ensemble_levels + 1):
                self.stack_new_level(X=X_train, y=y_train, level=level)

        self.save()

    def stack_new_level(self, X, y, level):
        self.stack_new_level_core(X=X, y=y, level=level)
        self.stack_new_level_aux(X=X, y=y, level=level+1)

    def stack_new_level_core(self, X, y, level=1):
        base_model_names, base_model_paths, base_model_types = self.get_models_info(model_names=self.models_level[level-1])

        use_orig_features = True
        stacker_models = self.get_models(self.hyperparameters)

        stacker_models = [
            StackerEnsembleModel(path=self.path, name=stacker_model.name + '_STACKER_l' + str(level), model_base=stacker_model, base_model_names=base_model_names,
                                 base_model_paths_dict=base_model_paths, base_model_types_dict=base_model_types, use_orig_features=use_orig_features,
                                 num_classes=self.num_classes)
            for stacker_model in stacker_models]
        X_train_init = self.get_inputs_to_stacker(X, level_start=0, level_end=level, fit=True)

        self.train_multi(X_train=X_train_init, y_train=y, X_test=None, y_test=None, models=stacker_models, hyperparameter_tune=False, feature_prune=False, level=level)

    def stack_new_level_aux(self, X, y, level):
        stack_loc = self.models_level_auxiliary
        X_train_stack_preds = self.get_inputs_to_stacker(X, level_start=0, level_end=level, fit=True)
        self.generate_weighted_ensemble(X=X_train_stack_preds, y=y, level=level, k_fold=0, stack_loc=stack_loc)

        # self.generate_stack_log_reg(X=X_train_stack_preds, y=y, level=level, k_fold=0, stack_loc=stack_loc)
        # self.generate_stack_log_reg(X=X_train_stack_preds, y=y, level=level, k_fold=self.kfolds, stack_loc=stack_loc)

    def generate_weighted_ensemble(self, X, y, level, k_fold=0, stack_loc=None):
        weighted_ensemble_model = WeightedEnsembleModel(path=self.path, name='weighted_ensemble_l' + str(level), base_model_names=self.models_level[level-1],
                                                        base_model_paths_dict=self.model_paths, base_model_types_dict=self.model_types,
                                                        num_classes=self.num_classes)

        self.train_multi(X_train=X, y_train=y, X_test=None, y_test=None, models=[weighted_ensemble_model], hyperparameter_tune=False, feature_prune=False, stack_loc=stack_loc, kfolds=k_fold, level=level)
        self.model_best = weighted_ensemble_model.name  # TODO: Make it the max val score model!

    def generate_stack_log_reg(self, X, y, level, k_fold=0, stack_loc=None):
        base_model_names, base_model_paths, base_model_types = self.get_models_info(model_names=self.models_level[level-1])
        stacker_model_lr = get_preset_stacker_model(path=self.path, problem_type=self.problem_type, objective_func=self.objective_func, num_classes=self.num_classes)
        name_new = stacker_model_lr.name + '_STACKER_k' + str(k_fold) + '_l' + str(level)

        stacker_model_lr = StackerEnsembleModel(path=self.path, name=name_new, model_base=stacker_model_lr, base_model_names=base_model_names, base_model_paths_dict=base_model_paths, base_model_types_dict=base_model_types,
                                                use_orig_features=False,
                                                num_classes=self.num_classes)

        self.train_multi(X_train=X, y_train=y, X_test=None, y_test=None, models=[stacker_model_lr], hyperparameter_tune=False, feature_prune=False, stack_loc=stack_loc, kfolds=k_fold, level=level)

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

    def get_inputs_to_model(self, model, X, level_start, fit=False):
        if type(model) == str:
            model = self.load_model(model)
        model_level = self.get_model_level(model.name)
        if model_level >= 1:
            X = self.get_inputs_to_stacker(X=X, level_start=level_start, level_end=model_level-1, fit=fit)
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

    def load_model(self, model_name: str):
        if self.low_memory:
            return self.model_types[model_name].load(path=self.model_paths[model_name], reset_paths=self.reset_paths)
        else:
            return self.models[model_name]

    def _get_dummy_stacker(self, level, use_orig_features=True):
        model_names = self.models_level[level - 1]
        dummy_stacker = StackerEnsembleModel(
            path='', name='',
            model_base=AbstractModel(path='', name='', model=None, problem_type=self.problem_type, objective_func=self.objective_func),
            base_model_names=model_names, base_model_paths_dict=self.model_paths,
            base_model_types_dict=self.model_types, use_orig_features=use_orig_features, num_classes=self.num_classes
        )
        return dummy_stacker

    def get_models_info(self, model_names):
        model_names = copy.deepcopy(model_names)
        model_paths = {model_name: self.model_paths[model_name] for model_name in model_names}
        model_types = {model_name: self.model_types[model_name] for model_name in model_names}
        return model_names, model_paths, model_types

    def leaderboard(self):
        model_names = self.model_names
        score_val = []
        fit_time = []
        pred_time = []
        stack_level = []
        for model_name in model_names:
            score_val.append(self.model_performance.get(model_name))
            fit_time.append(self.model_fit_times.get(model_name))
            pred_time.append(self.model_pred_times.get(model_name))
            stack_level.append(self.get_model_level(model_name))
        df = pd.DataFrame(data={
            'model': model_names,
            'score_val': score_val,
            'fit_time': fit_time,
            'pred_time': pred_time,
            'stack_level': stack_level,
        })
        df_sorted = df.sort_values(by=['score_val', 'model'], ascending=False)
        return df_sorted

    def info(self):
        model_count = len(self.model_names)
        if self.model_best is not None:
            best_model = self.model_best
        else:
            best_model = self.model_best_core
        best_model_score_val = self.model_performance.get(best_model)
        # fit_time = None
        stack_levels = self.max_level
        problem_type = self.problem_type
        objective_func = self.objective_func.name
        # TODO:
        #  Disk size of models
        #  Raw feature count
        #  Train row count
        #  num_classes
        #  Kfolds
        #  HPO time
        #  Bag time
        #  Feature prune time
        #  Exception count / models failed count
        #  True model count (models * kfold)
        #  Best Model Error
        #  AutoGluon version fit on
        #  Max memory usage
        #  CPU count used / GPU count used
        #  Date of fit

        info = {
            'model_count': model_count,
            'best_model': best_model,
            'best_model_score_val': best_model_score_val,
            # 'fit_time': fit_time,
            'stack_levels': stack_levels,
            'problem_type': problem_type,
            'objective_func': objective_func,
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
