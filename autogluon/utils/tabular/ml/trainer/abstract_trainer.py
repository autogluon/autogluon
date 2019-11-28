from pandas import DataFrame, Series
from typing import List
import numpy as np
import pandas as pd
import copy
import time
import traceback
from collections import defaultdict
from sklearn.model_selection import train_test_split


from ..constants import BINARY, MULTICLASS, REGRESSION
from ...utils.loaders import load_pkl
from ...utils.savers import save_pkl
from ..utils import get_pred_from_proba, generate_kfold
from ..models.abstract.abstract_model import AbstractModel
from ..tuning.feature_pruner import FeaturePruner
from ..models.tabular_nn.tabular_nn_model import TabularNeuralNetModel
from ...metrics import accuracy, root_mean_squared_error, scorer_expects_y_pred
from ..models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from ..tuning.ensemble_selection import EnsembleSelection
from ..trainer.model_presets.presets import get_preset_stacker_model
from ..models.ensemble.stacker_ensemble_model import StackerEnsembleModel
from ..models.ensemble.weighted_ensemble_model import WeightedEnsembleModel


# TODO: Dynamic model loading for ensemble models during prediction, only load more models if prediction is uncertain. This dynamically reduces inference time.
class AbstractTrainer:
    trainer_file_name = 'trainer.pkl'

    def __init__(self, path: str, problem_type: str, scheduler_options=None, objective_func=None,
                 num_classes=None, low_memory=False, feature_types_metadata={}, kfolds=0, stack_levels=0,
                 compute_feature_importance=False):
        self.path = path
        self.problem_type = problem_type
        self.feature_types_metadata = feature_types_metadata
        if objective_func is not None:
            self.objective_func = objective_func
        elif self.problem_type == BINARY:
            self.objective_func = accuracy
        elif self.problem_type == MULTICLASS:
            self.objective_func = accuracy
        else:
            self.objective_func = root_mean_squared_error

        self.objective_func_expects_y_pred = scorer_expects_y_pred(scorer=self.objective_func)

        self.num_classes = num_classes
        self.low_memory = low_memory
        self.compute_feature_importance = compute_feature_importance
        self.kfolds = kfolds  # int number of folds to do model bagging, < 2 means disabled
        self.bagged_mode = True if self.kfolds >= 2 else False
        if self.bagged_mode:
            self.stack_levels = stack_levels
            self.stack_mode = True if self.stack_levels >= 1 else False  # TODO: Add as param, only do if bagged_mode = True
        else:
            self.stack_levels = 0
            self.stack_mode = False

        # self.models_level[0] # Includes base models
        # self.models_level[1] # Stacker level 1, includes weighted ensembles of level 0 (base)
        # self.models_level[2] # Stacker level 2, includes weighted ensembles of level 1
        self.models_level = defaultdict(list)
        self.models_level_auxiliary = defaultdict(list)

        self.model_best = None

        self.model_performance = {}
        self.model_paths = {}
        self.model_types = {}
        self.model_fit_times = {}
        self.model_pred_times = {}
        self.models = {}
        self.model_weights = None
        self.reset_paths = False
        self.feature_importance = {}
        # Things stored
        self.hpo_results = {} # Stores summary of HPO process
        self.hpo_model_names = [] # stores additional models produced during HPO
        # Scheduler attributes:
        if scheduler_options is not None:
            self.scheduler_func = scheduler_options[0] # unpack tuple
            self.scheduler_options = scheduler_options[1]
        else:
            self.scheduler_func = None
            self.scheduler_options = None
        # nthreads_per_trial = self.scheduler_options['resource']['num_cpus']
        # ngpus_per_trial = self.scheduler_options['resource']['num_gpus']

    @property
    def model_names(self):
        model_names = []
        levels = np.sort(list(self.models_level.keys()))
        print(levels)
        for level in levels:
            model_names += self.models_level[level]
        return model_names

    @property
    def max_level(self):
        return np.sort(list(self.models_level.keys()))[-1]

    @property
    def max_level_auxiliary(self):
        return np.sort(list(self.models_level_auxiliary.keys()))[-1]

    def get_models(self, hyperparameters):
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

    # Note: This should not be used for time-series data
    @staticmethod
    def get_cv(X, y, n_splits, model: AbstractModel, random_state=0):
        kfolds = generate_kfold(X, n_splits, random_state)
        models = []
        oof = []

        print('training models...')
        for i, kfold in enumerate(kfolds):
            name = model.name + '_cv_' + str(i+1) + '_of_' + str(n_splits)
            path = model.path + 'cv_' + str(i + 1) + '_of_' + str(n_splits) + '/'

            model_cv = copy.deepcopy(model)
            model_cv.name = name
            model_cv.path = path

            # X_train, y_train, X_test, y_test = kfold
            train_index, test_index = kfold
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model_cv.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
            print(model_cv.score(X=X_test, y=y_test))
            models.append(model_cv)

            y_pred_prob = model_cv.predict_proba(X=X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            oof_part = pd.DataFrame(data=y_pred_prob, index=X_test.index)
            oof.append(oof_part)
        oof_pred_proba = pd.concat(oof, ignore_index=False)
        oof_pred_proba = oof_pred_proba.reindex(X.index)
        print(oof_pred_proba)
        return oof_pred_proba, models

    def train(self, X_train, y_train, X_test=None, y_test=None, hyperparameter_tune=True, feature_prune=False,
              holdout_frac=0.1, hyperparameters= {}):
        raise NotImplementedError

    def train_single(self, X_train, y_train, X_test, y_test, model, objective_func=accuracy):
        print('fitting', model.name, '...')
        model.feature_types_metadata = self.feature_types_metadata # TODO: move this into model creation process?
        model_fit_kwargs = {}
        if self.scheduler_options is not None:
            model_fit_kwargs = {'num_cpus': self.scheduler_options['resource']['num_cpus'],
                'num_gpus': self.scheduler_options['resource']['num_gpus'] } # Additional configurations for model.fit
        if self.bagged_mode:
            if type(model) != BaggedEnsembleModel:
                model = BaggedEnsembleModel(path=model.path[:-(len(model.name) + 1)], name=model.name + '_BAGGED', model_base=model)
            model.fit(X=X_train, y=y_train, k_fold=self.kfolds, **model_fit_kwargs)  # TODO: k_fold should be a parameter somewhere. Should it be a param to BaggedEnsembleModel?
        else:
            model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, **model_fit_kwargs)

        if self.compute_feature_importance:
            self.feature_importance[model.name] = self._compute_model_feature_importance(model, X_test, y_test)
        return model

    def _compute_model_feature_importance(self, model, X_test, y_test):
        # Excluding vectorizers features from evaluation because usually there are too many of these
        vectorizer_cols = [] if 'vectorizers' not in model.feature_types_metadata else model.feature_types_metadata['vectorizers']
        features_to_use = [col for col in X_test.columns if col not in vectorizer_cols]
        print(f'Calculating feature importance for the following features: {features_to_use}')
        feature_importance = model.debug_feature_gain(X_test=X_test, Y_test=y_test, model=model, features_to_use=features_to_use)
        return feature_importance

    def train_single_full(self, X_train, y_train, X_test, y_test, model: AbstractModel, feature_prune=False, hyperparameter_tune=True):
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
                    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options))
            except Exception as err:
                traceback.print_tb(err.__traceback__)
                print('Warning: Exception caused ' + model.name + ' to fail during hyperparameter tuning... Skipping model.')
                print(err)
                del model
            else:
                self.hpo_model_names += list(sorted(hpo_models.keys()))
                self.model_paths.update(hpo_models)
                self.model_performance.update(hpo_model_performances)
                self.hpo_results[model.name] = hpo_results
                self.model_types.update({name: type(model) for name in sorted(hpo_models.keys())})
        else:
            self.train_and_save(X_train, y_train, X_test, y_test, model)
        self.save()

    def train_multi(self, X_train, y_train, X_test, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False):
        for i, model in enumerate(models):
            self.train_single_full(X_train, y_train, X_test, y_test, model, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
        if self.bagged_mode:  # TODO: Maybe toggle this based on if we have sufficient time left in our time budget after HPO
            # TODO: Maybe generate weighted_ensemble prior to bagging, and only bag models which were given weight in the initial weighted_ensemble
            for i, hpo_model_name in enumerate(self.hpo_model_names):
                model_hpo = self.load_model(hpo_model_name)
                if type(model_hpo) == TabularNeuralNetModel:  # TODO: Remove this after fixing TabularNeuralNetModel
                    model_hpo = model_hpo.create_unfit_copy()
                model_bagged = BaggedEnsembleModel(path=model_hpo.path[:-(len(model_hpo.name) + 1)], name=model_hpo.name + '_' + str(i) + '_BAGGED', model_base=model_hpo)
                # TODO: Throws exception on Neural Network since trained object is not pickle-able. Fix this to enable bagging for NN by creating new base model in BaggedEnsembleModel with trained model's hyperparams
                self.train_and_save(X_train, y_train, X_test, y_test, model_bagged)
                self.save()
        else:
            self.models_level[0] += self.hpo_model_names  # Update model list with (potentially empty) list of new models created during HPO
        unique_names = []
        for item in self.models_level[0]:
            if item not in unique_names: unique_names.append(item)
        self.models_level[0] = unique_names # make unique and preserve order

    # TODO: Handle case where all models have negative weight, currently crashes due to pruning
    def train_multi_and_ensemble(self, X_train, y_train, X_test, y_test, models: List[AbstractModel],
                                 hyperparameter_tune=True, feature_prune=False):
        self.train_multi(X_train, y_train, X_test, y_test, models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
        # if not hyperparameter_tune: # TODO: we store and print model_performance after HPO
        for model_name in self.models_level[0]:
            if model_name not in self.model_performance:
                model = self.load_model(model_name)
                self.model_performance[model_name] = model.score(X_test, y_test)
            print("Performance of %s model: %s" % (model_name, self.model_performance[model_name]))
        if len(self.models_level[0]) == 0:
            raise ValueError('AutoGluon did not successfully train any models')

        # TODO: Add validation oof score!
        # TODO: Move up 1 level, use the preprocessing directly to get pred_proba! Will be much faster!
        if self.bagged_mode:
            self.generate_weighted_ensemble(X=X_train, y=y_train, level=0)
        else:
            self.generate_weighted_ensemble(X=X_test, y=y_test, level=0)

        if self.stack_mode:
            self.compute_optimal_stacking_ensemble(models=self.models_level[0], X=X_train, y=y_train, level=0)
            self.generate_weighted_ensemble(X=X_train, y=y_train, level=1)

        # print('Score of weighted ensemble:', ensemble_weighted_score)
        # self.model_performance['weighted_ensemble'] = ensemble_weighted_score
        # print('optimal weights:', self.model_weights)

        # TODO: Consider having this be a separate call outside of train, to use on a fitted trainer object
        if self.compute_feature_importance:
            self._compute_ensemble_feature_importance()
            with pd.option_context('display.max_rows', 10000, 'display.max_columns', 10):
                print('Ensemble feature importance:')
                print(self.feature_importance['weighted_ensemble'].sort_values(ascending=False))

        self.save()

    def generate_weighted_ensemble(self, X, y, level):
        # TODO: Add validation oof score!
        # TODO: Move up 1 level, use the preprocessing directly to get pred_proba! Will be much faster!
        model_weights = self.compute_optimal_voting_ensemble_weights(models=self.models_level[level], X=X, y=y, bagged_mode=self.bagged_mode)
        weighted_ensemble_model = WeightedEnsembleModel(path=self.path, name='weighted_ensemble_l' + str(level+1), base_model_names=self.models_level[level], base_model_paths_dict=self.model_paths, base_model_types_dict=self.model_types, base_model_weights=model_weights)
        self.save_model(weighted_ensemble_model)
        self.models_level_auxiliary[level].append(weighted_ensemble_model.name)  # TODO: level+1, update preprocessing!
        self.model_paths[weighted_ensemble_model.name] = weighted_ensemble_model.path
        self.model_types[weighted_ensemble_model.name] = type(weighted_ensemble_model)
        self.model_best = weighted_ensemble_model.name

    # TODO: Remove all feature importance code, it is outdated and has to be reworked
    def _compute_ensemble_feature_importance(self):
        norm_model_weights = self.model_weights / np.sum(self.model_weights)
        model_name_to_weight = {name: weight for name, weight in zip(self.models_level[0], norm_model_weights)}
        models_feature_importance = pd.DataFrame()
        for model in self.models_level[0]:
            models_feature_importance[model] = self.feature_importance[model] * model_name_to_weight[model]
        models_feature_importance.fillna(0, inplace=True)
        self.feature_importance['weighted_ensemble'] = models_feature_importance.sum(axis=1)

    def train_and_save(self, X_train, y_train, X_test, y_test, model: AbstractModel):
        print('training', model.name)
        try:
            fit_start_time = time.time()
            model = self.train_single(X_train, y_train, X_test, y_test, model, objective_func=self.objective_func)
            fit_end_time = time.time()
            if type(model) == BaggedEnsembleModel:
                score = model.score_with_y_pred_proba(y=y_train, y_pred_proba=model.oof_pred_proba)
            else:
                score = model.score(X=X_test, y=y_test)
            pred_end_time = time.time()
        except Exception as err:
            traceback.print_tb(err.__traceback__)
            print('Warning: Exception caused ' + model.name + ' to fail during training... Skipping model.')
            print(err)
            del model
        else:
            self.models_level[0].append(model.name)
            self.model_performance[model.name] = score
            self.model_paths[model.name] = model.path
            self.model_types[model.name] = type(model)
            print('Score of ' + model.name + ':', score)
            print('Fit Runtime:  ', model.name, '=', fit_end_time - fit_start_time, 's')
            print('Score Runtime:', model.name, '=', pred_end_time - fit_end_time, 's')
            # TODO: Should model have fit-time/pred-time information?
            # TODO: Add to HPO
            self.model_fit_times[model.name] = fit_end_time - fit_start_time
            self.model_pred_times[model.name] = pred_end_time - fit_end_time
            self.save_model(model=model)
            if self.low_memory:
                del model

    def predict(self, X):
        X = self.get_inputs_to_model(model_name=self.model_best, X=X, level_start=0, fit=False)  # TODO: Move to predict_model / predict_proba_model?
        return self.predict_model(X, self.model_best)
        # return self.predict_voting_ensemble(models=self.models_level[0], X_test=X, weights=self.model_weights)

    def predict_proba(self, X):
        X = self.get_inputs_to_model(model_name=self.model_best, X=X, level_start=0, fit=False)
        return self.predict_proba_model(X, self.model_best)
        # return self.predict_proba_voting_ensemble(models=self.models_level[0], X_test=X, weights=self.model_weights)

    def predict_model(self, X, model):
        if type(model) == str:
            model = self.load_model(model)
        return model.predict(X=X)

    def predict_proba_model(self, X, model):
        if type(model) == str:
            model = self.load_model(model)
        return model.predict_proba(X=X)

    def get_inputs_to_model(self, model_name, X, level_start, fit=False):
        model_level = self.get_model_level(model_name)
        # if self.model_types[model_name] == WeightedEnsembleModel:
            # TODO: Hack to get it working with WeightedEnsembleModel due to it being different from other models
            #  Eventually convert WeightedEnsembleModel to same format as other stackers and remove
            # model_level -= 1
        if model_level >= 1:
            X = self.get_inputs_to_stacker(X=X, level_start=level_start, level_end=model_level, fit=fit)
        return X

    # TODO: REWORK, NOT ACCURATE! Use self.model_best
    def score(self, X, y, weights=None):
        if weights is None:
            weights = self.model_weights
        elif weights == 'voting':
            weights = [1 / len(self.models_level[0])] * len(self.models_level[0])
        if self.objective_func_expects_y_pred:
            y_pred_ensemble = self.predict_voting_ensemble(models=self.models_level[0], X_test=X, weights=weights)
            return self.objective_func(y, y_pred_ensemble)
        else:
            y_pred_proba_ensemble = self.predict_proba_voting_ensemble(models=self.models_level[0], X_test=X, weights=weights)
            return self.objective_func(y, y_pred_proba_ensemble)

    def score_with_y_pred_proba(self, y, y_pred_proba):
        if self.objective_func_expects_y_pred:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            return self.objective_func(y, y_pred)
        else:
            return self.objective_func(y, y_pred_proba)

    def score_with_y_pred_proba_weighted(self, y, y_pred_probas, weights=None):
        if weights is None:
            weights = self.model_weights
        elif weights == 'voting':
            weights = [1 / len(self.models_level[0])] * len(self.models_level[0])
        preds_norm = [pred * weight for pred, weight in zip(y_pred_probas, weights)]
        preds_ensemble = np.sum(preds_norm, axis=0)
        if self.objective_func_expects_y_pred:
            y_pred = get_pred_from_proba(y_pred_proba=preds_ensemble, problem_type=self.problem_type)
            return self.objective_func(y, y_pred)
        else:
            return self.objective_func(y, preds_ensemble)

    # TODO: Remove
    def score_bagged(self, y, weights=None):
        oof_y_pred_probas = []
        for model_name in self.models_level[0]:
            model = self.load_model(model_name)
            oof_y_pred_probas.append(model.oof_pred_proba)
        return self.score_with_y_pred_proba_weighted(y=y, y_pred_probas=oof_y_pred_probas, weights=weights)

    def autotune(self, X_train, X_holdout, y_train, y_holdout, model_base: AbstractModel):
        feature_pruner = FeaturePruner(model_base=model_base)
        X_train, X_test, y_train, y_test = self.generate_train_test_split(X_train, y_train)
        feature_pruner.tune(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_holdout=X_holdout, y_holdout=y_holdout)
        features_to_keep = feature_pruner.features_in_iter[feature_pruner.best_iteration]
        print(features_to_keep)
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

    def predict_proba_voting_ensemble(self, models, X_test, weights=None):
        if weights is None:
            weights = [1/len(models)]*len(models)
        model_index_to_ignore = []
        for index, weight in enumerate(weights):
            if weight == 0:
                model_index_to_ignore.append(index)
        models_to_predict_on = [model for index, model in enumerate(models) if index not in model_index_to_ignore]
        models_to_predict_on_weights = [weight for index, weight in enumerate(weights) if index not in model_index_to_ignore]
        pred_probas = self.pred_proba_predictions(models=models_to_predict_on, X_test=X_test)
        preds_norm = [pred * weight for pred, weight in zip(pred_probas, models_to_predict_on_weights)]
        preds_ensemble = np.sum(preds_norm, axis=0)
        return preds_ensemble

    def predict_voting_ensemble(self, models, X_test, weights=None):
        y_pred_proba_ensemble = self.predict_proba_voting_ensemble(models=models, X_test=X_test, weights=weights)
        y_pred_ensemble = get_pred_from_proba(y_pred_proba=y_pred_proba_ensemble, problem_type=self.problem_type)
        return y_pred_ensemble

    def predict_voting_ensemble_optimize(self, models, X_test, y_test):
        optimal_weights = self.compute_optimal_voting_ensemble_weights(models=models, X=X_test, y=y_test)
        return self.predict_voting_ensemble(models=models, X_test=X_test, weights=optimal_weights)

    # Ensemble Selection (https://dl.acm.org/citation.cfm?id=1015432)
    def compute_optimal_voting_ensemble_weights(self, models, X, y, bagged_mode=False):
        if bagged_mode:
            pred_probas = []
            for model in models:
                if type(model) is str:
                    model = self.load_model(model)
                pred_probas.append(model.oof_pred_proba)
        else:
            pred_probas = self.pred_proba_predictions(models=models, X_test=X)
        ensemble_selection = EnsembleSelection(ensemble_size=100, problem_type=self.problem_type, metric=self.objective_func)
        ensemble_selection.fit(predictions=pred_probas, labels=y, identifiers=None)
        return ensemble_selection.weights_

    def get_inputs_to_stacker(self, X, level_start, level_end, fit=False):
        if fit:
            model_names = self.models_level[level_end-1]
            dummy_stacker = StackerEnsembleModel(path='', name='', stacker_model=AbstractModel(path='', name='', model=None, problem_type=self.problem_type, objective_func=self.objective_func), base_model_names=model_names, base_model_paths_dict=self.model_paths, base_model_types_dict=self.model_types, use_orig_features=True, num_classes=self.num_classes)
            X = dummy_stacker.preprocess(X=X, fit=True, compute_base_preds=True)
        else:
            for level in range(level_start, level_end):
                if level >= 1:
                    model_names_last = self.models_level[level-1]
                    dummy_stacker_last = StackerEnsembleModel(path='', name='', stacker_model=AbstractModel(path='', name='', model=None, problem_type=self.problem_type, objective_func=self.objective_func), base_model_names=model_names_last, base_model_paths_dict=self.model_paths, base_model_types_dict=self.model_types, use_orig_features=True, num_classes=self.num_classes)
                    cols_to_drop = dummy_stacker_last.stack_columns
                else:
                    cols_to_drop = []

                model_names = self.models_level[level]
                dummy_stacker = StackerEnsembleModel(path='', name='', stacker_model=AbstractModel(path='', name='', model=None, problem_type=self.problem_type, objective_func=self.objective_func), base_model_names=model_names, base_model_paths_dict=self.model_paths, base_model_types_dict=self.model_types, use_orig_features=True, num_classes=self.num_classes)
                X = dummy_stacker.preprocess(X=X, fit=False, compute_base_preds=True)

                if len(cols_to_drop) > 0:
                    X = X.drop(cols_to_drop, axis=1)
        return X

    def compute_optimal_stacking_ensemble(self, models, X, y, level=0):
        pred_probas = []
        model_names = []
        base_model_names, base_model_paths, base_model_types = self.get_models_info(model_names=self.models_level[level])
        for model in models:
            if type(model) is str:
                model = self.load_model(model)
            pred_probas.append(model.oof_pred_proba)
            model_names.append(model.name)

        use_orig_features = True
        stacker_models = self.get_models(self.hyperparameters)

        stacker_models = [StackerEnsembleModel(path=self.path, name=stacker_model.name + '_STACKER_l' + str(level+1), stacker_model=stacker_model, base_model_names=base_model_names, base_model_paths_dict=base_model_paths, base_model_types_dict=base_model_types, use_orig_features=use_orig_features, num_classes=self.num_classes)
                          for stacker_model in stacker_models]
        X_train_init = stacker_models[0].preprocess(X=X, fit=True, compute_base_preds=True)
        for stacker_model in stacker_models:
            stacker_model.fit(X=X_train_init, y=y, compute_base_preds=False, k_fold=self.kfolds, random_state=level+1)
            # print('bagged stacker fit!')
            # score = bagged_stacker_model.score_with_y_pred_proba(y=y, y_pred_proba=bagged_stacker_model.oof_pred_proba)
            score = stacker_model.model.score_with_y_pred_proba(y=y, y_pred_proba=stacker_model.oof_pred_proba)
            # print('bagged_stacker_score:', score)

            self.save_model(stacker_model)
            self.models_level[level+1].append(stacker_model.name)
            self.model_paths[stacker_model.name] = stacker_model.path
            self.model_types[stacker_model.name] = type(stacker_model)
            self.model_performance[stacker_model.name] = score

        # TODO: MOVE THIS OUT
        self.generate_stack_log_reg(X_train_init=X_train_init, y=y, level=0, k_fold=0)
        self.generate_stack_log_reg(X_train_init=X_train_init, y=y, level=0, k_fold=self.kfolds)
        X_train_init = self.get_inputs_to_stacker(X, level_start=0, level_end=2, fit=True)
        self.generate_stack_log_reg(X_train_init=X_train_init, y=y, level=1, k_fold=0)
        self.generate_stack_log_reg(X_train_init=X_train_init, y=y, level=1, k_fold=self.kfolds)

    def generate_stack_log_reg(self, X_train_init, y, level, k_fold=0):
        base_model_names, base_model_paths, base_model_types = self.get_models_info(model_names=self.models_level[level])
        # base_model_names = copy.deepcopy(self.models_level[level])
        # base_model_paths = copy.deepcopy(self.model_paths)
        # base_model_types = copy.deepcopy(self.model_types)
        stacker_model_lr = get_preset_stacker_model(path=self.path, problem_type=self.problem_type, objective_func=self.objective_func, num_classes=self.num_classes)
        name_new = stacker_model_lr.name + '_STACKER_l' + str(level+1) + '_k' + str(k_fold)

        stacker_model_lr = StackerEnsembleModel(path=self.path, name=name_new, stacker_model=stacker_model_lr, base_model_names=base_model_names, base_model_paths_dict=base_model_paths, base_model_types_dict=base_model_types,
                                                use_orig_features=False,
                                                num_classes=self.num_classes)
        stacker_model_lr.fit(X=X_train_init, y=y, compute_base_preds=False, k_fold=k_fold, random_state=level+1)
        # score = stacker_model_lr.model.score_with_y_pred_proba(y=y, y_pred_proba=stacker_model_lr.oof_pred_proba)
        # print('log_reg_stacker_score:', score)
        self.save_model(stacker_model_lr)
        self.models_level_auxiliary[level+1].append(stacker_model_lr.name)
        self.model_paths[stacker_model_lr.name] = stacker_model_lr.path
        self.model_types[stacker_model_lr.name] = type(stacker_model_lr)
        if stacker_model_lr.bagged_mode:
            score = stacker_model_lr.score_with_y_pred_proba(y=y, y_pred_proba=stacker_model_lr.oof_pred_proba)
            self.model_performance[stacker_model_lr.name] = score

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

    def get_models_info(self, model_names):
        model_names = copy.deepcopy(model_names)
        model_paths = {model_name: self.model_paths[model_name] for model_name in model_names}
        model_types = {model_name: self.model_types[model_name] for model_name in model_names}
        return model_names, model_paths, model_types

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
