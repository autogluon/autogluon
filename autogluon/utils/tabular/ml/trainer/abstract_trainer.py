from pandas import DataFrame, Series
from typing import List
import numpy as np
import pandas as pd
import copy
import time
import traceback
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


# TODO: Dynamic model loading for ensemble models during prediction, only load more models if prediction is uncertain. This dynamically reduces inference time.
class AbstractTrainer:
    trainer_file_name = 'trainer.pkl'

    def __init__(self, path: str, problem_type: str, scheduler_options=None, objective_func=None,
                 num_classes=None, low_memory=False, feature_types_metadata={},
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
        self.model_names = []
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
        if type(model) == BaggedEnsembleModel:
            X = pd.concat([X_train, X_test], ignore_index=True)  # TODO: Consider doing earlier so this isn't repeated for each model
            y = pd.concat([y_train, y_test], ignore_index=True)
            model.fit(X=X, y=y, k_fold=10, **model_fit_kwargs)  # TODO: k_fold should be a parameter somewhere. Should it be a param to BaggedEnsembleModel?
        else:
            model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, **model_fit_kwargs)

        if self.compute_feature_importance:
            self.feature_importance[model.name] = self._compute_model_feature_importance(model, X_test, y_test)

    def _compute_model_feature_importance(self, model, X_test, y_test):
        # Excluding vectorizers features from evaluation because usually there are too many of these
        vectorizer_cols = [] if 'vectorizers' not in model.feature_types_metadata else model.feature_types_metadata['vectorizers']
        features_to_use = [col for col in X_test.columns if col not in vectorizer_cols]
        print(f'Calculating feature importance for the following features: {features_to_use}')
        feature_importance = model.debug_feature_gain(X_test=X_test, Y_test=y_test, model=model, features_to_use=features_to_use)
        return feature_importance

    def train_single_full(self, X_train, y_train, X_test, y_test, model: AbstractModel, feature_prune=False, hyperparameter_tune=True):
        bagged_mode = type(model) == BaggedEnsembleModel
        model.feature_types_metadata = self.feature_types_metadata  # TODO: Don't set feature_types_metadata here
        if bagged_mode:  # TODO: Don't set feature_types_metadata here
            model.model_base.feature_types_metadata = self.feature_types_metadata
        if feature_prune:
            self.autotune(X_train=X_train, X_holdout=X_test, y_train=y_train, y_holdout=y_test, model_base=model)  # TODO: Update to use CV instead of holdout
        if hyperparameter_tune:
            if self.scheduler_func is None or self.scheduler_options is None:
                raise ValueError("scheduler_options cannot be None when hyperparameter_tune = True")
            # Moved split into lightGBM. TODO: need to do same for other models that use their own splits as well. Old code was:  model.hyperparameter_tune(pd.concat([X_train, X_test], ignore_index=True), pd.concat([y_train, y_test], ignore_index=True))
            # hpo_models (dict): keys = model_names, values = model_paths
            try:  # TODO: Make exception handling more robust? Return successful HPO models?
                if bagged_mode:
                    model = model.model_base
                    # TODO: If HPO, do bagging at the end, current it just skips them.
                    hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X_train=X_train, X_test=X_test,
                                                                        Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options))
                else:
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
        bagged_mode = type(models[0]) == BaggedEnsembleModel  # TODO: trainer parameter?
        if bagged_mode:  # TODO: Maybe toggle this based on if we have sufficient time left in our time budget after HPO
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
            self.model_names += self.hpo_model_names  # Update model list with (potentially empty) list of new models created during HPO
        unique_names = []
        for item in self.model_names:
            if item not in unique_names: unique_names.append(item)
        self.model_names = unique_names # make unique and preserve order

    # TODO: Handle case where all models have negative weight, currently crashes due to pruning
    def train_multi_and_ensemble(self, X_train, y_train, X_test, y_test, models: List[AbstractModel],
                                 hyperparameter_tune=True, feature_prune=False):
        self.train_multi(X_train, y_train, X_test, y_test, models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
        # if not hyperparameter_tune: # TODO: we store and print model_performance after HPO
        for model_name in self.model_names:
            if model_name not in self.model_performance:
                model = self.load_model(model_name)
                self.model_performance[model_name] = model.score(X_test, y_test)
            print("Performance of %s model: %s" % (model_name, self.model_performance[model_name]))
        if len(self.model_names) == 0:
            raise ValueError('AutoGluon did not successfully train any models')

        bagged_mode = self.model_types[self.model_names[0]] == BaggedEnsembleModel
        if bagged_mode:
            X_test = pd.concat([X_train, X_test], ignore_index=True)
            y_test = pd.concat([y_train, y_test], ignore_index=True)

        if not bagged_mode:
            ensemble_voting_score = self.score(X=X_test, y=y_test, weights='voting')  # TODO: Might want to remove this to avoid needless computation
        else:
            ensemble_voting_score = self.score_bagged(y=y_test, weights='voting')

        self.model_performance['ensemble.equal_weights'] = ensemble_voting_score
        print('Score of voting ensemble:', ensemble_voting_score)
        self.model_weights = self.compute_optimal_voting_ensemble_weights(models=self.model_names, X_test=X_test, y_test=y_test, bagged_mode=bagged_mode)

        if not bagged_mode:
            ensemble_weighted_score = self.score(X=X_test, y=y_test)
        else:
            ensemble_weighted_score = self.score_bagged(y=y_test)

        print('Score of weighted ensemble:', ensemble_weighted_score)
        self.model_performance['weighted_ensemble'] = ensemble_weighted_score
        print('optimal weights:', self.model_weights)

        # TODO: Consider having this be a separate call outside of train, to use on a fitted trainer object
        if self.compute_feature_importance:
            self._compute_ensemble_feature_importance()
            with pd.option_context('display.max_rows', 10000, 'display.max_columns', 10):
                print('Ensemble feature importance:')
                print(self.feature_importance['weighted_ensemble'].sort_values(ascending=False))

        self.save()

    def _compute_ensemble_feature_importance(self):
        norm_model_weights = self.model_weights / np.sum(self.model_weights)
        model_name_to_weight = {name: weight for name, weight in zip(self.model_names, norm_model_weights)}
        models_feature_importance = pd.DataFrame()
        for model in self.model_names:
            models_feature_importance[model] = self.feature_importance[model] * model_name_to_weight[model]
        models_feature_importance.fillna(0, inplace=True)
        self.feature_importance['weighted_ensemble'] = models_feature_importance.sum(axis=1)

    def train_and_save(self, X_train, y_train, X_test, y_test, model: AbstractModel):
        print('training', model.name)
        try:
            fit_start_time = time.time()
            self.train_single(X_train, y_train, X_test, y_test, model, objective_func=self.objective_func)
            fit_end_time = time.time()
            if type(model) == BaggedEnsembleModel:
                y = pd.concat([y_train, y_test], ignore_index=True)
                score = model.score_with_y_pred_proba(y=y, y_pred_proba=model.oof_pred_proba)
            else:
                score = model.score(X=X_test, y=y_test)
            pred_end_time = time.time()
        except Exception as err:
            traceback.print_tb(err.__traceback__)
            print('Warning: Exception caused ' + model.name + ' to fail during training... Skipping model.')
            print(err)
            del model
        else:
            self.model_names.append(model.name)
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
        return self.predict_voting_ensemble(models=self.model_names, X_test=X, weights=self.model_weights)

    def predict_proba(self, X):
        return self.predict_proba_voting_ensemble(models=self.model_names, X_test=X, weights=self.model_weights)

    def score(self, X, y, weights=None):
        if weights is None:
            weights = self.model_weights
        elif weights == 'voting':
            weights = [1/len(self.model_names)]*len(self.model_names)
        if self.objective_func_expects_y_pred:
            y_pred_ensemble = self.predict_voting_ensemble(models=self.model_names, X_test=X, weights=weights)
            return self.objective_func(y, y_pred_ensemble)
        else:
            y_pred_proba_ensemble = self.predict_proba_voting_ensemble(models=self.model_names, X_test=X, weights=weights)
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
            weights = [1/len(self.model_names)]*len(self.model_names)
        preds_norm = [pred * weight for pred, weight in zip(y_pred_probas, weights)]
        preds_ensemble = np.sum(preds_norm, axis=0)
        if self.objective_func_expects_y_pred:
            y_pred = get_pred_from_proba(y_pred_proba=preds_ensemble, problem_type=self.problem_type)
            return self.objective_func(y, y_pred)
        else:
            return self.objective_func(y, preds_ensemble)

    def score_bagged(self, y, weights=None):
        oof_y_pred_probas = []
        for model_name in self.model_names:
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
        optimal_weights = self.compute_optimal_voting_ensemble_weights(models=models, X_test=X_test, y_test=y_test)
        return self.predict_voting_ensemble(models=models, X_test=X_test, weights=optimal_weights)

    # Ensemble Selection (https://dl.acm.org/citation.cfm?id=1015432)
    def compute_optimal_voting_ensemble_weights(self, models, X_test, y_test, bagged_mode=False):
        if bagged_mode:
            pred_probas = []
            for model in models:
                if type(model) is str:
                    model = self.load_model(model)
                pred_probas.append(model.oof_pred_proba)
        else:
            pred_probas = self.pred_proba_predictions(models=models, X_test=X_test)
        ensemble_selection = EnsembleSelection(ensemble_size=100, problem_type=self.problem_type, metric=self.objective_func)
        ensemble_selection.fit(predictions=pred_probas, labels=y_test, identifiers=None)
        return ensemble_selection.weights_

    # TODO: Implement stacking
    # def compute_optimal_stacking_ensemble(self, models, X, y):
    #     pred_probas = []
    #     model_names = []
    #     for model in models:
    #         if type(model) is str:
    #             model = self.load_model(model)
    #         pred_probas.append(model.oof_pred_proba)
    #         model_names.append(model.name)
    #
    #     stacker_model = get_preset_stacker_model(path=self.path, problem_type=self.problem_type, objective_func=self.objective_func, num_classes=self.num_classes)
    #     print(stacker_model)
    #     bagged_stacker_model = BaggedEnsembleModel(path=stacker_model.path[:-(len(stacker_model.name) + 1)], name=stacker_model.name + '_BAGGED', model_base=stacker_model)
    #
    #     X_stacker = pd.DataFrame(data=np.asarray(pred_probas).T, columns=model_names)
    #
    #     bagged_stacker_model.fit(X=X_stacker, y=y, k_fold=5, random_state=1)
    #     stacker_model_full = StackerEnsembleModel(path=self.path, name=bagged_stacker_model.name + '_STACKER', stacker_model=bagged_stacker_model, base_model_names=self.model_names, base_model_paths_dict=self.model_paths, base_model_types_dict=self.model_types)
    #     print('bagged stacker fit!')
    #     score = bagged_stacker_model.score_with_y_pred_proba(y=y, y_pred_proba=bagged_stacker_model.oof_pred_proba)
    #     print('bagged_stacker_score:', score)
    #
    #     self.stacker_model_full = stacker_model_full
    #
    #
    #     return bagged_stacker_model

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
