
from pandas import DataFrame, Series
from typing import List
import numpy as np
import pandas as pd
import copy
import random

from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from tabular.utils.loaders import load_pkl
from tabular.utils.savers import save_pkl
from tabular.ml.utils import get_pred_from_proba
from tabular.ml.models.abstract_model import AbstractModel
from tabular.ml.tuning.autotune import AutoTune

from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


class AbstractTrainer:
    trainer_file_name = 'trainer.pkl'

    def __init__(self, path: str, problem_type: str, scheduler_options, objective_func=None, num_classes=None, low_memory=False, feature_types_metadata={}, compute_feature_importance=False):
        self.path = path
        self.problem_type = problem_type
        self.feature_types_metadata = feature_types_metadata
        if objective_func is not None:
            self.objective_func = objective_func
        elif self.problem_type == BINARY:
            self.objective_func = accuracy_score
        elif self.problem_type == MULTICLASS:
            self.objective_func = accuracy_score
        else:
            self.objective_func = mean_absolute_error
        self.num_classes = num_classes
        self.low_memory = low_memory
        self.compute_feature_importance = compute_feature_importance
        self.model_names = []
        self.model_performance = {}
        self.model_paths = {}
        self.model_types = {}
        self.models = {}
        self.model_weights = None
        self.reset_paths = False
        self.feature_importance = {}
        # Things stored
        self.hpo_results = {} # Stores summary of HPO process
        self.hpo_model_names = [] # stores additional models produced during HPO
        # Scheduler attributes:
        self.scheduler_func = scheduler_options[0] # unpack tuple
        self.scheduler_options = scheduler_options[1]

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

    @staticmethod
    def generate_kfold(X, n_splits, random_state=0):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        kf.get_n_splits(X)
        kfolds = []
        for train_index, test_index in kf.split(X):
            kfolds.append([train_index, test_index])
        return kfolds

    # Note: This should not be used for time-series data
    @staticmethod
    def get_cv(X, y, n_splits, model: AbstractModel, random_state=0):
        kfolds = AbstractTrainer.generate_kfold(X, n_splits, random_state)
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

    @staticmethod
    def get_oof_preds(kfolds, kmodels):
        oof = []
        for k, fold in enumerate(kfolds):
            _, _, X_test, y_test = fold
            y_pred_proba = kmodels[k].predict_proba(X_test)
            oof.append(y_pred_proba)
        return oof

    def train(self, X_train, y_train, X_test=None, y_test=None):
        raise NotImplementedError

    def train_single(self, X_train, X_test, y_train, y_test, model, objective_func=accuracy_score):
        print('fitting', model.name, '...')
        model.feature_types_metadata = self.feature_types_metadata # TODO: move this into model creation process?
        model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
        score = model.score(X=X_test, y=y_test)
        print('Score of', model.name, ':', score)
        if self.compute_feature_importance:
            self.feature_importance[model.name] = self._compute_model_feature_importance(model, X_test, y_test)

    def _compute_model_feature_importance(self, model, X_test, y_test):
        # Excluding vectorizers features from evaluation because usually there are too many of these
        vectorizer_cols = [] if 'vectorizers' not in model.feature_types_metadata else model.feature_types_metadata['vectorizers']
        features_to_use = [col for col in X_test.columns if col not in vectorizer_cols]
        print(f'Calculating feature importance for the following features: {features_to_use}')
        feature_importance = model.debug_feature_gain(X_test=X_test, Y_test=y_test, model=model, features_to_use=features_to_use)
        return feature_importance

    def train_single_full(self, X_train, X_test, y_train, y_test, model: AbstractModel, feature_prune=False, hyperparameter_tune=True):
        model.feature_types_metadata = self.feature_types_metadata
        if feature_prune:
            self.autotune(X_train=X_train, X_holdout=X_test, y_train=y_train, y_holdout=y_test, model_base=model)  # TODO: Update to use CV instead of holdout
        if hyperparameter_tune:
            # Moved split into lightGBM. TODO: need to do same for other models that use their own splits as well. Old code was:  model.hyperparameter_tune(pd.concat([X_train, X_test], ignore_index=True), pd.concat([y_train, y_test], ignore_index=True))
            # hpo_models (dict): keys = model_names, values = model_paths
            hpo_models, hpo_results = model.hyperparameter_tune(X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options))
            self.hpo_model_names += list(sorted(hpo_models.keys()))
            self.model_paths.update(hpo_models)
            self.hpo_results[model.name] = hpo_results
            self.model_types.update({name: type(model) for name in sorted(hpo_models.keys())})
        else:
            self.train_and_save(X_train, X_test, y_train, y_test, model)
        self.save()
    
    def train_multi(self, X_train, X_test, y_train, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False):
        for i, model in enumerate(models):
            self.train_single_full(X_train, X_test, y_train, y_test, model, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
        self.model_names += self.hpo_model_names # Update model list with (potentially empty) list of new models created during HPO
        unique_names = []
        for item in self.model_names:
            if item not in unique_names: unique_names.append(item)
        self.model_names = unique_names # make unique and preserve order
    
    # TODO: Handle case where all models have negative weight, currently crashes due to pruning
    def train_multi_and_ensemble(self, X_train, X_test, y_train, y_test, models: List[AbstractModel], hyperparameter_tune=True, feature_prune=False):
        self.train_multi(X_train, X_test, y_train, y_test, models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
        for model_name in self.model_names:
            model = self.load_model(model_name)
            if model is not None:
                print(model_name, model.score(X_test, y_test))

        y_pred_ensemble = self.predict_voting_ensemble(X_test=X_test, models=self.model_names)
        acc_ensemble = self.objective_func(y_true=y_test, y_pred=y_pred_ensemble)
        self.model_performance['ensemble.equal_weights'] = acc_ensemble
        print('Score of ensemble:', acc_ensemble)

        ##########
        # TODO: Comment out if you want to use basic same-weight voting ensemble
        # TODO: Make this maximize optimization function, not logloss
        self.model_weights = self.compute_optimal_voting_ensemble_weights(models=self.model_names, X_test=X_test, y_test=y_test)

        y_pred_ensemble_optimal = self.predict_voting_ensemble(X_test=X_test, models=self.model_names, weights=self.model_weights)
        acc_ensemble_optimize = self.objective_func(y_true=y_test, y_pred=y_pred_ensemble_optimal)
        print('acc ensemble_optimize:', acc_ensemble_optimize)
        self.model_performance['ensemble.optimized'] = acc_ensemble_optimize
        print('optimal weights:', self.model_weights)
        ##########

        # TODO: Consider having this be a separate call outside of train, to use on a fitted trainer object
        if self.compute_feature_importance:
            self._compute_ensemble_feature_importance()
            with pd.option_context('display.max_rows', 10000, 'display.max_columns', 10):
                print('Ensemble feature importance:')
                print(self.feature_importance['ensemble.optimized'].sort_values(ascending=False))

        self.save()

    def _compute_ensemble_feature_importance(self):
        norm_model_weights = self.model_weights / np.sum(self.model_weights)
        model_name_to_weight = {name: weight for name, weight in zip(self.model_names, norm_model_weights)}
        models_feature_importance = pd.DataFrame()
        for model in self.model_names:
            models_feature_importance[model] = self.feature_importance[model] * model_name_to_weight[model]
        models_feature_importance.fillna(0, inplace=True)
        self.feature_importance['ensemble.optimized'] = models_feature_importance.sum(axis=1)

    def train_and_save(self, X_train, X_test, y_train, y_test, model: AbstractModel):
        print('training', model.name)
        self.train_single(X_train, X_test, y_train, y_test, model, objective_func=self.objective_func)
        self.model_names.append(model.name)
        self.model_performance[model.name] = model.score(X_test, y_test)
        self.model_paths[model.name] = model.path
        self.model_types[model.name] = type(model)
        self.save_model(model=model)
        if self.low_memory:
            del model

    @staticmethod
    def train_ensemble(X_train, X_test, y_train, y_test, model_base: AbstractModel, objective_func=accuracy_score):
        oof_pred_proba, models_cv = AbstractTrainer.get_cv(X=X_train, y=y_train, n_splits=5, model=model_base)
        for model in models_cv:
            print(model.score(X=X_test, y=y_test))
        cv_preds = []
        for model in models_cv:
            cv_pred = model.predict_proba(X_test)
            cv_preds.append(cv_pred)
        num_models = len(cv_preds)
        cv_preds_norm = [cv_pred / num_models for cv_pred in cv_preds]
        y_pred_proba_ensemble = np.sum(cv_preds_norm, axis=0)
        y_pred_ensemble = np.argmax(y_pred_proba_ensemble, axis=1)
        print('ensemble:', objective_func(y_true=y_test, y_pred=y_pred_ensemble))
        return oof_pred_proba, y_pred_proba_ensemble, models_cv

    def predict(self, X):
        return self.predict_voting_ensemble(models=self.model_names, X_test=X, weights=self.model_weights)

    def predict_proba(self, X):
        return self.predict_proba_voting_ensemble(models=self.model_names, X_test=X, weights=self.model_weights)

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.objective_func(y_true=y, y_pred=y_pred)

    def autotune(self, X_train, X_holdout, y_train, y_holdout, model_base: AbstractModel):
        autotuner = AutoTune(model_base=model_base)
        X_train, X_test, y_train, y_test = self.generate_train_test_split(X_train, y_train)
        autotuner.tune(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_holdout=X_holdout, y_holdout=y_holdout)
        features_to_keep = autotuner.features_in_iter[autotuner.best_iteration]
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
        pred_probas = self.pred_proba_predictions(models=models, X_test=X_test)
        preds_norm = [pred * weight for pred, weight in zip(pred_probas, weights)]
        preds_ensemble = np.sum(preds_norm, axis=0)
        return preds_ensemble

    def predict_voting_ensemble(self, models, X_test, weights=None):
        y_pred_proba_ensemble = self.predict_proba_voting_ensemble(models=models, X_test=X_test, weights=weights)
        y_pred_ensemble = get_pred_from_proba(y_pred_proba=y_pred_proba_ensemble, problem_type=self.problem_type)
        return y_pred_ensemble

    def predict_voting_ensemble_optimize(self, models, X_test, y_test):
        optimal_weights = self.compute_optimal_voting_ensemble_weights(models=models, X_test=X_test, y_test=y_test)
        return self.predict_voting_ensemble(models=models, X_test=X_test, weights=optimal_weights)

    def compute_optimal_voting_ensemble_weights(self, models, X_test, y_test, iterations=10):
        pred_probas = self.pred_proba_predictions(models=models, X_test=X_test)
        # TODO: Use accuracy instead of logloss?
        if (self.problem_type == BINARY) or (self.problem_type == MULTICLASS):
            def opt_func(weights):
                final_prediction = 0
                for weight, prediction in zip(weights, pred_probas):
                    final_prediction += weight * prediction

                # if len(final_prediction.shape) == 1:
                #     y_pred_ensemble = [1 if pred >= 0.5 else 0 for pred in final_prediction]
                # else:
                #     y_pred_ensemble = np.argmax(final_prediction, axis=1)
                # acc = accuracy_score(y_test, y_pred_ensemble)
                # print(acc)
                if (self.problem_type == MULTICLASS) and (final_prediction.shape[1] != len(set(y_test))):
                    # Need to provide lablels since y_test does not contain all classes.
                    log_loss_val = log_loss(y_test, final_prediction, labels=range(max(final_prediction.shape[1], len(set(y_test)))))
                else:
                    log_loss_val = log_loss(y_test, final_prediction)
                return log_loss_val
        else:
            def opt_func(weights):
                final_prediction = 0
                for weight, prediction in zip(weights, pred_probas):
                    final_prediction += weight * prediction
                log_loss_val = mean_absolute_error(y_test, final_prediction)
                return log_loss_val

        cur_best = None
        # TODO: Parallelize?
        for i in range(iterations):
            starting_values = [random.random() for _ in range(len(models))]
            starting_values = starting_values / np.sum(starting_values)

            res = minimize(opt_func, starting_values, method='Nelder-Mead')

            if cur_best is None:
                cur_best = res
            elif cur_best['fun'] > res['fun']:
                # print('new best!')
                cur_best = res

        best_weights = cur_best['x']

        rerun = False
        rerun_models = []
        # TODO: Crashes when all model weights are negative. Handle this by returning model which always predicts 0 for regression (or most frequent class in binary/multiclass), or raise this as a dedicated error
        for i, weight in enumerate(best_weights):
            if weight < 0:
                print('model', i, 'has negative weight, setting to 0 and re-running optimization...')
                rerun = True
        if rerun:
            print('rerunning...')
            print('Ensemble Score Prior: {best_score}'.format(best_score=cur_best['fun']))
            print('Best Weights Prior: {weights}'.format(weights=cur_best['x']))

            for i, weight in enumerate(best_weights):
                if weight >= 0:
                    rerun_models.append(models[i])
            updated_weights = self.compute_optimal_voting_ensemble_weights(models=rerun_models, X_test=X_test, y_test=y_test, iterations=iterations)
            updated_weight_index = 0
            for i, weight in enumerate(best_weights):
                if weight >= 0:
                    # print('updated weight from', weight, 'to', updated_weights[updated_weight_index])
                    best_weights[i] = updated_weights[updated_weight_index]
                    updated_weight_index += 1
                else:
                    # print('updated weight from', weight, 'to', 0)
                    best_weights[i] = 0
        else:
            print('Ensemble Score: {best_score}'.format(best_score=cur_best['fun']))
            print('Best Weights: {weights}'.format(weights=best_weights))

        return best_weights

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

