import gc, copy
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from skopt.utils import use_named_args # TODO: remove
from skopt import gp_minimize # TODO: remove


# TODO: Move these files:
from tabular.ml.models.abstract_model import AbstractModel
from tabular.ml.utils import construct_dataset
from tabular.callbacks.lgb.callbacks import record_evaluation_custom, early_stopping_custom
from tabular.ml.trainer.abstract_trainer import AbstractTrainer
from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from tabular.ml.tuning.hyperparameters.lgbm_spaces import LGBMSpaces
from tabular.ml.models.utils import lgb_utils
from tabular.ml.tuning.hyperparameters.defaults.lgbm.parameters import get_param_baseline


class LGBModel(AbstractModel):
    def __init__(self, path, name, problem_type, objective_func,
                 num_classes=None, gbm_options={}, features=None, debug=0):
        model = None
        super().__init__(path=path, name=name, model=model, problem_type=problem_type, objective_func=objective_func, features=features, debug=debug)
        self.params = get_param_baseline(problem_type=problem_type, num_classes=num_classes) # get default hyperparameters
        if gbm_options is not None:
            self.params.update(gbm_options) # update with user-specified settings
        
        self.metric_types = self.params['metric'].split(',')
        self.eval_metric_name = self.objective_func.name
        self.is_higher_better = True
        self.best_iteration = None
        self.eval_results = {}
        self.model_name_checkpointing_0 = 'model_checkpoint_0.pkl'
        self.model_name_checkpointing_1 = 'model_checkpoint_1.pkl'
        self.model_name_trained = 'model_trained.pkl'
        self.eval_result_path = 'eval_result.pkl'
        self.latest_model_checkpoint = 'model_checkpoint_latest.pointer'

    def get_eval_metric(self):
        return lgb_utils.func_generator(metric=self.objective_func, is_higher_better=True, needs_pred_proba=not self.metric_needs_y_pred, problem_type=self.problem_type)

    # TODO: Avoid deleting X_train and X_test to not corrupt future runs
    def fit(self, X_train=None, Y_train=None, X_test=None, Y_test=None, dataset_train=None, dataset_val=None, **kwargs):
        eval_metric = self.get_eval_metric()
        dataset_train, dataset_val = self.generate_datasets(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, dataset_train=dataset_train, dataset_val=dataset_val)
        gc.collect()
        self.eval_results = {}
        callbacks = []
        valid_names = ['train_set']
        valid_sets = [dataset_train]
        if dataset_val is not None:
            callbacks += [
                early_stopping_custom(150, metrics_to_use=[('valid_set', self.eval_metric_name)], max_diff=None, ignore_dart_warning=True, verbose=False, manual_stop_file=self.path + 'stop.txt'),
            ]
            valid_names = ['valid_set'] + valid_names
            valid_sets = [dataset_val] + valid_sets

        callbacks += [
            record_evaluation_custom(self.path + self.eval_result_path, eval_result={}, interval=1000),
            # save_model_callback(self.path + self.model_name_checkpointing_0, latest_model_checkpoint=self.path + self.latest_model_checkpoint, interval=400, offset=0),
            # save_model_callback(self.path + self.model_name_checkpointing_1, latest_model_checkpoint=self.path + self.latest_model_checkpoint, interval=400, offset=200),
            # lgb.reset_parameter(learning_rate=lambda iter: alpha * (0.999 ** iter)),
        ]
        # lr_over_time = lambda iter: 0.05 * (0.99 ** iter)
        # alpha = 0.1
        num_boost_round = self.params.pop('num_boost_round')
        print('Training Gradient Boosting Model for %s rounds...' % num_boost_round)
        print("with the following hyperparameter settings:")
        print(self.params)
        train_params = {
            'params': self.params,
            'train_set': dataset_train,
            'num_boost_round': num_boost_round, 
            'valid_sets': valid_sets,
            'valid_names': valid_names,
            'evals_result': self.eval_results,
            'callbacks': callbacks,
            'verbose_eval': 10,
        }
        if type(eval_metric) != str:
            train_params['feval'] = eval_metric
        self.model = lgb.train(**train_params)
        self.params['num_boost_round'] = num_boost_round # re-set this value after training
        # del dataset_train
        # del dataset_val
        # print('running gc...')
        # gc.collect()
        # print('ran garbage collection...')
        self.best_iteration = self.model.best_iteration
        # self.model.save_model(self.path + 'model.txt')
        # model_json = self.model.dump_model()
        #
        # with open(self.path + 'model.json', 'w+') as f:
        #     json.dump(model_json, f, indent=4)
        # save_pkl.save(path=self.path + self.model_name_trained, object=self)  # TODO: saving self instead of model, not consistent with save callbacks
        # save_pointer.save(path=self.path + self.latest_model_checkpoint, content_path=self.path + self.model_name_trained)

    def predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)
        if self.problem_type == REGRESSION:
            return self.model.predict(X)

        y_pred_proba = self.model.predict(X)
        if (self.problem_type == BINARY):
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif self.problem_type == MULTICLASS:
            return y_pred_proba
        else:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 2:  # Should this ever happen?
                return y_pred_proba
            else:  # Should this ever happen?
                return y_pred_proba[:, 1]

    def cv(self, X=None, y=None, k_fold=5, dataset_train=None):
        if dataset_train is None:
            dataset_train, _ = self.generate_datasets(X_train=X, Y_train=y)
        gc.collect()
        params = copy.deepcopy(self.params)
        eval_metric = self.get_eval_metric()
        # TODO: Either edit lgb.cv to return models / oof preds or make custom implementation!
        cv_params = {
            'params': params,
            'train_set': dataset_train,
            'num_boost_round': self.num_boost_round,
            'nfold': k_fold,
            'early_stopping_rounds': 150,
            'verbose_eval': 10,
            'seed': 0,
        }
        if type(eval_metric) != str:
            cv_params['feval'] = eval_metric
            cv_params['params']['metric'] = 'None'
        else:
            cv_params['params']['metric'] = eval_metric
        if self.problem_type == REGRESSION:
            cv_params['stratified'] = False

        print('Current parameters:\n', params)
        eval_hist = lgb.cv(**cv_params)  # TODO: Try to use customer early stopper to enable dart
        best_score = eval_hist[self.eval_metric_name + '-mean'][-1]
        print('Best num_boost_round:', len(eval_hist[self.eval_metric_name + '-mean']))
        print('Best CV score:', best_score)
        return best_score

    def convert_to_weight(self, X: DataFrame):
        print(X)
        w = X['count']
        X = X.drop(['count'], axis=1)
        return X, w

    def generate_datasets(self, X_train: DataFrame, Y_train: Series, X_test=None, Y_test=None, dataset_train=None, dataset_val=None, save=False):
        W_train = None  # TODO: Add weight support
        W_test = None  # TODO: Add weight support
        if X_train is not None:
            X_train = self.preprocess(X_train)
        if X_test is not None:
            X_test = self.preprocess(X_test)
        # TODO: Try creating multiple Datasets for subsets of features, then combining with Dataset.add_features_from(), this might avoid memory spike
        if not dataset_train:
            # X_train, W_train = self.convert_to_weight(X=X_train)
            dataset_train = construct_dataset(x=X_train, y=Y_train, location=self.path + 'datasets/train', params=self.params, save=save, weight=W_train)
            # dataset_train = construct_dataset_lowest_memory(X=X_train, y=Y_train, location=self.path + 'datasets/train', params=self.params)
        if (not dataset_val) and (X_test is not None) and (Y_test is not None):
            # X_test, W_test = self.convert_to_weight(X=X_test)
            dataset_val = construct_dataset(x=X_test, y=Y_test, location=self.path + 'datasets/val', reference=dataset_train, params=self.params, save=save, weight=W_test)
            # dataset_val = construct_dataset_lowest_memory(X=X_test, y=Y_test, location=self.path + 'datasets/val', reference=dataset_train, params=self.params)
        return dataset_train, dataset_val

    def debug_features_to_use(self, X_test_in):
        feature_splits = self.model.feature_importance()
        total_splits = feature_splits.sum()
        feature_names = list(X_test_in.columns.values)
        feature_count = len(feature_names)
        feature_importances = pd.DataFrame(data=feature_names, columns=['feature'])
        feature_importances['splits'] = feature_splits
        feature_importances_unused = feature_importances[feature_importances['splits'] == 0]
        feature_importances_used = feature_importances[feature_importances['splits'] >= (total_splits/feature_count)]
        print(feature_importances_unused)
        print(feature_importances_used)
        print('feature_importances_unused:', len(feature_importances_unused))
        print('feature_importances_used:', len(feature_importances_used))
        features_to_use = list(feature_importances_used['feature'].values)
        print(features_to_use)
        return features_to_use

    def hyperparameter_tune(self, X_train, X_test, y_train, y_test, spaces=None, scheduler_options=None): # scheduler_options unused for now
        print("Beginning hyperparameter tuning for Gradient Boosting Model...")
        X = pd.concat([X_train, X_test], ignore_index=True)
        y = pd.concat([y_train, y_test], ignore_index=True)
        if spaces is None:
            spaces = LGBMSpaces(problem_type=self.problem_type, objective_func=self.objective_func, num_classes=None).get_hyperparam_spaces_baseline()

        X = self.preprocess(X)
        dataset_train, _ = self.generate_datasets(X_train=X, Y_train=y)
        space = spaces[0]
        param_baseline = self.params

        @use_named_args(space)
        def objective(**params):
            print(params)
            new_params = copy.deepcopy(param_baseline)
            new_params['verbose'] = -1
            for param in params:
                new_params[param] = params[param]

            new_model = copy.deepcopy(self)
            new_model.params = new_params
            score = new_model.cv(dataset_train=dataset_train)

            print(score)
            if self.is_higher_better:
                score = -score

            return score

        reg_gp = gp_minimize(objective, space, verbose=True, n_jobs=1, n_calls=15)

        print('best score: {}'.format(reg_gp.fun))

        optimal_params = copy.deepcopy(param_baseline)
        for i, param in enumerate(space):
            optimal_params[param.name] = reg_gp.x[i]

        self.params = optimal_params
        print(self.params)

        # TODO: final fit should not be here eventually
        self.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
        self.save()
        return ({self.name: self.path}, {}) # dummy hpo_info

"""" OLD code: 

    def hyperparameter_tune(self, X, y, spaces=None):
        if spaces is None:
            spaces = LGBMSpaces(problem_type=self.problem_type, objective_func=self.objective_func, num_classes=None).get_hyperparam_spaces_baseline()

        X = self.preprocess(X)
        dataset_train, _ = self.generate_datasets(X_train=X, Y_train=y)

        print('starting skopt')
        space = spaces[0]

        param_baseline = self.params

        @use_named_args(space)
        def objective(**params):
            print(params)
            new_params = copy.deepcopy(param_baseline)
            new_params['verbose'] = -1
            for param in params:
                new_params[param] = params[param]

            new_model = copy.deepcopy(self)
            new_model.params = new_params
            score = new_model.cv(dataset_train=dataset_train)

            print(score)
            if self.is_higher_better:
                score = -score

            return score

        reg_gp = gp_minimize(objective, space, verbose=True, n_jobs=1, n_calls=15)

        print('best score: {}'.format(reg_gp.fun))

        optimal_params = copy.deepcopy(param_baseline)
        for i, param in enumerate(space):
            optimal_params[param.name] = reg_gp.x[i]

        self.params = optimal_params
        print(self.params)
        return optimal_params

"""