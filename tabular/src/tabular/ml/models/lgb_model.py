
from tabular.ml.models.abstract_model import AbstractModel
from tabular.ml.utils import construct_dataset
from tabular.callbacks.lgb.callbacks import record_evaluation_custom, early_stopping_custom
from tabular.ml.trainer.abstract_trainer import AbstractTrainer
from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from tabular.ml.tuning.hyperparameters.lgbm_spaces import LGBMSpaces
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import gc

import copy
from skopt.utils import use_named_args
from skopt import gp_minimize


class LGBModel(AbstractModel):
    def __init__(self, path, name, params, num_boost_round, problem_type, objective_func, features=None, debug=0):
        model = None
        super().__init__(path=path, name=name, model=model, problem_type=problem_type, objective_func=objective_func, features=features, debug=debug)
        self.params = params
        self.objective = self.params['objective']
        self.metric_types = self.params['metric'].split(',')
        if 'binary_error' in self.metric_types:
            self.eval_metric = 'binary_error'
        elif 'multi_error' in self.metric_types:
            self.eval_metric = 'multi_error'
        elif 'l2' in self.metric_types:
            self.eval_metric = 'l2'
        else:
            self.eval_metric = self.metric_types[-1]

        self.num_boost_round = num_boost_round
        self.best_iteration = None
        self.eval_results = {}

        self.model_name_checkpointing_0 = 'model_checkpoint_0.pkl'
        self.model_name_checkpointing_1 = 'model_checkpoint_1.pkl'
        self.model_name_trained = 'model_trained.pkl'
        self.eval_result_path = 'eval_result.pkl'
        self.latest_model_checkpoint = 'model_checkpoint_latest.pointer'

    # TODO: Avoid deleting X_train and X_test to not corrupt future runs
    def fit(self, X_train=None, Y_train=None, X_test=None, Y_test=None, dataset_train=None, dataset_val=None):
        dataset_train, dataset_val = self.generate_datasets(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, dataset_train=dataset_train, dataset_val=dataset_val)
        gc.collect()

        self.eval_results = {}
        callbacks = []
        valid_names = ['train_set']
        valid_sets = [dataset_train]
        if dataset_val is not None:
            callbacks += [
                early_stopping_custom(100, metrics_to_use=[('valid_set', self.eval_metric)], max_diff=None, ignore_dart_warning=True, verbose=False, manual_stop_file=self.path + 'stop.txt'),
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
        print('TRAINING', self.num_boost_round, ' boosting rounds')
        print(self.params)
        self.model = lgb.train(params=self.params, train_set=dataset_train, num_boost_round=self.num_boost_round, valid_sets=valid_sets, valid_names=valid_names,
                          evals_result=self.eval_results,
                          callbacks=callbacks,
                          # keep_training_booster=True
                          )

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

    def hyperparameter_tune(self, X, y, spaces=None):
        if spaces is None:
            spaces = LGBMSpaces(problem_type=self.problem_type, objective_func=self.objective_func, num_classes=None).get_hyperparam_spaces_baseline()

        X = self.preprocess(X)
        kfolds = AbstractTrainer.generate_kfold(X=X, n_splits=5) # TODO: should depend on dataset size...

        kfolds_datasets = []
        for train_index, test_index in kfolds:
            dataset_train, dataset_val = self.generate_datasets(X_train=X.iloc[train_index], Y_train=y.iloc[train_index], X_test=X.iloc[test_index], Y_test=y.iloc[test_index])
            kfolds_datasets.append([dataset_train, dataset_val])

        print('starting skopt')
        space = spaces[0]

        param_baseline = self.params

        # TODO: Make CV splits prior, don't redo
        @use_named_args(space)
        def objective(**params):
            print(params)
            new_params = copy.deepcopy(param_baseline)
            new_params['verbose'] = -1
            for param in params:
                new_params[param] = params[param]

            # TODO: Fix early_stopping_rounds, not stopping on best acc

            scores = []
            for dataset_train, dataset_val in kfolds_datasets:
                new_model = copy.deepcopy(self)
                new_model.params = new_params
                new_model.fit(dataset_train=dataset_train, dataset_val=dataset_val)
                model_score = new_model.eval_results['valid_set'][new_model.eval_metric][new_model.best_iteration-1]
                scores.append(model_score)

            score = np.mean(scores)
            print(score)
            return score

        reg_gp = gp_minimize(objective, space, verbose=True, n_jobs=1, n_calls=15)

        print('best score: {}'.format(reg_gp.fun))

        optimal_params = copy.deepcopy(param_baseline)
        for i, param in enumerate(space):
            optimal_params[param.name] = reg_gp.x[i]

        self.params = optimal_params
        print(self.params)
        return optimal_params

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
