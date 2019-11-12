import contextlib
import shutil
import tempfile
import torch
from pathlib import Path

import numpy as np
import pandas as pd
from fastai.basic_data import DatasetType
from fastai.basic_train import load_learner
from fastai.callbacks import SaveModelCallback
from fastai.data_block import FloatList
from fastai.metrics import mean_absolute_error, accuracy
from fastai.tabular import tabular_learner, TabularList, FillMissing, Categorify, Normalize
from fastai.utils.mod_display import progress_disabled_ctx
from skopt import gp_minimize
from skopt.utils import use_named_args
import copy


from tabular.utils.loaders import load_pkl
from tabular.ml.constants import REGRESSION, BINARY
from tabular.ml.models.abstract_model import AbstractModel
from tabular.ml.trainer.abstract_trainer import AbstractTrainer
from tabular.ml.tuning.hyperparameters.nn_tab_spaces import NNTabularSpaces
from tabular.utils.savers import save_pkl

#https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
torch.multiprocessing.set_sharing_strategy('file_system')

LABEL = '__label__'

@contextlib.contextmanager
def make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


# TODO: Crashes when sent data to infer which has NaN's in a column that no NaN's existed during training
class NNTabularModel(AbstractModel):
    model_internals_file_name = 'model-internals.pkl'

    def __init__(self, path, name, params, problem_type, objective_func, features=None, debug=0, max_unique_categorical_values=10000):
        super().__init__(path=path, name=name, model=None, problem_type=problem_type, objective_func=objective_func, features=features, debug=debug)
        self.params = params
        self.procs = [FillMissing, Categorify, Normalize]
        self.cat_names = []
        self.cont_names = []
        self.max_unique_categorical_values = max_unique_categorical_values
        self.eval_result = None

    def predict(self, X, preprocess=True):
        return super().predict(X, preprocess)

    def __get_feature_type_if_present(self, feature_type):
        return self.feature_types_metadata[feature_type] if feature_type in self.feature_types_metadata else []

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, **kwargs):
        X_train = self.preprocess(X_train)
        if X_test is not None:
            X_test = self.preprocess(X_test)
        df_train, train_idx, val_idx = self._generate_datasets(X_train, Y_train, X_test, Y_test)
        self.cat_names = self.__get_feature_type_if_present('object') + self.__get_feature_type_if_present('bool')

        try:
            X_train_stats = X_train.describe(include='all').T.reset_index()
            cat_cols_to_drop = X_train_stats[(X_train_stats['unique'] > self.max_unique_categorical_values) | (X_train_stats['unique'].isna())]['index'].values
        except:
            cat_cols_to_drop = []
        cat_cols_to_keep = [col for col in X_train.columns.values if (col not in cat_cols_to_drop)]
        cat_cols_to_use = [col for col in self.cat_names if col in cat_cols_to_keep]
        print(f'Using {len(cat_cols_to_use)}/{len(self.cat_names)} categorical features')
        self.cat_names = cat_cols_to_use
        self.cat_names = [feature for feature in self.cat_names if feature in list(X_train.columns)]

        self.cont_names = self.__get_feature_type_if_present('float') + self.__get_feature_type_if_present('int') + self.__get_feature_type_if_present('datetime')  # + self.__get_feature_type_if_present('vectorizers')  # Disabling vectorizers until more performance testing is done
        self.cont_names = [feature for feature in self.cont_names if feature in list(X_train.columns)]
        print(f'Using {len(self.cont_names)} cont features')

        label_class = FloatList if self.problem_type == REGRESSION else None
        data = (TabularList.from_df(df_train, path=self.path, cat_names=self.cat_names, cont_names=self.cont_names, procs=self.procs)
                .split_by_idxs(train_idx, val_idx)
                .label_from_df(cols=LABEL, label_cls=label_class)
                .databunch(bs=self.params['nn.tabular.bs'] if len(X_train) > self.params['nn.tabular.bs'] else 32))

        metrics_map = {
            'accuracy': accuracy,
            'mean_absolute_error': mean_absolute_error,
        }

        nn_metric = metrics_map[self.params['nn.tabular.metric']]

        # TODO: calculate max emb concat layer size and use 1st layer as that value and 2nd in between number of classes and the value
        if self.problem_type == REGRESSION or self.problem_type == BINARY:
            layers = [200, 100]
        else:
            base_size = max(len(data.classes) * 2, 100)
            layers = [base_size * 2, base_size]

        self.model = tabular_learner(data, layers=layers, ps=[self.params['nn.tabular.dropout']], emb_drop=self.params['nn.tabular.dropout'], metrics=nn_metric)
        print(self.model.model)

        save_callback_mode = 'min' if self.params['nn.tabular.metric'] == 'mean_absolute_error' else 'auto'
        with make_temp_directory() as temp_dir:
            save_callback = SaveModelCallback(self.model, monitor=self.params['nn.tabular.metric'], mode=save_callback_mode, name=self.name)
            with progress_disabled_ctx(self.model) as model:
                original_path = model.path
                model.path = Path(temp_dir)
                model.fit_one_cycle(self.params['nn.tabular.epochs'], self.params['nn.tabular.lr'], callbacks=save_callback)

                # Load the best one and export it
                model.load(self.name)

                self.eval_result = model.validate()[1].numpy().reshape(-1)[0]

                print(f'Model validation metrics: {self.eval_result}')
                model.path = original_path

    def _generate_datasets(self, X_train, Y_train, X_test, Y_test):
        df_train = pd.concat([X_train, X_test], ignore_index=True)
        df_train[LABEL] = pd.concat([Y_train, Y_test], ignore_index=True)
        train_idx = np.arange(len(X_train))
        val_idx = np.arange(len(X_test)) + len(X_train)

        return df_train, train_idx, val_idx

    def predict_proba(self, X, preprocess=True):
        self.model.data.add_test(TabularList.from_df(X, cat_names=self.cat_names, cont_names=self.cont_names, procs=self.procs))
        with progress_disabled_ctx(self.model) as model:
            preds, _ = model.get_preds(ds_type=DatasetType.Test)

        if self.problem_type == REGRESSION:
            return preds.numpy().reshape(-1)
        if self.problem_type == BINARY:
            return preds[:, 1].numpy()
        else:
            return preds.numpy()

    def save(self):
        # Export model
        save_pkl.save_with_fn(f'{self.path}{self.model_internals_file_name}', self.model, lambda m, buffer: m.export(buffer, destroy=True))
        self.model = None
        super().save()

    @classmethod
    def load(cls, path: str, reset_paths=False):
        obj = super().load(path, reset_paths=reset_paths)
        obj.model = load_pkl.load_with_fn(f'{obj.path}{obj.model_internals_file_name}', lambda p: load_learner(obj.path, p))
        return obj

    def hyperparameter_tune(self, X, y, spaces=None):
        return self.params  # TODO: Disabled currently, model must first be fixed to handle custom objective functions
        # if spaces is None:
        #     spaces = NNTabularSpaces(problem_type=self.problem_type, objective_func=self.objective_func, num_classes=None).get_hyperparam_spaces_baseline()
        #
        # X = self.preprocess(X)
        # kfolds = AbstractTrainer.generate_kfold(X=X, n_splits=5)
        #
        # kfolds_datasets = []
        # for train_index, test_index in kfolds:
        #     kfolds_datasets.append([X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]])
        #
        # print('starting skopt')
        # space = spaces[0]
        #
        # param_baseline = self.params
        #
        # # TODO: Make CV splits prior, don't redo
        # @use_named_args(space)
        # def objective(**params):
        #     print(params)
        #     new_params = copy.deepcopy(param_baseline)
        #     new_params['verbose'] = -1
        #     for param in params:
        #         new_params[param] = params[param]
        #
        #     scores = []
        #     fold = 0
        #     for X_train, Y_train, X_test, Y_test in kfolds_datasets:
        #         fold += 1
        #         print(f'----- Running fold {fold}')
        #         new_model = copy.deepcopy(self)
        #         new_model.params = new_params
        #         new_model.fit(X_train, Y_train, X_test, Y_test)
        #         model_score = new_model.eval_result
        #         scores.append(model_score)
        #
        #     score = np.mean(scores)
        #     print(score)
        #
        #     minimize_factor = 1 if self.params['nn.tabular.metric'] == 'mean_absolute_error' else -1
        #
        #     return score * minimize_factor
        #
        # reg_gp = gp_minimize(objective, space, verbose=True, n_jobs=1, n_calls=15)
        #
        # print('best score: {}'.format(reg_gp.fun))
        #
        # optimal_params = copy.deepcopy(param_baseline)
        # for i, param in enumerate(space):
        #     optimal_params[param.name] = reg_gp.x[i]
        #
        # self.params = optimal_params
        # print(self.params)
        # return optimal_params
