import contextlib
import shutil
import tempfile
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

from f3_grail_data_frame_utilities.loaders import load_pkl
from tabular.ml.constants import REGRESSION, BINARY
from tabular.ml.models.abstract_model import AbstractModel
from tabular.utils.savers import save_pkl

LABEL = '__label__'

@contextlib.contextmanager
def make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


class NNTabularModel(AbstractModel):
    model_internals_file_name = 'model-internals.pkl'

    def __init__(self, path, name, params, problem_type, objective_func, features=None, debug=0, max_unique_categorical_values=10000):
        super().__init__(path=path, name=name, model=None, problem_type=problem_type, objective_func=objective_func, features=features, debug=debug)
        self.params = params
        self.procs = [FillMissing, Categorify, Normalize]
        self.bs = params['nn.tabular.bs']
        self.ps = params['nn.tabular.ps']
        self.emb_drop = params['nn.tabular.emb_drop']
        self.lr = params['nn.tabular.lr']
        self.epochs = params['nn.tabular.epochs']
        self.metric = params['nn.tabular.metric']
        self.cat_names = []
        self.cont_names = []
        self.feature_types_metadata = {}
        self.max_unique_categorical_values = max_unique_categorical_values

    def predict(self, X, preprocess=True):
        return super().predict(X, preprocess)

    def __get_feature_type_if_present(self, feature_type):
        return self.feature_types_metadata[feature_type] if feature_type in self.feature_types_metadata else []

    def fit(self, X_train, Y_train, X_test=None, Y_test=None):
        X_train = self.preprocess(X_train)  # TODO: Handle cases where features have been removed due to tuning, currently crashes!
        if X_test is not None:
            X_test = self.preprocess(X_test)
        df_train, train_idx, val_idx = self._generate_datasets(X_train, Y_train, Y_test, X_test)
        self.cat_names = self.__get_feature_type_if_present('object') + self.__get_feature_type_if_present('bool')

        try:
            X_train_stats = X_train.describe(include='all').T.reset_index()
            cols_to_drop = X_train_stats[(X_train_stats['unique'] > self.max_unique_categorical_values) | (X_train_stats['unique'].isna())]['index'].values
        except:
            cols_to_drop = []
        cols_to_keep = [col for col in list(X_train.columns) if col not in cols_to_drop]
        cols_to_use = [col for col in self.cat_names if col in cols_to_keep]
        print(f'Using {len(cols_to_use)}/{len(self.cat_names)} categorical features')
        self.cat_names = cols_to_use

        self.cont_names = self.__get_feature_type_if_present('float') + self.__get_feature_type_if_present('int')
        print(f'Using {len(self.cont_names)} cont features')

        label_class = FloatList if self.problem_type == REGRESSION else None
        data = (TabularList.from_df(df_train, path=self.path, cat_names=self.cat_names, cont_names=self.cont_names, procs=self.procs)
                .split_by_idxs(train_idx, val_idx)
                .label_from_df(cols=LABEL, label_cls=label_class)
                .databunch(bs=self.bs if len(X_train) > self.bs else 32))

        metrics_map = {
            'accuracy': accuracy,
            'mean_absolute_error': mean_absolute_error,
        }

        nn_metric = metrics_map[self.metric]

        # TODO: calculate max emb concat layer size and use 1st layer as that value and 2nd in between number of classes and the value
        if self.problem_type == REGRESSION or self.problem_type == BINARY:
            layers = [200, 100]
        else:
            base_size = max(len(data.classes) * 2, 100)
            layers = [base_size * 2, base_size]

        self.model = tabular_learner(data, layers=layers, ps=self.ps, emb_drop=self.emb_drop, metrics=nn_metric)
        print(self.model.model)

        save_callback_mode = 'min' if self.metric == 'mean_absolute_error' else 'auto'
        with make_temp_directory() as temp_dir:
            save_callback = SaveModelCallback(self.model, monitor=self.metric, mode=save_callback_mode, name=self.name)
            with progress_disabled_ctx(self.model) as model:
                original_path = model.path
                model.path = Path(temp_dir)
                model.fit_one_cycle(self.epochs, self.lr, callbacks=save_callback)

                # Load the best one and export it
                model.load(self.name)
                print(f'Model validation metrics: {model.validate()}')
                model.path = original_path

    def _generate_datasets(self, X_train, Y_train, Y_test, X_test):
        df_train = pd.concat([X_train.copy(), X_test.copy()]).reset_index(drop=True)
        df_train[LABEL] = pd.concat([Y_train.copy(), Y_test.copy()]).reset_index(drop=True)
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
        obj = super().load(path)
        obj.model = load_pkl.load_with_fn(f'{obj.path}{obj.model_internals_file_name}', lambda p: load_learner(obj.path, p))
        return obj
