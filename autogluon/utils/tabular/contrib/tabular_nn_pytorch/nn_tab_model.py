import contextlib
import logging
import shutil
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from fastai.basic_data import DatasetType
from fastai.basic_train import load_learner
from fastai.callbacks import SaveModelCallback, EarlyStoppingCallback
from fastai.data_block import FloatList
from fastai.layers import LabelSmoothingCrossEntropy
from fastai.metrics import mean_absolute_error, accuracy, root_mean_squared_error, AUROC, mean_squared_error, r2_score, FBeta, Precision, Recall
from fastai.tabular import tabular_learner, TabularList, FillMissing, Categorify, Normalize
from fastai.utils.mod_display import progress_disabled_ctx
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from autogluon.utils.tabular.contrib.tabular_nn_pytorch.hyperparameters.parameters import get_param_baseline
from autogluon.utils.tabular.contrib.tabular_nn_pytorch.hyperparameters.searchspaces import get_default_searchspace
from autogluon.utils.tabular.ml.constants import REGRESSION, BINARY, MULTICLASS
from autogluon.utils.tabular.ml.models.abstract.abstract_model import AbstractModel
from autogluon.utils.tabular.ml.models.tabular_nn.categorical_encoders import OrdinalMergeRaresHandleUnknownEncoder
from autogluon.utils.tabular.utils.loaders import load_pkl
from autogluon.utils.tabular.utils.savers import save_pkl

# FIXME: Has a leak somewhere, training additional models in a single python script will slow down training for each additional model. Gets very slow after 20+ models (10x+ slowdown)
#  Slowdown does not appear to impact Mac OS
# Reproduced with raw torch: https://github.com/pytorch/pytorch/issues/31867
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
# https://github.com/pytorch/pytorch/issues/973
# https://pytorch.org/docs/master/multiprocessing.html#file-system-file-system
# Slowdown bug not experienced on Linux if 'torch.multiprocessing.set_sharing_strategy('file_system')' commented out
# NOTE: If below line is commented out, Torch uses many file descriptors. If issues arise, increase ulimit through 'ulimit -n 2048' or larger. Default on Linux is 1024.
# torch.multiprocessing.set_sharing_strategy('file_system')

LABEL = '__label__'

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


# TODO: Add to contrib

# TODO: On regression problems, can sometimes give insane predictions on unseen data. Cap the y valuees between min and max seen, else this can give REALLY bad results to test data.
# TODO: Takes extremely long (infinite?) time prior to training start if many (10000) continuous features from ngrams, debug
class NNFastAiTabularModel(AbstractModel):
    model_internals_file_name = 'model-internals.pkl'
    unique_category_str = '!missing!'
    metrics_map = {
        # Regression
        'root_mean_squared_error': root_mean_squared_error,
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'r2': r2_score,
        # Not supported: median_absolute_error

        # Classification
        'accuracy': accuracy,

        'f1': FBeta(beta=1),
        'f1_macro': FBeta(beta=1, average='macro'),
        'f1_micro': FBeta(beta=1, average='micro'),
        'f1_weighted': FBeta(beta=1, average='weigthed'),  # this one has some issues

        'roc_auc': AUROC(),

        'precision': Precision(),
        'precision_macro': Precision(average='macro'),
        'precision_micro': Precision(average='micro'),
        'precision_weigthed': Precision(average='weigthed'),

        'recall': Recall(),
        'recall_macro': Recall(average='macro'),
        'recall_micro': Recall(average='micro'),
        'recall_weigthed': Recall(average='weigthed'),
        # Not supported: pac_score

    }

    def __init__(self, path, name, problem_type, objective_func, stopping_metric=None, hyperparameters=None, features=None, feature_types_metadata=None,
                 debug=0, max_unique_categorical_values=10000,
                 y_scaler=None):
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features,
                         feature_types_metadata=feature_types_metadata, debug=debug)
        self.procs = [FillMissing, Categorify, Normalize]
        self.cat_names = []
        self.cont_names = []
        self.max_unique_categorical_values = max_unique_categorical_values
        self.eval_result = None

        self.col_after_transformer = None
        self.col_transformer = None
        self.y_scaler = y_scaler

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self, problem_type):
        spaces = {}
        return spaces

    def preprocess(self, X, fit=False):
        if fit:
            self.col_after_transformer = list(X.columns)
            self.col_transformer, self.col_after_transformer = self._construct_transformer(X=X)
            X = self.col_transformer.fit_transform(X)
        else:
            X = self.col_transformer.transform(X)
        X = pd.DataFrame(data=X, columns=self.col_after_transformer)
        X = super().preprocess(X)
        return X

    def __get_feature_type_if_present(self, feature_type):
        return self.feature_types_metadata[feature_type] if feature_type in self.feature_types_metadata else []

    def _construct_transformer(self, X):
        transformers = []
        if len(self.cont_names) > 0:
            continuous_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                # ('scaler', StandardScaler())
            ])
            transformers.append(('continuous', continuous_transformer, self.cont_names))
        if len(self.cat_names) > 0:
            ordinal_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('ordinal', OrdinalMergeRaresHandleUnknownEncoder(max_levels=self.max_unique_categorical_values))
                # returns 0-n when max_category_levels = n-1. category n is reserved for unknown test-time categories.
            ])
            transformers.append(('ordinal', ordinal_transformer, self.cat_names))
        cols_to_use = self.cont_names + self.cat_names
        col_after_transformer = [col for col in X.columns if col in cols_to_use]
        return ColumnTransformer(transformers=transformers), col_after_transformer

    def preprocess_train(self, X_train, Y_train, X_test, Y_test):
        self.cat_names = self.__get_feature_type_if_present('object') + self.__get_feature_type_if_present('bool')
        if self.problem_type == REGRESSION and self.y_scaler is not None:
            Y_train_norm = pd.Series(self.y_scaler.fit_transform(Y_train.values.reshape(-1, 1)).reshape(-1))
            Y_test_norm = pd.Series(self.y_scaler.transform(Y_test.values.reshape(-1, 1)).reshape(-1)) if Y_test is not None else None
            logger.log(15, f'Training with scaled targets: {self.y_scaler} - !!! NN training metric will be different from the final results !!!')
        else:
            Y_train_norm = Y_train
            Y_test_norm = Y_test
        try:
            X_train_stats = X_train.describe(include='all').T.reset_index()
            cat_cols_to_drop = X_train_stats[(X_train_stats['unique'] > self.max_unique_categorical_values) | (X_train_stats['unique'].isna())]['index'].values
        except:
            cat_cols_to_drop = []
        cat_cols_to_keep = [col for col in X_train.columns.values if (col not in cat_cols_to_drop)]
        cat_cols_to_use = [col for col in self.cat_names if col in cat_cols_to_keep]
        logger.log(15, f'Using {len(cat_cols_to_use)}/{len(self.cat_names)} categorical features')
        self.cat_names = cat_cols_to_use
        self.cat_names = [feature for feature in self.cat_names if feature in list(X_train.columns)]
        self.cont_names = self.__get_feature_type_if_present('float') + self.__get_feature_type_if_present('int') + self.__get_feature_type_if_present(
            'datetime')  # + self.__get_feature_type_if_present('vectorizers')  # Disabling vectorizers until more performance testing is done
        self.cont_names = [feature for feature in self.cont_names if feature in list(X_train.columns)]
        logger.log(15, f'Using {len(self.cont_names)} cont features')
        X_train = self.preprocess(X_train, fit=True)
        if X_test is not None:
            X_test = self.preprocess(X_test)
        df_train, train_idx, val_idx = self._generate_datasets(X_train, Y_train_norm, X_test, Y_test_norm)
        label_class = FloatList if self.problem_type == REGRESSION else None
        data = (TabularList.from_df(df_train, path=self.path, cat_names=self.cat_names, cont_names=self.cont_names, procs=self.procs)
                .split_by_idxs(train_idx, val_idx)
                .label_from_df(cols=LABEL, label_cls=label_class)
                .databunch(bs=self.params['bs'] if len(X_train) > self.params['bs'] else 32))
        return data

    def _generate_datasets(self, X_train, Y_train, X_test, Y_test):
        df_train = pd.concat([X_train, X_test], ignore_index=True)
        df_train[LABEL] = pd.concat([Y_train, Y_test], ignore_index=True)
        train_idx = np.arange(len(X_train))
        val_idx = np.arange(len(X_test)) + len(X_train)

        return df_train, train_idx, val_idx

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, **kwargs):
        logger.log(15, f'Fitting Neural Network with parameters {self.params}...')
        data = self.preprocess_train(X_train, Y_train, X_test, Y_test)

        nn_metric, objective_func_name = self.__get_objective_func_name()
        objective_func_name_to_monitor = self.__get_objective_func_to_monitor(objective_func_name)
        objective_optim_mode = 'min' if objective_func_name in [
            'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'r2'  # Regression objectives
        ] else 'auto'

        # TODO: calculate max emb concat layer size and use 1st layer as that value and 2nd in between number of classes and the value
        if self.problem_type in [REGRESSION, BINARY]:
            layers = [200, 100]
        else:
            base_size = max(len(data.classes) * 2, 100)
            layers = [base_size * 2, base_size]

        early_stopping_fn = partial(EarlyStoppingCallback, monitor=objective_func_name_to_monitor, mode=objective_optim_mode,
                                    min_delta=self.params['early.stopping.min_delta'], patience=self.params['early.stopping.patience'])

        loss_func = None
        if self.problem_type in [BINARY, MULTICLASS] and self.params.get('smoothing', 0.0) > 0.0:
            loss_func = LabelSmoothingCrossEntropy(self.params['smoothing'])

        self.model = tabular_learner(
            data, layers=layers, ps=[self.params['dropout']], emb_drop=self.params['dropout'], metrics=nn_metric,
            loss_func=loss_func, callback_fns=[early_stopping_fn]
        )
        logger.log(15, self.model.model)

        with make_temp_directory() as temp_dir:
            save_callback = SaveModelCallback(self.model, monitor=objective_func_name_to_monitor, mode=objective_optim_mode, name=self.name)
            with progress_disabled_ctx(self.model) as model:
                original_path = model.path
                model.path = Path(temp_dir)
                model.fit_one_cycle(self.params['epochs'], self.params['lr'], callbacks=save_callback)

                # Load the best one and export it
                model.load(self.name)

                if objective_func_name == 'log_loss':
                    self.eval_result = model.validate()[0]
                else:
                    self.eval_result = model.validate()[1].numpy().reshape(-1)[0]

                logger.log(15, f'Model validation metrics: {self.eval_result}')
                model.path = original_path

    def __get_objective_func_name(self):
        objective_func_name = self.objective_func.name
        if objective_func_name in self.metrics_map.keys():
            nn_metric = self.metrics_map[objective_func_name]
        elif objective_func_name is None:
            objective_func_name = self.params['metric']
            nn_metric = self.metrics_map[self.params['metric']]
        else:
            nn_metric = None
        return nn_metric, objective_func_name

    def __get_objective_func_to_monitor(self, objective_func_name):
        monitor_obj_func = {
            'roc_auc': 'auroc',

            'f1': 'f_beta',
            'f1_macro': 'f_beta',
            'f1_micro': 'f_beta',
            'f1_weighted': 'f_beta',

            'precision_macro': 'precision',
            'precision_micro': 'precision',
            'precision_weigthed': 'precision',

            'recall_macro': 'recall',
            'recall_micro': 'recall',
            'recall_weigthed': 'recall',
            'log_loss': 'valid_loss',
        }
        objective_func_name_to_monitor = objective_func_name
        if objective_func_name in monitor_obj_func:
            objective_func_name_to_monitor = monitor_obj_func[objective_func_name]
        return objective_func_name_to_monitor

    def predict(self, X, preprocess=True):
        return super().predict(X, preprocess)

    def predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)
        self.model.data.add_test(TabularList.from_df(X, cat_names=self.cat_names, cont_names=self.cont_names, procs=self.procs))
        with progress_disabled_ctx(self.model) as model:
            preds, _ = model.get_preds(ds_type=DatasetType.Test)
        if self.problem_type == REGRESSION:
            if self.y_scaler is not None:
                return self.y_scaler.inverse_transform(preds.numpy()).reshape(-1)
            else:
                return preds.numpy().reshape(-1)
        if self.problem_type == BINARY:
            return preds[:, 1].numpy()
        else:
            return preds.numpy()

    def save(self, file_prefix='', directory=None, return_filename=False, verbose=True):
        # Export model
        save_pkl.save_with_fn(f'{self.path}{self.model_internals_file_name}', self.model, lambda m, buffer: m.export(buffer, destroy=True), verbose=verbose)
        self.model = {}
        return super().save(file_prefix=file_prefix, directory=directory, return_filename=return_filename, verbose=verbose)

    @classmethod
    def load(cls, path: str, file_prefix='', reset_paths=False, verbose=True):
        obj = super().load(path, file_prefix=file_prefix, reset_paths=reset_paths, verbose=verbose)
        obj.model = load_pkl.load_with_fn(f'{obj.path}{obj.model_internals_file_name}', lambda p: load_learner(obj.path, p), verbose=verbose)
        return obj

    def _get_default_searchspace(self, problem_type):
        return get_default_searchspace(problem_type)

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):
        self.fit(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, **kwargs)
        hpo_model_performances = {self.name: self.score(X_test, Y_test)}
        hpo_results = {}
        self.save()
        hpo_models = {self.name: self.path}

        return hpo_models, hpo_model_performances, hpo_results
