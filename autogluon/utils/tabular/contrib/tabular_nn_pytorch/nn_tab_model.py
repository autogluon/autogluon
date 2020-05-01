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
from fastai.metrics import mean_absolute_error, accuracy, root_mean_squared_error, AUROC
from fastai.tabular import tabular_learner, TabularList, FillMissing, Categorify, Normalize
from fastai.utils.mod_display import progress_disabled_ctx
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from autogluon.utils.tabular.contrib.tabular_nn_pytorch.hyperparameters.parameters import get_param_baseline
from autogluon.utils.tabular.ml.constants import REGRESSION, BINARY
from autogluon.utils.tabular.ml.models.abstract.abstract_model import AbstractModel
from autogluon.utils.tabular.ml.models.tabular_nn.categorical_encoders import OrdinalMergeRaresHandleUnknownEncoder
from autogluon.utils.tabular.utils.loaders import load_pkl
from autogluon.utils.tabular.utils.savers import save_pkl

# FIXME: Has a leak somewhere, training additional models in a single python script will slow down training for each additional model. Gets very slow after 20+ models (10x+ slowdown)
#  Slowdown does not appear to impact Mac OS
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
# https://github.com/pytorch/pytorch/issues/973
# https://pytorch.org/docs/master/multiprocessing.html#file-system-file-system
# Slowdown bug not experienced on Linux if 'torch.multiprocessing.set_sharing_strategy('file_system')' commented out
# NOTE: If below line is commented out, Torch uses many file descriptors. If issues arise, increase ulimit through 'ulimit -n 2048' or larger. Default on Linux is 1024.
# torch.multiprocessing.set_sharing_strategy('file_system')

LABEL = '__label__'


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
# TODO: Model performance on Regression is dependent on the scale of y. If y is very large, learning rate has to be increased dramatically. Can be fixed by y being divided by ex: 1000. Debug the root cause, this shouldn't be happening.

class NNTabularModel(AbstractModel):
    model_internals_file_name = 'model-internals.pkl'
    unique_category_str = '!missing!'

    def __init__(self, path, name, problem_type, objective_func, hyperparameters=None, features=None, feature_types_metadata=None, debug=0, max_unique_categorical_values=10000,
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

    def predict(self, X):
        return super().predict(X)

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

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, **kwargs):
        self.cat_names = self.__get_feature_type_if_present('object') + self.__get_feature_type_if_present('bool')

        if self.problem_type == REGRESSION and self.y_scaler is not None:
            Y_train_norm = pd.Series(self.y_scaler.fit_transform(Y_train.values.reshape(-1, 1)).reshape(-1))
            Y_test_norm = pd.Series(self.y_scaler.transform(Y_test.values.reshape(-1, 1)).reshape(-1)) if Y_test is not None else None
            print(f'Training with scaled targets: {self.y_scaler} - !!! NN training metric will be different from the final results !!!')
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
        print(f'Using {len(cat_cols_to_use)}/{len(self.cat_names)} categorical features')
        self.cat_names = cat_cols_to_use
        self.cat_names = [feature for feature in self.cat_names if feature in list(X_train.columns)]

        self.cont_names = self.__get_feature_type_if_present('float') + self.__get_feature_type_if_present('int') + self.__get_feature_type_if_present(
            'datetime')  # + self.__get_feature_type_if_present('vectorizers')  # Disabling vectorizers until more performance testing is done
        self.cont_names = [feature for feature in self.cont_names if feature in list(X_train.columns)]
        print(f'Using {len(self.cont_names)} cont features')

        X_train = self.preprocess(X_train, fit=True)
        if X_test is not None:
            X_test = self.preprocess(X_test)

        df_train, train_idx, val_idx = self._generate_datasets(X_train, Y_train_norm, X_test, Y_test_norm)
        label_class = FloatList if self.problem_type == REGRESSION else None
        data = (TabularList.from_df(df_train, path=self.path, cat_names=self.cat_names, cont_names=self.cont_names, procs=self.procs)
                .split_by_idxs(train_idx, val_idx)
                .label_from_df(cols=LABEL, label_cls=label_class)
                .databunch(bs=self.params['nn.tabular.bs'] if len(X_train) > self.params['nn.tabular.bs'] else 32))

        metrics_map = {
            'accuracy': accuracy,
            'roc_auc': AUROC(),
            'mean_absolute_error': mean_absolute_error,
            'root_mean_squared_error': root_mean_squared_error
        }

        objective_func_name = self.objective_func.name
        if objective_func_name in metrics_map.keys():
            nn_metric = metrics_map[objective_func_name]
        else:
            objective_func_name = self.params['nn.tabular.metric']
            nn_metric = metrics_map[self.params['nn.tabular.metric']]

        # TODO: calculate max emb concat layer size and use 1st layer as that value and 2nd in between number of classes and the value
        if self.problem_type == REGRESSION or self.problem_type == BINARY:
            layers = [200, 100]
        else:
            base_size = max(len(data.classes) * 2, 100)
            layers = [base_size * 2, base_size]

        self.model = tabular_learner(data, layers=layers, ps=[self.params['nn.tabular.dropout']], emb_drop=self.params['nn.tabular.dropout'], metrics=nn_metric)
        print(self.model.model)

        objective_func_name_to_monitor = objective_func_name if objective_func_name != 'roc_auc' else 'auroc'
        save_callback_mode = 'min' if objective_func_name in ['root_mean_squared_error', 'mean_absolute_error'] else 'auto'
        with make_temp_directory() as temp_dir:
            save_callback = SaveModelCallback(self.model, monitor=objective_func_name_to_monitor, mode=save_callback_mode, name=self.name)
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

    def predict_proba(self, X):
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


"""
DEFECT:
On Linux, 

File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 280, in train_and_save
    self.train_single(X_train, y_train, X_test, y_test, model, objective_func=self.objective_func)
  File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 152, in train_single
    model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, **model_fit_kwargs)
  File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/contrib/tabular_nn_pytorch/nn_tab_model.py", line 161, in fit
    model.fit_one_cycle(self.params['nn.tabular.epochs'], self.params['nn.tabular.lr'], callbacks=save_callback)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastai-1.0.55-py3.6.egg/fastai/train.py", line 22, in fit_one_cycle
    learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastai-1.0.55-py3.6.egg/fastai/basic_train.py", line 200, in fit
    fit(epochs, self, metrics=self.metrics, callbacks=self.callbacks+callbacks)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastai-1.0.55-py3.6.egg/fastai/basic_train.py", line 99, in fit
    for xb,yb in progress_bar(learn.data.train_dl, parent=pbar):
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastprogress-0.1.21-py3.6.egg/fastprogress/fastprogress.py", line 72, in __iter__
    for i,o in enumerate(self._gen):
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastai-1.0.55-py3.6.egg/fastai/basic_data.py", line 75, in __iter__
    for b in self.dl: yield self.proc_batch(b)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/utils/data/dataloader.py", line 582, in __next__
    return self._process_next_batch(batch)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/utils/data/dataloader.py", line 608, in _process_next_batch
    raise batch.exc_type(batch.exc_msg)
Warning: Exception caused GrailNNTabularModel_0 to fail during training... Skipping model.
Traceback (most recent call last):
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/utils/data/_utils/worker.py", line 99, in _worker_loop
    samples = collate_fn([dataset[i] for i in batch_indices])
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastai-1.0.55-py3.6.egg/fastai/torch_core.py", line 127, in data_collate
    return torch.utils.data.dataloader.default_collate(to_data(batch))
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/utils/data/_utils/collate.py", line 68, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/utils/data/_utils/collate.py", line 68, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/utils/data/_utils/collate.py", line 68, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/utils/data/_utils/collate.py", line 68, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/utils/data/_utils/collate.py", line 41, in default_collate
    storage = batch[0].storage()._new_shared(numel)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/storage.py", line 124, in _new_shared
    return cls._new_using_filename(size)
RuntimeError: error executing torch_shm_manager at "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/bin/torch_shm_manager" at /pytorch/torch/lib/libshm/core.cpp:99

Traceback (most recent call last):
  File "autogluon/utils/tabular/sandbox/ames/run_learner.py", line 28, in <module>
    learner = task.fit(time_limits=30, train_data=X, label=LABEL, output_directory=path_model_prefix, hyperparameter_tune=False, id_columns=[], feature_generator=feature_generator,)
  File "/home/ubuntu/workspace/autogluon/autogluon/task/tabular_prediction/tabular_prediction.py", line 177, in fit
    holdout_frac=holdout_frac, hyperparameters=hyperparameters)
  File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/ml/learner/default_learner.py", line 58, in fit
    hyperparameters=hyperparameters)
  File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/ml/trainer/auto_trainer.py", line 31, in train
    hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
  File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 225, in train_multi_and_ensemble
    self.train_multi(X_train, y_train, X_test, y_test, models, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
  File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 203, in train_multi
    self.train_single_full(X_train, y_train, X_test, y_test, model, hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune)
  File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 198, in train_single_full
    self.train_and_save(X_train, y_train, X_test, y_test, model)
  File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 280, in train_and_save
    self.train_single(X_train, y_train, X_test, y_test, model, objective_func=self.objective_func)
  File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 152, in train_single
    model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, **model_fit_kwargs)
  File "/home/ubuntu/workspace/autogluon/autogluon/utils/tabular/contrib/tabular_nn_pytorch/nn_tab_model.py", line 161, in fit
    model.fit_one_cycle(self.params['nn.tabular.epochs'], self.params['nn.tabular.lr'], callbacks=save_callback)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastai-1.0.55-py3.6.egg/fastai/train.py", line 22, in fit_one_cycle
    learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastai-1.0.55-py3.6.egg/fastai/basic_train.py", line 200, in fit
    fit(epochs, self, metrics=self.metrics, callbacks=self.callbacks+callbacks)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastai-1.0.55-py3.6.egg/fastai/basic_train.py", line 99, in fit
    for xb,yb in progress_bar(learn.data.train_dl, parent=pbar):
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastprogress-0.1.21-py3.6.egg/fastprogress/fastprogress.py", line 72, in __iter__
    for i,o in enumerate(self._gen):
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/fastai-1.0.55-py3.6.egg/fastai/basic_data.py", line 75, in __iter__
    for b in self.dl: yield self.proc_batch(b)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/utils/data/dataloader.py", line 193, in __iter__
    return _DataLoaderIter(self)
  File "/home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/utils/data/dataloader.py", line 469, in __init__
    w.start()
  File "/home/ubuntu/anaconda3/lib/python3.6/multiprocessing/process.py", line 105, in start
    self._popen = self._Popen(self)
  File "/home/ubuntu/anaconda3/lib/python3.6/multiprocessing/context.py", line 223, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "/home/ubuntu/anaconda3/lib/python3.6/multiprocessing/context.py", line 277, in _Popen
    return Popen(process_obj)
  File "/home/ubuntu/anaconda3/lib/python3.6/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/home/ubuntu/anaconda3/lib/python3.6/multiprocessing/popen_fork.py", line 66, in _launch
    self.pid = os.fork()

"""

"""
DEFECT: Upon ending script (Success)

Segmentation fault: 11

Stack trace:
  [bt] (0) /home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2b64150) [0x7febd47ba150]
  [bt] (1) /lib/x86_64-linux-gnu/libc.so.6(+0x354b0) [0x7febe72d54b0]
  [bt] (2) /home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/lib/libcaffe2_gpu.so(std::__detail::_Map_base<int, std::pair<int const, std::vector<cudnnContext*, std::allocator<cudnnContext*> > >, std::allocator<std::pair<int const, std::vector<cudnnContext*, std::allocator<cudnnContext*> > > >, std::__detail::_Select1st, std::equal_to<int>, std::hash<int>, std::__detail::_Mod_range_hashing, std::__detail::_Default_ranged_hash, std::__detail::_Prime_rehash_policy, std::__detail::_Hashtable_traits<false, false, true>, true>::operator[](int const&)+0x2e) [0x7feb4b41593e]
  [bt] (3) /home/ubuntu/workspace/autogluon/venv/lib/python3.6/site-packages/torch-1.1.0-py3.6-linux-x86_64.egg/torch/lib/libcaffe2_gpu.so(+0x14561e1) [0x7feb4b4141e1]
  [bt] (4) /home/ubuntu/anaconda3/bin/../lib/libstdc++.so.6(+0xaaad1) [0x7febd1865ad1]
  [bt] (5) /lib/x86_64-linux-gnu/libc.so.6(+0x39ff8) [0x7febe72d9ff8]
  [bt] (6) /lib/x86_64-linux-gnu/libc.so.6(+0x3a045) [0x7febe72da045]
  [bt] (7) /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf7) [0x7febe72c0837]
  [bt] (8) python(+0x1c847b) [0x5628943e747b]



"""
