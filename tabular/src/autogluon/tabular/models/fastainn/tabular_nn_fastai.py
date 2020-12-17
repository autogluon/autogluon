import contextlib
import copy
import logging
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from autogluon.core.constants import REGRESSION, BINARY, MULTICLASS
from autogluon.core.utils import try_import_fastai_v1
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.multiprocessing_utils import is_fork_enabled
from autogluon.core.utils.savers import save_pkl
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from .metrics import get_objective_func_name, get_objective_func_to_monitor
from .. import AbstractNeuralNetworkModel
from ..abstract.model_trial import skip_hpo
from ..tabular_nn.tabular_nn_preprocessor import TabularNeuralNetPreprocessor
from ...features.feature_metadata import R_OBJECT

# FIXME: Has a leak somewhere, training additional models in a single python script will slow down training for each additional model. Gets very slow after 20+ models (10x+ slowdown)
#  Slowdown does not appear to impact Mac OS
# Reproduced with raw torch: https://github.com/pytorch/pytorch/issues/31867
# https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935
# https://github.com/pytorch/pytorch/issues/973
# https://pytorch.org/docs/master/multiprocessing.html#file-system-file-system
# Slowdown bug not experienced on Linux if 'torch.multiprocessing.set_sharing_strategy('file_system')' commented out
# NOTE: If below line is commented out, Torch uses many file descriptors. If issues arise, increase ulimit through 'ulimit -n 2048' or larger. Default on Linux is 1024.
# torch.multiprocessing.set_sharing_strategy('file_system')

# MacOS issue: torchvision==0.7.0 + torch==1.6.0 can cause segfaults; use torch==1.2.0 torchvision==0.4.0

LABEL = '__label__'

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


# TODO: Takes extremely long time prior to training start if many (10000) continuous features from ngrams, debug - explore TruncateSVD option to reduce input dimensionality
# TODO: currently fastai automatically detect and use CUDA if available - add code to honor autogluon settings
class NNFastAiTabularModel(AbstractNeuralNetworkModel):
    """ Class for fastai v1 neural network models that operate on tabular data.

        Hyperparameters:
            y_scaler: on a regression problems, the model can give unreasonable predictions on unseen data.
            This attribute allows to pass a scaler for y values to address this problem. Please note that intermediate
            iteration metrics will be affected by this transform and as a result intermediate iteration scores will be
            different from the final ones (these will be correct).
            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

            'layers': list of hidden layers sizes; None - use model's heuristics; default is None

            'emb_drop': embedding layers dropout; defaut is 0.1

            'ps': linear layers dropout - list of values applied to every layer in `layers`; default is [0.1]

            'bs': batch size; default is 256

            'lr': maximum learning rate for one cycle policy; default is 1e-2;
            see also https://fastai1.fast.ai/train.html#fit_one_cycle, One-cycle policy paper: https://arxiv.org/abs/1803.09820

            'epochs': number of epochs; default is 30

            # Early stopping settings. See more details here: https://fastai1.fast.ai/callbacks.tracker.html#EarlyStoppingCallback
            'early.stopping.min_delta': 0.0001,
            'early.stopping.patience': 10,

            'smoothing': If > 0, then use LabelSmoothingCrossEntropy loss function for binary/multi-class classification;
            otherwise use default loss function for this type of problem; default is 0.0.
            See: https://docs.fast.ai/layers.html#LabelSmoothingCrossEntropy
    """

    model_internals_file_name = 'model-internals.pkl'
    unique_category_str = '!missing!'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cat_columns = []
        self.cont_columns = []
        self.col_after_transformer = None
        self.y_scaler = None
        self.processor = None
        self.feature_columns = None
        self.feature_arraycol_map = None

    def process_train_data(self, X, params):
        if self.processor is not None:
            Warning("Attempting to process training data for TabularNeuralNetModel, but previously already did this.")
        X = super().preprocess(X)
        max_category_levels = params['proc.max_category_levels']
        skew_threshold = params['proc.skew_threshold']
        use_ngram_features = params['use_ngram_features']
        impute_strategy = params['proc.impute_strategy']
        embed_min_categories = params['proc.embed_min_categories']

        self._types_of_features, X = self._get_types_of_features(
            X, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features)

        self.processor = TabularNeuralNetPreprocessor(self._types_of_features, self.unique_category_str,
            impute_strategy, max_category_levels)

        X = self.processor.fit_transform(X)

        self.__register_feature_types(max_category_levels)

        logger.log(15, f'Using {len(self.cat_columns)} categorical columns, {len(self.cont_columns)} continuous columns')

        X = pd.DataFrame(X, columns=self.feature_columns)
        return X

    def __register_feature_types(self, max_category_levels):
        # OrderedDict of feature-name -> list of column-indices in df corresponding to this feature
        self.feature_arraycol_map = self.processor.get_feature_arraycol_map(max_category_levels)
        self.feature_columns = []
        self.cat_columns = []
        self.cont_columns = []
        for col, col_idxs in self.feature_arraycol_map.items():
            if len(col_idxs) == 1:
                self.feature_columns.append([col_idxs[0], col])
                self.__append_feature_type(col, col)
            else:
                for i, col_id in enumerate(col_idxs):
                    _col = f'{col}.{i}'
                    self.feature_columns.append([col_id, _col])
                    self.__append_feature_type(col, _col)
        sorted(self.feature_columns, key=lambda x: x[0])
        self.feature_columns = [x[1] for x in self.feature_columns]

    def process_test_data(self, X, params):
        X = pd.DataFrame(data=X, columns=self.col_after_transformer)
        X = super().preprocess(X)
        X = pd.DataFrame(self.processor.transform(X), columns=self.feature_columns)
        return X

    def __append_feature_type(self, col, col_to_add):
        if self.__is_feature_is_in_any_of_types(col, ['embed']):
            self.cat_columns.append(col_to_add)
        elif self.__is_feature_is_in_any_of_types(col, ['continuous', 'skewed', 'onehot']):
            self.cont_columns.append(col_to_add)

    def __is_feature_is_in_any_of_types(self, feature, types):
        for t in types:
            if feature in self._types_of_features[t]:
                return True
        return False

    def process_train(self, X_train, y_train, X_val, y_val, params, **kwargs):
        from fastai.data_block import FloatList
        from fastai.tabular import TabularList
        from fastai.tabular import FillMissing, Categorify, Normalize
        from fastai.core import defaults

        # TODO add NaN imputation logic

        if self.problem_type == REGRESSION and self.y_scaler is not None:
            y_train_norm = pd.Series(self.y_scaler.fit_transform(y_train.values.reshape(-1, 1)).reshape(-1))
            y_val_norm = pd.Series(self.y_scaler.transform(y_val.values.reshape(-1, 1)).reshape(-1)) if y_val is not None else None
            logger.log(0, f'Training with scaled targets: {self.y_scaler} - !!! NN training metric will be different from the final results !!!')
        else:
            y_train_norm = y_train
            y_val_norm = y_val

        X_train = self.process_train_data(X_train, params=params)
        if X_val is not None:
            X_val = self.process_test_data(X_val, params=params)

        df_train, train_idx, val_idx = self._generate_datasets(X_train, y_train_norm, X_val, y_val_norm)
        label_class = FloatList if self.problem_type == REGRESSION else None
        procs = [FillMissing, Categorify, Normalize]

        # additional workers are helping only when fork is enabled; in other mp modes, communication overhead reduces performance
        num_workers = defaults.cpus if is_fork_enabled() else 0
        data = (TabularList.from_df(df_train, path=self.path, cat_names=self.cat_columns, cont_names=self.cont_columns, procs=procs)
                .split_by_idxs(train_idx, val_idx)
                .label_from_df(cols=LABEL, label_cls=label_class)
                .databunch(bs=self.params['bs'] if len(X_train) > self.params['bs'] else 32, num_workers=num_workers))
        return data

    def _fit(self, X_train, y_train, X_val=None, y_val=None, time_limit=None, **kwargs):
        try_import_fastai_v1()
        from fastai.layers import LabelSmoothingCrossEntropy
        from fastai.tabular import tabular_learner
        from fastai.utils.mod_display import progress_disabled_ctx
        from .callbacks import EarlyStoppingCallbackWithTimeLimit, SaveModelCallback

        start_time = time.time()

        params = self.params.copy()

        self.y_scaler = params.get('y_scaler', None)
        if self.y_scaler is not None:
            self.y_scaler = copy.deepcopy(self.y_scaler)

        logger.log(15, f'Fitting Neural Network with parameters {params}...')
        data = self.process_train(X_train, y_train, X_val, y_val, params)

        nn_metric, objective_func_name = get_objective_func_name(self.problem_type, self.stopping_metric)
        objective_func_name_to_monitor = get_objective_func_to_monitor(objective_func_name)
        objective_optim_mode = 'min' if objective_func_name in [
            'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'r2'  # Regression objectives
        ] else 'auto'

        # TODO: calculate max emb concat layer size and use 1st layer as that value and 2nd in between number of classes and the value
        if params.get('layers', None) is not None:
            layers = params['layers']
        elif self.problem_type in [REGRESSION, BINARY]:
            layers = [200, 100]
        else:
            base_size = max(len(data.classes) * 2, 100)
            layers = [base_size * 2, base_size]

        loss_func = None
        if self.problem_type in [BINARY, MULTICLASS] and params.get('smoothing', 0.0) > 0.0:
            loss_func = LabelSmoothingCrossEntropy(params['smoothing'])

        ps = params['ps']
        if type(ps) != list:
            ps = [ps]

        if time_limit:
            time_elapsed = time.time() - start_time
            time_left = time_limit - time_elapsed
        else:
            time_left = None

        best_epoch_stop = params.get("best_epoch", None)  # Use best epoch for refit_full.
        early_stopping_fn = partial(EarlyStoppingCallbackWithTimeLimit, monitor=objective_func_name_to_monitor, mode=objective_optim_mode,
                                    min_delta=params['early.stopping.min_delta'], patience=params['early.stopping.patience'],
                                    time_limit=time_left, best_epoch_stop=best_epoch_stop)

        self.model = tabular_learner(
            data, layers=layers, ps=ps, emb_drop=params['emb_drop'], metrics=nn_metric,
            loss_func=loss_func, callback_fns=[early_stopping_fn]
        )
        logger.log(15, self.model.model)

        with make_temp_directory() as temp_dir:
            save_callback = SaveModelCallback(self.model, monitor=objective_func_name_to_monitor, mode=objective_optim_mode, name=self.name,
                                              best_epoch_stop=best_epoch_stop)
            with progress_disabled_ctx(self.model) as model:
                original_path = model.path
                model.path = Path(temp_dir)
                model.fit_one_cycle(params['epochs'], params['lr'], callbacks=save_callback)

                # Load the best one and export it
                model.load(self.name)

                if objective_func_name == 'log_loss':
                    eval_result = model.validate()[0]
                else:
                    eval_result = model.validate()[1].numpy().reshape(-1)[0]

                logger.log(15, f'Model validation metrics: {eval_result}')
                model.path = original_path
            self.params_trained['best_epoch'] = save_callback.best_epoch

    def _generate_datasets(self, X_train, y_train, X_val, y_val):
        df_train = pd.concat([X_train, X_val], ignore_index=True)
        df_train[LABEL] = pd.concat([y_train, y_val], ignore_index=True)
        train_idx = np.arange(len(X_train))
        if X_val is None:
            val_idx = train_idx  # use validation set for refit_full case - it's not going to be used for early stopping
        else:
            val_idx = np.arange(len(X_val)) + len(X_train)
        return df_train, train_idx, val_idx

    def _predict_proba(self, X, **kwargs):
        from fastai.basic_data import DatasetType
        from fastai.tabular import TabularList
        from fastai.utils.mod_display import progress_disabled_ctx
        from fastai.tabular import FillMissing, Categorify, Normalize

        X = self.preprocess(X, **kwargs)
        X = self.process_test_data(X, params=self.params)

        procs = [FillMissing, Categorify, Normalize]

        single_row = len(X) == 1
        # fastai has issues predicting on a single row, duplicating the row as a workaround
        if single_row:
            X = pd.concat([X, X]).reset_index(drop=True)

        self.model.data.add_test(TabularList.from_df(X, cat_names=self.cat_columns, cont_names=self.cont_columns, procs=procs))
        with progress_disabled_ctx(self.model) as model:
            preds, _ = model.get_preds(ds_type=DatasetType.Test)
        if single_row:
            preds = preds[:1, :]
        if self.problem_type == REGRESSION:
            if self.y_scaler is not None:
                return self.y_scaler.inverse_transform(preds.numpy()).reshape(-1)
            else:
                return preds.numpy().reshape(-1)
        if self.problem_type == BINARY:
            return preds[:, 1].numpy()
        else:
            return preds.numpy()

    def save(self, path: str = None, verbose=True) -> str:
        __model = self.model
        self.model = None
        path_final = super().save(path=path, verbose=verbose)
        self.model = __model

        # Export model
        save_pkl.save_with_fn(f'{path_final}{self.model_internals_file_name}', self.model, lambda m, buffer: m.export(buffer), verbose=verbose)
        return path_final

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        from fastai.basic_train import load_learner
        model = super().load(path, reset_paths=reset_paths, verbose=verbose)
        model.model = load_pkl.load_with_fn(f'{model.path}{model.model_internals_file_name}', lambda p: load_learner(model.path, p), verbose=verbose)
        return model

    def _set_default_params(self):
        """ Specifies hyperparameter values to use by default """
        default_params = get_param_baseline(self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type, num_classes=None)

    # TODO: add warning regarding dataloader leak: https://github.com/pytorch/pytorch/issues/31867
    # TODO: Add HPO
    def hyperparameter_tune(self, **kwargs):
        return skip_hpo(self, **kwargs)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_OBJECT],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
