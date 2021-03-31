import copy
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.core.constants import REGRESSION, BINARY, MULTICLASS
from autogluon.core.utils import try_import_fastai_v1
from autogluon.core.utils.files import make_temp_directory
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.multiprocessing_utils import is_fork_enabled
from autogluon.core.utils.savers import save_pkl
from autogluon.core.features.types import R_OBJECT, R_INT, R_FLOAT, R_DATETIME, R_CATEGORY, R_BOOL

from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from autogluon.core.models import AbstractModel
from autogluon.core.models.abstract.model_trial import skip_hpo

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
MISSING = '__!#ag_internal_missing#!__'

logger = logging.getLogger(__name__)


# TODO: Takes extremely long time prior to training start if many (10000) continuous features from ngrams, debug - explore TruncateSVD option to reduce input dimensionality
# TODO: currently fastai automatically detect and use CUDA if available - add code to honor autogluon settings
class NNFastAiTabularModel(AbstractModel):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cat_columns = None
        self.cont_columns = None
        self.columns_fills = None
        self.procs = None
        self.y_scaler = None
        self._inner_features = None
        self._load_model = None  # Whether to load inner model when loading.

    def _preprocess_train(self, X, y, X_val, y_val, num_workers):
        from fastai.data_block import FloatList
        from fastai.tabular import TabularList

        X = self.preprocess(X, fit=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        from fastai.tabular import FillMissing, Categorify, Normalize
        self.procs = [FillMissing, Categorify, Normalize]

        if self.problem_type == REGRESSION and self.y_scaler is not None:
            y_norm = pd.Series(self.y_scaler.fit_transform(y.values.reshape(-1, 1)).reshape(-1))
            y_val_norm = pd.Series(self.y_scaler.transform(y_val.values.reshape(-1, 1)).reshape(-1)) if y_val is not None else None
            logger.log(0, f'Training with scaled targets: {self.y_scaler} - !!! NN training metric will be different from the final results !!!')
        else:
            y_norm = y
            y_val_norm = y_val

        logger.log(15, f'Using {len(self.cont_columns)} cont features')
        df_train, train_idx, val_idx = self._generate_datasets(X, y_norm, X_val, y_val_norm)
        label_class = FloatList if self.problem_type == REGRESSION else None

        # Copy cat_columns and cont_columns because TabularList is mutating the list
        data = (TabularList.from_df(df_train, path=self.path,
                                    cat_names=self.cat_columns.copy(), cont_names=self.cont_columns.copy(), procs=self.procs)
                .split_by_idxs(train_idx, val_idx)
                .label_from_df(cols=LABEL, label_cls=label_class)
                .databunch(bs=self.params['bs'] if len(X) > self.params['bs'] else 32, num_workers=num_workers))
        return data

    def _preprocess(self, X: pd.DataFrame, fit=False, **kwargs):
        X = super()._preprocess(X=X, **kwargs)
        if fit:
            self.cat_columns = self.feature_metadata.get_features(valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL])
            self.cont_columns = self.feature_metadata.get_features(valid_raw_types=[R_INT, R_FLOAT, R_DATETIME])
            try:
                X_stats = X.describe(include='all').T.reset_index()
                cat_cols_to_drop = X_stats[(X_stats['unique'] > self.params.get('max_unique_categorical_values', 10000)) | (X_stats['unique'].isna())]['index'].values
            except:
                cat_cols_to_drop = []
            cat_cols_to_keep = [col for col in X.columns.values if (col not in cat_cols_to_drop)]
            cat_cols_to_use = [col for col in self.cat_columns if col in cat_cols_to_keep]
            logger.log(15, f'Using {len(cat_cols_to_use)}/{len(self.cat_columns)} categorical features')
            self.cat_columns = cat_cols_to_use
            self.cat_columns = [feature for feature in self.cat_columns if feature in list(X.columns)]
            self.cont_columns = [feature for feature in self.cont_columns if feature in list(X.columns)]

            self.columns_fills = {}
            for c in self.cat_columns:
                self.columns_fills[c] = MISSING
            for c in self.cont_columns:
                self.columns_fills[c] = X[c].mean()
            self._inner_features = self.cat_columns + self.cont_columns
        return self._fill_missing(X)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[self._inner_features].copy()
        for c in self.cat_columns:
            df[c] = df[c].cat.add_categories(MISSING)
            df[c] = df[c].fillna(self.columns_fills[c])
        for c in self.cont_columns:
            df[c] = df[c].fillna(self.columns_fills[c])
        return df

    def _fit(self,
             X,
             y,
             X_val=None,
             y_val=None,
             time_limit=None,
             num_cpus=None,
             num_gpus=0,
             sample_weight=None,
             **kwargs):
        try_import_fastai_v1()
        import torch
        from fastai.layers import LabelSmoothingCrossEntropy
        from fastai.tabular import tabular_learner
        from fastai.utils.mod_display import progress_disabled_ctx
        from fastai.core import defaults
        from .callbacks import EarlyStoppingCallbackWithTimeLimit, SaveModelCallback

        start_time = time.time()
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for NNFastAiTabularModel, this model will ignore them in training.")

        params = self._get_model_params()

        self.y_scaler = params.get('y_scaler', None)
        if self.y_scaler is not None:
            self.y_scaler = copy.deepcopy(self.y_scaler)

        if num_cpus is None:
            num_cpus = defaults.cpus
        # additional workers are helping only when fork is enabled; in other mp modes, communication overhead reduces performance
        num_workers = int(num_cpus / 2)
        if not is_fork_enabled():
            num_workers = 0
        if num_gpus is not None:
            if num_gpus == 0:
                # TODO: Does not obviously impact inference speed
                defaults.device = torch.device('cpu')
            else:
                defaults.device = torch.device('cuda')

        logger.log(15, f'Fitting Neural Network with parameters {params}...')
        data = self._preprocess_train(X, y, X_val, y_val, num_workers=num_workers)

        nn_metric, objective_func_name = self.__get_objective_func_name()
        objective_func_name_to_monitor = self.__get_objective_func_to_monitor(objective_func_name)
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

    def _generate_datasets(self, X, y, X_val, y_val):
        df_train = pd.concat([X, X_val], ignore_index=True)
        df_train[LABEL] = pd.concat([y, y_val], ignore_index=True)
        train_idx = np.arange(len(X))
        if X_val is None:
            val_idx = train_idx  # use validation set for refit_full case - it's not going to be used for early stopping
        else:
            val_idx = np.arange(len(X_val)) + len(X)
        return df_train, train_idx, val_idx

    def __get_objective_func_name(self):
        from fastai.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, accuracy, FBeta, AUROC, Precision, Recall, r2_score

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
            'f1_weighted': FBeta(beta=1, average='weighted'),  # this one has some issues

            'roc_auc': AUROC(),

            'precision': Precision(),
            'precision_macro': Precision(average='macro'),
            'precision_micro': Precision(average='micro'),
            'precision_weighted': Precision(average='weighted'),

            'recall': Recall(),
            'recall_macro': Recall(average='macro'),
            'recall_micro': Recall(average='micro'),
            'recall_weighted': Recall(average='weighted'),
            'log_loss': None,
            # Not supported: pac_score
        }

        # Unsupported metrics will be replaced by defaults for a given problem type
        objective_func_name = self.stopping_metric.name
        if objective_func_name not in metrics_map.keys():
            if self.problem_type == REGRESSION:
                objective_func_name = 'mean_squared_error'
            else:
                objective_func_name = 'log_loss'
            logger.warning(f'Metric {self.stopping_metric.name} is not supported by this model - using {objective_func_name} instead')

        if objective_func_name in metrics_map.keys():
            nn_metric = metrics_map[objective_func_name]
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
            'precision_weighted': 'precision',

            'recall_macro': 'recall',
            'recall_micro': 'recall',
            'recall_weighted': 'recall',
            'log_loss': 'valid_loss',
        }
        objective_func_name_to_monitor = objective_func_name
        if objective_func_name in monitor_obj_func:
            objective_func_name_to_monitor = monitor_obj_func[objective_func_name]
        return objective_func_name_to_monitor

    def _predict_proba(self, X, **kwargs):
        from fastai.basic_data import DatasetType
        from fastai.tabular import TabularList
        from fastai.utils.mod_display import progress_disabled_ctx

        X = self.preprocess(X, **kwargs)

        single_row = len(X) == 1
        # fastai has issues predicting on a single row, duplicating the row as a workaround
        if single_row:
            X = pd.concat([X, X]).reset_index(drop=True)

        # Copy cat_columns and cont_columns because TabularList is mutating the list
        self.model.data.add_test(TabularList.from_df(
            X, cat_names=self.cat_columns.copy(), cont_names=self.cont_columns.copy(), procs=self.procs))
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
        self._load_model = self.model is not None
        __model = self.model
        self.model = None
        path = super().save(path=path, verbose=verbose)
        self.model = __model
        # Export model
        if self._load_model:
            save_pkl.save_with_fn(f'{path}{self.model_internals_file_name}', self.model, lambda m, buffer: m.export(buffer), verbose=verbose)
        self._load_model = None
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model = super().load(path, reset_paths=reset_paths, verbose=verbose)
        if model._load_model:
            from fastai.basic_train import load_learner
            model.model = load_pkl.load_with_fn(f'{model.path}{model.model_internals_file_name}', lambda p: load_learner(model.path, p), verbose=verbose)
        model._load_model = None
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
    def _hyperparameter_tune(self, **kwargs):
        return skip_hpo(self, **kwargs)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_OBJECT],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
