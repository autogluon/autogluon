import copy
import logging
import psutil
import time
from builtins import classmethod
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from autogluon.common.features.types import R_OBJECT, R_INT, R_FLOAT, R_DATETIME, R_CATEGORY, R_BOOL, S_TEXT_SPECIAL, S_TEXT_NGRAM, S_TEXT_AS_CATEGORY
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.constants import REGRESSION, BINARY, QUANTILE
from autogluon.core.models import AbstractModel
from autogluon.core.utils import try_import_fastai
from autogluon.core.utils.exceptions import TimeLimitExceeded, NotEnoughMemoryError
from autogluon.core.utils.files import make_temp_directory
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace

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
            see also https://docs.fast.ai/callback.schedule.html#Learner.fit_one_cycle,
            One-cycle policy paper: https://arxiv.org/abs/1803.09820

            'epochs': number of epochs; default is 30

            # Early stopping settings. See more details here: https://docs.fast.ai/callback.tracker.html#EarlyStoppingCallback
            'early.stopping.min_delta': 0.0001,
            'early.stopping.patience': 10,
    """

    model_internals_file_name = 'model-internals.pkl'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cat_columns = None
        self.cont_columns = None
        self.columns_fills = None
        self.procs = None
        self.y_scaler = None
        self._cont_normalization = None
        self._load_model = None  # Whether to load inner model when loading.

    def _preprocess_train(self, X, y, X_val, y_val):
        from fastai.tabular.core import TabularPandas
        from fastai.data.block import RegressionBlock, CategoryBlock
        from fastai.data.transforms import IndexSplitter
        from fastcore.basics import range_of

        X = self.preprocess(X, fit=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        from fastai.tabular.core import Categorify
        self.procs = [Categorify]

        if self.problem_type in [REGRESSION, QUANTILE] and self.y_scaler is not None:
            y_norm = pd.Series(self.y_scaler.fit_transform(y.values.reshape(-1, 1)).reshape(-1))
            y_val_norm = pd.Series(self.y_scaler.transform(y_val.values.reshape(-1, 1)).reshape(-1)) if y_val is not None else None
            logger.log(0, f'Training with scaled targets: {self.y_scaler} - !!! NN training metric will be different from the final results !!!')
        else:
            y_norm = y
            y_val_norm = y_val

        logger.log(15, f'Using {len(self.cont_columns)} cont features')
        df_train, train_idx, val_idx = self._generate_datasets(X, y_norm, X_val, y_val_norm)
        y_block = RegressionBlock() if self.problem_type in [REGRESSION, QUANTILE] else CategoryBlock()

        # Copy cat_columns and cont_columns because TabularList is mutating the list
        data = TabularPandas(
            df_train,
            cat_names=self.cat_columns.copy(),
            cont_names=self.cont_columns.copy(),
            procs=self.procs,
            y_block=y_block,
            y_names=LABEL,
            splits=IndexSplitter(val_idx)(range_of(df_train)),
        )
        return data

    def _preprocess(self, X: pd.DataFrame, fit=False, **kwargs):
        X = super()._preprocess(X=X, **kwargs)
        if fit:
            self.cont_columns = self._feature_metadata.get_features(valid_raw_types=[R_INT, R_FLOAT, R_DATETIME])
            self.cat_columns = self._feature_metadata.get_features(valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL])
            self._cont_normalization = (X[self.cont_columns].mean(), X[self.cont_columns].std())

            num_cat_cols_og = len(self.cat_columns)
            if self.cat_columns:
                try:
                    X_stats = X[self.cat_columns].describe(include='all').T.reset_index()
                    cat_cols_to_drop = list(X_stats[(X_stats['unique'] > self.params.get('max_unique_categorical_values', 10000)) | (X_stats['unique'].isna())]['index'].values)
                except:
                    cat_cols_to_drop = []
                if len(cat_cols_to_drop) != 0:
                    cat_cols_to_drop = set(cat_cols_to_drop)
                    self.cat_columns = [col for col in self.cat_columns if (col not in cat_cols_to_drop)]
            num_cat_cols_use = len(self.cat_columns)
            logger.log(15, f'Using {num_cat_cols_use}/{num_cat_cols_og} categorical features')

            nullable_numeric_features = self._feature_metadata.get_features(valid_raw_types=[R_FLOAT, R_DATETIME], invalid_special_types=[S_TEXT_SPECIAL])
            self.columns_fills = dict()
            for c in nullable_numeric_features:  # No need to do this for int features, int can't have null
                self.columns_fills[c] = X[c].mean()
        X = self._fill_missing(X)
        if self.cont_columns:
            cont_mean, cont_std = self._cont_normalization
            X[self.cont_columns] = (X[self.cont_columns] - cont_mean) / cont_std
            X = X.copy()  # Removes potential DF fragmentation
        return X

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIXME: Consider representing categories as int
        if self.columns_fills:
            df = df.fillna(self.columns_fills, inplace=False, downcast=False)
        else:
            df = df.copy()
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
        try_import_fastai()
        from fastai.tabular.model import tabular_config
        from fastai.tabular.learner import tabular_learner
        from fastai import torch_core
        from .callbacks import AgSaveModelCallback, EarlyStoppingCallbackWithTimeLimit
        from .quantile_helpers import HuberPinballLoss

        import torch
        torch.set_num_threads(num_cpus)

        start_time = time.time()
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for NNFastAiTabularModel, this model will ignore them in training.")

        params = self._get_model_params()

        self.y_scaler = params.get('y_scaler', None)
        if self.y_scaler is None:
            if self.problem_type == REGRESSION:
                self.y_scaler = sklearn.preprocessing.StandardScaler()
            elif self.problem_type == QUANTILE:
                self.y_scaler = sklearn.preprocessing.MinMaxScaler()
        else:
            self.y_scaler = copy.deepcopy(self.y_scaler)

        if num_gpus is not None:
            # TODO: Control CPU vs GPU usage during inference
            if num_gpus == 0:
                torch_core.default_device(use_cuda=False)
            else:
                # TODO: respect CUDA_VISIBLE_DEVICES to select proper GPU
                torch_core.default_device(use_cuda=True)

        logger.log(15, f'Fitting Neural Network with parameters {params}...')
        data = self._preprocess_train(X, y, X_val, y_val)

        nn_metric, objective_func_name = self.__get_objective_func_name(self.stopping_metric)
        objective_func_name_to_monitor = self.__get_objective_func_to_monitor(objective_func_name)
        objective_optim_mode = np.less if objective_func_name in [
            'log_loss',
            'root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error',  # Regression objectives
            'pinball_loss',  # Quantile objective
        ] else np.greater

        # TODO: calculate max emb concat layer size and use 1st layer as that value and 2nd in between number of classes and the value
        if params.get('layers', None) is not None:
            layers = params['layers']
        elif self.problem_type in [REGRESSION, BINARY]:
            layers = [200, 100]
        elif self.problem_type == QUANTILE:
            base_size = max(len(self.quantile_levels) * 4, 128)
            layers = [base_size, base_size, base_size]
        else:
            base_size = max(data.c * 2, 100)
            layers = [base_size * 2, base_size]

        loss_func = None
        if self.problem_type == QUANTILE:
            loss_func = HuberPinballLoss(self.quantile_levels, alpha=self.params['alpha'])

        best_epoch_stop = params.get("best_epoch", None)  # Use best epoch for refit_full.
        batch_size = self._get_batch_size(X)
        dls = data.dataloaders(bs=batch_size)

        # Make deterministic
        from fastai.torch_core import set_seed
        set_seed(0, True)
        dls.rng.seed(0)

        if self.problem_type == QUANTILE:
            dls.c = len(self.quantile_levels)

        self.model = tabular_learner(
            dls, layers=layers, metrics=nn_metric,
            config=tabular_config(ps=params['ps'], embed_p=params['emb_drop']),
            loss_func=loss_func,
        )
        logger.log(15, self.model.model)

        fname = 'model'
        save_callback = AgSaveModelCallback(
            monitor=objective_func_name_to_monitor, comp=objective_optim_mode, fname=fname,
            best_epoch_stop=best_epoch_stop, with_opt=True
        )

        if time_limit is not None:
            time_elapsed = time.time() - start_time
            time_left = time_limit - time_elapsed
            if time_left <= time_limit * 0.7:  # if 30% of time was spent preprocessing, likely not enough time to train model
                raise TimeLimitExceeded
        else:
            time_left = None

        early_stopping = EarlyStoppingCallbackWithTimeLimit(
            monitor=objective_func_name_to_monitor,
            comp=objective_optim_mode,
            min_delta=params['early.stopping.min_delta'],
            patience=params['early.stopping.patience'],
            time_limit=time_left, best_epoch_stop=best_epoch_stop
        )

        callbacks = [save_callback, early_stopping]

        with make_temp_directory() as temp_dir:
            with self.model.no_bar():
                with self.model.no_logging():
                    original_path = self.model.path
                    self.model.path = Path(temp_dir)

                    len_val = len(X_val) if X_val is not None else 0
                    epochs = self._get_epochs_number(samples_num=len(X) + len_val, epochs=params['epochs'], batch_size=batch_size, time_left=time_left)
                    if epochs == 0:
                        # Stop early if there is not enough time to train a full epoch
                        raise TimeLimitExceeded

                    self.model.fit_one_cycle(epochs, params['lr'], cbs=callbacks)

                    # Load the best one and export it
                    self.model = self.model.load(fname)

                    if objective_func_name == 'log_loss':
                        eval_result = self.model.validate(dl=dls.valid)[0]
                    else:
                        eval_result = self.model.validate(dl=dls.valid)[1]

                    logger.log(15, f'Model validation metrics: {eval_result}')
                    self.model.path = original_path

            self.params_trained['epochs'] = epochs
            self.params_trained['best_epoch'] = save_callback.best_epoch

    def _get_batch_size(self, X, default_batch_size_for_small_inputs=32):
        bs = self.params['bs']
        if bs == 'auto':
            bs = 512 if len(X) >= 200000 else 256
        bs = bs if len(X) > bs else default_batch_size_for_small_inputs

        if self.params['bs'] == 'auto':
            logger.log(15, f'Automated batch size selection: {bs}')

        return bs

    def _get_epochs_number(self, samples_num, epochs, batch_size, time_left=None, min_batches_count=30, default_epochs=30):
        if epochs == 'auto':
            batches_count = int(samples_num / batch_size) + 1
            if not time_left:
                return default_epochs
            elif batches_count < min_batches_count:
                return default_epochs
            else:
                est_batch_time = self._measure_batch_times(min_batches_count)
                est_epoch_time = batches_count * est_batch_time * 1.1
                est_max_epochs = int(time_left / est_epoch_time)
                epochs = min(default_epochs, est_max_epochs)
                epochs = max(epochs, 0)
                logger.log(15, f'Automated epochs selection: training for {epochs} epoch(s). Estimated time budget use {epochs * est_epoch_time:.2f} / {time_left:.2f} sec')
        return epochs

    def _measure_batch_times(self, min_batches_count):
        from fastai.callback.core import CancelFitException
        from .callbacks import BatchTimeTracker
        batch_time_tracker_callback = BatchTimeTracker(batches_to_measure=min_batches_count)
        try:
            with self.model.no_bar():
                with self.model.no_logging():
                    self.model.fit(1, lr=0, cbs=[batch_time_tracker_callback])
        except CancelFitException:
            pass  # expected early exit
        batch_time = batch_time_tracker_callback.batch_measured_time
        return batch_time

    def _generate_datasets(self, X, y, X_val, y_val):
        df_train = pd.concat([X, X_val], ignore_index=True)
        df_train[LABEL] = pd.concat([y, y_val], ignore_index=True)
        train_idx = np.arange(len(X))
        if X_val is None:
            # use validation set for refit_full case - it's not going to be used for early stopping
            val_idx = np.array([0, 1]) + len(train_idx)
            df_train = pd.concat([df_train, df_train[:2]], ignore_index=True)
        else:
            val_idx = np.arange(len(X_val)) + len(X)
        return df_train, train_idx, val_idx

    def __get_objective_func_name(self, stopping_metric):
        metrics_map = self.__get_metrics_map()

        # Unsupported metrics will be replaced by defaults for a given problem type
        objective_func_name = stopping_metric.name
        if objective_func_name not in metrics_map.keys():
            if self.problem_type == REGRESSION:
                objective_func_name = 'mean_squared_error'
            elif self.problem_type == QUANTILE:
                objective_func_name = 'pinball_loss'
            else:
                objective_func_name = 'log_loss'
            logger.warning(f'Metric {stopping_metric.name} is not supported by this model - using {objective_func_name} instead')

        nn_metric = metrics_map.get(objective_func_name, None)

        return nn_metric, objective_func_name

    def __get_objective_func_to_monitor(self, objective_func_name):
        monitor_obj_func = {
            **{k: m.name if hasattr(m, 'name') else m.__name__ for k, m in self.__get_metrics_map().items() if m is not None},
            'log_loss': 'valid_loss'
        }
        objective_func_name_to_monitor = objective_func_name
        if objective_func_name in monitor_obj_func:
            objective_func_name_to_monitor = monitor_obj_func[objective_func_name]
        return objective_func_name_to_monitor

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)

        single_row = len(X) == 1
        # fastai has issues predicting on a single row, duplicating the row as a workaround
        if single_row:
            X = pd.concat([X, X]).reset_index(drop=True)
        # Copy cat_columns and cont_columns because TabularList is mutating the list
        # TODO: This call has very high fixed cost with many features (0.7s for a single row with 3k features)
        #  Primarily due to normalization performed on the inputs
        test_dl = self.model.dls.test_dl(X, inplace=True)
        with self.model.no_bar():
            with self.model.no_logging():
                preds, _ = self.model.get_preds(dl=test_dl)
        if single_row:
            preds = preds[:1, :]
        if self.problem_type == REGRESSION:
            if self.y_scaler is not None:
                return self.y_scaler.inverse_transform(preds.numpy()).reshape(-1)
            else:
                return preds.numpy().reshape(-1)
        elif self.problem_type == QUANTILE:
            from .quantile_helpers import isotonic
            if self.y_scaler is not None:
                preds = self.y_scaler.inverse_transform(preds.numpy()).reshape(-1, len(self.quantile_levels))
            else:
                preds = preds.numpy().reshape(-1, len(self.quantile_levels))
            return isotonic(preds, self.quantile_levels)
        elif self.problem_type == BINARY:
            return preds[:, 1].numpy()
        else:
            return preds.numpy()

    def save(self, path: str = None, verbose=True) -> str:
        from .fastai_helpers import export
        self._load_model = self.model is not None
        __model = self.model
        self.model = None
        path = super().save(path=path, verbose=verbose)
        self.model = __model
        # Export model
        if self._load_model:
            save_pkl.save_with_fn(
                f'{path}{self.model_internals_file_name}',
                self.model,
                pickle_fn=lambda m, buffer: export(m, buffer),
                verbose=verbose
            )
        self._load_model = None
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        from fastai.learner import load_learner
        model = super().load(path, reset_paths=reset_paths, verbose=verbose)
        if model._load_model:
            model.model = load_pkl.load_with_fn(f'{model.path}{model.model_internals_file_name}', lambda p: load_learner(p), verbose=verbose)
        model._load_model = None
        return model

    def _set_default_params(self):
        """ Specifies hyperparameter values to use by default """
        default_params = get_param_baseline(self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type, num_classes=None)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_default_resources(self):
        # psutil.cpu_count(logical=False) is faster in training than psutil.cpu_count()
        num_cpus = psutil.cpu_count(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus

    def __get_metrics_map(self):
        from fastai.metrics import rmse, mse, mae, accuracy, FBeta, RocAucBinary, Precision, Recall, R2Score
        from .fastai_helpers import medae
        from .quantile_helpers import HuberPinballLoss
        metrics_map = {
            # Regression
            'root_mean_squared_error': rmse,
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'r2': R2Score(),
            'median_absolute_error': medae,

            # Classification
            'accuracy': accuracy,

            'f1': FBeta(beta=1),
            'f1_macro': FBeta(beta=1, average='macro'),
            'f1_micro': FBeta(beta=1, average='micro'),
            'f1_weighted': FBeta(beta=1, average='weighted'),  # this one has some issues

            'roc_auc': RocAucBinary(),

            'precision': Precision(),
            'precision_macro': Precision(average='macro'),
            'precision_micro': Precision(average='micro'),
            'precision_weighted': Precision(average='weighted'),

            'recall': Recall(),
            'recall_macro': Recall(average='macro'),
            'recall_micro': Recall(average='micro'),
            'recall_weighted': Recall(average='weighted'),
            'log_loss': None,

            'pinball_loss': HuberPinballLoss(quantile_levels=self.quantile_levels)
            # Not supported: pac_score
        }
        return metrics_map

    def _estimate_memory_usage(self, X, **kwargs):
        return 10 * get_approximate_df_mem_usage(X).sum()

    def _more_tags(self):
        return {'can_refit_full': True}
