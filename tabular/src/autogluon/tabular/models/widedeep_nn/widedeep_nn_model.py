import logging
import time
from inspect import isclass
from typing import Union

import numpy as np
import pandas as pd
import sklearn
import torch
from autogluon.common.features.types import R_OBJECT, S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL
from autogluon.core import metrics
from autogluon.core.constants import BINARY, REGRESSION, MULTICLASS, QUANTILE
from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.utils.files import make_temp_directory
from autogluon.core.utils.try_import import try_import_pytorch_widedeep
from common.src.autogluon.common.features.types import R_INT, R_FLOAT, R_DATETIME, R_BOOL, R_CATEGORY
from torchmetrics import F1Score

from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from .utils import set_seed

logger = logging.getLogger(__name__)


class WideDeepNNModel(AbstractModel):

    def __init__(self, path: str = None, name: str = None, problem_type: str = None, eval_metric: Union[str, metrics.Scorer] = None, hyperparameters=None):
        super().__init__(path, name, problem_type, eval_metric, hyperparameters)
        self.y_scaler = None
        self.columns_fills = None

    # TODO: Leverage sample_weight
    # TODO: Experiment with text and image data
    # TODO: How to leverage GPU?
    # TODO: Missing value handling?
    def _fit(self, X, y, X_val=None, y_val=None, sample_weight=None, time_limit=None, **kwargs):
        start_time = time.time()

        try_import_pytorch_widedeep()

        from pytorch_widedeep import Trainer
        import pytorch_widedeep.training.trainer
        from pytorch_widedeep.callbacks import ModelCheckpoint
        from .callbacks import EarlyStoppingCallbackWithTimeLimit

        # Deterministic training
        set_seed(0, True)

        # TODO: confirm if this is reproducible on linux
        # DataLoaders are very slow if defaults are used
        pytorch_widedeep.training.trainer.n_cpus = 0

        params = self._get_model_params()
        logger.log(15, f'Fitting with parameters {params}...')

        X_train, X_valid, cont_cols, embed_cols, val_split = self.__prepare_datasets(X, y, X_val, y_val)

        nn_metric = self.__get_nn_metric(self.stopping_metric, self.num_classes)
        monitor_metric = self.__get_monitor_metric(nn_metric)
        objective = self.__get_objective(self.problem_type, self.stopping_metric)
        pred_dim = self.__get_output_dim(self.problem_type, self.num_classes)

        model = self._construct_wide_deep_model(
            model_type=self.params['type'],
            column_idx=self._tab_preprocessor.column_idx,
            embed_input=None if embed_cols is None else self._tab_preprocessor.cat_embed_input,
            continuous_cols=cont_cols,
            pred_dim=pred_dim,
            **self.params.get('model_args', {})
        )
        logger.log(15, model)

        best_epoch = None
        best_epoch_stop = params.get("best_epoch", None)  # Use best epoch for refit_full.
        batch_size = self.__get_batch_size(params)

        tab_opt = torch.optim.Adam(model.deeptabular.parameters(), lr=(params['lr']))
        tab_sch = torch.optim.lr_scheduler.OneCycleLR(  # howard superconvergence schedule
            tab_opt,
            max_lr=(params['lr']),
            epochs=(params['epochs']),
            steps_per_epoch=int(np.ceil(len(X_train['X_tab']) / batch_size)),
            pct_start=0.25,
            final_div_factor=1e5
        )

        with make_temp_directory() as temp_dir:
            checkpoint_path_prefix = f'{temp_dir}/model'

            model_checkpoint = ModelCheckpoint(
                filepath=checkpoint_path_prefix,
                verbose=kwargs.get('verbosity', 2),
                save_best_only=True,
                max_save=1,
                monitor=monitor_metric
            )

            if time_limit is not None:
                time_elapsed = time.time() - start_time
                time_left = time_limit - time_elapsed
                if time_left <= time_limit * 0.7:  # if 30% of time was spent preprocessing, likely not enough time to train model
                    raise TimeLimitExceeded
            else:
                time_left = None

            early_stopping = EarlyStoppingCallbackWithTimeLimit(
                monitor=monitor_metric,
                mode='auto',
                min_delta=params['early.stopping.min_delta'],
                patience=params['early.stopping.patience'],
                time_limit=time_left,
                best_epoch_stop=best_epoch_stop
            )

            trainer = Trainer(
                model,
                objective=objective,
                metrics=[m for m in [nn_metric] if m is not None],
                optimizers=tab_opt,
                lr_schedulers=tab_sch,
                callbacks=[model_checkpoint, early_stopping],
                verbose=kwargs.get('verbosity', 2),
            )

            trainer.fit(
                X_train=X_train,
                X_val=X_valid,
                n_epochs=params['epochs'],
                batch_size=batch_size,
                val_split=val_split,
            )
            best_epoch = model_checkpoint.best_epoch
            trainer.model.load_state_dict(torch.load(model_checkpoint.old_files[-1]))

        self.model = trainer

        # TODO: add dynamic epochs selection
        self.params_trained['epochs'] = params['epochs']
        self.params_trained['best_epoch'] = best_epoch

    def __get_batch_size(self, params):
        batch_size = params['bs']
        # SAINT need larger batches because it is using information between rows
        if params['type'] == 'SAINT':
            batch_size *= 2
        return batch_size

    def __prepare_datasets(self, X, y, X_val, y_val):
        from pytorch_widedeep.preprocessing import TabPreprocessor

        # prepare wide, crossed, embedding and continuous columns
        cont_cols = self._feature_metadata.get_features(valid_raw_types=[R_INT, R_FLOAT, R_DATETIME])
        cat_cols = self._feature_metadata.get_features(valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL])

        X = self.preprocess(X)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        # TODO: refactor common things between widedeep and fastai models
        # TODO: add high-cardinality categorical variables trimming
        self.y_scaler = self.params.get('y_scaler', None)
        if self.y_scaler is None:
            if self.problem_type == REGRESSION:
                self.y_scaler = sklearn.preprocessing.StandardScaler()
            elif self.problem_type == QUANTILE:
                self.y_scaler = sklearn.preprocessing.MinMaxScaler()
        else:
            self.y_scaler = copy.deepcopy(self.y_scaler)

        nullable_numeric_features = self._feature_metadata.get_features(valid_raw_types=[R_FLOAT, R_DATETIME], invalid_special_types=[S_TEXT_SPECIAL])
        self.columns_fills = dict()
        for c in nullable_numeric_features:  # No need to do this for int features, int can't have null
            self.columns_fills[c] = X[c].mean()
        X = self._fill_missing(X, self.columns_fills)
        if X_val is not None:
            X_val = self._fill_missing(X_val, self.columns_fills)

        if self.problem_type in [REGRESSION, QUANTILE] and self.y_scaler is not None:
            y_norm = pd.Series(self.y_scaler.fit_transform(y.values.reshape(-1, 1)).reshape(-1))
            y_val_norm = pd.Series(self.y_scaler.transform(y_val.values.reshape(-1, 1)).reshape(-1)) if y_val is not None else None
            logger.log(0, f'Training with scaled targets: {self.y_scaler} - !!! NN training metric will be different from the final results !!!')
        else:
            y_norm = y
            y_val_norm = y_val

        embed_cols, for_transformer = self.__get_embedding_columns_metadata(X, cat_cols)
        self._tab_preprocessor = TabPreprocessor(
            embed_cols=embed_cols,
            continuous_cols=cont_cols,
            for_transformer=for_transformer
        )

        X_tab = self._tab_preprocessor.fit_transform(X)
        X_train = {'X_tab': X_tab, 'target': y_norm.values}
        if X_val is not None and y_val is not None:
            X_valid = {'X_tab': self._tab_preprocessor.transform(X_val), 'target': y_val_norm.values}
            val_split = None
        else:
            X_valid = None
            val_split = 0.1

        return X_train, X_valid, cont_cols, embed_cols, val_split

    def __get_monitor_metric(self, nn_metric):
        if nn_metric is None:
            return 'val_loss'
        monitor_metric = nn_metric
        if isclass(monitor_metric):
            metric = monitor_metric()
            if hasattr(metric, '_name'):
                return f'val_{metric._name}'
        elif not isclass(monitor_metric):
            monitor_metric = monitor_metric.__class__
        monitor_metric = f'val_{monitor_metric.__name__}'
        return monitor_metric

    def __get_embedding_columns_metadata(self, X, cat_cols):
        for_transformer = self.params['type'] in ['tab_transformer', 'ft_transformer', 'tab_perciever', 'tab_fastformer']
        if for_transformer:
            embed_cols = cat_cols
        else:
            embed_cols = []
            for cat_feat in cat_cols:
                num_categories = len(X[cat_feat].cat.categories)
                embed_cols.append((cat_feat, min(600, round(1.6 * num_categories ** 0.56))))
        if len(embed_cols) == 0:
            embed_cols = None
            for_transformer = False
        return embed_cols, for_transformer

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)
        X = self._fill_missing(X, self.columns_fills)
        X_tab = self._tab_preprocessor.transform(X)
        if self.problem_type != REGRESSION:
            preds = self.model.predict_proba(X_tab=X_tab)
        else:
            preds = self.model.predict(X_tab=X_tab)
            if self.y_scaler is not None:
                preds = self.y_scaler.inverse_transform(preds.reshape(-1,1)).reshape(-1)

        if self.problem_type == BINARY:
            return preds[:, 1]
        else:
            return preds

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_OBJECT],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @staticmethod
    def _construct_wide_deep_model(model_type, column_idx, embed_input, continuous_cols, pred_dim, **model_args):
        from pytorch_widedeep.models import TabMlp, WideDeep, TabResnet, SAINT, TabTransformer, FTTransformer, TabPerceiver, ContextAttentionMLP, SelfAttentionMLP, TabNet, TabFastFormer

        model_cls = dict(
            tab_mlp=TabMlp,
            context_attention_mlp=ContextAttentionMLP,
            self_attention_mlp=SelfAttentionMLP,
            tabresnet=TabResnet,
            tabnet=TabNet,
            SAINT=SAINT,
            tab_transformer=TabTransformer,
            ft_transformer=FTTransformer,
            tab_fastformer=TabFastFormer,
            tab_perciever=TabPerceiver,
        ).get(model_type, None)

        if model_cls is None:
            raise ValueError(f'Unknown model type {model_type}')

        model = model_cls(
            column_idx=column_idx,
            cat_embed_input=embed_input,
            continuous_cols=continuous_cols,
            **model_args
        )
        model = WideDeep(deeptabular=model, pred_dim=pred_dim)
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

    def __get_objective(self, problem_type, stopping_metric):
        if problem_type == BINARY:
            return 'binary'
        elif problem_type == MULTICLASS:
            return 'multiclass'
        elif problem_type == REGRESSION:
            # See supported objectives: https://pytorch-widedeep.readthedocs.io/en/latest/trainer.html#pytorch_widedeep.training.Trainer
            return {
                'root_mean_squared_error': 'root_mean_squared_error',
                'mean_squared_error': 'regression',
                'mean_absolute_error': 'mean_absolute_error',
            }.get(stopping_metric, 'regression')
        else:
            raise ValueError(f'Unsupported problem type {problem_type}')

    def __get_output_dim(self, problem_type, num_classes):
        return {
            BINARY: 1,
            MULTICLASS: num_classes,
            REGRESSION: 1,
        }.get(problem_type, 1)

    def __get_metrics_map(self, num_classes):
        from pytorch_widedeep.metrics import Accuracy, R2Score
        import torchmetrics as tm
        num_classes = 2 if num_classes is None else num_classes
        metrics_map = {
            # Regression
            'root_mean_squared_error': tm.MeanSquaredError(squared=False),
            'mean_squared_error': tm.MeanSquaredError(),
            'mean_absolute_error': tm.MeanAbsoluteError(),
            'r2': R2Score,
            # Not supported: 'median_absolute_error': None,

            # Classification
            'accuracy': Accuracy,

            'f1': F1Score(num_classes=num_classes),
            'f1_macro': tm.F1Score(average='macro', num_classes=num_classes),
            'f1_micro': tm.F1Score(average='micro', num_classes=num_classes),
            'f1_weighted': tm.F1Score(average='weighted', num_classes=num_classes),

            'roc_auc': tm.AUROC(num_classes=num_classes),

            'precision': tm.Precision(num_classes=num_classes),
            'precision_macro': tm.Precision(average='macro', num_classes=num_classes),
            'precision_micro': tm.Precision(average='micro', num_classes=num_classes),
            'precision_weighted': tm.Precision(average='weighted', num_classes=num_classes),

            'recall': tm.Recall(num_classes=num_classes),
            'recall_macro': tm.Recall(average='macro', num_classes=num_classes),
            'recall_micro': tm.Recall(average='micro', num_classes=num_classes),
            'recall_weighted': tm.Recall(average='weighted', num_classes=num_classes),
            'log_loss': None,

            # Not supported: 'pinball_loss': None
            # Not supported: pac_score
        }
        return metrics_map

    def __get_nn_metric(self, stopping_metric, num_classes):
        metrics_map = self.__get_metrics_map(num_classes)

        # Unsupported metrics will be replaced by defaults for a given problem type
        objective_func_name = stopping_metric.name
        if objective_func_name not in metrics_map.keys():
            if self.problem_type == REGRESSION:
                objective_func_name = 'mean_squared_error'
            else:
                objective_func_name = 'log_loss'
            logger.warning(f'Metric {stopping_metric.name} is not supported by this model - using {objective_func_name} instead')

        nn_metric = metrics_map.get(objective_func_name, None)

        return nn_metric

    def _fill_missing(self, df: pd.DataFrame, columns_fills) -> pd.DataFrame:
        if columns_fills:
            df = df.fillna(columns_fills, inplace=False, downcast=False)
        else:
            df = df.copy()
        return df

    def _more_tags(self):
        return {'can_refit_full': True}
