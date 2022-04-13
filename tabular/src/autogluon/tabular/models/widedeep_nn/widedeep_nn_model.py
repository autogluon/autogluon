import logging
import sys
import time
from typing import Union

import numpy as np
import torch

from autogluon.common.features.types import R_OBJECT, S_TEXT_NGRAM, S_TEXT_AS_CATEGORY
from autogluon.core import metrics
from autogluon.core.constants import BINARY, REGRESSION, MULTICLASS
from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.utils.files import make_temp_directory
from autogluon.core.utils.try_import import try_import_pytorch_widedeep
from autogluon.common.features.types import R_INT, R_FLOAT, R_DATETIME, R_BOOL, R_CATEGORY
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from .metrics import get_nn_metric, get_objective, get_monitor_metric
from .preprocessing_utils import MissingFiller, TargetScaler, CategoricalFeaturesFilter
from .utils import set_seed

logger = logging.getLogger(__name__)


class WideDeepNNModel(AbstractModel):

    def __init__(self, path: str = None, name: str = None, problem_type: str = None, eval_metric: Union[str, metrics.Scorer] = None, hyperparameters=None):
        super().__init__(path, name, problem_type, eval_metric, hyperparameters)
        self.y_scaler = None
        self.missing_filler = None
        self.cont_normalization = None

    # TODO: Leverage sample_weight
    # TODO: Experiment with text and image data
    def _fit(self,
             X,
             y,
             X_val=None,
             y_val=None,
             sample_weight=None,
             time_limit=None,
             num_cpus=None,
             num_gpus=0,
             **kwargs):
        start_time = time.time()

        try_import_pytorch_widedeep()

        from pytorch_widedeep import Trainer
        from pytorch_widedeep.callbacks import ModelCheckpoint
        from .callbacks import EarlyStoppingCallbackWithTimeLimit
        import pytorch_widedeep

        # Deterministic training
        set_seed(0, True)

        params = self._get_model_params()
        logger.log(15, f'Fitting with parameters {params}...')

        X_train, X_valid, cont_cols, embed_cols, val_split = self.__prepare_datasets_before_fit(X, y, X_val, y_val)

        nn_metric = get_nn_metric(self.problem_type, self.stopping_metric, self.num_classes)
        monitor_metric = get_monitor_metric(nn_metric)
        objective = get_objective(self.problem_type, self.stopping_metric)
        pred_dim = self.num_classes if self.problem_type == MULTICLASS else 1

        model = self.__construct_wide_deep_model(
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
        tab_sch = torch.optim.lr_scheduler.OneCycleLR(  # superconvergence schedule
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

            system_params = {}
            if num_cpus is not None:
                # Specifying number of CPUs will cause `unclosed socket <zmq.Socket(zmq.PUSH)` - forcing all workers to be local
                system_params['num_workers'] = 0
            if num_gpus is not None:
                # TODO: Control CPU vs GPU usage during inference
                if num_gpus == 0:
                    system_params['device'] = 'cpu'
                    # Temp workaround: https://github.com/jrzaurin/pytorch-widedeep/issues/89
                    pytorch_widedeep.models.wide_deep.use_cuda = False
                    pytorch_widedeep.models.wide_deep.device = 'cpu'
                else:
                    # TODO: respect CUDA_VISIBLE_DEVICES to select proper GPU
                    system_params['device'] = 'cuda'
                    # Temp workaround: https://github.com/jrzaurin/pytorch-widedeep/issues/89
                    pytorch_widedeep.models.wide_deep.use_cuda = True
                    pytorch_widedeep.models.wide_deep.device = 'cuda'

            trainer = Trainer(
                model,
                objective=objective,
                metrics=[m for m in [nn_metric] if m is not None],
                optimizers=tab_opt,
                lr_schedulers=tab_sch,
                callbacks=[model_checkpoint, early_stopping],
                verbose=kwargs.get('verbosity', 2),
                **system_params
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

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)
        X = self.missing_filler.transform(X)
        X_tab = self._tab_preprocessor.transform(X)
        if self.problem_type != REGRESSION:
            preds = self.model.predict_proba(X_tab=X_tab)
        else:
            preds = self.model.predict(X_tab=X_tab)
            preds = self.y_scaler.inverse_transform(preds)

        if self.problem_type == BINARY:
            return preds[:, 1]
        else:
            return preds

    def __get_batch_size(self, params):
        batch_size = params['bs']
        # SAINT need larger batches because it is using information between rows
        if params['type'] == 'SAINT':
            batch_size *= 2
        return batch_size

    def __prepare_datasets_before_fit(self, X, y, X_val, y_val):
        from pytorch_widedeep.preprocessing import TabPreprocessor

        # prepare wide, crossed, embedding and continuous columns
        cont_cols = self._feature_metadata.get_features(valid_raw_types=[R_INT, R_FLOAT, R_DATETIME])

        cat_cols = self._feature_metadata.get_features(valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL])
        cat_cols = CategoricalFeaturesFilter.filter(X, cat_cols, self.params.get('max_unique_categorical_values', 10000))

        X = self.preprocess(X)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.missing_filler = MissingFiller(self._feature_metadata)
        X = self.missing_filler.fit_transform(X)
        if X_val is not None:
            X_val = self.missing_filler.transform(X_val)

        self.y_scaler = TargetScaler(self.problem_type, self.params.get('y_scaler', None))
        y_norm, y_val_norm = self.y_scaler.fit_transform(y, y_val)

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

    @staticmethod
    def __construct_wide_deep_model(model_type, column_idx, embed_input, continuous_cols, pred_dim, **model_args):
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

    def _more_tags(self):
        return {'can_refit_full': True}
