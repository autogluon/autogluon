import inspect
import logging
import time
from typing import Dict

import numpy as np

from autogluon.common.features.types import R_INT, R_FLOAT, R_DATETIME, R_BOOL, R_CATEGORY
from autogluon.common.features.types import R_OBJECT, S_TEXT_NGRAM, S_TEXT_AS_CATEGORY
from autogluon.core.constants import BINARY, REGRESSION, MULTICLASS
from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.utils.files import make_temp_directory
from autogluon.core.utils.try_import import try_import_pytorch_widedeep
from autogluon.features.nn_transforms import MissingFiller, TargetScaler, CategoricalFeaturesFilter
from .hyperparameters.parameters import get_param_baseline
from .hyperparameters.searchspaces import get_default_searchspace
from .metrics import get_nn_metric, get_objective, get_monitor_metric

logger = logging.getLogger(__name__)


class WideDeepNNModel(AbstractModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        import torch
        from .utils import set_seed

        # Deterministic training
        set_seed(0, True)

        params = self._get_model_params()
        logger.log(15, f'Fitting with parameters {params}...')

        X_train, X_valid, cont_cols, embed_cols, val_split = self._preprocess_fit(X, y, X_val, y_val)

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

        tab_opt = torch.optim.Adam(model.deeptabular.parameters(), lr=params['lr'])
        tab_sch = torch.optim.lr_scheduler.OneCycleLR(  # superconvergence schedule
            tab_opt,
            max_lr=params['lr'],
            epochs=params['epochs'],
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
                start_time=start_time,
                time_limit=time_left,
                best_epoch_stop=best_epoch_stop
            )

            if num_cpus is not None:
                system_params = {'num_workers': num_cpus}

            if num_gpus is not None:
                # TODO: Control CPU vs GPU usage during inference
                if num_gpus == 0:
                    system_params['device'] = 'cpu'
                else:
                    # TODO: respect CUDA_VISIBLE_DEVICES to select proper GPU
                    system_params['device'] = 'cuda'

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

            # Torch expects longs for categoricals - converting if present (see dionis dataset)
            for ds in [X_train, X_valid]:
                for k, d in ds.items():
                    if ds[k].dtype == np.uint16:
                        ds[k] = ds[k].astype(np.int64)

            trainer.fit(
                X_train=X_train,
                X_val=X_valid,
                n_epochs=params['epochs'],
                batch_size=batch_size,
                val_split=val_split,
                drop_last=True
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
        from pytorch_widedeep.models import SAINT
        batch_size = params['bs']
        # SAINT need larger batches because it is using information between rows
        if params['type'] == SAINT:
            batch_size *= 2
        return batch_size

    def _preprocess_fit(self, X, y, X_val, y_val):
        from pytorch_widedeep.preprocessing import TabPreprocessor

        # prepare wide, crossed, embedding and continuous columns
        cont_cols = self._feature_metadata.get_features(valid_raw_types=[R_INT, R_FLOAT, R_DATETIME])
        if not cont_cols:
            cont_cols = None

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
        from pytorch_widedeep.models import TabTransformer, FTTransformer, TabFastFormer, TabPerceiver
        for_transformer = self.params['type'] in [TabTransformer, FTTransformer, TabFastFormer, TabPerceiver]
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
        from pytorch_widedeep.models import WideDeep

        if (model_type is None) or (not inspect.isclass(model_type)):
            raise ValueError(f'Unknown model type {model_type}')

        model = model_type(
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
        return get_default_searchspace(self.problem_type)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    # TODO: Fix refit-full (currently implemented using 10% as a validation set because early stopping is requires stopping metric
    #  - re-implement refit callback to use only best_epoch
    def _more_tags(self):
        return {'can_refit_full': False}

    def _get_default_resources(self):
        # Optimal is likely 1, but specifying number of CPUs > 0 is causing crashes in jupyter (but works in IPython/CLI runs)
        # Forcing all workers to be local
        # Another observed message on older instances: `unclosed socket <zmq.Socket(zmq.PUSH)`
        num_cpus = 0
        num_gpus = 0
        return num_cpus, num_gpus

    def _merge_params(self, params):
        return {**params, **self.params.get('model_args', {})}

    def get_minimum_resources(self) -> Dict[str, int]:
        # Overriding this method becuase num_cpus = 0 is a valid value in data loaders.
        # 0 == don't use multi-processing. See comments in _get_default_resources().
        return {
            'num_cpus': 0,
        }


class WideDeepTabMlp(WideDeepNNModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_widedeep.models import TabMlp
        self.params['type'] = TabMlp
        if not self.params.get('use_vanilla', False):
            self.params['model_args'] = self._merge_params({'mlp_batchnorm': True, 'mlp_linear_first': True, 'mlp_batchnorm_last': True})


class WideDeepContextAttentionMLP(WideDeepNNModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_widedeep.models import ContextAttentionMLP
        self.params['type'] = ContextAttentionMLP


class WideDeepSelfAttentionMLP(WideDeepNNModel):

    def __init__(self, **kwargs):
        from pytorch_widedeep.models import SelfAttentionMLP
        super().__init__(**kwargs)
        self.params['type'] = SelfAttentionMLP


class WideDeepTabResnet(WideDeepNNModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_widedeep.models import TabResnet
        self.params['type'] = TabResnet
        if not self.params.get('use_vanilla', False):
            self.params['model_args'] = self._merge_params({'mlp_batchnorm': True, 'mlp_linear_first': True, 'mlp_batchnorm_last': True})


class WideDeepTabNet(WideDeepNNModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_widedeep.models import TabNet
        self.params['type'] = TabNet


class WideDeepSAINT(WideDeepNNModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_widedeep.models import SAINT
        self.params['type'] = SAINT
        if not self.params.get('use_vanilla', False):
            self.params['model_args'] = self._merge_params({'mlp_batchnorm': True, 'mlp_linear_first': True, 'mlp_batchnorm_last': True})


class WideDeepTabTransformer(WideDeepNNModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_widedeep.models import TabTransformer
        self.params['type'] = TabTransformer
        if not self.params.get('use_vanilla', False):
            self.params['model_args'] = self._merge_params({'mlp_batchnorm': True, 'mlp_linear_first': True, 'mlp_batchnorm_last': True, 'embed_continuous': True})


class WideDeepTabFTTransformer(WideDeepNNModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_widedeep.models import FTTransformer
        self.params['type'] = FTTransformer
        if not self.params.get('use_vanilla', False):
            self.params['model_args'] = self._merge_params({'mlp_batchnorm': True, 'mlp_linear_first': True, 'mlp_batchnorm_last': True})


class WideDeepTabFastFormer(WideDeepNNModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_widedeep.models import TabFastFormer
        self.params['type'] = TabFastFormer
        if not self.params.get('use_vanilla', False):
            self.params['model_args'] = self._merge_params({'mlp_batchnorm': True, 'mlp_linear_first': True, 'mlp_batchnorm_last': True})


class WideDeepTabPerceiver(WideDeepNNModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pytorch_widedeep.models import TabPerceiver
        self.params['type'] = TabPerceiver
        if not self.params.get('use_vanilla', False):
            self.params['model_args'] = self._merge_params({'mlp_batchnorm': True, 'mlp_linear_first': True, 'mlp_batchnorm_last': True})
