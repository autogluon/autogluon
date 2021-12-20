import logging

import numpy as np
import pytorch_widedeep.training.trainer

from autogluon.common.features.types import R_OBJECT
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from .utils import set_seed
import torch

logger = logging.getLogger(__name__)


class WideDeepNNModel(AbstractModel):
    # TODO: Leverage time_limit
    # TODO: Leverage sample_weight
    # TODO: Experiment with text and image data
    # TODO: Enable usage of the other NN models (TabNet, TabTransformer, etc.)
    # TODO: How to leverage GPU?
    # TODO: Missing value handling?
    def _fit(self,
             X,
             y,
             X_val=None,
             y_val=None,
             sample_weight=None,
             time_limit=None,
             **kwargs):
        # TODO: Add try_import_pytorch_widedeep() to enable a more helpful error message if pytorch_widedeep is not installed
        #  Refer to other model implementations for examples
        from pytorch_widedeep import Trainer
        from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
        from pytorch_widedeep.models import Wide, TabMlp, WideDeep, TabResnet, SAINT, TabTransformer
        from pytorch_widedeep.metrics import Accuracy, R2Score
        # TODO: Use this to get user-specified params instead of hard-coding
        # params = self._get_model_params()

        set_seed(0, True)

        X = self.preprocess(X)

        # prepare wide, crossed, embedding and continuous columns
        # TODO: Either don't use cross_cols or find a way to automatically determine them in a fully automated fashion
        # TODO: Find a way to automatically determine embed dimensions

        embed_cols = []
        cont_cols = self.feature_metadata.get_features(valid_raw_types=['float', 'int'])
        cat_cols = self.feature_metadata.get_features(valid_raw_types=['category'])
        for cat_feat in cat_cols:
            num_categories = len(X[cat_feat].cat.categories)
            if num_categories >= 32:
                embed_cols.append((cat_feat, 32))
            elif num_categories >= 16:
                embed_cols.append((cat_feat, 16))
            elif num_categories >= 8:
                embed_cols.append((cat_feat, 8))

        wide_cols = cat_cols

        if self.problem_type == BINARY:
            objective = 'binary'
            metrics = [Accuracy]
            pred_dim = 1
        elif self.problem_type == MULTICLASS:
            objective = 'multiclass'
            metrics = [Accuracy]
            pred_dim = self.num_classes
        elif self.problem_type == REGRESSION:
            objective = 'regression'
            metrics = [R2Score]
            pred_dim = 1
        else:
            raise ValueError(f'{self.name} does not support the problem_type {self.problem_type}.')

        # wide
        self._wide_preprocessor = WidePreprocessor(wide_cols=wide_cols)
        X_wide = self._wide_preprocessor.fit_transform(X)

        wide = Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=pred_dim)

        # deeptabular
        self._tab_preprocessor = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
        X_tab = self._tab_preprocessor.fit_transform(X)

        self.name = f"{self.params['type']}_{self.name}"
        if self.params['type'] == 'tabmlp':
            model = TabMlp(
                # mlp_hidden_dims=[200, 100],
                column_idx=self._tab_preprocessor.column_idx,
                embed_input=self._tab_preprocessor.embeddings_input,
                continuous_cols=cont_cols,
            )

            # wide and deep
            # model = WideDeep(wide=wide, deeptabular=deeptabular, pred_dim=pred_dim)

        elif self.params['type'] == 'tabresnet':
            model = TabResnet(
                # blocks_dims=[16, 16, 16],
                column_idx=self._tab_preprocessor.column_idx,
                embed_input=self._tab_preprocessor.embeddings_input,
                continuous_cols=cont_cols,
                # blocks_dims=[200,200,200],
                # cont_norm_layer="layernorm",
                # concat_cont_first=False,
                # mlp_hidden_dims=[200, 100],
                # mlp_dropout=0.1,
            )
        elif self.params['type'] == 'SAINT':
            model = SAINT(
                column_idx=self._tab_preprocessor.column_idx,
                embed_input=self._tab_preprocessor.embeddings_input,
                continuous_cols=cont_cols,
                # embed_continuous_activation="leaky_relu",
            )
        elif self.params['type'] == 'tab_transformer':
            model = TabTransformer(
                column_idx=self._tab_preprocessor.column_idx,
                embed_input=self._tab_preprocessor.embeddings_input,
                continuous_cols=cont_cols,
            )
        else:
            raise ValueError(f'Unknown model type {self.params["type"]}')

        model = WideDeep(deeptabular=model, pred_dim=pred_dim)

        X_train = dict()
        X_train['X_wide'] = X_wide
        X_train['X_tab'] = X_tab
        X_train['target'] = y.values

        if X_val is not None and y_val is not None:
            X_val_in = dict()
            X_val_in['X_wide'] = self._wide_preprocessor.transform(X_val)
            X_val_in['X_tab'] = self._tab_preprocessor.transform(X_val)
            X_val_in['target'] = y_val.values
            val_split = None
        else:
            X_val_in = None
            val_split = 0.1

        logger.log(15, model)


        # train the model
        # TODO: Add custom metric support (Convert arbitrary AG metric)

        # DataLoaders are very slow if defaults are used
        # TODO: confirm if this is reproducible on linux
        pytorch_widedeep.training.trainer.n_cpus = 0

        n_epochs = 3
        wide_opt = torch.optim.Adam(model.deeptabular.parameters(), lr=0.02)
        wide_sch = torch.optim.lr_scheduler.CosineAnnealingLR(wide_opt, T_max=n_epochs, eta_min=1e-5)

        trainer = Trainer(model, objective=objective, metrics=metrics)
        # FIXME: Does not return best epoch, instead returns final epoch
        #  Very important to return best epoch, otherwise model can be far worse than ideal
        # FIXME: Add early stopping
        trainer.fit(
            X_train=X_train,
            X_val=X_val_in,
            n_epochs=n_epochs,
            optimizers=[wide_opt],
            lr_schedulers=[wide_sch],
            batch_size=256,
            val_split=val_split,
        )

        self.model = trainer

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)
        X_wide = self._wide_preprocessor.transform(X)
        X_tab = self._tab_preprocessor.transform(X)
        if self.problem_type != REGRESSION:
            preds = self.model.predict_proba(X_wide=X_wide, X_tab=X_tab)
        else:
            preds = self.model.predict(X_wide=X_wide, X_tab=X_tab)

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
