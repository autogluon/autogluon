import logging
from typing import Dict, List

from autogluon.core.models import AbstractModel
from autogluon.core.trainer.abstract_trainer import AbstractTrainer
from autogluon.core.utils import generate_train_test_split

from .model_presets.presets import get_preset_models, MODEL_TYPES
from .model_presets.presets_distill import get_preset_models_distillation
from ..models.lgb.lgb_model import LGBModel

logger = logging.getLogger(__name__)


# This Trainer handles model training details
class AutoTrainer(AbstractTrainer):
    def construct_model_templates(self, hyperparameters, **kwargs):
        path = kwargs.pop('path', self.path)
        problem_type = kwargs.pop('problem_type', self.problem_type)
        eval_metric = kwargs.pop('eval_metric', self.eval_metric)
        quantile_levels = kwargs.pop('quantile_levels', self.quantile_levels)
        invalid_model_names = kwargs.pop('invalid_model_names', self._get_banned_model_names())
        silent = kwargs.pop('silent', self.verbosity < 3)
        ag_args_fit = kwargs.pop('ag_args_fit', None)
        if quantile_levels is not None:
            if ag_args_fit is None:
                ag_args_fit = dict()
            ag_args_fit = ag_args_fit.copy()
            ag_args_fit['quantile_levels'] = quantile_levels

        return get_preset_models(path=path,
                                 problem_type=problem_type,
                                 eval_metric=eval_metric,
                                 hyperparameters=hyperparameters,
                                 ag_args_fit=ag_args_fit,
                                 invalid_model_names=invalid_model_names,
                                 silent=silent, **kwargs)

    def fit(self,
            X,
            y,
            hyperparameters,
            X_val=None,
            y_val=None,
            X_unlabeled=None,
            holdout_frac=0.1,
            num_stack_levels=0,
            core_kwargs: dict = None,
            aux_kwargs: dict = None,
            time_limit=None,
            infer_limit=None,
            infer_limit_batch_size=None,
            use_bag_holdout=False,
            groups=None,
            **kwargs):
        for key in kwargs:
            logger.warning(f'Warning: Unknown argument passed to `AutoTrainer.fit()`. Argument: {key}')

        if use_bag_holdout:
            if self.bagged_mode:
                logger.log(20, f'use_bag_holdout={use_bag_holdout}, will use tuning_data as holdout (will not be used for early stopping).')
            else:
                logger.warning(f'Warning: use_bag_holdout={use_bag_holdout}, but bagged mode is not enabled. use_bag_holdout will be ignored.')

        if (y_val is None) or (X_val is None):
            if not self.bagged_mode or use_bag_holdout:
                if groups is not None:
                    raise AssertionError(f'Validation data must be manually specified if use_bag_holdout and groups are both specified.')
                if self.bagged_mode:
                    # Need at least 2 samples of each class in train data after split for downstream k-fold splits
                    # to ensure each k-fold has at least 1 sample of each class in training data
                    min_cls_count_train = 2
                else:
                    min_cls_count_train = 1
                X, X_val, y, y_val = generate_train_test_split(
                    X,
                    y,
                    problem_type=self.problem_type,
                    test_size=holdout_frac,
                    random_state=self.random_state,
                    min_cls_count_train=min_cls_count_train,
                )
                logger.log(20, f'Automatically generating train/validation split with holdout_frac={holdout_frac}, Train Rows: {len(X)}, Val Rows: {len(X_val)}')
        elif self.bagged_mode:
            if not use_bag_holdout:
                # TODO: User could be intending to blend instead. Add support for blend stacking.
                #  This error message is necessary because when calculating out-of-fold predictions for user, we want to return them in the form given in train_data,
                #  but if we merge train and val here, it becomes very confusing from a users perspective, especially because we reset index, making it impossible to match
                #  the original train_data to the out-of-fold predictions from `predictor.get_oof_pred_proba()`.
                raise AssertionError('X_val, y_val is not None, but bagged mode was specified. '
                                     'If calling from `TabularPredictor.fit()`, `tuning_data` should be None.\n'
                                     'Default bagged mode does not use tuning data / validation data. '
                                     'Instead, all data (`train_data` and `tuning_data`) should be combined and specified as `train_data`.\n'
                                     'To avoid this error and use `tuning_data` as holdout data in bagged mode, '
                                     'specify the following:\n'
                                     '\tpredictor.fit(..., tuning_data=tuning_data, use_bag_holdout=True)')

        self._train_multi_and_ensemble(X=X,
                                       y=y,
                                       X_val=X_val,
                                       y_val=y_val,
                                       X_unlabeled=X_unlabeled,
                                       hyperparameters=hyperparameters,
                                       num_stack_levels=num_stack_levels,
                                       time_limit=time_limit,
                                       core_kwargs=core_kwargs,
                                       aux_kwargs=aux_kwargs,
                                       infer_limit=infer_limit,
                                       infer_limit_batch_size=infer_limit_batch_size,
                                       groups=groups)

    def construct_model_templates_distillation(self, hyperparameters, **kwargs):
        path = kwargs.pop('path', self.path)
        problem_type = kwargs.pop('problem_type', self.problem_type)
        eval_metric = kwargs.pop('eval_metric', self.eval_metric)
        invalid_model_names = kwargs.pop('invalid_model_names', self._get_banned_model_names())
        silent = kwargs.pop('silent', self.verbosity < 3)

        # TODO: QUANTILE VERSION?

        return get_preset_models_distillation(
            path=path,
            problem_type=problem_type,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            invalid_model_names=invalid_model_names,
            silent=silent,
            **kwargs,
        )

    def _get_default_proxy_model_class(self):
        return LGBModel

    def compile_models(self, model_names='all', with_ancestors=False, compiler_configs: dict = None) -> List[str]:
        """Ensures that compiler_configs maps to the correct models if the user specified the same keys as in hyperparameters such as RT, XT, etc."""
        if compiler_configs is not None:
            model_types_map = self._get_model_types_map()
            compiler_configs_new = dict()
            for k in compiler_configs:
                if k in model_types_map:
                    compiler_configs_new[model_types_map[k]] = compiler_configs[k]
                else:
                    compiler_configs_new[k] = compiler_configs[k]
            compiler_configs = compiler_configs_new
        return super().compile_models(model_names=model_names, with_ancestors=with_ancestors, compiler_configs=compiler_configs)

    def _get_model_types_map(self) -> Dict[str, AbstractModel]:
        return MODEL_TYPES
