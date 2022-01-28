import logging
from typing import List

from autogluon.core.trainer.abstract_trainer import AbstractTrainer
from autogluon.core.models.ensemble.linear_aggregator_model import LinearAggregatorModel
from autogluon.core.utils import default_holdout_frac, generate_train_test_split, extract_column
from autogluon.core.constants import QUANTILE

from .model_presets.presets import get_preset_models
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

    def fit(self, X, y, hyperparameters, X_val=None, y_val=None, X_unlabeled=None, holdout_frac=0.1, num_stack_levels=0, core_kwargs: dict = None, aux_kwargs: dict = None, time_limit=None, use_bag_holdout=False, groups=None, **kwargs):
        for key in kwargs:
            logger.warning(f'Warning: Unknown argument passed to `AutoTrainer.fit()`. Argument: {key}')

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
                raise AssertionError('X_val, y_val is not None, but bagged mode was specified. If calling from `TabularPredictor.fit()`, `tuning_data` must be None.\n'
                                     'Bagged mode does not use tuning data / validation data. Instead, all data (`train_data` and `tuning_data`) should be combined and specified as `train_data`.\n'
                                     'Bagging/Stacking with a held-out validation set (blend stacking) is not yet supported.')

        self._train_multi_and_ensemble(X, y, X_val, y_val, X_unlabeled=X_unlabeled, hyperparameters=hyperparameters,
                                       num_stack_levels=num_stack_levels, time_limit=time_limit,
                                       core_kwargs=core_kwargs, aux_kwargs=aux_kwargs, groups=groups)

    def stack_new_level_aux(self, X, y, base_model_names: List[str], level,
                            fit=True, stack_name='aux1', time_limit=None, name_suffix: str = None, get_models_func=None, check_if_best=True, **kwargs) -> List[str]:
        aux_models = super().stack_new_level_aux(X, y, base_model_names, level, fit, stack_name, time_limit, name_suffix, get_models_func, check_if_best, **kwargs)
        if self.problem_type == QUANTILE:
            X_stack_preds = self.get_inputs_to_stacker(X, base_models=base_model_names, fit=fit, use_orig_features=True)
            if self.weight_evaluation:
                X, w = extract_column(X, self.sample_weight)  # TODO: consider redesign with w as separate arg instead of bundled inside X
                if w is not None:
                    X_stack_preds[self.sample_weight] = w.values / w.mean()
            hyperparameter_tune_kwargs = None
            ag_args = kwargs.get('ag_args', None)
            excluded_model_types = kwargs.get('excluded_model_types', None)
            if ag_args is not None:
                hyperparameter_tune_kwargs = ag_args.get('hyperparameter_tune_kwargs', None)
            aux_models += self.generate_quantile_aggregator(X=X_stack_preds, y=y,
                                                            level=level, base_model_names=base_model_names, k_fold=0, n_repeats=1,
                                                            stack_name=stack_name, time_limit=time_limit, name_suffix=name_suffix,
                                                            get_models_func=get_models_func, check_if_best=check_if_best,
                                                            excluded_model_types=excluded_model_types,
                                                            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs)

        return aux_models

    def generate_quantile_aggregator(self, X, y, level, base_model_names, k_fold=0, n_repeats=1, stack_name=None,
                                     hyperparameters=None, time_limit=None, name_suffix: str = None,
                                     save_bag_folds=None, check_if_best=True, child_hyperparameters=None,
                                     get_models_func=None, excluded_model_types=None, hyperparameter_tune_kwargs=None) -> List[str]:
        if get_models_func is None:
            get_models_func = self.construct_model_templates
        if len(base_model_names) == 0:
            logger.log(20, 'No base models to train on, skipping quantile aggregator...')
            return []

        if child_hyperparameters is None:
            child_hyperparameters = {}

        if save_bag_folds is None:
            can_infer_dict = self.get_models_attribute_dict('can_infer', models=base_model_names)
            if False in can_infer_dict.values():
                save_bag_folds = False
            else:
                save_bag_folds = True

        aggr_model, aggr_model_args_fit = get_models_func(
            hyperparameters={
                'default': {
                    'QUANTILE_AGGR': [child_hyperparameters],
                }
            },
            ensemble_type=LinearAggregatorModel,
            ensemble_kwargs=dict(
                base_model_names=base_model_names,
                base_model_paths_dict=self.get_models_attribute_dict(attribute='path', models=base_model_names),
                base_model_types_dict=self.get_models_attribute_dict(attribute='type', models=base_model_names),
                base_model_types_inner_dict=self.get_models_attribute_dict(attribute='type_inner',
                                                                           models=base_model_names),
                base_model_performances_dict=self.get_models_attribute_dict(attribute='val_score',
                                                                            models=base_model_names),
                hyperparameters=hyperparameters,
                random_state=level + self.random_state,
            ),
            ag_args={'name_bag_suffix': '', 'hyperparameter_tune_kwargs': hyperparameter_tune_kwargs},
            ag_args_ensemble={'save_bag_folds': save_bag_folds},
            excluded_model_types=excluded_model_types,
            name_suffix=name_suffix,
            level=level,
        )
        if aggr_model_args_fit:
            hyperparameter_tune_kwargs = {
                model_name: aggr_model_args_fit[model_name]['hyperparameter_tune_kwargs']
                for model_name in aggr_model_args_fit if 'hyperparameter_tune_kwargs' in aggr_model_args_fit[model_name]
            }
        w = None
        if self.weight_evaluation:
            X, w = extract_column(X, self.sample_weight)

        models = self._train_multi(
            X=X,
            y=y,
            models=aggr_model,
            k_fold=k_fold,
            n_repeats=n_repeats,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            feature_prune=False,
            stack_name=stack_name,
            level=level,
            time_limit=time_limit,
            ens_sample_weight=w)
        for weighted_ensemble_model_name in models:
            if check_if_best and weighted_ensemble_model_name in self.get_model_names():
                if self.model_best is None:
                    self.model_best = weighted_ensemble_model_name
                else:
                    best_score = self.get_model_attribute(self.model_best, 'val_score')
                    cur_score = self.get_model_attribute(weighted_ensemble_model_name, 'val_score')
                    if cur_score > best_score:
                        # new best model
                        self.model_best = weighted_ensemble_model_name
        return models

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
