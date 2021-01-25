import copy, logging, time
import os
from typing import Dict
import numpy as np
import pandas as pd
from collections import defaultdict

from autogluon.core.utils.utils import generate_kfold
from autogluon.core.constants import MULTICLASS

from ..abstract.abstract_model import AbstractModel
from .bagged_ensemble_model import BaggedEnsembleModel
from ...features.feature_metadata import FeatureMetadata, R_FLOAT, S_STACK

logger = logging.getLogger(__name__)


# TODO: Currently, if this is a stacker above level 1, it will be very slow taking raw input due to each stacker needing to repeat computation on the base models.
#  To solve this, this model must know full context of stacker, and only get preds once for each required model
#  This is already done in trainer, but could be moved internally.
class StackerEnsembleModel(BaggedEnsembleModel):
    def __init__(self, base_model_names=None, base_models_dict=None, base_model_paths_dict=None, base_model_types_dict=None, base_model_types_inner_dict=None, base_model_performances_dict=None, **kwargs):
        super().__init__(**kwargs)
        if base_model_names is None:
            base_model_names = []
        if base_models_dict is None:
            base_models_dict = {}
        if base_model_paths_dict is None:
            base_model_paths_dict = {}
        if base_model_types_dict is None:
            base_model_types_dict = {}
        self.base_model_names = base_model_names
        self.base_models_dict: Dict[str, AbstractModel] = base_models_dict  # String name -> Model objects
        self.base_model_paths_dict = base_model_paths_dict
        self.base_model_types_dict = base_model_types_dict

        if (base_model_performances_dict is not None) and (base_model_types_inner_dict is not None):
            if self.params['max_base_models_per_type'] > 0:
                self.base_model_names = self.limit_models_per_type(models=self.base_model_names, model_types=base_model_types_inner_dict, model_scores=base_model_performances_dict, max_base_models_per_type=self.params['max_base_models_per_type'])
            if self.params['max_base_models'] > 0:
                self.base_model_names = self.limit_models(models=self.base_model_names, model_scores=base_model_performances_dict, max_base_models=self.params['max_base_models'])

        for model_name, model in self.base_models_dict.items():
            if model_name not in self.base_model_names:
                self.base_models_dict.pop(model_name)

        self.stack_column_prefix_lst = copy.deepcopy(self.base_model_names)
        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(stack_column_prefix_lst=self.stack_column_prefix_lst)
        self.stack_column_prefix_to_model_map = {stack_column_prefix: self.base_model_names[i] for i, stack_column_prefix in enumerate(self.stack_column_prefix_lst)}

    @staticmethod
    def limit_models_per_type(models, model_types, model_scores, max_base_models_per_type):
        model_type_groups = defaultdict(list)
        for model in models:
            model_type_groups[model_types[model]].append((model, model_scores[model]))
        models_remain = []
        for key in model_type_groups:
            models_remain += sorted(model_type_groups[key], key=lambda x: x[1], reverse=True)[:max_base_models_per_type]
        models_valid = [model for model, score in models_remain]
        return models_valid

    def limit_models(self, models, model_scores, max_base_models):
        model_types = {model: '' for model in models}
        return self.limit_models_per_type(models=models, model_types=model_types, model_scores=model_scores, max_base_models_per_type=max_base_models)

    def _set_default_params(self):
        default_params = {'use_orig_features': True, 'max_base_models': 25, 'max_base_models_per_type': 5}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()

    def preprocess(self, X, fit=False, compute_base_preds=True, infer=True, model_pred_proba_dict=None, **kwargs):
        if self.stack_column_prefix_lst:
            if infer:
                if set(self.stack_columns).issubset(set(list(X.columns))):
                    compute_base_preds = False  # TODO: Consider removing, this can be dangerous but the code to make this work otherwise is complex (must rewrite predict_proba)
            if compute_base_preds:
                X_stacker = []
                for stack_column_prefix in self.stack_column_prefix_lst:
                    base_model_name = self.stack_column_prefix_to_model_map[stack_column_prefix]
                    if fit:
                        base_model_type = self.base_model_types_dict[base_model_name]
                        base_model_path = self.base_model_paths_dict[base_model_name]
                        y_pred_proba = base_model_type.load_oof(path=base_model_path)
                    elif model_pred_proba_dict and base_model_name in model_pred_proba_dict:
                        y_pred_proba = model_pred_proba_dict[base_model_name]
                    else:
                        base_model = self.load_base_model(base_model_name)
                        y_pred_proba = base_model.predict_proba(X)
                    X_stacker.append(y_pred_proba)  # TODO: This could get very large on a high class count problem. Consider capping to top N most frequent classes and merging least frequent
                X_stacker = self.pred_probas_to_df(X_stacker, index=X.index)
                if self.params['use_orig_features']:
                    X = pd.concat([X_stacker, X], axis=1)
                else:
                    X = X_stacker
            elif not self.params['use_orig_features']:
                X = X[self.stack_columns]
        X = super().preprocess(X, **kwargs)
        return X

    def pred_probas_to_df(self, pred_proba: list, index=None) -> pd.DataFrame:
        if self.problem_type == MULTICLASS:
            pred_proba = np.concatenate(pred_proba, axis=1)
            pred_proba = pd.DataFrame(pred_proba, columns=self.stack_columns)
        else:
            pred_proba = pd.DataFrame(data=np.asarray(pred_proba).T, columns=self.stack_columns)
        if index is not None:
            pred_proba.set_index(index, inplace=True)
        return pred_proba

    def _fit(self, X_train, y_train, k_fold=5, k_fold_start=0, k_fold_end=None, n_repeats=1, n_repeat_start=0, compute_base_preds=True, time_limit=None, **kwargs):
        start_time = time.time()
        # TODO: This could be preprocess=True in general, just have preprocess=False for child models
        X_train = self.preprocess(X=X_train, preprocess=False, fit=True, compute_base_preds=compute_base_preds)
        if time_limit is not None:
            time_limit = time_limit - (time.time() - start_time)
        self._add_stack_to_feature_metadata()
        super()._fit(X_train=X_train, y_train=y_train, k_fold=k_fold, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, time_limit=time_limit, **kwargs)

    def set_contexts(self, path_context):
        path_root_orig = self.path_root
        super().set_contexts(path_context=path_context)
        for model, model_path in self.base_model_paths_dict.items():
            model_local_path = model_path.split(path_root_orig, 1)[1]
            self.base_model_paths_dict[model] = self.path_root + model_local_path

    def set_stack_columns(self, stack_column_prefix_lst):
        if self.problem_type == MULTICLASS:
            stack_columns = [stack_column_prefix + '_' + str(cls) for stack_column_prefix in stack_column_prefix_lst for cls in range(self.num_classes)]
            num_pred_cols_per_model = self.num_classes
        else:
            stack_columns = stack_column_prefix_lst
            num_pred_cols_per_model = 1
        return stack_columns, num_pred_cols_per_model

    def _hyperparameter_tune(self, X_train, y_train, k_fold, scheduler_options, compute_base_preds=True, **kwargs):
        if len(self.models) != 0:
            raise ValueError('self.models must be empty to call hyperparameter_tune, value: %s' % self.models)
        self._add_stack_to_feature_metadata()

        preprocess_kwargs = {'compute_base_preds': compute_base_preds}
        return super()._hyperparameter_tune(X_train=X_train, y_train=y_train, k_fold=k_fold, scheduler_options=scheduler_options, preprocess_kwargs=preprocess_kwargs, **kwargs)

    def _get_init_args(self):
        init_args = dict(
            base_model_names=self.base_model_names,
            base_models_dict=self.base_models_dict,
            base_model_paths_dict=self.base_model_paths_dict,
            base_model_types_dict=self.base_model_types_dict,
        )
        init_args.update(super()._get_init_args())
        return init_args

    def load_base_model(self, model_name):
        if model_name in self.base_models_dict.keys():
            model = self.base_models_dict[model_name]
        else:
            model_type = self.base_model_types_dict[model_name]
            model_path = self.base_model_paths_dict[model_name]
            model = model_type.load(model_path)
        return model

    def get_info(self):
        info = super().get_info()
        stacker_info = dict(
            num_base_models=len(self.base_model_names),
            base_model_names=self.base_model_names,
        )
        children_info = info.pop('children_info')
        info['stacker_info'] = stacker_info
        info['children_info'] = children_info  # Ensure children_info is last in order
        return info

    def _add_stack_to_feature_metadata(self):
        if len(self.models) == 0:
            type_map_raw = {column: R_FLOAT for column in self.stack_columns}
            type_group_map_special = {S_STACK: self.stack_columns}
            stacker_feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)
            if self.feature_metadata is None:  # TODO: This is probably not the best way to do this
                self.feature_metadata = stacker_feature_metadata
            else:
                existing_stack_features = self.feature_metadata.get_features(required_special_types=[S_STACK])
                if set(stacker_feature_metadata.get_features()) != set(existing_stack_features):
                    self.feature_metadata = self.feature_metadata.join_metadata(stacker_feature_metadata)
