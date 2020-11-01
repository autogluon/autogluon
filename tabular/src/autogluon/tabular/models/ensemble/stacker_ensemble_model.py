import copy, logging, time
import os
from typing import Dict
import numpy as np
import pandas as pd
from collections import defaultdict

from autogluon.core.utils.utils import generate_kfold
from ..abstract.abstract_model import AbstractModel
from .bagged_ensemble_model import BaggedEnsembleModel
from ...constants import MULTICLASS
from ...features.feature_metadata import FeatureMetadata, R_FLOAT, S_STACK

logger = logging.getLogger(__name__)


# TODO: Currently, if this is a stacker above level 1, it will be very slow taking raw input due to each stacker needing to repeat computation on the base models.
#  To solve this, this model must know full context of stacker, and only get preds once for each required model
#  This is already done in trainer, but could be moved internally.
class StackerEnsembleModel(BaggedEnsembleModel):
    def __init__(self, base_model_names=None, base_models_dict=None, base_model_paths_dict=None, base_model_types_dict=None, base_model_types_inner_dict=None, base_model_performances_dict=None, use_orig_features=True, **kwargs):
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
        self.use_orig_features = use_orig_features

        if (base_model_performances_dict is not None) and (base_model_types_inner_dict is not None):
            if self.params['max_models_per_type'] > 0:
                self.base_model_names = self.limit_models_per_type(models=self.base_model_names, model_types=base_model_types_inner_dict, model_scores=base_model_performances_dict, max_models_per_type=self.params['max_models_per_type'])
            if self.params['max_models'] > 0:
                self.base_model_names = self.limit_models(models=self.base_model_names, model_scores=base_model_performances_dict, max_models=self.params['max_models'])

        for model_name, model in self.base_models_dict.items():
            if model_name not in self.base_model_names:
                self.base_models_dict.pop(model_name)

        self.stack_column_prefix_lst = copy.deepcopy(self.base_model_names)
        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(stack_column_prefix_lst=self.stack_column_prefix_lst)
        self.stack_column_prefix_to_model_map = {stack_column_prefix: self.base_model_names[i] for i, stack_column_prefix in enumerate(self.stack_column_prefix_lst)}

    @staticmethod
    def limit_models_per_type(models, model_types, model_scores, max_models_per_type):
        model_type_groups = defaultdict(list)
        for model in models:
            model_type = model_types[model]
            model_type_groups[model_type].append((model, model_scores[model]))
        for key in model_type_groups:
            model_type_groups[key] = sorted(model_type_groups[key], key=lambda x: x[1], reverse=True)
        for key in model_type_groups:
            model_type_groups[key] = model_type_groups[key][:max_models_per_type]
        models_remain = []
        for key in model_type_groups:
            models_remain += model_type_groups[key]
        models_valid = [model for model, score in models_remain]
        return models_valid

    def limit_models(self, models, model_scores, max_models):
        model_types = {model: '' for model in models}
        return self.limit_models_per_type(models=models, model_types=model_types, model_scores=model_scores, max_models_per_type=max_models)

    def _set_default_params(self):
        default_params = {'max_models': 25, 'max_models_per_type': 5}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

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
                if self.use_orig_features:
                    X = pd.concat([X_stacker, X], axis=1)
                else:
                    X = X_stacker
            elif not self.use_orig_features:
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

    def _fit(self, X, y, k_fold=5, k_fold_start=0, k_fold_end=None, n_repeats=1, n_repeat_start=0, compute_base_preds=True, time_limit=None, **kwargs):
        start_time = time.time()
        # TODO: This could be preprocess=True in general, just have preprocess=False for child models
        X = self.preprocess(X=X, preprocess=False, fit=True, compute_base_preds=compute_base_preds)
        if time_limit is not None:
            time_limit = time_limit - (time.time() - start_time)
        if len(self.models) == 0:
            type_map_raw = {column: R_FLOAT for column in self.stack_columns}
            type_group_map_special = {S_STACK: self.stack_columns}
            stacker_feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)
            if self.feature_metadata is None:  # TODO: This is probably not the best way to do this
                self.feature_metadata = stacker_feature_metadata
            else:
                self.feature_metadata = self.feature_metadata.join_metadata(stacker_feature_metadata)
        super()._fit(X=X, y=y, k_fold=k_fold, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, time_limit=time_limit, **kwargs)

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

    # TODO: Currently double disk usage, saving model in HPO and also saving model in stacker
    def hyperparameter_tune(self, X, y, k_fold, scheduler_options=None, compute_base_preds=True, **kwargs):
        if len(self.models) != 0:
            raise ValueError('self.models must be empty to call hyperparameter_tune, value: %s' % self.models)

        if len(self.models) == 0:
            type_map_raw = {column: R_FLOAT for column in self.stack_columns}
            type_group_map_special = {S_STACK: self.stack_columns}
            stacker_feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)
            if self.feature_metadata is None:  # TODO: This is probably not the best way to do this
                self.feature_metadata = stacker_feature_metadata
            else:
                self.feature_metadata = self.feature_metadata.join_metadata(stacker_feature_metadata)
        self.model_base.feature_metadata = self.feature_metadata  # TODO: Move this

        # TODO: Preprocess data here instead of repeatedly
        X = self.preprocess(X=X, preprocess=False, fit=True, compute_base_preds=compute_base_preds)
        kfolds = generate_kfold(X=X, y=y, n_splits=k_fold, stratified=self.is_stratified(), random_state=self._random_state, n_repeats=1)

        train_index, test_index = kfolds[0]
        X_train, X_val = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        orig_time = scheduler_options[1]['time_out']
        scheduler_options[1]['time_out'] = orig_time * 0.8  # TODO: Scheduler doesn't early stop on final model, this is a safety net. Scheduler should be updated to early stop
        hpo_models, hpo_model_performances, hpo_results = self.model_base.hyperparameter_tune(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, scheduler_options=scheduler_options, **kwargs)
        scheduler_options[1]['time_out'] = orig_time

        stackers = {}
        stackers_performance = {}
        for i, (model_name, model_path) in enumerate(hpo_models.items()):
            child: AbstractModel = self._child_type.load(path=model_path)
            y_pred_proba = child.predict_proba(X_val)

            # TODO: Create new StackerEnsemble Here
            stacker = copy.deepcopy(self)
            stacker.name = stacker.name + os.path.sep + str(i)
            stacker.set_contexts(self.path_root + stacker.name + os.path.sep)

            if self.problem_type == MULTICLASS:
                oof_pred_proba = np.zeros(shape=(len(X), len(y.unique())))
            else:
                oof_pred_proba = np.zeros(shape=len(X))
            oof_pred_model_repeats = np.zeros(shape=len(X))
            oof_pred_proba[test_index] += y_pred_proba
            oof_pred_model_repeats[test_index] += 1

            stacker.model_base = None
            child.set_contexts(stacker.path + child.name + os.path.sep)
            stacker.save_model_base(child.convert_to_template())

            stacker._k = k_fold
            stacker._k_fold_end = 1
            stacker._n_repeats = 1
            stacker._oof_pred_proba = oof_pred_proba
            stacker._oof_pred_model_repeats = oof_pred_model_repeats
            child.name = child.name + '_fold_0'
            child.set_contexts(stacker.path + child.name + os.path.sep)
            if not self.save_bagged_folds:
                child.model = None
            if stacker.low_memory:
                stacker.save_child(child, verbose=False)
                stacker.models.append(child.name)
            else:
                stacker.models.append(child)
            stacker.val_score = child.val_score
            stacker._add_child_times_to_bag(model=child)

            stacker.save()
            stackers[stacker.name] = stacker.path
            stackers_performance[stacker.name] = stacker.val_score

        # TODO: hpo_results likely not correct because no renames
        return stackers, stackers_performance, hpo_results

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
            use_orig_features=self.use_orig_features,
        )
        children_info = info.pop('children_info')
        info['stacker_info'] = stacker_info
        info['children_info'] = children_info  # Ensure children_info is last in order
        return info
