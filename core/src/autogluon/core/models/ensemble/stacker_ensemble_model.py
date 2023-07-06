import copy
import logging
import os
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.features.types import R_FLOAT, S_STACK
from autogluon.common.utils.path_converter import PathConverter

from ...constants import MULTICLASS, QUANTILE, SOFTCLASS
from ..abstract.abstract_model import AbstractModel
from .bagged_ensemble_model import BaggedEnsembleModel

logger = logging.getLogger(__name__)


# TODO: Currently, if this is a stacker above level 1, it will be very slow taking raw input due to each stacker needing to repeat computation on the base models.
#  To solve this, this model must know full context of stacker, and only get preds once for each required model
#  This is already done in trainer, but could be moved internally.
class StackerEnsembleModel(BaggedEnsembleModel):
    """
    Stack ensemble meta-model which functions identically to :class:`BaggedEnsembleModel` with the additional capability to leverage base models.

    By specifying base models during init, stacker models can use the base model predictions as features during training and inference.

    This property allows for significantly improved model quality in many situations compared to non-stacking alternatives.

    Stacker models can act as base models to other stacker models, enabling multi-layer stack ensembling.
    """

    def __init__(
        self,
        base_model_names=None,
        base_models_dict=None,
        base_model_paths_dict=None,
        base_model_types_dict=None,
        base_model_types_inner_dict=None,
        base_model_performances_dict=None,
        **kwargs,
    ):
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
        self.base_model_paths_dict = {key: os.path.relpath(val, self.path) for key, val in base_model_paths_dict.items()}
        self.base_model_types_dict = base_model_types_dict

        # TODO: Consider deleting these variables after initialization
        self._base_model_performances_dict = base_model_performances_dict
        self._base_model_types_inner_dict = base_model_types_inner_dict

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)
        base_model_performances_dict = self._base_model_performances_dict
        base_model_types_inner_dict = self._base_model_types_inner_dict
        if (base_model_performances_dict is not None) and (base_model_types_inner_dict is not None):
            if self.params["max_base_models_per_type"] > 0:
                self.base_model_names = self.limit_models_per_type(
                    models=self.base_model_names,
                    model_types=base_model_types_inner_dict,
                    model_scores=base_model_performances_dict,
                    max_base_models_per_type=self.params["max_base_models_per_type"],
                )
            if self.params["max_base_models"] > 0:
                self.base_model_names = self.limit_models(
                    models=self.base_model_names, model_scores=base_model_performances_dict, max_base_models=self.params["max_base_models"]
                )

        for model_name, model in self.base_models_dict.items():
            if model_name not in self.base_model_names:
                self.base_models_dict.pop(model_name)

        self.stack_column_prefix_lst = copy.deepcopy(self.base_model_names)
        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(stack_column_prefix_lst=self.stack_column_prefix_lst)
        self.stack_column_prefix_to_model_map = {
            stack_column_prefix: self.base_model_names[i] for i, stack_column_prefix in enumerate(self.stack_column_prefix_lst)
        }

        self._add_stack_to_feature_metadata()

    @staticmethod
    def limit_models_per_type(models, model_types, model_scores, max_base_models_per_type):
        model_type_groups = defaultdict(list)
        for model in models:
            model_type_groups[model_types[model]].append((model, model_scores[model]))
        models_remain = []
        for key in model_type_groups:
            models_remain += sorted(model_type_groups[key], key=lambda x: x[1], reverse=True)[:max_base_models_per_type]
        models_valid_set = set([model for model, score in models_remain])
        # Important: Ensure ordering of `models_valid` is the same as `models`
        models_valid = [model for model in models if model in models_valid_set]
        return models_valid

    def limit_models(self, models, model_scores, max_base_models):
        model_types = {model: "" for model in models}
        return self.limit_models_per_type(models=models, model_types=model_types, model_scores=model_scores, max_base_models_per_type=max_base_models)

    def _set_default_params(self):
        default_params = {"use_orig_features": True, "max_base_models": 25, "max_base_models_per_type": 5}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()

    def preprocess(self, X, fit=False, compute_base_preds=True, infer=True, model_pred_proba_dict=None, **kwargs):
        if self.stack_column_prefix_lst:
            if infer:
                if set(self.stack_columns).issubset(set(list(X.columns))):
                    compute_base_preds = (
                        False  # TODO: Consider removing, this can be dangerous but the code to make this work otherwise is complex (must rewrite predict_proba)
                    )
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
                    X_stacker.append(
                        y_pred_proba
                    )  # TODO: This could get very large on a high class count problem. Consider capping to top N most frequent classes and merging least frequent
                X_stacker = self.pred_probas_to_df(X_stacker, index=X.index)
                if self.params["use_orig_features"]:
                    X = pd.concat([X_stacker, X], axis=1)
                else:
                    X = X_stacker
            elif not self.params["use_orig_features"]:
                X = X[self.stack_columns]
        X = super().preprocess(X, **kwargs)
        return X

    def pred_probas_to_df(self, pred_proba: list, index=None) -> pd.DataFrame:
        if self.problem_type in [MULTICLASS, SOFTCLASS, QUANTILE]:
            pred_proba = np.concatenate(pred_proba, axis=1)
            pred_proba = pd.DataFrame(pred_proba, columns=self.stack_columns)
        else:
            pred_proba = pd.DataFrame(data=np.asarray(pred_proba).T, columns=self.stack_columns)
        if index is not None:
            pred_proba.set_index(index, inplace=True)
        return pred_proba

    def _fit(self, X, y, compute_base_preds=True, time_limit=None, **kwargs):
        start_time = time.time()
        # TODO: This could be preprocess_nonadaptive=True in general, just have preprocess_nonadaptive=False for child models
        X = self.preprocess(X=X, preprocess_nonadaptive=False, fit=True, compute_base_preds=compute_base_preds)
        if time_limit is not None:
            time_limit = time_limit - (time.time() - start_time)
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)

    def set_stack_columns(self, stack_column_prefix_lst):
        if self.problem_type in [MULTICLASS, SOFTCLASS]:
            stack_columns = [stack_column_prefix + "_" + str(cls) for stack_column_prefix in stack_column_prefix_lst for cls in range(self.num_classes)]
            num_pred_cols_per_model = self.num_classes
        elif self.problem_type == QUANTILE:
            stack_columns = [stack_column_prefix + "_" + str(q) for stack_column_prefix in stack_column_prefix_lst for q in self.quantile_levels]
            num_pred_cols_per_model = len(self.quantile_levels)
        else:
            stack_columns = stack_column_prefix_lst
            num_pred_cols_per_model = 1
        return stack_columns, num_pred_cols_per_model

    def _hyperparameter_tune(self, X, y, k_fold, hpo_executor, compute_base_preds=True, **kwargs):
        if len(self.models) != 0:
            raise ValueError("self.models must be empty to call hyperparameter_tune, value: %s" % self.models)

        preprocess_kwargs = {"compute_base_preds": compute_base_preds}
        return super()._hyperparameter_tune(X=X, y=y, k_fold=k_fold, hpo_executor=hpo_executor, preprocess_kwargs=preprocess_kwargs, **kwargs)

    def get_params(self):
        init_args = dict(
            base_model_names=self.base_model_names,
            base_models_dict=self.base_models_dict,
            base_model_paths_dict=self.base_model_paths_dict,
            base_model_types_dict=self.base_model_types_dict,
            base_model_performances_dict=self._base_model_performances_dict,
            base_model_types_inner_dict=self._base_model_types_inner_dict,
        )
        init_args.update(super().get_params())
        return init_args

    def load_base_model(self, model_name):
        if model_name in self.base_models_dict.keys():
            model = self.base_models_dict[model_name]
        else:
            model_type = self.base_model_types_dict[model_name]
            model_path = os.path.join(self.path, self.base_model_paths_dict[model_name])
            model = model_type.load(model_path)
        return model

    def get_info(self):
        info = super().get_info()
        stacker_info = dict(
            num_base_models=len(self.base_model_names),
            base_model_names=self.base_model_names,
        )
        children_info = info.pop("children_info")
        info["stacker_info"] = stacker_info
        info["children_info"] = children_info  # Ensure children_info is last in order
        return info

    def _add_stack_to_feature_metadata(self):
        if len(self.models) == 0:
            type_map_raw = {column: R_FLOAT for column in self.stack_columns}
            type_group_map_special = {S_STACK: self.stack_columns}
            stacker_feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)
            if self.feature_metadata is None:  # TODO: This is probably not the best way to do this
                self.feature_metadata = stacker_feature_metadata
            else:
                # FIXME: This is a hack, stack feature special types should be already present in feature_metadata, not added here
                existing_stack_features = self.feature_metadata.get_features(required_special_types=[S_STACK])
                # HACK: Currently AutoGluon auto-adds all base learner prediction features into self.feature_metadata.
                # The two lines below are here because if feature pruning would crash if it prunes a base learner prediction feature.
                existing_features = self.feature_metadata.get_features()
                stacker_feature_metadata = stacker_feature_metadata.keep_features([feature for feature in existing_features if feature in type_map_raw])
                if set(stacker_feature_metadata.get_features()) != set(existing_stack_features):
                    self.feature_metadata = self.feature_metadata.add_special_types(stacker_feature_metadata.get_type_map_special())
