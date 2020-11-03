import logging
from collections import defaultdict

import pandas as pd

from .stacker_ensemble_model import StackerEnsembleModel
from .greedy_weighted_ensemble_model import GreedyWeightedEnsembleModel

logger = logging.getLogger(__name__)


# TODO: Optimize predict speed when fit on kfold, can simply sum weights
class WeightedEnsembleModel(StackerEnsembleModel):
    def __init__(self, base_model_names, base_model_paths_dict, base_model_types_dict, **kwargs):
        model_0 = base_model_types_dict[base_model_names[0]].load(path=base_model_paths_dict[base_model_names[0]], verbose=False)
        super().__init__(model_base=model_0, base_model_names=base_model_names, base_model_paths_dict=base_model_paths_dict, base_model_types_dict=base_model_types_dict, use_orig_features=False, **kwargs)
        child_hyperparameters = kwargs.get('_tmp_greedy_hyperparameters', None)  # TODO: Rework to avoid this hack
        self.model_base = GreedyWeightedEnsembleModel(path='', name='greedy_ensemble', num_classes=self.num_classes, base_model_names=self.stack_column_prefix_lst, problem_type=self.problem_type, eval_metric=self.eval_metric, stopping_metric=self.stopping_metric, hyperparameters=child_hyperparameters)
        self._child_type = type(self.model_base)
        self.low_memory = False

    def _fit(self, X, y, k_fold=5, k_fold_start=0, k_fold_end=None, n_repeats=1, n_repeat_start=0, compute_base_preds=True, time_limit=None, **kwargs):
        super()._fit(X, y, k_fold=k_fold, k_fold_start=k_fold_start, k_fold_end=k_fold_end, n_repeats=n_repeats, n_repeat_start=n_repeat_start, compute_base_preds=compute_base_preds, time_limit=time_limit, **kwargs)
        stack_columns = []
        for model in self.models:
            model = self.load_child(model, verbose=False)
            stack_columns = stack_columns + [stack_column for stack_column in model.base_model_names if stack_column not in stack_columns]
        self.stack_column_prefix_lst = [stack_column for stack_column in self.stack_column_prefix_lst if stack_column in stack_columns]
        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(stack_column_prefix_lst=self.stack_column_prefix_lst)
        min_stack_column_prefix_to_model_map = {k: v for k, v in self.stack_column_prefix_to_model_map.items() if k in self.stack_column_prefix_lst}
        self.base_model_names = [base_model_name for base_model_name in self.base_model_names if base_model_name in min_stack_column_prefix_to_model_map.values()]
        self.stack_column_prefix_to_model_map = min_stack_column_prefix_to_model_map

    def _get_model_weights(self):
        weights_dict = defaultdict(int)
        num_models = len(self.models)
        for model in self.models:
            model: GreedyWeightedEnsembleModel = self.load_child(model, verbose=False)
            model_weight_dict = model._get_model_weights()
            for key in model_weight_dict.keys():
                weights_dict[key] += model_weight_dict[key]
        for key in weights_dict:
            weights_dict[key] = weights_dict[key] / num_models
        return weights_dict

    def compute_feature_importance(self, X, y, features_to_use=None, is_oof=True, **kwargs) -> pd.Series:
        logger.warning('Warning: non-raw feature importance calculation is not valid for weighted ensemble since it does not have features, returning ensemble weights instead...')
        if is_oof:
            feature_importance = pd.Series(self._get_model_weights()).sort_values(ascending=False)
        else:
            logger.warning('Warning: Feature importance calculation is not yet implemented for WeightedEnsembleModel on unseen data, returning generic feature importance...')
            feature_importance = pd.Series(self._get_model_weights()).sort_values(ascending=False)
            # TODO: Rewrite preprocess() in greedy_weighted_ensemble_model to enable
            # feature_importance = super().compute_feature_importance(X=X, y=y, features_to_use=features_to_use, preprocess=preprocess, is_oof=is_oof, **kwargs)
        return feature_importance
