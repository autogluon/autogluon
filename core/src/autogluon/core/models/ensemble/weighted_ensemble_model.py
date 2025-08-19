from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from ..greedy_ensemble.greedy_weighted_ensemble_model import GreedyWeightedEnsembleModel
from .stacker_ensemble_model import StackerEnsembleModel

logger = logging.getLogger(__name__)


# TODO: v0.1 see if this can be removed and logic moved to greedy weighted ensemble model -> Use StackerEnsembleModel as stacker instead
# TODO: Optimize predict speed when fit on kfold, can simply sum weights
class WeightedEnsembleModel(StackerEnsembleModel):
    """
    Weighted ensemble meta-model that implements Ensemble Selection: https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

    A :class:`autogluon.core.models.GreedyWeightedEnsembleModel` must be specified as the `model_base` to properly function.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.low_memory = False

    def _fit(self, X, y, **kwargs):
        super()._fit(X, y, **kwargs)
        stack_columns = []
        for model in self.models:
            model = self.load_child(model, verbose=False)
            stack_columns = stack_columns + [
                stack_column for stack_column in model.base_model_names if stack_column not in stack_columns
            ]
        self.stack_column_prefix_lst = [
            stack_column for stack_column in self.stack_column_prefix_lst if stack_column in stack_columns
        ]
        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(
            stack_column_prefix_lst=self.stack_column_prefix_lst
        )
        min_stack_column_prefix_to_model_map = {
            k: v for k, v in self.stack_column_prefix_to_model_map.items() if k in self.stack_column_prefix_lst
        }
        self.base_model_names = [
            base_model_name
            for base_model_name in self.base_model_names
            if base_model_name in min_stack_column_prefix_to_model_map.values()
        ]
        self.stack_column_prefix_to_model_map = min_stack_column_prefix_to_model_map
        return self

    def _get_model_weights(self) -> dict:
        weights_dict = defaultdict(int)
        num_models = len(self.models)
        for model in self.models:
            model: GreedyWeightedEnsembleModel = self.load_child(model, verbose=False)
            model_weight_dict = model._get_model_weights()
            for key in model_weight_dict.keys():
                weights_dict[key] += model_weight_dict[key]
        for key in weights_dict:
            weights_dict[key] = weights_dict[key] / num_models
        weights_dict = dict(weights_dict)
        return weights_dict

    def compute_feature_importance(self, X, y, features=None, is_oof=True, **kwargs) -> pd.DataFrame:
        logger.warning(
            "Warning: non-raw feature importance calculation is not valid for weighted ensemble since it does not have features, returning ensemble weights instead..."
        )
        if is_oof:
            fi = pd.Series(self._get_model_weights()).sort_values(ascending=False)
        else:
            logger.warning(
                "Warning: Feature importance calculation is not yet implemented for WeightedEnsembleModel on unseen data, returning generic feature importance..."
            )
            fi = pd.Series(self._get_model_weights()).sort_values(ascending=False)

        fi_df = fi.to_frame(name="importance")
        fi_df["stddev"] = np.nan
        fi_df["p_score"] = np.nan
        fi_df["n"] = np.nan

        # TODO: Rewrite preprocess() in greedy_weighted_ensemble_model to enable
        # fi_df = super().compute_feature_importance(X=X, y=y, features_to_use=features_to_use, preprocess=preprocess, is_oof=is_oof, **kwargs)
        return fi_df

    def _set_default_params(self):
        default_params = {"use_orig_features": False}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()

    def _more_tags(self):
        """
        This model can generate out-of-fold (oof) predictions by predicting directly on the training data.
        This will make the result slightly overfit, but the weighted ensemble has limited degrees of freedom intentionally, making the overfitting negligible.
        """
        tags = {
            "can_get_oof_from_train": True,
            "print_weights": True,
        }
        return tags
