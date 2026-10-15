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

    By default the `model_base` is a :class:`autogluon.core.models.GreedyWeightedEnsembleModel`. Any model fit on the
    base models' predictions can take its place (see `ensemble_model` in `TabularPredictor.fit`). The ensemble keeps
    the base models with a stack column among the model's `features` after fit, so a model that uses only some of
    them can drop the others' columns from `self.features`, as greedy ensemble selection does. A model may also
    implement `_get_model_weights()`, a `{base model: weight}` dict; without it, no weights are reported.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.low_memory = False

    def _fit(self, X, y, **kwargs):
        super()._fit(X, y, **kwargs)
        # Keep the base models whose predictions a child reads: those with a stack column among the child's
        # features. Greedy ensemble selection keeps only the models it gave weight to.
        child_features = set()
        for model in self.models:
            model = self.load_child(model, verbose=False)
            child_features.update(model.features)
        self.stack_column_prefix_lst = [
            stack_column_prefix
            for stack_column_prefix in self.stack_column_prefix_lst
            if any(
                stack_column in child_features
                for stack_column in self.set_stack_columns(stack_column_prefix_lst=[stack_column_prefix])[0]
            )
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

    def _child_has_weights(self) -> bool:
        """Whether the child reports global per-base-model weights, as greedy ensemble selection does."""
        child_type = getattr(self, "_child_type", None)
        return child_type is None or hasattr(child_type, "_get_model_weights")

    def _get_model_weights(self) -> dict:
        """The base models' weights, averaged over the children; empty if the child reports none."""
        if not self._child_has_weights():
            return {}
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
        if not self._child_has_weights():
            # No weights to report, so the importance is computed like any stacker's.
            return super().compute_feature_importance(X=X, y=y, features=features, is_oof=is_oof, **kwargs)
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
            "print_weights": self._child_has_weights(),
        }
        return tags
