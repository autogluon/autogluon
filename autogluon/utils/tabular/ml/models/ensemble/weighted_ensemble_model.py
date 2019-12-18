import logging
import numpy as np

from .stacker_ensemble_model import StackerEnsembleModel
from ...tuning.ensemble_selection import EnsembleSelection

logger = logging.getLogger(__name__)


# TODO: Do Kfold to determine if its overfit, observing the val score is way overly optimistic on AUC at l2, despite being worse on test than l1 (OpenML Amazon_employee_access)
class WeightedEnsembleModel(StackerEnsembleModel):
    def __init__(self, path, name, base_model_names, base_model_paths_dict, base_model_types_dict, base_model_weights=None, base_model_performances_dict=None, num_classes=None, hyperparameters=None, debug=0):
        self.base_model_weights = base_model_weights

        if self.base_model_weights is not None:
            base_model_names, self.base_model_weights = self.remove_zero_weight_models(self.base_model_names, self.base_model_weights)

        model_0 = base_model_types_dict[base_model_names[0]].load(path=base_model_paths_dict[base_model_names[0]], verbose=False)
        super().__init__(path=path, name=name, model_base=model_0, base_model_names=base_model_names, base_model_paths_dict=base_model_paths_dict, base_model_types_dict=base_model_types_dict, base_model_performances_dict=base_model_performances_dict, use_orig_features=False, num_classes=num_classes, hyperparameters=hyperparameters, debug=debug)
        self.model_base = None

    # TODO: Add kfold support? Might be a useful debugging tool in optimizing weighted_ensemble to do better on average
    def fit(self, X, y, k_fold=5, random_state=1, compute_base_preds=True, **kwargs):
        X = self.preprocess(X=X, preprocess=False, fit=True, compute_base_preds=compute_base_preds)
        pred_probas = self.convert_pred_probas_df_to_list(X)

        ensemble_selection = EnsembleSelection(ensemble_size=100, problem_type=self.problem_type, metric=self.objective_func)
        ensemble_selection.fit(predictions=pred_probas, labels=y, identifiers=None)
        self.base_model_weights = ensemble_selection.weights_
        self.oof_pred_proba = self.predict_proba(X=X, preprocess=True)

        self.base_model_names, self.base_model_weights = self.remove_zero_weight_models(self.base_model_names, self.base_model_weights)
        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(base_model_names=self.base_model_names)

    def predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X, preprocess=False, model=None)
        pred_probas = self.convert_pred_probas_df_to_list(X)
        return self.weight_pred_probas(pred_probas, weights=self.base_model_weights)

    def convert_pred_probas_df_to_list(self, pred_probas_df) -> list:
        pred_probas = []
        for i, model in enumerate(self.base_model_names):
            index_start = i * self.num_pred_cols_per_model
            index_end = (i + 1) * self.num_pred_cols_per_model
            model_cols = self.stack_columns[index_start:index_end]
            pred_proba = pred_probas_df[model_cols].values
            if self.num_pred_cols_per_model == 1:
                pred_proba = pred_proba.flatten()
            pred_probas.append(pred_proba)
        return pred_probas

    @staticmethod
    def weight_pred_probas(pred_probas, weights):
        preds_norm = [pred * weight for pred, weight in zip(pred_probas, weights)]
        preds_ensemble = np.sum(preds_norm, axis=0)
        return preds_ensemble

    @staticmethod
    def remove_zero_weight_models(base_model_names, base_model_weights):
        base_models_to_keep = []
        base_model_weights_to_keep = []
        for i, weight in enumerate(base_model_weights):
            if weight != 0:
                base_models_to_keep.append(base_model_names[i])
                base_model_weights_to_keep.append(weight)
        return base_models_to_keep, base_model_weights_to_keep
