import numpy as np
import pandas as pd

from ..abstract.abstract_model import AbstractModel
from .bagged_ensemble_model import BaggedEnsembleModel
from ...constants import MULTICLASS


# TODO: Inherit from BaggedEnsembleModel?
class StackerEnsembleModel(AbstractModel):
    def __init__(self, path, name, stacker_model: AbstractModel, base_model_names, base_model_paths_dict, base_model_types_dict, use_orig_features=True, num_classes=None, debug=0):
        self.base_model_names = base_model_names
        self.base_model_paths_dict = base_model_paths_dict
        self.base_model_types_dict = base_model_types_dict
        self.bagged_mode = None
        self.use_orig_features = use_orig_features
        self.num_classes = num_classes
        # self.oof_pred_proba = stacker_model.oof_pred_proba
        super().__init__(path=path, name=name, model=stacker_model, problem_type=stacker_model.problem_type, objective_func=stacker_model.objective_func, debug=debug)

        if self.problem_type == MULTICLASS:
            self.stack_columns = [model_name + '_' + str(cls) for model_name in self.base_model_names for cls in range(self.num_classes)]
        else:
            self.stack_columns = self.base_model_names

    # TODO: Add option to also include X features in X_stacker
    def preprocess(self, X, fit=False, compute_base_preds=True, infer=True):
        if infer:
            if (set(self.stack_columns).issubset(set(list(X.columns)))):
                compute_base_preds = False  # TODO: Consider removing, this can be dangerous but the code to make this work otherwise is complex (must rewrite predict_proba)
        if compute_base_preds:
            X_stacker = []
            for model_name in self.base_model_names:
                model_type = self.base_model_types_dict[model_name]
                model_path = self.base_model_paths_dict[model_name]
                model = model_type.load(model_path)
                if fit:
                    y_pred_proba = model.oof_pred_proba
                else:
                    y_pred_proba = model.predict_proba(X)
                X_stacker.append(y_pred_proba)
            if self.problem_type == MULTICLASS:
                X_stacker = np.concatenate(X_stacker, axis=1)
                X_stacker = pd.DataFrame(X_stacker, columns=self.stack_columns, index=X.index)
            else:
                X_stacker = pd.DataFrame(data=np.asarray(X_stacker).T, columns=self.stack_columns)
            if self.use_orig_features:
                X = pd.concat([X_stacker, X], axis=1)
            else:
                X = X_stacker
        elif not self.use_orig_features:
            X = X[self.stack_columns]
        X = super().preprocess(X)
        return X

    def fit(self, X, y, k_fold=5, random_state=1, compute_base_preds=True, **kwargs):
        X = self.preprocess(X=X, fit=True, compute_base_preds=compute_base_preds)
        if k_fold >= 2:
            self.bagged_mode = True
            self.model = BaggedEnsembleModel(path=self.model.path[:-(len(self.model.name) + 1)], name=self.model.name + '_BAGGED', model_base=self.model)
            self.model.fit(X=X, y=y, k_fold=k_fold, random_state=random_state)
            self.oof_pred_proba = self.model.oof_pred_proba  # TODO: Just have stacker_ensemble_model inherit BaggedEnsemble
        else:
            self.bagged_mode = False
            self.model.fit(X_train=X, Y_train=y)
            self.oof_pred_proba = y  # TODO: Remove

        # self.oof_pred_proba = self.predict_proba(X)
