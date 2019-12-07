import copy, logging
import numpy as np
import pandas as pd

from ..abstract.abstract_model import AbstractModel
from .bagged_ensemble_model import BaggedEnsembleModel
from ...constants import MULTICLASS

logger = logging.getLogger(__name__)

# TODO: Currently, if this is a stacker above level 1, it will be very slow taking raw input due to each stacker needing to repeat computation on the base models.
    #  To solve this, this model must know full context of stacker, and only get preds once for each required model
    #  This is already done in trainer, but could be moved internally.
class StackerEnsembleModel(BaggedEnsembleModel):
    def __init__(self, path, name, model_base: AbstractModel, base_model_names, base_model_paths_dict, base_model_types_dict, use_orig_features=True, num_classes=None, debug=0):
        super().__init__(path=path, name=name, model_base=model_base, debug=debug)
        self.base_model_names = base_model_names
        self.base_model_paths_dict = base_model_paths_dict
        self.base_model_types_dict = base_model_types_dict
        self.bagged_mode = None
        self.use_orig_features = use_orig_features
        self.num_classes = num_classes

        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(base_model_names=self.base_model_names)

    def preprocess(self, X, preprocess=True, fit=False, compute_base_preds=True, infer=True, model=None):
        if infer:
            if (set(self.stack_columns).issubset(set(list(X.columns)))):
                compute_base_preds = False  # TODO: Consider removing, this can be dangerous but the code to make this work otherwise is complex (must rewrite predict_proba)
        if compute_base_preds:
            X_stacker = []
            for model_name in self.base_model_names:
                model_type = self.base_model_types_dict[model_name]
                model_path = self.base_model_paths_dict[model_name]
                model_loaded = model_type.load(model_path)
                if fit:
                    y_pred_proba = model_loaded.oof_pred_proba
                else:
                    y_pred_proba = model_loaded.predict_proba(X)
                X_stacker.append(y_pred_proba)  # TODO: This could get very large on a high class count problem. Consider capping to top N most frequent classes and merging least frequent
            X_stacker = self.pred_probas_to_df(X_stacker)
            X_stacker.index = X.index
            if self.use_orig_features:
                X = pd.concat([X_stacker, X], axis=1)
            else:
                X = X_stacker
        elif not self.use_orig_features:
            X = X[self.stack_columns]
        if preprocess:
            X = super().preprocess(X, model=model)
        return X

    def pred_probas_to_df(self, pred_proba: list) -> pd.DataFrame:
        if self.problem_type == MULTICLASS:
            pred_proba = np.concatenate(pred_proba, axis=1)
            pred_proba = pd.DataFrame(pred_proba, columns=self.stack_columns)
        else:
            pred_proba = pd.DataFrame(data=np.asarray(pred_proba).T, columns=self.stack_columns)
        return pred_proba

    def fit(self, X, y, k_fold=5, random_state=1, compute_base_preds=True, **kwargs):
        X = self.preprocess(X=X, preprocess=False, fit=True, compute_base_preds=compute_base_preds)
        if self.feature_types_metadata is None:  # TODO: This is probably not the best way to do this
            self.feature_types_metadata = {'float': self.stack_columns}
        else:
            self.feature_types_metadata = copy.deepcopy(self.feature_types_metadata)
            if 'float' in self.feature_types_metadata.keys():
                self.feature_types_metadata['float'] += self.stack_columns
            else:
                self.feature_types_metadata['float'] = self.stack_columns
        if k_fold >= 2:
            super().fit(X=X, y=y, k_fold=k_fold, random_state=random_state)
            self.bagged_mode = True
        else:
            self.models = [copy.deepcopy(self.model_base)]
            self.model_base = None
            self.bagged_mode = False
            self.models[0].set_contexts(path_context=self.path + self.models[0].name + '/')
            self.models[0].feature_types_metadata = self.feature_types_metadata  # TODO: Move this
            self.models[0].fit(X_train=X, Y_train=y)
            self.oof_pred_proba = self.models[0].predict_proba(X=X)  # TODO: Cheater value, will be overfit to valid set

        return self.models, self.oof_pred_proba

    def set_stack_columns(self, base_model_names):
        if self.problem_type == MULTICLASS:
            stack_columns = [model_name + '_' + str(cls) for model_name in base_model_names for cls in range(self.num_classes)]
            num_pred_cols_per_model = self.num_classes
        else:
            stack_columns = base_model_names
            num_pred_cols_per_model = 1
        return stack_columns, num_pred_cols_per_model