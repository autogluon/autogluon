import numpy as np
import pandas as pd
import copy

from ..abstract.abstract_model import AbstractModel
from .bagged_ensemble_model import BaggedEnsembleModel
from ...constants import MULTICLASS
from ....utils.loaders import load_pkl
from ....utils.savers import save_pkl


# TODO: Inherit from BaggedEnsembleModel?
class StackerEnsembleModel(AbstractModel):
    def __init__(self, path, name, stacker_model: AbstractModel, base_model_names, base_model_paths_dict, base_model_types_dict, use_orig_features=True, num_classes=None, debug=0):
        self.base_model_names = base_model_names
        self.base_model_paths_dict = base_model_paths_dict
        self.base_model_types_dict = base_model_types_dict
        self.bagged_mode = None
        self.use_orig_features = use_orig_features
        self.num_classes = num_classes

        self._model_type = None
        self._model_name = None

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
                X_stacker.append(y_pred_proba)  # TODO: This could get very large on a high class count problem. Consider capping to top N most frequent classes and merging least frequent
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
        if self.feature_types_metadata is None:  # TODO: This is probably not the best way to do this
            self.feature_types_metadata = {'float': self.stack_columns}
        else:
            self.feature_types_metadata = copy.deepcopy(self.feature_types_metadata)
            if 'float' in self.feature_types_metadata.keys():
                self.feature_types_metadata['float'] += self.stack_columns
            else:
                self.feature_types_metadata['float'] = self.stack_columns
        if k_fold >= 2:
            self.bagged_mode = True
            self.model = BaggedEnsembleModel(path=self.path, name=self.model.name + '_BAGGED', model_base=self.model)
            self.model.feature_types_metadata = self.feature_types_metadata  # TODO: MOVE THIS
            self.model.fit(X=X, y=y, k_fold=k_fold, random_state=random_state)
            self.oof_pred_proba = self.model.oof_pred_proba  # TODO: Just have stacker_ensemble_model inherit BaggedEnsemble
        else:
            self.bagged_mode = False
            self.model.set_contexts(path_context=self.path + self.model.name + '/')
            self.model.feature_types_metadata = self.feature_types_metadata  # TODO: MOVE THIS
            self.model.fit(X_train=X, Y_train=y)
            self.oof_pred_proba = y  # TODO: Remove

        # self.oof_pred_proba = self.predict_proba(X)

    @classmethod
    def load(cls, path, file_prefix="", reset_paths=False):
        path = path + file_prefix
        load_path = path + cls.model_file_name
        if not reset_paths:
            obj = load_pkl.load(path=load_path)
        else:
            obj = load_pkl.load(path=load_path)
            obj.set_contexts(path)
        obj.models = []
        child_path = obj.create_contexts(obj.path + obj._model_name + '/')
        child_type = obj._model_type
        child_model = child_type.load(path=child_path, reset_paths=reset_paths)
        obj.model = child_model

        obj._model_type = None
        obj._model_name = None
        return obj

    def save(self, file_prefix ="", directory = None, return_filename=False):
        if directory is None:
            directory = self.path
        directory = directory + file_prefix

        self.model.path = self.create_contexts(self.path + self.model.name + '/')
        self.model.save()
        self._model_type = type(self.model)
        self._model_name = self.model.name

        file_name = directory + self.model_file_name
        self.model = None

        save_pkl.save(path=file_name, object=self)
        if return_filename:
            return file_name
