import copy
import numpy as np

from ..abstract.abstract_model import AbstractModel
from ...utils import generate_kfold
from ...constants import MULTICLASS, REGRESSION
from ....utils.loaders import load_pkl
from ....utils.savers import save_pkl


class BaggedEnsembleModel(AbstractModel):
    def __init__(self, path, name, model_base: AbstractModel, debug=0):
        self.model_base = model_base
        self.models = []
        self._model_names = None
        self._model_types = None
        self.oof_pred_proba = None  # TODO: Remove this? Move it internally into trainer
        self.n_repeats = 1  # TODO: Add as param or move to fit
        try:
            self.feature_types_metadata = self.model_base.feature_types_metadata
        except:
            self.feature_types_metadata = None
        super().__init__(path=path, name=name, model=None, problem_type=self.model_base.problem_type, objective_func=self.model_base.objective_func, debug=debug)

    def preprocess(self, X):
        return self.models[0].preprocess(X)

    # TODO: Likely will caues issues involving memory since later models will have earlier models still in memory during training.
    #  Add option for low_memory to save models as they are trained, and only load model for prediction.
    def fit(self, X, y, k_fold=5, random_state=0, **kwargs):
        print('BAGGING', self.name)
        self.model_base.feature_types_metadata = self.feature_types_metadata  # TODO: Don't pass this here
        if self.problem_type == REGRESSION:
            stratified = False
        else:
            stratified = True
        # random_state = 0  # TODO: This should be a value sent in and shared across all bags. If stacking, it should be incremented by 1 for each stacking layer

        # TODO: Preprocess data here instead of repeatedly
        kfolds = generate_kfold(X=X, y=y, n_splits=k_fold, stratified=stratified, random_state=random_state, n_repeats=self.n_repeats)

        if self.problem_type == MULTICLASS:
            oof_pred_proba = np.zeros(shape=(len(X), len(y.unique())))
        else:
            oof_pred_proba = np.zeros(shape=len(X))

        models = []
        for i, fold in enumerate(kfolds):
            train_index, test_index = fold
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            fold_model = copy.deepcopy(self.model_base)
            fold_model.name = fold_model.name + '_fold_' + str(i)
            fold_model.path = fold_model.create_contexts(self.path + fold_model.name + '/')
            fold_model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, **kwargs)
            pred_proba = fold_model.predict_proba(X_test)
            models.append(fold_model)
            oof_pred_proba[test_index] += pred_proba
        oof_pred_proba = oof_pred_proba / self.n_repeats

        self.models = models
        self.model_base = None
        self.oof_pred_proba = oof_pred_proba
        return self.models, oof_pred_proba

    def predict_proba(self, X, preprocess=True):
        X = self.preprocess(X)
        pred_proba = self.models[0].predict_proba(X=X, preprocess=False)
        for model in self.models[1:]:
            pred_proba += model.predict_proba(X=X, preprocess=False)
        pred_proba = pred_proba / len(self.models)

        return pred_proba

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
        for i, model_name in enumerate(obj._model_names):
            child_path = obj.create_contexts(obj.path + model_name + '/')
            child_type = obj._model_types[i]
            child_model = child_type.load(path=child_path, reset_paths=reset_paths)
            obj.models.append(child_model)

        obj._model_types = None
        obj._model_names = None
        return obj

    def save(self, file_prefix ="", directory = None, return_filename=False):
        if directory is None:
            directory = self.path
        directory = directory + file_prefix

        self._model_types = []
        self._model_names = []
        for child in self.models:
            child.path = self.create_contexts(self.path + child.name + '/')
            child.save()
            self._model_types.append(type(child))
            self._model_names.append(child.name)

        file_name = directory + self.model_file_name

        self.models = None

        save_pkl.save(path=file_name, object=self)
        if return_filename:
            return file_name
