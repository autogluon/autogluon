import copy, logging, time
import numpy as np

from ..abstract.abstract_model import AbstractModel
from ...utils import generate_kfold
from ...constants import MULTICLASS, REGRESSION
from ....utils.loaders import load_pkl
from ....utils.savers import save_pkl
from ....utils.exceptions import TimeLimitExceeded

logger = logging.getLogger(__name__)


class BaggedEnsembleModel(AbstractModel):
    def __init__(self, path, name, model_base: AbstractModel, hyperparameters=None, debug=0):
        self.model_base = model_base
        self._child_type = type(self.model_base)
        self.models = []
        self._oof_pred_proba = None
        self._n_repeats = 0
        self.low_memory = True
        try:
            feature_types_metadata = self.model_base.feature_types_metadata
        except:
            feature_types_metadata = None
        super().__init__(path=path, name=name, model=None, problem_type=self.model_base.problem_type, objective_func=self.model_base.objective_func, feature_types_metadata=feature_types_metadata, hyperparameters=hyperparameters, debug=debug)

    # TODO: This assumes bagged ensemble has a complete k_fold and no partial k_fold models, this is likely fine but will act incorrectly if called when only a partial k_fold has been completed
    #  Solving this is memory intensive, requires all oof_pred_probas from all n_repeats, so its probably not worth it.
    @property
    def oof_pred_proba(self):
        return self._oof_pred_proba / self._n_repeats

    def preprocess(self, X, model=None):
        if model is None:
            if len(self.models) == 0:
                return X
            model = self.models[0]
        if type(model) == str:
            model = self.load_child(model)
        return model.preprocess(X)

    # TODO: compute_base_preds is unused here, it is present for compatibility with StackerEnsembleModel, consider merging the two.
    def fit(self, X, y, k_fold=5, n_repeats=1, n_repeat_start=0, random_state=0, compute_base_preds=False, time_limit=None, **kwargs):
        if n_repeat_start != self._n_repeats:
            raise ValueError('n_repeat_start must equal self._n_repeats, values: (' + str(n_repeat_start) + ', ' + str(self._n_repeats) + ')')
        if n_repeats <= n_repeat_start:
            raise ValueError('n_repeats must be greater than n_repeat_start, values: (' + str(n_repeats) + ', ' + str(n_repeat_start) + ')')
        fold_start = n_repeat_start * k_fold
        fold_end = n_repeats * k_fold
        start_time = time.time()
        if self.problem_type == REGRESSION:
            stratified = False
        else:
            stratified = True

        # TODO: Preprocess data here instead of repeatedly
        kfolds = generate_kfold(X=X, y=y, n_splits=k_fold, stratified=stratified, random_state=random_state, n_repeats=n_repeats)

        if self.problem_type == MULTICLASS:
            oof_pred_proba = np.zeros(shape=(len(X), len(y.unique())))
        else:
            oof_pred_proba = np.zeros(shape=len(X))

        if self.model_base is None:
            model_base = self.load_model_base()
        else:
            model_base = self.model_base
        model_base.feature_types_metadata = self.feature_types_metadata  # TODO: Don't pass this here

        models = []
        num_folds = len(kfolds)
        time_limit_fold = None
        for i in range(fold_start, fold_end):
            fold = kfolds[i]
            if time_limit:
                time_elapsed = time.time() - start_time
                time_left = time_limit - time_elapsed
                required_time_per_fold = time_left / (num_folds - i)
                time_limit_fold = required_time_per_fold * 0.8
                if i > 0:
                    expected_time_required = time_elapsed * (num_folds / i)
                    expected_remaining_time_required = expected_time_required / (num_folds / (num_folds - i))
                    if expected_remaining_time_required > time_left:
                        raise TimeLimitExceeded
                if time_left <= 0:
                    raise TimeLimitExceeded

            train_index, test_index = fold
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            fold_model = copy.deepcopy(model_base)
            fold_model.name = fold_model.name + '_fold_' + str(i)
            fold_model.path = fold_model.create_contexts(self.path + fold_model.name + '/')
            fold_model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, time_limit=time_limit_fold, **kwargs)
            pred_proba = fold_model.predict_proba(X_test)
            if self.low_memory:
                self.save_child(fold_model, verbose=False)
                models.append(fold_model.name)
            else:
                models.append(fold_model)
            oof_pred_proba[test_index] += pred_proba

        self.models += models

        if self.model_base is not None:
            self.save_model_base(self.model_base)
            self.model_base = None

        if self._oof_pred_proba is None:
            self._oof_pred_proba = oof_pred_proba
        else:
            self._oof_pred_proba += oof_pred_proba

        self._n_repeats = n_repeats

    # FIXME: Defective if model does not apply same preprocessing in all bags!
    #  No model currently violates this rule, but in future it could happen
    def predict_proba(self, X, preprocess=True):
        model = self.load_child(self.models[0])
        if preprocess:
            X = self.preprocess(X, model=model)
        pred_proba = model.predict_proba(X=X, preprocess=False)
        for model in self.models[1:]:
            model = self.load_child(model)
            pred_proba += model.predict_proba(X=X, preprocess=False)
        pred_proba = pred_proba / len(self.models)

        return pred_proba

    def load_child(self, model, verbose=False):
        if type(model) == str:
            child_path = self.create_contexts(self.path + model + '/')
            return self._child_type.load(path=child_path, verbose=verbose)
        else:
            return model

    def save_child(self, model, verbose=False):
        child = self.load_child(model)
        child.path = self.create_contexts(self.path + child.name + '/')
        child.save(verbose=verbose)

    @classmethod
    def load(cls, path, file_prefix="", reset_paths=False, low_memory=True, verbose=True):
        path = path + file_prefix
        load_path = path + cls.model_file_name
        obj = load_pkl.load(path=load_path, verbose=verbose)
        if reset_paths:
            obj.set_contexts(path)
        if low_memory:
            pass
        else:
            for i, model_name in enumerate(obj.models):
                if type(model_name) == str:
                    child_path = obj.create_contexts(obj.path + model_name + '/')
                    child_model = obj._child_type.load(path=child_path, reset_paths=reset_paths, verbose=True)
                    obj.models[i] = child_model
        return obj

    def load_model_base(self):
        return load_pkl.load(path=self.path + 'utils/model_template.pkl')

    def save_model_base(self, model_base):
        save_pkl.save(path=self.path + 'utils/model_template.pkl', object=model_base)

    def save(self, file_prefix="", directory=None, return_filename=False, verbose=True, save_children=False):
        if directory is None:
            directory = self.path
        directory = directory + file_prefix

        if save_children:
            model_names = []
            for child in self.models:
                child = self.load_child(child)
                child.path = self.create_contexts(self.path + child.name + '/')
                child.save(verbose=False)
                model_names.append(child.name)
            self.models = model_names

        file_name = directory + self.model_file_name

        save_pkl.save(path=file_name, object=self, verbose=verbose)
        if return_filename:
            return file_name
