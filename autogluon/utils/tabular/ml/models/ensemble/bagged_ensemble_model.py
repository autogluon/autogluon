import copy, logging, time
import numpy as np

from ..abstract.abstract_model import AbstractModel
from ...utils import generate_kfold
from ...constants import MULTICLASS, REGRESSION
from ....utils.loaders import load_pkl
from ....utils.savers import save_pkl
from ....utils.exceptions import TimeLimitExceeded

logger = logging.getLogger(__name__)


# TODO: Add metadata object with info like score on each model, train time on each model, etc.
class BaggedEnsembleModel(AbstractModel):
    def __init__(self, path: str, name: str, model_base: AbstractModel, hyperparameters=None, random_state=0, debug=0):
        self.model_base = model_base
        self._child_type = type(self.model_base)
        self.models = []
        self._oof_pred_proba = None
        self._oof_pred_model_repeats = None
        self._n_repeats = 0  # Number of n_repeats with at least 1 model fit, if kfold=5 and 8 models have been fit, _n_repeats is 2
        self._n_repeats_finished = 0  # Number of n_repeats finished, if kfold=5 and 8 models have been fit, _n_repeats_finished is 1
        self._k_fold_end = 0  # Number of models fit in current n_repeat (0 if completed), if kfold=5 and 8 models have been fit, _k_fold_end is 3
        self._k = None  # k models per n_repeat, equivalent to kfold value
        self._random_state = random_state
        self.low_memory = True
        try:
            feature_types_metadata = self.model_base.feature_types_metadata
        except:
            feature_types_metadata = None
        super().__init__(path=path, name=name, problem_type=self.model_base.problem_type, objective_func=self.model_base.objective_func, feature_types_metadata=feature_types_metadata, hyperparameters=hyperparameters, debug=debug)

    def is_valid(self):
        return self.is_fit() and (self._n_repeats == self._n_repeats_finished)

    def is_fit(self):
        return len(self.models) != 0

    # TODO: This assumes bagged ensemble has a complete k_fold and no partial k_fold models, this is likely fine but will act incorrectly if called when only a partial k_fold has been completed
    #  Solving this is memory intensive, requires all oof_pred_probas from all n_repeats, so its probably not worth it.
    @property
    def oof_pred_proba(self):
        # TODO: Require is_valid == True (add option param to ignore is_valid)
        oof_pred_model_repeats_without_0 = np.where(self._oof_pred_model_repeats == 0, 1, self._oof_pred_model_repeats)
        if self._oof_pred_proba.ndim == 2:
            oof_pred_model_repeats_without_0 = oof_pred_model_repeats_without_0[:, None]
        return self._oof_pred_proba / oof_pred_model_repeats_without_0

    def preprocess(self, X, model=None):
        if model is None:
            if len(self.models) == 0:
                return X
            model = self.models[0]
        if type(model) == str:
            model = self.load_child(model)
        return model.preprocess(X)

    # TODO: compute_base_preds is unused here, it is present for compatibility with StackerEnsembleModel, consider merging the two.
    def fit(self, X, y, k_fold=5, k_fold_start=0, k_fold_end=None, n_repeats=1, n_repeat_start=0, time_limit=None, **kwargs):
        if n_repeat_start != self._n_repeats_finished:
            raise ValueError('n_repeat_start must equal self._n_repeats_finished, values: (' + str(n_repeat_start) + ', ' + str(self._n_repeats_finished) + ')')
        if n_repeats <= n_repeat_start:
            raise ValueError('n_repeats must be greater than n_repeat_start, values: (' + str(n_repeats) + ', ' + str(n_repeat_start) + ')')
        if k_fold_end is None:
            k_fold_end = k_fold
        if k_fold_start != self._k_fold_end:
            raise ValueError('k_fold_start must equal previous k_fold_end, values: (' + str(k_fold_start) + ', ' + str(self._k_fold_end) + ')')
        if k_fold_start >= k_fold_end:
            # TODO: Remove this limitation if n_repeats > 1
            raise ValueError('k_fold_end must be greater than k_fold_start, values: (' + str(k_fold_end) + ', ' + str(k_fold_start) + ')')
        if (n_repeats - n_repeat_start) > 1:
            if k_fold_end != k_fold:
                # TODO: Remove this limitation
                raise ValueError('k_fold_end must equal k_fold when (n_repeats - n_repeat_start) > 1, values: (' + str(k_fold_end) + ', ' + str(k_fold) + ')')
        if self._k is not None:
            if self._k != k_fold:
                raise ValueError('k_fold must equal previously fit k_fold value for the current n_repeat, values: (' + str(k_fold) + ', ' + str(self._k) + ')')
        fold_start = n_repeat_start * k_fold + k_fold_start
        fold_end = (n_repeats-1) * k_fold + k_fold_end
        start_time = time.time()
        if self.problem_type == REGRESSION:
            stratified = False
        else:
            stratified = True

        # TODO: Preprocess data here instead of repeatedly
        kfolds = generate_kfold(X=X, y=y, n_splits=k_fold, stratified=stratified, random_state=self._random_state, n_repeats=n_repeats)

        if self.problem_type == MULTICLASS:
            oof_pred_proba = np.zeros(shape=(len(X), len(y.unique())))
        else:
            oof_pred_proba = np.zeros(shape=len(X))
        oof_pred_model_repeats = np.zeros(shape=len(X))

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
            oof_pred_model_repeats[test_index] += 1

        self.models += models

        if self.model_base is not None:
            self.save_model_base(self.model_base)
            self.model_base = None

        if self._oof_pred_proba is None:
            self._oof_pred_proba = oof_pred_proba
            self._oof_pred_model_repeats = oof_pred_model_repeats
        else:
            self._oof_pred_proba += oof_pred_proba
            self._oof_pred_model_repeats += oof_pred_model_repeats

        self._n_repeats = n_repeats
        if k_fold == k_fold_end:
            self._k = None
            self._k_fold_end = 0
            self._n_repeats_finished = self._n_repeats
        else:
            self._k = k_fold
            self._k_fold_end = k_fold_end
            self._n_repeats_finished = self._n_repeats - 1

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

    def score_with_oof(self, y):
        valid_indices = self._oof_pred_model_repeats > 0
        y = y[valid_indices]
        y_pred_proba = self.oof_pred_proba[valid_indices]

        return self.score_with_y_pred_proba(y=y, y_pred_proba=y_pred_proba)

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
