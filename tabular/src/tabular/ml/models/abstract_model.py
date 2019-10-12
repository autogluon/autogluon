
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import accuracy_score
from tabular.ml.utils import get_pred_from_proba
from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from sklearn.model_selection import RandomizedSearchCV
from tabular.utils.decorators import calculate_time
import inspect

from tabular.utils.loaders import load_pkl
from tabular.utils.savers import save_pkl


class AbstractModel:
    model_file_name = 'model.pkl'

    def __init__(self, path, name, model, problem_type=BINARY, objective_func=accuracy_score, features=None, debug=0):
        """ Creates a new model. 
            Args:
                path (str): directory where to store all outputs
                name (str): name of subdirectory inside path where model will be saved
        """
        self.name = name
        self.path = self.create_contexts(path + name + '/')
        self.model = model
        self.problem_type = problem_type
        self.objective_func = objective_func

        # TODO: Find a better way of determining this
        objective_func_args = inspect.signature(self.objective_func)
        if 'y_pred' in objective_func_args.parameters:
            self.objective_func_expects_y_pred = True
        else:  # TODO: This is present to handle things like sklearn.metrics.roc_auc_score() which are present in binary classification but expect y_pred_proba
            self.objective_func_expects_y_pred = False
        self.features = features
        self.debug = debug
        if type(model) == str:
            self.model = self.load_model(model)
        self.child_models = []
        self.params = None

    def set_contexts(self, path_context):
        self.path = self.create_contexts(path_context)

    def create_contexts(self, path_context):
        path = path_context
        return path

    def fit(self, X_train, Y_train, X_test=None, Y_test=None):
        X_train = self.preprocess(X_train)
        self.model = self.model.fit(X_train, Y_train)

    def predict(self, X, preprocess=True):
        y_pred_proba = self.predict_proba(X, preprocess=preprocess)
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        return y_pred

    def predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)
        if self.problem_type == REGRESSION:
            return self.model.predict(X)

        y_pred_proba = self.model.predict_proba(X)

        if self.problem_type == BINARY:
            if y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif y_pred_proba.shape[1] > 2:
            return y_pred_proba
        else:
            return y_pred_proba[:, 1]

    def score(self, X, y):
        if self.objective_func_expects_y_pred:
            y_pred = self.predict(X=X)
            return self.objective_func(y_true=y, y_pred=y_pred)
        else:
            y_pred_proba = self.predict_proba(X=X)
            return self.objective_func(y, y_pred_proba)

    # TODO: Add simple generic CV logic
    def cv(self, X, y, k_fold=5):
        raise NotImplementedError

    def preprocess(self, X):
        if self.features is not None:
            return X[self.features]
        return X

    def save(self):
        save_pkl.save(path=self.path + self.model_file_name, object=self)

    @classmethod
    def load(cls, path, reset_paths=False):
        load_path = path + cls.model_file_name
        if not reset_paths:
            return load_pkl.load(path=load_path)
        else:
            obj = load_pkl.load(path=load_path)
            obj.set_contexts(path)
            return obj

    # In trainer or model?
    def hyperparameter_tune(self, X, y, spaces=None):
        if spaces is None:
            print('skipping hyperparameter tuning, no spaces specified...')
            return {}
        # TODO: Make this specific to the objective func! Use https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        if self.problem_type == REGRESSION:
            scoring = 'neg_mean_absolute_error'  # self.objective_func.make_scorer()  # TODO FIXME
        else:
            scoring = 'accuracy'  # TODO FIXME

        model = copy.deepcopy(self)
        X = model.preprocess(X)
        # Set the parameters by cross-validation

        clf = RandomizedSearchCV(model.model, param_distributions=spaces, n_iter=10, cv=5,
                                 scoring=scoring)
        clf.fit(X, y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()

        self.model = self.model.__class__(**clf.best_params_)
        return clf.best_params_

    @calculate_time
    def debug_feature_gain(self, X_test, Y_test, model, features_to_use=None):
        sample_size = 10000
        if len(X_test) > sample_size:
            X_test = X_test.sample(sample_size, random_state=0)
            Y_test = Y_test.loc[X_test.index]
        else:
            X_test = X_test.copy()
            Y_test = Y_test.copy()

        X_test.reset_index(drop=True, inplace=True)
        Y_test.reset_index(drop=True, inplace=True)

        X_test = model.preprocess(X_test)

        if not features_to_use:
            features = X_test.columns.values
        else:
            features = features_to_use
        feature_count = len(features)

        model_score_base = model.score(X=X_test, y=Y_test)

        model_score_diff = []

        row_count = X_test.shape[0]
        rand_shuffle = np.random.randint(0, row_count, size=row_count)

        X_test_shuffled = X_test.iloc[rand_shuffle].reset_index(drop=True)
        compute_count = 200
        indices = [x for x in range(0, feature_count, compute_count)]

        # TODO: Make this faster by multi-threading?
        for i, indice in enumerate(indices):
            if indice + compute_count > feature_count:
                compute_count = feature_count - indice

            print(indice)
            x = [X_test.copy() for _ in range(compute_count)]  # TODO Make this much faster, only make this and concat it once. Then just update values and reset the values edited each iteration
            for j, val in enumerate(x):
                feature = features[indice+j]
                val[feature] = X_test_shuffled[feature]
            X_test_raw = pd.concat(x, ignore_index=True)
            Y_pred = model.predict(X_test_raw, preprocess=False)
            row_index = 0
            for j in range(compute_count):
                row_index_end = row_index + row_count
                Y_pred_cur = Y_pred[row_index:row_index_end]
                row_index = row_index_end
                accuracy = model.objective_func(y_true=Y_test, y_pred=Y_pred_cur)
                model_score_diff.append(model_score_base - accuracy)

        results = pd.Series(data=model_score_diff, index=features)
        results = results.sort_values(ascending=False)

        return results
        # self.save_debug()
