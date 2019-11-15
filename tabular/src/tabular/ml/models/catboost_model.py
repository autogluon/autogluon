
from tabular.ml.models.abstract_model import AbstractModel
from tabular.ml.models.utils.catboost_utils import construct_custom_catboost_metric
from catboost import CatBoostClassifier, CatBoostRegressor
from tabular.ml.constants import PROBLEM_TYPES_CLASSIFICATION


# TODO: Catboost crashes on multiclass problems where only two classes have significant member count.
#  Question: Do we turn these into binary classification and then convert to multiclass output in Learner? This would make the most sense.
class CatboostModel(AbstractModel):
    def __init__(self, path, name, problem_type, objective_func, options=None, debug=0):
        super().__init__(path=path, name=name, model=None, problem_type=problem_type, objective_func=objective_func, debug=debug)
        if options is None:
            options = {}

        if 'random_seed' not in options.keys():
            options['random_seed'] = 0  # Remove randomness for reproducibility
        options['eval_metric'] = construct_custom_catboost_metric(self.objective_func, True, not self.metric_needs_y_pred, self.problem_type)

        self.model_type = CatBoostClassifier if problem_type in PROBLEM_TYPES_CLASSIFICATION else CatBoostRegressor
        self.params = {**options}

        self.best_iteration = 0

    def preprocess(self, X):
        X = super().preprocess(X)
        categoricals = list(X.select_dtypes(include='category').columns)
        if categoricals:
            X = X.copy()
            for category in categoricals:
                current_categories = X[category].cat.categories
                if '__NaN__' in current_categories:
                    X[category] = X[category].fillna('__NaN__')
                else:
                    X[category] = X[category].cat.add_categories('__NaN__').fillna('__NaN__')
        return X

    def fit(self, X_train, Y_train, X_test=None, Y_test=None, **kwargs):
        X_train = self.preprocess(X_train)
        if X_test is not None:
            X_test = self.preprocess(X_test)
            eval_set = (X_test, Y_test)
            early_stopping_rounds = 50
        else:
            eval_set = None
            early_stopping_rounds = None

        cat_features = list(X_train.select_dtypes(include='category').columns)

        self.model = self.model_type(
            **self.params,
        )

        print('Catboost Model params:')
        print(self.model.get_params())

        # TODO: Add more control over these params (specifically verbose and early_stopping_rounds)
        self.model.fit(
            X_train, Y_train,
            cat_features=cat_features,
            eval_set=eval_set,
            verbose=True,
            early_stopping_rounds=early_stopping_rounds,
        )

        self.best_iteration = self.model.tree_count_ - 1
