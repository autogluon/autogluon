import logging
from .....try_import import try_import_catboost

from ..abstract.abstract_model import AbstractModel
from .hyperparameters.parameters import get_param_baseline
from .catboost_utils import construct_custom_catboost_metric
from ...constants import PROBLEM_TYPES_CLASSIFICATION
from ......core import Int, Real

logger = logging.getLogger(__name__)

# TODO: Catboost crashes on multiclass problems where only two classes have significant member count.
#  Question: Do we turn these into binary classification and then convert to multiclass output in Learner? This would make the most sense.
# TODO: Consider having Catboost variant that converts all categoricals to numerical as done in RFModel, was showing improved results in some problems.
class CatboostModel(AbstractModel):
    def __init__(self, path, name, problem_type, objective_func, hyperparameters=None, features=None, debug=0):
        super().__init__(path=path, name=name, model=None, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features, debug=debug)
        try_import_catboost()
        from catboost import CatBoostClassifier, CatBoostRegressor
        self.model_type = CatBoostClassifier if problem_type in PROBLEM_TYPES_CLASSIFICATION else CatBoostRegressor
        self.best_iteration = 0

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        self._set_default_param_value('random_seed', 0)  # Remove randomness for reproducibility
        self._set_default_param_value('eval_metric', construct_custom_catboost_metric(self.objective_func, True, not self.metric_needs_y_pred, self.problem_type))

    def _get_default_searchspace(self, problem_type):
        spaces = {
            'learning_rate': Real(lower=5e-3, upper=0.2, default=0.1, log=True),
            'depth': Int(lower=5, upper=8, default=6),
            'l2_leaf_reg': Real(lower=1, upper=5, default=3),
        }

        return spaces

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
            early_stopping_rounds = 150
        else:
            eval_set = None
            early_stopping_rounds = None

        cat_features = list(X_train.select_dtypes(include='category').columns)

        invalid_params = ['num_threads', 'num_gpus']
        for invalid in invalid_params:
            if invalid in self.params:
                self.params.pop(invalid)
        logger.log(15, 'Catboost model hyperparameters:')
        logger.log(15, self.params)

        self.model = self.model_type(
            **self.params,
        )

        # print('Catboost Model params:')
        # print(self.model.get_params())

        # TODO: Add more control over these params (specifically early_stopping_rounds)
        verbosity = kwargs.get('verbosity', 2)
        if verbosity <= 1:
            verbose = False
        elif verbosity == 2:
            verbose = False
        elif verbosity == 3:
            verbose = 20
        else:
            verbose = True

        self.model.fit(
            X_train, Y_train,
            cat_features=cat_features,
            eval_set=eval_set,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
        )

        self.best_iteration = self.model.tree_count_ - 1
