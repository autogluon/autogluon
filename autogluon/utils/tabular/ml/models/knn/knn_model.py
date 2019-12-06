import logging

from ..sklearn.sklearn_model import SKLearnModel

logger = logging.getLogger(__name__)

# TODO: Pass in num_classes?
class KNNModel(SKLearnModel):
    def preprocess(self, X):
        cat_columns = X.select_dtypes(['category']).columns
        X = X.drop(cat_columns, axis=1).fillna(0)  # TODO: Test if crash when all columns are categorical
        return X

    # TODO: Enable HPO for KNN
    def _get_default_searchspace(self, problem_type):
        spaces = {}
        return spaces

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):
        # verbosity = kwargs.get('verbosity', 2)
        self.fit(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, **kwargs)
        hpo_model_performances = {self.name: self.score(X_test, Y_test)}
        hpo_results = {}
        self.save()
        hpo_models = {self.name: self.path}

        return hpo_models, hpo_model_performances, hpo_results
