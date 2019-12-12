import logging
from ..sklearn.sklearn_model import SKLearnModel

logger = logging.getLogger(__name__)

# TODO: Pass in num_classes?
class RFModel(SKLearnModel):
    def preprocess(self, X):
        X = super().preprocess(X)
        X = X.fillna(0)
        return X

    # TODO: Add in documentation that Categorical default is the first index
    # TODO: enable HPO for RF models
    def _get_default_searchspace(self, problem_type):
        spaces = {
            # 'n_estimators': Int(lower=10, upper=1000, default=300),
            # 'max_features': Categorical(['auto', 0.5, 0.25]),
            # 'criterion': Categorical(['gini', 'entropy']),
        }

        return spaces

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options=None, **kwargs):

        self.fit(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, **kwargs)
        hpo_model_performances = {self.name: self.score(X_test, Y_test)}
        hpo_results = {}
        self.save()
        hpo_models = {self.name: self.path}

        return hpo_models, hpo_model_performances, hpo_results
