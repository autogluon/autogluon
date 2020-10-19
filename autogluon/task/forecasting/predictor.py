from ..base.base_predictor import BasePredictor
from .preset_configs import preset_models


class ForecastingPredictor(BasePredictor):

    def __init__(self, learner):
        self.learner = learner

    def get_preset_models(self):
        return preset_models

    def predict(self, X):
        forecasts, tss = self.learner.predict(X)
        return forecasts, tss

    def evaluate(self, dataset):
        return self.learner.score(dataset)

    def best_configs(self):
        return self.learner.get_best_configs()

    def evaluate_predictions(self, y_true, y_pred):
        pass

    @classmethod
    def load(cls, output_directory):
        pass

    def save(self, output_directory):
        pass

    def predict_proba(self, X):
        pass


