from ..models.mqcnn.mqcnn_model import MQCNNModel


class AbstractLearner:

    def __init__(self):
        self.preset_models = {"mqcnn": MQCNNModel}
        self.model = None
        self.best_configs = {}

    def fit(self, hyperparameters, hyperparameter_tune, train_ds, test_ds, metric):
        for model, hps in hyperparameters.items():
            if model not in self.preset_models.keys():
                raise ValueError(f"Model {model} was not available yet.")
            self.model = self.preset_models[model](hyperparameters=hps)
            if hyperparameter_tune:
                self.model.hyperparameter_tune(train_data=train_ds, test_data=test_ds, metric=metric)
                best_configs = self.model.best_configs
                self.model.params = best_configs
                self.best_configs[self.model.name] = best_configs
            self.model.fit(test_ds)

    def predict(self, test_ds):
        forecasts, tss = self.model.predict(test_ds)
        return forecasts, tss

    def get_best_configs(self):
        return self.best_configs

    def score(self, test_ds):
        return self.model.score(test_ds)