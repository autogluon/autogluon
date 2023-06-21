import numpy as np


class DummyQuantileRegressor:
    """Adds support for quantile regression to dummy models"""

    def __init__(self, quantile_levels: list):
        self.quantile_levels = quantile_levels
        self.models = []

    def fit(self, X, y):
        from sklearn.dummy import DummyRegressor

        for quantile in self.quantile_levels:
            m = DummyRegressor(quantile=quantile)
            m.fit(X, y)
            self.models.append(m)

    def predict(self, X):
        predictions = []
        for m in self.models:
            predictions.append(m.predict(X))
        predictions = np.array(predictions).T
        return predictions
