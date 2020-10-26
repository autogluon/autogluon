from gluonts.evaluation.backtest import make_evaluation_predictions
from ....utils.loaders import load_pickle
from ....utils.savers import save_pickle
import core.utils.savers.save_pkl as saver
import core.utils.loaders.load_pkl as loader


class AbstractModel:

    def __init__(self, hyperparameters=None, model=None):
        self.set_default_parameters()
        self.params = {}
        if hyperparameters is not None:
            self.params.update(hyperparameters)
        self.model = model
        self.name = None

    def save(self, path):
        saver.save(path, self)

    @classmethod
    def load(cls, path):
        return loader.load(path)

    def set_default_parameters(self):
        pass

    def create_model(self):
        pass

    def fit(self, train_ds):
        pass
        # self.model = self.model.train(train_ds)

    def predict(self, test_ds, num_samples=100):
        pass

    def hyperparameter_tune(self, train_data, test_data, metric, scheduler_options, **kwargs):
        pass

    def score(self, y, y_true):
        pass

