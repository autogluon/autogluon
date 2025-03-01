from autogluon.tabular.models.knn.knn_model import KNNModel
from autogluon.tabular.testing import FitHelper


def test_knn():
    model_cls = KNNModel
    model_hyperparameters = {"n_neighbors": 2}

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
