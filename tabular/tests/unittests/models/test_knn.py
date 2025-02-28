from autogluon.tabular.models.knn.knn_model import KNNModel


def test_knn(fit_helper):
    model_cls = KNNModel
    model_hyperparameters = {"n_neighbors": 2}

    fit_helper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
