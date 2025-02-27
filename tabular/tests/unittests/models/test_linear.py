from autogluon.tabular.models.lr.lr_model import LinearModel


def test_linear(fit_helper):
    model_cls = LinearModel
    model_hyperparameters = {}

    fit_helper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
