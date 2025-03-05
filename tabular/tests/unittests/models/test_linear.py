from autogluon.tabular.models.lr.lr_model import LinearModel
from autogluon.tabular.testing import FitHelper


def test_linear():
    model_cls = LinearModel
    model_hyperparameters = {}

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
