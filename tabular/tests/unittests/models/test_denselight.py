from autogluon.tabular.models.denselight.denselight_model import DenseLightModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_epochs": 2, "patience": 2, "batch_size": 32}


def test_denselight():
    model_cls = DenseLightModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
    )
