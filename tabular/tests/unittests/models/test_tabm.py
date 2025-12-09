from autogluon.tabular.models.tabm.tabm_model import TabMModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_epochs": 5}


def test_tabm():
    model_cls = TabMModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
    )
