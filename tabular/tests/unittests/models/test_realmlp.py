from autogluon.tabular.models.realmlp.realmlp_model import RealMLPModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_epochs": 5}


def test_realmlp():
    model_cls = RealMLPModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        # verify_load_wo_cuda=True,  # TODO: RealMLP doesn't yet work for GPU -> CPU
    )
