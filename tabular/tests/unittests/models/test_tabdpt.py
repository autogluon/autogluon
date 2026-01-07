from autogluon.tabular.models.tabdpt.tabdpt_model import TabDPTModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {}


def test_tabdpt():
    model_cls = TabDPTModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
    )
