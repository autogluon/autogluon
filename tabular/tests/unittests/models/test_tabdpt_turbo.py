from autogluon.tabular.models.tabdpt.tabdpt_turbo_model import TabDPTTurboModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {}


def test_tabdpt_turbo():
    model_cls = TabDPTTurboModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
        # TabDPT returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )
